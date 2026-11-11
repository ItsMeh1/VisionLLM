import { useEffect, useMemo, useRef, useState } from 'react'
import { ImageGrid } from '../ImageGrid'
import { WebcamCapture } from '../WebcamCapture'
import { Button } from '../ui'
import { IconCamera, IconPlus } from '../Icons'
import { useProject } from '../../lib/store'
import { useApp } from '../../lib/app'
import { stratifiedSplit } from '../../lib/analysis'
import { datasetDownloadUrl } from '../../lib/engine'
import { CLASS_COLORS, type ClassDef, type ImageRecord, type Split } from '../../lib/types'

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

function ClassNameField({ cls, onRename }: { cls: ClassDef; onRename: (name: string) => void }) {
  const [draft, setDraft] = useState(cls.name)
  useEffect(() => setDraft(cls.name), [cls.name])
  return (
    <input
      className="class-name-input"
      value={draft}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={() => {
        const next = draft.trim()
        if (next && next !== cls.name) onRename(next)
        else setDraft(cls.name)
      }}
      onKeyDown={(event) => {
        if (event.key === 'Enter') (event.target as HTMLInputElement).blur()
        if (event.key === 'Escape') {
          setDraft(cls.name)
          ;(event.target as HTMLInputElement).blur()
        }
      }}
      aria-label={`Name for ${cls.name}`}
    />
  )
}

export function ExamplesPanel({ onAnnotate }: { onAnnotate: (imageId?: string, classId?: string) => void }) {
  const { project, images, addFiles, addBlob, saveProject, saveImages, removeImages } = useProject()
  const { notify } = useApp()
  const [importing, setImporting] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [camera, setCamera] = useState(false)
  const [selecting, setSelecting] = useState(false)
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [filter, setFilter] = useState<'all' | 'needs' | 'train' | 'val'>('all')
  const [newClass, setNewClass] = useState('')
  const fileInput = useRef<HTMLInputElement | null>(null)
  const folderInput = useRef<HTMLInputElement | null>(null)
  const classes = project?.classes ?? []
  const task = project?.task ?? 'detect'

  const unlabelled = images.filter((image) =>
    task === 'classify' ? !image.label : image.boxes.length === 0 && image.shapes.length === 0,
  )

  const classImages = useMemo(() => {
    const map = new Map<string, ImageRecord[]>()
    for (const cls of classes) map.set(cls.id, [])
    for (const image of images) {
      if (task === 'classify') {
        if (image.label && map.has(image.label)) map.get(image.label)?.push(image)
        continue
      }
      const seen = new Set<string>()
      for (const box of image.boxes) {
        if (map.has(box.cls) && !seen.has(box.cls)) {
          seen.add(box.cls)
          map.get(box.cls)?.push(image)
        }
      }
      for (const shape of image.shapes) {
        if (map.has(shape.cls) && !seen.has(shape.cls)) {
          seen.add(shape.cls)
          map.get(shape.cls)?.push(image)
        }
      }
    }
    return map
  }, [images, classes, task])

  const classObjects = useMemo(() => {
    const map = new Map<string, number>()
    for (const cls of classes) map.set(cls.id, 0)
    for (const image of images) {
      if (task === 'classify') {
        if (image.label && map.has(image.label)) map.set(image.label, (map.get(image.label) ?? 0) + 1)
        continue
      }
      for (const box of image.boxes) if (map.has(box.cls)) map.set(box.cls, (map.get(box.cls) ?? 0) + 1)
      for (const shape of image.shapes) if (map.has(shape.cls)) map.set(shape.cls, (map.get(shape.cls) ?? 0) + 1)
    }
    return map
  }, [images, classes, task])

  const visible = useMemo(() => {
    if (filter === 'needs') return unlabelled
    if (filter === 'train') return images.filter((image) => image.split === 'train')
    if (filter === 'val') return images.filter((image) => image.split === 'val')
    return images
  }, [filter, images, unlabelled])

  if (!project) return null

  const selectedImages = images.filter((image) => selected.has(image.id))

  const importFiles = async (files: File[]) => {
    if (!files.length) return
    setImporting(true)
    try {
      const report = await addFiles(files, 'upload')
      if (report.added) {
        notify(
          `Added ${report.added} image${report.added === 1 ? '' : 's'}. Originals were kept unchanged.`,
          'ok',
        )
        if (report.records[0]) onAnnotate(report.records[0].id)
      }
      if (report.skipped.length) {
        notify(`${report.skipped.length} file${report.skipped.length === 1 ? ' was' : 's were'} skipped. ${report.skipped[0]}`, 'error')
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), 'error')
    } finally {
      setImporting(false)
    }
  }

  const pickFiles = () => {
    fileInput.current?.click()
  }

  const toggle = (id: string) => {
    setSelected((current) => {
      const next = new Set(current)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const assignSplit = async (split: Split) => {
    if (!selectedImages.length) return
    await saveImages(selectedImages.map((image) => ({ ...image, split })))
    notify(`Moved ${selectedImages.length} image${selectedImages.length === 1 ? '' : 's'} to ${split === 'train' ? 'training' : 'validation'}.`, 'ok')
    setSelected(new Set())
  }

  const assignLabel = async (classId: string) => {
    if (!selectedImages.length || !classId) return
    await saveImages(selectedImages.map((image) => ({ ...image, label: classId })))
    notify(`Labelled ${selectedImages.length} image${selectedImages.length === 1 ? '' : 's'}.`, 'ok')
    setSelected(new Set())
  }

  const removeSelected = async () => {
    if (!selectedImages.length) return
    if (!window.confirm(`Remove ${selectedImages.length} image${selectedImages.length === 1 ? '' : 's'} from this project? The browser copies are deleted; your files on disk are untouched.`)) return
    await removeImages(selectedImages.map((image) => image.id))
    setSelected(new Set())
    notify('Images removed from the project.', 'ok')
  }

  const rebalance = async () => {
    await saveImages(stratifiedSplit(project, images, 0.2))
    notify('Training and validation sets rebalanced.', 'ok')
  }

  const addClass = async () => {
    const name = newClass.trim()
    if (!name) return
    if (project.classes.some((cls) => cls.name.toLowerCase() === name.toLowerCase())) {
      notify('That class already exists.', 'error')
      return
    }
    const cls: ClassDef = {
      id: uid(),
      name,
      color: CLASS_COLORS[project.classes.length % CLASS_COLORS.length],
    }
    await saveProject({ classes: [...project.classes, cls] })
    setNewClass('')
  }

  const renameClass = async (id: string, name: string) => {
    if (project.classes.some((cls) => cls.id !== id && cls.name.toLowerCase() === name.toLowerCase())) {
      notify('Another class already uses that name.', 'error')
      return
    }
    await saveProject({ classes: project.classes.map((cls) => (cls.id === id ? { ...cls, name } : cls)) })
  }

  return (
    <section
      className={dragging ? 'card examples-card dragging' : 'card examples-card'}
      onDragOver={(event) => {
        event.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault()
        setDragging(false)
        void importFiles([...event.dataTransfer.files])
      }}
    >
      <header className="card-head">
        <div>
          <div className="card-title-row">
            <span className="step-badge">1</span>
            <h2>Examples</h2>
            {images.length ? <span className="card-count">{images.length}</span> : null}
          </div>
          <p className="card-sub">Show the model what each thing looks like.</p>
        </div>
        <div className="card-actions">
          <Button size="sm" variant="primary" onClick={pickFiles} busy={importing}>
            <IconPlus size={16} /> Add images
          </Button>
          <button type="button" className="icon-button" onClick={() => setCamera(true)} aria-label="Capture with camera" title="Capture with camera">
            <IconCamera />
          </button>
        </div>
      </header>

      <div className="class-grid">
        {project.classes.map((cls) => {
          const list = classImages.get(cls.id) ?? []
          const count = classObjects.get(cls.id) ?? 0
          return (
            <article key={cls.id} className="class-card" style={{ borderColor: `${cls.color}33` }}>
              <div className="class-card-head">
                <span className="class-dot" style={{ background: cls.color }} aria-hidden />
                <ClassNameField cls={cls} onRename={(name) => void renameClass(cls.id, name)} />
                <span className="class-count" title={project.task === 'classify' ? 'images' : 'boxes'}>
                  {count}
                </span>
              </div>
              <div className="class-thumbs">
                {list.slice(0, 4).map((image) => (
                  <button
                    key={image.id}
                    type="button"
                    className="class-thumb"
                    onClick={() => onAnnotate(image.id, cls.id)}
                    title={`Open ${image.name}`}
                  >
                    <img src={image.thumbnail} alt="" loading="lazy" />
                  </button>
                ))}
                {!list.length ? (
                  <span className="class-empty">
                    {project.task === 'classify' ? 'Label an image to add it here.' : 'Draw boxes to add examples here.'}
                  </span>
                ) : null}
              </div>
              <div className="class-card-foot">
                <button
                  type="button"
                  className="text-button"
                  onClick={() => onAnnotate(list[0]?.id, cls.id)}
                  disabled={!list.length}
                >
                  {project.task === 'classify' ? 'Review' : project.task === 'segment' ? 'Paint' : 'Draw boxes'}
                </button>
              </div>
            </article>
          )
        })}

        <form
          className="class-add-card"
          onSubmit={(event) => {
            event.preventDefault()
            void addClass()
          }}
        >
          <label className="class-add-label" htmlFor="new-class-name">
            New class
          </label>
          <div className="class-add-row">
            <input
              id="new-class-name"
              value={newClass}
              onChange={(event) => setNewClass(event.target.value)}
              placeholder="e.g. bolt"
              maxLength={40}
            />
            <button type="submit" className="icon-button filled" aria-label="Add class" disabled={!newClass.trim()}>
              <IconPlus size={16} />
            </button>
          </div>
          <span className="class-add-hint">One class per thing you want to find.</span>
        </form>
      </div>

      <div className="library-head">
        <div className="library-title">
          <strong>{images.length ? `${images.length} example${images.length === 1 ? '' : 's'}` : 'Your examples'}</strong>
          {images.length ? <span>{unlabelled.length ? `${unlabelled.length} need${unlabelled.length === 1 ? 's' : ''} work` : 'all annotated'}</span> : null}
        </div>
        <div className="library-tools">
          <div className="filter-chips" role="group" aria-label="Filter examples">
            {([
              ['all', 'All', images.length],
              ['needs', 'Needs work', unlabelled.length],
              ['train', 'Training', images.filter((image) => image.split === 'train').length],
              ['val', 'Validation', images.filter((image) => image.split === 'val').length],
            ] as const).map(([key, label, count]) => (
              <button
                key={key}
                type="button"
                className={filter === key ? 'chip active' : 'chip'}
                onClick={() => setFilter(key)}
                aria-pressed={filter === key}
              >
                {label}
                {count ? <span>{count}</span> : null}
              </button>
            ))}
          </div>
          <button
            type="button"
            className={selecting ? 'text-button on' : 'text-button'}
            onClick={() => {
              setSelecting((current) => !current)
              setSelected(new Set())
            }}
          >
            {selecting ? 'Done' : 'Select'}
          </button>
        </div>
      </div>

      {selecting && selectedImages.length ? (
        <div className="bulk-bar">
          <span>{selectedImages.length} selected</span>
          <div className="bulk-actions">
            <button type="button" className="text-button" onClick={() => void assignSplit('train')}>
              Move to training
            </button>
            <button type="button" className="text-button" onClick={() => void assignSplit('val')}>
              Move to validation
            </button>
            {project.task === 'classify' ? (
              <select
                className="mini-select"
                defaultValue=""
                onChange={(event) => {
                  if (event.target.value) void assignLabel(event.target.value)
                  event.target.value = ''
                }}
                aria-label="Assign a class"
              >
                <option value="">Label as…</option>
                {project.classes.map((cls) => (
                  <option key={cls.id} value={cls.id}>{cls.name}</option>
                ))}
              </select>
            ) : null}
            <button type="button" className="text-button danger" onClick={() => void removeSelected()}>
              Remove
            </button>
            <button type="button" className="text-button" onClick={() => setSelected(new Set())}>
              Clear
            </button>
          </div>
        </div>
      ) : null}

      {images.length ? (
        <div className="library-grid">
          <ImageGrid
            images={visible}
            classes={project.classes}
            task={project.task}
            selected={selected}
            onToggle={toggle}
            onOpen={(image) => onAnnotate(image.id)}
            mode={selecting ? 'select' : 'browse'}
            emptyLabel={filter === 'needs' ? 'Every example has an annotation. Nice.' : 'No images match this view.'}
          />
        </div>
      ) : (
        <div className="examples-empty">
          <div className="empty-art" aria-hidden>
            <span className="empty-frame" />
            <span className="empty-dot" />
            <span className="empty-plus">+</span>
          </div>
          <h3>Add your first examples</h3>
          <p>Drop a folder of photos here, or choose a few images. JPEG, PNG, WebP and GIF are welcome.</p>
          <div className="empty-actions">
            <Button variant="primary" onClick={pickFiles}>Choose images</Button>
            <Button onClick={() => setCamera(true)}>Use camera</Button>
          </div>
        </div>
      )}

      {images.length ? (
        <div className="examples-foot">
          <span>
            {images.filter((image) => image.split === 'train').length} training ·{' '}
            {images.filter((image) => image.split === 'val').length} validation
          </span>
          <details className="dataset-menu">
            <summary>Dataset settings</summary>
            <div className="dataset-menu-body">
              <button type="button" className="text-button" onClick={() => void rebalance()}>
                Rebalance split
              </button>
              <button type="button" className="text-button" onClick={() => folderInput.current?.click()}>
                Add a folder
              </button>
              {project.task === 'detect' ? (
                <a className="text-button" href={datasetDownloadUrl(project.id)} download>
                  Download YOLO annotations
                </a>
              ) : null}
              <p>Originals are stored exactly as imported. Annotations are coordinates, never painted on the image.</p>
            </div>
          </details>
        </div>
      ) : null}

      <input
        ref={fileInput}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(event) => {
          void importFiles([...(event.target.files ?? [])])
          event.target.value = ''
        }}
      />
      <input
        ref={folderInput}
        type="file"
        multiple
        hidden
        // @ts-expect-error non-standard folder picker attributes
        webkitdirectory=""
        directory=""
        onChange={(event) => {
          void importFiles([...(event.target.files ?? [])])
          event.target.value = ''
        }}
      />

      {dragging ? <div className="drop-overlay">Drop images to add them to your examples</div> : null}

      <WebcamCapture
        open={camera}
        onClose={() => setCamera(false)}
        onCapture={async (blob) => {
          const record = await addBlob(blob, `camera-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`)
          if (record) {
            notify('Camera frame added as a PNG.', 'ok')
            onAnnotate(record.id)
          }
        }}
      />
    </section>
  )
}
