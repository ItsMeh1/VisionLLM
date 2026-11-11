import { useMemo, useRef, useState } from 'react'
import { Badge, Button, Callout, Panel, PanelHead, Segmented } from '../components/ui'
import { ImageGrid } from '../components/ImageGrid'
import { ModelInputPreview } from '../components/ModelInputPreview'
import { WebcamCapture } from '../components/WebcamCapture'
import { useProject } from '../lib/store'
import { useApp } from '../lib/app'
import { stratifiedSplit } from '../lib/analysis'
import { datasetDownloadUrl } from '../lib/engine'
import { formatBytes } from '../lib/image'
import { navigate } from '../lib/router'
import type { Split } from '../lib/types'

export function DatasetScreen() {
  const { project, images, addFiles, addBlob, removeImages, saveImages } = useProject()
  const { notify, online } = useApp()
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [skipped, setSkipped] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [camera, setCamera] = useState(false)
  const [filter, setFilter] = useState<'all' | 'unlabelled' | 'train' | 'val'>('all')
  const [labelChoice, setLabelChoice] = useState('')
  const fileInput = useRef<HTMLInputElement | null>(null)
  const folderInput = useRef<HTMLInputElement | null>(null)

  if (!project) return null

  const selectedImages = images.filter((image) => selected.has(image.id))
  const focus = selectedImages[0] ?? images[0] ?? null
  const unlabelledCount = images.filter((image) =>
    project.task === 'classify' ? !image.label : image.boxes.length === 0 && image.shapes.length === 0,
  ).length

  const visible = useMemo(() => {
    if (filter === 'unlabelled') {
      return images.filter((image) =>
        project.task === 'classify' ? !image.label : image.boxes.length === 0 && image.shapes.length === 0,
      )
    }
    if (filter === 'train' || filter === 'val') return images.filter((image) => image.split === filter)
    return images
  }, [images, filter, project.task])

  const importFiles = async (files: File[], origin: 'upload' | 'folder') => {
    if (!files.length) return
    setBusy(true)
    try {
      const report = await addFiles(files, origin)
      if (report.added) notify(`Imported ${report.added} images. Originals were kept unchanged.`, 'ok')
      setSkipped(report.skipped)
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), 'error')
    } finally {
      setBusy(false)
    }
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
    if (!selected.size) return
    await saveImages(selectedImages.map((image) => ({ ...image, split })))
    notify(`${selected.size} images moved to ${split === 'train' ? 'training' : 'validation'}.`, 'ok')
    setSelected(new Set())
  }

  const assignLabel = async (classId: string) => {
    if (!selected.size || !classId) return
    await saveImages(selectedImages.map((image) => ({ ...image, label: classId })))
    notify(`Labelled ${selected.size} images.`, 'ok')
    setSelected(new Set())
  }

  const rebalance = async () => {
    const next = stratifiedSplit(project, images, 0.2)
    await saveImages(next)
    notify('Split rebalanced: 20% held out for validation, stratified by class.', 'ok')
  }

  const removeSelected = async () => {
    if (!selected.size) return
    if (!window.confirm(`Remove ${selected.size} image${selected.size === 1 ? '' : 's'} from this project? The browser copies are deleted; your original files on disk are not touched.`)) return
    await removeImages([...selected])
    setSelected(new Set())
    notify('Images removed from the project.', 'ok')
  }

  return (
    <div className="screen dataset-screen">
      <Panel className="import-panel">
        <PanelHead
          eyebrow="01 · Dataset"
          title="Bring in images"
          note="Files are read into this browser only. Nothing is re-encoded or resized, and the trainer receives a separate copy."
          actions={
            <div className="button-row">
              <Button size="sm" onClick={() => fileInput.current?.click()} busy={busy}>
                Choose images
              </Button>
              <Button size="sm" onClick={() => folderInput.current?.click()} busy={busy}>
                Choose folder
              </Button>
              <Button size="sm" variant="ghost" onClick={() => setCamera(true)}>
                Camera
              </Button>
            </div>
          }
        />
        <div
          className={dragging ? 'dropzone dragging' : 'dropzone'}
          onDragOver={(event) => {
            event.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault()
            setDragging(false)
            void importFiles([...event.dataTransfer.files], 'upload')
          }}
          onClick={() => fileInput.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter') fileInput.current?.click()
          }}
        >
          <span className="dropzone-mark" aria-hidden>＋</span>
          <span>
            <strong>Drop images or a folder here</strong>
            <span className="dropzone-sub mono">JPEG · PNG · WebP · BMP · GIF · TIFF</span>
          </span>
        </div>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          multiple
          hidden
          onChange={(event) => {
            void importFiles([...(event.target.files ?? [])], 'upload')
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
            void importFiles([...(event.target.files ?? [])], 'folder')
            event.target.value = ''
          }}
        />
        {skipped.length ? (
          <div className="import-log">
            <span className="label">Not imported</span>
            <ul>
              {skipped.slice(0, 6).map((line) => (
                <li key={line} className="mono">{line}</li>
              ))}
            </ul>
            {skipped.length > 6 ? <span className="mono">+{skipped.length - 6} more</span> : null}
          </div>
        ) : null}
      </Panel>

      <div className="dataset-columns">
        <Panel className="grid-panel">
          <PanelHead
            eyebrow="Library"
            title={`${images.length} image${images.length === 1 ? '' : 's'}`}
            actions={
              <Segmented
                ariaLabel="Filter images"
                value={filter}
                onChange={setFilter}
                options={[
                  { value: 'all', label: 'All' },
                  { value: 'unlabelled', label: `Unlabelled ${unlabelledCount}` },
                  { value: 'train', label: 'Train' },
                  { value: 'val', label: 'Validation' },
                ]}
              />
            }
          />
          {selected.size ? (
            <div className="bulk-bar">
              <span className="mono">{selected.size} selected</span>
              <div className="button-row">
                <Button size="sm" variant="ghost" onClick={() => void assignSplit('train')}>
                  Train
                </Button>
                <Button size="sm" variant="ghost" onClick={() => void assignSplit('val')}>
                  Validation
                </Button>
                {project.task === 'classify' ? (
                  <div className="inline-select">
                    <select value={labelChoice} onChange={(event) => setLabelChoice(event.target.value)} aria-label="Assign class">
                      <option value="">Assign class…</option>
                      {project.classes.map((cls) => (
                        <option key={cls.id} value={cls.id}>{cls.name}</option>
                      ))}
                    </select>
                    <Button size="sm" variant="ghost" onClick={() => void assignLabel(labelChoice)} disabled={!labelChoice}>
                      Set
                    </Button>
                  </div>
                ) : null}
                <Button size="sm" variant="ghost" onClick={() => setSelected(new Set())}>
                  Clear
                </Button>
                <Button size="sm" variant="danger" onClick={() => void removeSelected()}>
                  Remove
                </Button>
              </div>
            </div>
          ) : null}
          <ImageGrid
            images={visible}
            classes={project.classes}
            task={project.task}
            selected={selected}
            onToggle={toggle}
            onOpen={() => navigate(`/p/${project.id}/annotate`)}
            emptyLabel={images.length ? 'No images match this filter.' : 'Import a few images to begin.'}
          />
        </Panel>

        <aside className="dataset-side">
          <Panel>
            <div className="label">Preparation</div>
            <div className="mini-stats">
              <div>
                <span className="num">{images.filter((image) => image.split === 'train').length}</span>
                <span className="label">train</span>
              </div>
              <div>
                <span className="num">{images.filter((image) => image.split === 'val').length}</span>
                <span className="label">validation</span>
              </div>
              <div>
                <span className="num">{unlabelledCount}</span>
                <span className="label">unlabelled</span>
              </div>
            </div>
            <Button size="sm" onClick={() => void rebalance()} disabled={images.length < 3} className="block">
              Rebalance split (80 / 20)
            </Button>
            <p className="field-hint">
              Stratified by class, so rare classes appear in both sides. Validation images are never trained on.
            </p>
          </Panel>

          <Panel>
            <div className="label">Model input preview</div>
            {focus ? (
              <>
                <ModelInputPreview
                  blob={focus.blob}
                  size={project.task === 'detect' ? 512 : project.task === 'segment' ? 384 : 224}
                  boxes={project.task === 'detect' ? focus.boxes : []}
                  classes={project.classes}
                  caption="What the network sees"
                />
                <dl className="meta-list">
                  <div><dt>Original</dt><dd className="mono">{focus.width}×{focus.height}</dd></div>
                  <div><dt>File</dt><dd className="mono">{formatBytes(focus.bytes)}</dd></div>
                  <div><dt>Name</dt><dd title={focus.name}>{focus.name}</dd></div>
                  <div><dt>Split</dt><dd><Badge tone={focus.split === 'val' ? 'info' : 'neutral'}>{focus.split}</Badge></dd></div>
                </dl>
              </>
            ) : (
              <p className="field-hint">Import an image to compare the original with the letterboxed model input.</p>
            )}
          </Panel>

          <Callout tone="info" title="Originals are never modified">
            Every image is stored as the exact bytes you imported. Boxes and masks are stored separately as coordinates,
            so you can change an annotation without touching a pixel.
          </Callout>

          {project.task === 'detect' ? (
            <Panel>
              <div className="label">Interoperability</div>
              <p className="field-hint">Download the annotations as YOLO text files with a data.yaml and the original images.</p>
              <a className="text-link" href={datasetDownloadUrl(project.id)} download>
                dataset-yolo.zip ↓
              </a>
            </Panel>
          ) : null}

          {!online ? (
            <Callout tone="warn" title="Training engine offline">
              Browsing and annotation work without it. Start <span className="mono">npm run engine</span> before training.
            </Callout>
          ) : null}

          <a className="text-link" href={`#/p/${project.id}/annotate`}>
            Next: {project.task === 'detect' ? 'draw boxes' : project.task === 'classify' ? 'label images' : 'paint masks'} →
          </a>
        </aside>
      </div>

      <WebcamCapture
        open={camera}
        onClose={() => setCamera(false)}
        onCapture={async (blob) => {
          const record = await addBlob(blob, `camera-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.png`)
          if (record) notify('Frame captured and added as a PNG.', 'ok')
        }}
      />
    </div>
  )
}
