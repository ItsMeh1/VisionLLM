import { useEffect, useMemo, useRef, useState } from 'react'
import { ImageGrid } from '../ImageGrid'
import { WebcamCapture } from '../WebcamCapture'
import { Button } from '../ui'
import { IconCamera, IconPlus } from '../Icons'
import { useProject } from '../../lib/store'
import { useApp } from '../../lib/app'
import { datasetDownloadUrl } from '../../lib/engine'
import { CLASS_COLORS, type ClassDef, type ImageRecord } from '../../lib/types'

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

const MIME_EXTENSION: Record<string, string> = {
  'image/avif': 'avif',
  'image/bmp': 'bmp',
  'image/gif': 'gif',
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/svg+xml': 'svg',
  'image/tiff': 'tif',
  'image/webp': 'webp',
}
const DROP_IMAGE_EXT = /\.(?:jpe?g|png|webp|gif|bmp|tiff?|avif)(?:$|[?#])/i
const MAX_DROPPED_URLS = 20

function extensionFor(mime: string): string {
  return MIME_EXTENSION[mime] ?? 'png'
}

function droppedImageName(rawUrl: string, mime: string): string {
  const fallback = `dropped-image.${extensionFor(mime)}`
  try {
    const url = new URL(rawUrl)
    if (url.protocol === 'data:' || url.protocol === 'blob:') return fallback
    const base = decodeURIComponent(url.pathname.split('/').filter(Boolean).pop() ?? '').replace(/[/\\]/g, '-')
    if (!base) return fallback
    if (DROP_IMAGE_EXT.test(base)) return base
    return `${base.replace(/\.[^.]+$/, '') || 'dropped-image'}.${extensionFor(mime)}`
  } catch {
    return fallback
  }
}

function filesFromTransfer(transfer: DataTransfer): File[] {
  const files = transfer.files ? Array.from(transfer.files) : []
  if (files.length) return files
  for (const item of Array.from(transfer.items ?? [])) {
    if (item.kind !== 'file') continue
    const file = item.getAsFile()
    if (file) files.push(file)
  }
  return files
}

function imageUrlsFromTransfer(transfer: DataTransfer): string[] {
  const shared: Array<{ value: string; allowRelative: boolean }> = []
  const htmlImages: Array<{ value: string; allowRelative: boolean }> = []
  const read = (format: string) => {
    try {
      return transfer.getData(format)
    } catch {
      return ''
    }
  }

  for (const format of ['text/uri-list', 'text/plain', 'text/x-moz-url']) {
    for (const value of read(format).split(/\r?\n/)) shared.push({ value, allowRelative: false })
  }

  const html = read('text/html')
  if (html) {
    try {
      const doc = new DOMParser().parseFromString(html, 'text/html')
      for (const image of Array.from(doc.querySelectorAll('img'))) {
        const src =
          image.getAttribute('src') ?? image.getAttribute('srcset')?.split(',')[0]?.trim().split(/\s+/)[0]
        if (src) htmlImages.push({ value: src, allowRelative: true })
      }
    } catch {
      // Ignore malformed HTML fragments and fall through to the shared URL formats.
    }
  }

  // A dragged image wrapped in a link can put the page URL in text/uri-list. The
  // HTML fragment is more specific, so use its image sources when they exist.
  const candidates = htmlImages.length ? htmlImages : shared
  const seen = new Set<string>()
  const urls: string[] = []
  for (const { value, allowRelative } of candidates) {
    if (urls.length >= MAX_DROPPED_URLS) break
    const raw = value.trim()
    if (!raw || raw.startsWith('#')) continue
    if (!allowRelative && !/^(https?:\/\/|data:image\/|blob:)/i.test(raw)) continue
    let url: URL
    try {
      url = new URL(raw, window.location.href)
    } catch {
      continue
    }
    if (!['http:', 'https:', 'data:', 'blob:'].includes(url.protocol)) continue
    if (!seen.has(url.href)) {
      seen.add(url.href)
      urls.push(url.href)
    }
  }
  return urls
}

async function downloadDroppedImage(url: string): Promise<File> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), 20_000)
  try {
    const response = await fetch(url, { signal: controller.signal, credentials: 'same-origin' })
    if (!response.ok) throw new Error(`The image request failed with HTTP ${response.status}.`)
    const blob = await response.blob()
    const mime = ((response.headers.get('content-type') ?? blob.type) || '').split(';')[0].trim().toLowerCase()
    if (!mime.startsWith('image/')) throw new Error('That link did not return an image.')
    return new File([blob], droppedImageName(response.url || url, mime), { type: mime })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error('The image took too long to load.')
    }
    if (error instanceof TypeError) {
      throw new Error('The other site did not allow this app to read the image. Save it and drop the file instead.')
    }
    throw error
  } finally {
    window.clearTimeout(timer)
  }
}

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
  const [filter, setFilter] = useState<'all' | 'needs'>('all')
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

  const visible = useMemo(() => (filter === 'needs' ? unlabelled : images), [filter, images, unlabelled])

  if (!project) return null

  const selectedImages = images.filter((image) => selected.has(image.id))

  const importFiles = async (files: File[], errors: string[] = []) => {
    if (!files.length) {
      if (errors.length) {
        notify(errors.length === 1 ? errors[0] : `${errors.length} images could not be added. ${errors[0]}`, 'error')
      }
      return
    }
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
      if (errors.length) {
        notify(errors.length === 1 ? errors[0] : `${errors.length} images could not be added. ${errors[0]}`, 'error')
      }
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), 'error')
    } finally {
      setImporting(false)
    }
  }

  const importDrop = async (transfer: DataTransfer) => {
    const files = filesFromTransfer(transfer)
    if (files.length) {
      await importFiles(files)
      return
    }

    const urls = imageUrlsFromTransfer(transfer)
    if (!urls.length) {
      notify('That drag did not include an image. Drag the image itself, or save it and drop the file.', 'error')
      return
    }

    setImporting(true)
    const fetched: File[] = []
    const failures: string[] = []
    try {
      await Promise.all(
        urls.map(async (url) => {
          try {
            fetched.push(await downloadDroppedImage(url))
          } catch (error) {
            failures.push(error instanceof Error ? error.message : String(error))
          }
        }),
      )
      await importFiles(fetched, failures)
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
        void importDrop(event.dataTransfer)
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
          <p>
            Drop a folder of photos here, choose a few images, or drag an image straight from another tab. JPEG, PNG,
            WebP and GIF are welcome.
          </p>
          <div className="empty-actions">
            <Button variant="primary" onClick={pickFiles}>Choose images</Button>
            <Button onClick={() => setCamera(true)}>Use camera</Button>
          </div>
        </div>
      )}

      {images.length ? (
        <div className="examples-foot">
          <span>Your images stay together in one library.</span>
          <details className="dataset-menu">
            <summary>Library tools</summary>
            <div className="dataset-menu-body">
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
