import type { ClassDef, ImageRecord, Task } from '../lib/types'

export function ImageGrid({
  images,
  classes,
  task,
  selected,
  onToggle,
  onOpen,
  mode = 'select',
  emptyLabel = 'No images yet.',
}: {
  images: ImageRecord[]
  classes: ClassDef[]
  task: Task
  selected: Set<string>
  onToggle: (id: string) => void
  onOpen: (image: ImageRecord) => void
  mode?: 'browse' | 'select'
  emptyLabel?: string
}) {
  if (!images.length) return <div className="grid-empty">{emptyLabel}</div>
  const colorOf = (id?: string) => classes.find((cls) => cls.id === id)?.color ?? 'var(--line-strong)'
  return (
    <div className="image-grid">
      {images.map((image) => {
        const count = image.boxes.length + image.shapes.length
        const missing = task === 'classify' ? !image.label : count === 0
        const active = selected.has(image.id)
        const classesInImage = task === 'classify'
          ? (image.label ? [image.label] : [])
          : [...new Set([...image.boxes.map((box) => box.cls), ...image.shapes.map((shape) => shape.cls)])]
        return (
          <div
            key={image.id}
            className={active ? 'tile selected' : 'tile'}
            role="button"
            tabIndex={0}
            onClick={() => (mode === 'browse' ? onOpen(image) : onToggle(image.id))}
            onDoubleClick={() => onOpen(image)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') onOpen(image)
              if (event.key === ' ' && mode === 'select') {
                event.preventDefault()
                onToggle(image.id)
              }
            }}
            aria-label={`${image.name}, ${image.width} by ${image.height}${missing ? ', needs an example' : ''}`}
          >
            <div className="tile-well">
              <img src={image.thumbnail} alt="" loading="lazy" draggable={false} />
              <div className="tile-flags">
                <span className={`flag split-${image.split}`}>{image.split === 'val' ? 'Validation' : 'Training'}</span>
                {missing ? <span className="flag flag-missing">Needs work</span> : null}
              </div>
              {mode === 'select' ? (
                <span className={active ? 'tile-check on' : 'tile-check'} aria-hidden>
                  {active ? '✓' : ''}
                </span>
              ) : null}
              {count > 0 ? <span className="tile-count">{count}</span> : null}
            </div>
            <div className="tile-meta">
              <span className="tile-name" title={image.name}>
                {image.name}
              </span>
              <span className="tile-dots" aria-hidden>
                {classesInImage.slice(0, 4).map((cls) => (
                  <span key={cls} className="tile-dot" style={{ background: colorOf(cls) }} />
                ))}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
