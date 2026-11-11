import { useEffect, useMemo, useState } from 'react'
import { AnnotationCanvas } from '../AnnotationCanvas'
import { Button, Callout, Segmented } from '../ui'
import { IconChevron, IconX } from '../Icons'
import { useProject } from '../../lib/store'
import { useApp } from '../../lib/app'
import { autolabel, shortEngineError } from '../../lib/engine'
import type { Box, ImageRecord, Shape, Task } from '../../lib/types'

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

export function AnnotatorDialog({
  onClose,
  initialImageId,
  initialClassId,
}: {
  onClose: () => void
  initialImageId?: string
  initialClassId?: string
}) {
  const { project, images, saveImages } = useProject()
  const { online, notify } = useApp()
  const task: Task = project?.task ?? 'detect'
  const [index, setIndex] = useState(() => {
    if (initialImageId) {
      const found = images.findIndex((image) => image.id === initialImageId)
      if (found >= 0) return found
    }
    if (initialClassId) {
      const found = images.findIndex((image) =>
        task === 'classify' ? image.label === initialClassId : [...image.boxes, ...image.shapes].some((entry) => entry.cls === initialClassId),
      )
      if (found >= 0) return found
    }
    const next = images.findIndex((image) =>
      task === 'classify' ? !image.label : image.boxes.length === 0 && image.shapes.length === 0,
    )
    return Math.max(0, next)
  })
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeClass, setActiveClass] = useState(initialClassId ?? project?.classes[0]?.id ?? '')
  const [tool, setTool] = useState<'select' | 'draw'>('draw')
  const [proposals, setProposals] = useState<Box[]>([])
  const [proposalNote, setProposalNote] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const image: ImageRecord | undefined = images[Math.min(index, Math.max(0, images.length - 1))]

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    setSelectedId(null)
    setProposals([])
    setProposalNote(null)
  }, [image?.id])

  const unlabelled = useMemo(
    () =>
      images.filter((entry) =>
        task === 'classify' ? !entry.label : entry.boxes.length === 0 && entry.shapes.length === 0,
      ),
    [images, task],
  )

  if (!project || !image) return null

  const update = async (patch: Partial<ImageRecord>) => {
    await saveImages([{ ...image, ...patch }])
  }

  const removeSelected = async () => {
    if (!selectedId) return
    await update({
      boxes: image.boxes.filter((box) => box.id !== selectedId),
      shapes: image.shapes.filter((shape) => shape.id !== selectedId),
    })
    setSelectedId(null)
  }

  const changeClass = async (classId: string) => {
    if (!selectedId) return
    await update({
      boxes: image.boxes.map((box) => (box.id === selectedId ? { ...box, cls: classId } : box)),
      shapes: image.shapes.map((shape) => (shape.id === selectedId ? { ...shape, cls: classId } : shape)),
    })
  }

  const suggest = async () => {
    if (!online) {
      notify('Start the local engine to use the pretrained detector.', 'error')
      return
    }
    setBusy(true)
    setProposalNote(null)
    try {
      const response = await autolabel(
        [{ name: `${image.id}.png`, blob: image.blob }],
        project.classes.map((cls) => cls.name),
        project.autolabelModel,
        project.autolabelConf,
      )
      const result = response.results[0]
      const mapped: Box[] = result.boxes
        .map((box): Box | null => {
          const cls = project.classes[box.cls]
          if (!cls) return null
          return { id: uid(), cls: cls.id, x: box.x, y: box.y, w: box.w, h: box.h, score: box.score, source: box.source }
        })
        .filter((box): box is Box => box !== null)
      setProposals(mapped)
      setProposalNote(
        mapped.length
          ? `${mapped.length} suggestion${mapped.length === 1 ? '' : 's'} from ${result.model}. Dashed boxes are not saved until you accept them.`
          : `The pretrained detector found nothing matching ${project.classes.map((cls) => cls.name).join(', ')}.`,
      )
    } catch (error) {
      notify(shortEngineError(error), 'error')
    } finally {
      setBusy(false)
    }
  }

  const acceptProposals = async () => {
    if (!proposals.length) return
    await update({ boxes: [...image.boxes, ...proposals.map((box) => ({ ...box, id: uid() }))] })
    setProposals([])
    setProposalNote(null)
    notify('Suggestions became editable boxes.', 'ok')
  }

  const selectedBox = image.boxes.find((box) => box.id === selectedId)
  const selectedShape = image.shapes.find((shape) => shape.id === selectedId)
  const go = (next: number) => setIndex(Math.max(0, Math.min(images.length - 1, next)))
  const annotated = image.boxes.length + image.shapes.length
  const done = task === 'classify' ? Boolean(image.label) : annotated > 0

  return (
    <div
      className="annotator-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div className="annotator-modal" role="dialog" aria-modal="true" aria-label="Annotate examples">
        <header className="annotator-head">
          <div className="annotator-heading">
            <span className="step-badge">{task === 'detect' ? 'Box' : task === 'classify' ? 'Label' : 'Mask'}</span>
            <div>
              <h2>{task === 'detect' ? 'Draw around each thing' : task === 'classify' ? 'Name what is shown' : 'Paint each region'}</h2>
              <p>{project.name} · annotations live beside the original, never on it</p>
            </div>
          </div>
          <div className="annotator-classes">
            {project.classes.map((cls, classIndex) => (
              <button
                key={cls.id}
                type="button"
                className={cls.id === activeClass ? 'class-chip active' : 'class-chip'}
                style={cls.id === activeClass ? { borderColor: cls.color, background: `${cls.color}14` } : undefined}
                onClick={() => setActiveClass(cls.id)}
                title={`${cls.name} · ${classIndex + 1}`}
                aria-pressed={cls.id === activeClass}
              >
                <span className="class-dot" style={{ background: cls.color }} aria-hidden />
                {cls.name}
              </button>
            ))}
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Close annotation">
            <IconX />
          </button>
        </header>

        <div className="annotator-layout">
          <aside className="filmstrip" aria-label="Images">
            {images.map((entry, entryIndex) => {
              const count = entry.boxes.length + entry.shapes.length
              const ready = task === 'classify' ? Boolean(entry.label) : count > 0
              return (
                <button
                  key={entry.id}
                  type="button"
                  className={entry.id === image.id ? 'film-thumb active' : 'film-thumb'}
                  onClick={() => setIndex(entryIndex)}
                  title={entry.name}
                  aria-label={`${entry.name}, ${ready ? 'annotated' : 'needs work'}`}
                >
                  <img src={entry.thumbnail} alt="" loading="lazy" />
                  <span className={ready ? 'film-dot ready' : 'film-dot'} aria-hidden />
                </button>
              )
            })}
          </aside>

          <div className="annotator-main">
            <div className="annotator-toolbar">
              {task !== 'classify' ? (
                <Segmented
                  ariaLabel="Annotation tool"
                  value={tool}
                  onChange={setTool}
                  options={[
                    { value: 'draw', label: task === 'segment' ? 'Polygon' : 'Draw box' },
                    { value: 'select', label: 'Select' },
                  ]}
                />
              ) : (
                <span className="annotator-mode">One label per image</span>
              )}
              <span className="annotator-file" title={image.name}>{image.name}</span>
              <div className="annotator-tools">
                {task === 'detect' ? (
                  <Button size="sm" variant="secondary" onClick={() => void suggest()} busy={busy} disabled={!online}>
                    Suggest boxes
                  </Button>
                ) : null}
                <Button size="sm" variant="ghost" onClick={() => void removeSelected()} disabled={!selectedId}>
                  Delete
                </Button>
                <div className="stepper">
                  <button type="button" className="icon-button" onClick={() => go(index - 1)} disabled={index === 0} aria-label="Previous image">
                    <IconChevron className="flip" />
                  </button>
                  <span>
                    {index + 1} / {images.length}
                  </span>
                  <button type="button" className="icon-button" onClick={() => go(index + 1)} disabled={index >= images.length - 1} aria-label="Next image">
                    <IconChevron />
                  </button>
                </div>
              </div>
            </div>

            <div className={task === 'classify' ? 'annotator-stage classify' : 'annotator-stage'}>
              {task === 'classify' ? (
                <div className="classify-stage">
                  <img src={image.thumbnail} alt={image.name} />
                  <div className="classify-choices">
                    <p className="classify-question">What does this image show?</p>
                    <div className="choice-grid">
                      {project.classes.map((cls) => (
                        <button
                          key={cls.id}
                          type="button"
                          className={image.label === cls.id ? 'choice active' : 'choice'}
                          onClick={() => void update({ label: cls.id })}
                          style={image.label === cls.id ? { borderColor: cls.color, boxShadow: `inset 0 -3px 0 ${cls.color}` } : undefined}
                        >
                          <span className="class-dot" style={{ background: cls.color }} aria-hidden />
                          {cls.name}
                        </button>
                      ))}
                    </div>
                    <p className="field-hint">The whole frame is the input. No boxes, no cropping.</p>
                  </div>
                </div>
              ) : (
                <AnnotationCanvas
                  image={image}
                  classes={project.classes}
                  task={task}
                  tool={tool}
                  activeClass={activeClass}
                  selectedId={selectedId}
                  onSelect={setSelectedId}
                  onChange={(patch) => void update(patch)}
                  onToolChange={setTool}
                  proposals={proposals}
                  showProposals={proposals.length > 0}
                />
              )}
            </div>

            <div className="annotator-foot">
              <span>{image.width} × {image.height} · original</span>
              <span>{done ? 'This example is ready' : 'Not annotated yet'}</span>
            </div>
          </div>

          <aside className="annotator-side">
            {proposalNote ? (
              <Callout
                tone="warn"
                title="Review suggestions"
                action={proposals.length ? <Button size="sm" variant="primary" onClick={() => void acceptProposals()}>Accept all</Button> : undefined}
              >
                <p>{proposalNote}</p>
                {proposals.length ? (
                  <button type="button" className="text-button" onClick={() => setProposals([])}>
                    Discard
                  </button>
                ) : null}
              </Callout>
            ) : null}

            <section className="annotator-section">
              <h3>{task === 'classify' ? 'Label' : task === 'segment' ? 'Regions' : 'Boxes'} on this image</h3>
              {task === 'classify' ? (
                <p className="annotator-note">
                  {image.label ? project.classes.find((cls) => cls.id === image.label)?.name : 'Not labelled yet'}
                </p>
              ) : (
                <ul className="annotation-list">
                  {[...image.shapes, ...image.boxes].map((entry) => {
                    const cls = project.classes.find((item) => item.id === entry.cls)
                    const isShape = 'points' in entry
                    return (
                      <li key={entry.id}>
                        <button
                          type="button"
                          className={entry.id === selectedId ? 'annotation-row active' : 'annotation-row'}
                          onClick={() => {
                            setSelectedId(entry.id)
                            setTool('select')
                          }}
                        >
                          <span className="class-dot" style={{ background: cls?.color }} aria-hidden />
                          <span>{cls?.name ?? 'unassigned'}</span>
                          <span className="annotation-kind">
                            {isShape ? `${(entry as Shape).points.length} pts` : entry.score ? `${Math.round((entry.score ?? 0) * 100)}%` : 'manual'}
                          </span>
                        </button>
                      </li>
                    )
                  })}
                  {!image.boxes.length && !image.shapes.length ? (
                    <li className="annotation-empty">Nothing drawn yet. Drag on the image to add a box.</li>
                  ) : null}
                </ul>
              )}
            </section>

            {selectedBox || selectedShape ? (
              <section className="annotator-section">
                <h3>Selected</h3>
                <label className="field">
                  <span className="field-label">Class</span>
                  <select value={(selectedBox ?? selectedShape)?.cls} onChange={(event) => void changeClass(event.target.value)}>
                    {project.classes.map((cls) => (
                      <option key={cls.id} value={cls.id}>{cls.name}</option>
                    ))}
                  </select>
                </label>
                <p className="annotator-note">
                  {selectedBox
                    ? `${(selectedBox.w * 100).toFixed(0)}% wide · ${(selectedBox.h * 100).toFixed(0)}% tall`
                    : `${selectedShape?.points.length} points`}
                </p>
                <button type="button" className="text-button danger" onClick={() => void removeSelected()}>
                  Delete annotation
                </button>
              </section>
            ) : null}

            <section className="annotator-section">
              <h3>Progress</h3>
              <p className="annotator-note">
                {unlabelled.length
                  ? `${unlabelled.length} image${unlabelled.length === 1 ? '' : 's'} still need${unlabelled.length === 1 ? 's' : ''} work.`
                  : 'Every image has an annotation.'}
              </p>
              {unlabelled.length ? (
                <button
                  type="button"
                  className="text-button"
                  onClick={() => {
                    const next = images.findIndex((entry) =>
                      task === 'classify' ? !entry.label : entry.boxes.length === 0 && entry.shapes.length === 0,
                    )
                    if (next >= 0) setIndex(next)
                  }}
                >
                  Jump to next
                </button>
              ) : null}
              <p className="annotator-hintline">
                {task === 'segment'
                  ? 'Click points around a region, then press Enter or click the first point to close it.'
                  : 'Tip: draw around every object. In Select, dragging empty space starts another box.'}
              </p>
            </section>
          </aside>
        </div>
      </div>
    </div>
  )
}
