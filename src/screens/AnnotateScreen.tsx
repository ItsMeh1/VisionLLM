import { useEffect, useMemo, useRef, useState } from 'react'
import { AnnotationCanvas } from '../components/AnnotationCanvas'
import { ModelInputPreview } from '../components/ModelInputPreview'
import { Button, Callout, Panel, Segmented } from '../components/ui'
import { useProject } from '../lib/store'
import { useApp } from '../lib/app'
import { autolabel, shortEngineError } from '../lib/engine'
import { navigate } from '../lib/router'
import type { Box, ImageRecord, Shape, Task } from '../lib/types'
import { TASK_META } from '../lib/types'

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

export function AnnotateScreen() {
  const { project: stored, images, saveImages } = useProject()
  const { online, notify } = useApp()
  const project = stored as NonNullable<typeof stored>
  const classes = project?.classes ?? []
  const task: Task = project?.task ?? 'detect'
  const [index, setIndex] = useState(0)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeClass, setActiveClass] = useState(classes[0]?.id ?? '')
  const [tool, setTool] = useState<'select' | 'draw'>('draw')
  const [proposals, setProposals] = useState<Box[]>([])
  const [proposalNote, setProposalNote] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const stripRef = useRef<HTMLDivElement | null>(null)

  const image: ImageRecord | undefined = images[Math.min(index, Math.max(0, images.length - 1))]

  useEffect(() => {
    if (!activeClass && classes.length) setActiveClass(classes[0].id)
  }, [classes, activeClass])

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

  const update = async (patch: Partial<ImageRecord>) => {
    if (!image) return
    await saveImages([{ ...image, ...patch }])
  }

  const removeSelected = async () => {
    if (!image || !selectedId) return
    await update({
      boxes: image.boxes.filter((box) => box.id !== selectedId),
      shapes: image.shapes.filter((shape) => shape.id !== selectedId),
    })
    setSelectedId(null)
  }

  const changeClass = async (classId: string) => {
    if (!image || !selectedId) return
    await update({
      boxes: image.boxes.map((box) => (box.id === selectedId ? { ...box, cls: classId } : box)),
      shapes: image.shapes.map((shape) => (shape.id === selectedId ? { ...shape, cls: classId } : shape)),
    })
  }

  const suggest = async () => {
    if (!image) return
    if (!online) {
      notify('Start the local engine to use the pretrained detector.', 'error')
      return
    }
    setBusy(true)
    setProposalNote(null)
    try {
      const response = await autolabel(
        [{ name: `${image.id}.png`, blob: image.blob }],
        classes.map((cls) => cls.name),
        project.autolabelModel,
        project.autolabelConf,
      )
      const result = response.results[0]
      const mapped: Box[] = result.boxes
        .map((box): Box | null => {
          const cls = classes[box.cls]
          if (!cls) return null
          return { id: uid(), cls: cls.id, x: box.x, y: box.y, w: box.w, h: box.h, score: box.score, source: box.source }
        })
        .filter((box): box is Box => box !== null)
      setProposals(mapped)
      setProposalNote(
        mapped.length
          ? `${mapped.length} proposals from ${result.model}. Dashed boxes are not saved until you accept them.`
          : `The pretrained detector found nothing matching ${classes.map((cls) => cls.name).join(', ')}. It saw: ${
              Object.keys(result.detected_categories).slice(0, 5).join(', ') || 'nothing'
            }.`,
      )
    } catch (error) {
      notify(shortEngineError(error), 'error')
    } finally {
      setBusy(false)
    }
  }

  const acceptProposals = async () => {
    if (!image || !proposals.length) return
    await update({ boxes: [...image.boxes, ...proposals.map((box) => ({ ...box, id: uid() }))] })
    setProposals([])
    setProposalNote(null)
    notify('Proposals became editable boxes.', 'ok')
  }

  if (!project) return null

  if (!images.length) {
    return (
      <div className="screen">
        <Panel>
          <div className="empty-block">
            <h3>No images to annotate yet</h3>
            <p className="empty-copy">Import a few images first; annotation happens on your originals.</p>
            <Button variant="primary" onClick={() => navigate(`/p/${project.id}/dataset`)}>
              Go to dataset
            </Button>
          </div>
        </Panel>
      </div>
    )
  }

  const selectedBox = image?.boxes.find((box) => box.id === selectedId)
  const selectedShape = image?.shapes.find((shape) => shape.id === selectedId)
  return (
    <div className="screen annotate-screen">
      <div className="annotate-toolbar">
        <div className="toolbar-group">
          <Segmented
            ariaLabel="Annotation tool"
            value={tool}
            onChange={setTool}
            options={[
              { value: 'select', label: 'Select' },
              { value: 'draw', label: task === 'segment' ? 'Polygon' : 'Draw box' },
            ]}
          />
        </div>
        <div className="class-strip" role="group" aria-label="Active class">
          {classes.map((cls, classIndex) => (
            <button
              key={cls.id}
              type="button"
              className={cls.id === activeClass ? 'class-chip active' : 'class-chip'}
              onClick={() => {
                setActiveClass(cls.id)
                if (selectedId) void changeClass(cls.id)
              }}
              title={`${cls.name} · press ${classIndex + 1}`}
              aria-pressed={cls.id === activeClass}
            >
              <span className="swatch" style={{ background: cls.color }} />
              {cls.name}
            </button>
          ))}
        </div>
        <div className="toolbar-actions">
          <Button size="sm" variant="ghost" onClick={() => void removeSelected()} disabled={!selectedId}>
            Delete
          </Button>
          <Button size="sm" onClick={() => void suggest()} busy={busy} disabled={task !== 'detect' || !online}>
            Suggest boxes
          </Button>
        </div>
      </div>

      <div className="annotate-body">
        <div className="annotate-strip" ref={stripRef}>
          {images.map((entry, entryIndex) => {
            const count = entry.boxes.length + entry.shapes.length
            return (
              <button
                key={entry.id}
                type="button"
                className={entry.id === image?.id ? 'strip-thumb active' : 'strip-thumb'}
                onClick={() => setIndex(entryIndex)}
                title={entry.name}
                aria-label={`${entry.name}, ${count} annotations`}
              >
                <img src={entry.thumbnail} alt="" loading="lazy" />
                <span className={count || entry.label ? 'strip-dot has' : 'strip-dot'} />
              </button>
            )
          })}
        </div>

        <div className="annotate-stage">
          {image ? (
            task === 'classify' ? (
              <div className="classify-stage">
                <img src={image.thumbnail} alt={image.name} />
                <div className="classify-choices">
                  <div className="label">What does this image show?</div>
                  <div className="choice-row">
                    {classes.map((cls) => (
                      <button
                        key={cls.id}
                        type="button"
                        className={image.label === cls.id ? 'choice active' : 'choice'}
                        onClick={() => void update({ label: cls.id })}
                        style={image.label === cls.id ? { borderColor: cls.color, boxShadow: `inset 0 -2px 0 ${cls.color}` } : undefined}
                      >
                        <span className="swatch" style={{ background: cls.color }} />
                        {cls.name}
                      </button>
                    ))}
                  </div>
                  <p className="field-hint">One label per image. No boxes, no cropping — the whole frame is the input.</p>
                </div>
              </div>
            ) : (
              <AnnotationCanvas
                image={image}
                classes={classes}
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
            )
          ) : null}
          <div className="annotate-footbar">
            <div className="image-meta mono">
              <span>{image?.name}</span>
              <span>{image?.width}×{image?.height}</span>
              <span>original · untouched</span>
            </div>
            <div className="nav-buttons">
              <Button size="sm" variant="ghost" onClick={() => setIndex(Math.max(0, index - 1))} disabled={index === 0}>
                ← Prev
              </Button>
              <span className="mono page-count">{index + 1} / {images.length}</span>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setIndex(Math.min(images.length - 1, index + 1))}
                disabled={index >= images.length - 1}
              >
                Next →
              </Button>
            </div>
          </div>
        </div>

        <aside className="annotate-side">
          {proposalNote ? (
            <Callout
              tone="warn"
              title="Review proposals"
              action={
                proposals.length ? (
                  <Button size="sm" variant="primary" onClick={() => void acceptProposals()}>
                    Accept all
                  </Button>
                ) : undefined
              }
            >
              <p>{proposalNote}</p>
              {proposals.length ? (
                <button type="button" className="text-link" onClick={() => setProposals([])}>
                  Discard
                </button>
              ) : null}
            </Callout>
          ) : null}

          <Panel>
            <div className="label">
              {task === 'segment' ? 'Masks' : task === 'classify' ? 'Label' : 'Boxes'} on this image
            </div>
            <ul className="annotation-list">
              {task !== 'classify'
                ? [...(image?.shapes ?? []), ...(image?.boxes ?? [])].map((entry) => {
                    const cls = classes.find((item) => item.id === entry.cls)
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
                          <span className="swatch" style={{ background: cls?.color }} />
                          <span>{cls?.name ?? 'unassigned'}</span>
                          <span className="mono">{isShape ? `${(entry as Shape).points.length} pts` : entry.score ? `${Math.round((entry.score ?? 0) * 100)}%` : 'manual'}</span>
                        </button>
                      </li>
                    )
                  })
                : null}
              {task === 'classify' ? (
                <li className="annotation-empty mono">
                  {image?.label ? classes.find((cls) => cls.id === image.label)?.name : 'not labelled yet'}
                </li>
              ) : null}
              {task !== 'classify' && !image?.boxes.length && !image?.shapes.length ? (
                <li className="annotation-empty mono">nothing drawn yet</li>
              ) : null}
            </ul>
          </Panel>

          {selectedBox || selectedShape ? (
            <Panel>
              <div className="label">Selected</div>
              <label className="field">
                <span className="label">Class</span>
                <select value={(selectedBox ?? selectedShape)?.cls} onChange={(event) => void changeClass(event.target.value)}>
                  {classes.map((cls) => (
                    <option key={cls.id} value={cls.id}>{cls.name}</option>
                  ))}
                </select>
              </label>
              <div className="mono selection-meta">
                {selectedBox
                  ? `x ${(selectedBox.x * 100).toFixed(1)}% · y ${(selectedBox.y * 100).toFixed(1)}% · w ${(selectedBox.w * 100).toFixed(1)}% · h ${(selectedBox.h * 100).toFixed(1)}%`
                  : `${selectedShape?.points.length} points`}
              </div>
              <Button size="sm" variant="danger" onClick={() => void removeSelected()} className="block">
                Delete annotation
              </Button>
            </Panel>
          ) : null}

          {task === 'detect' && image ? (
            <Panel>
              <div className="label">Preview</div>
              <ModelInputPreview blob={image.blob} size={512} boxes={image.boxes} classes={classes} caption="Training view" />
            </Panel>
          ) : null}

          <Panel>
            <div className="label">Progress</div>
            <p className="field-hint">
              {unlabelled.length
                ? `${unlabelled.length} image${unlabelled.length === 1 ? '' : 's'} still unlabelled.`
                : 'Every image has an annotation.'}
            </p>
            {unlabelled.length ? (
              <Button
                size="sm"
                variant="ghost"
                className="block"
                onClick={() => {
                  const next = images.findIndex((entry) =>
                    task === 'classify' ? !entry.label : entry.boxes.length === 0 && entry.shapes.length === 0,
                  )
                  if (next >= 0) setIndex(next)
                }}
              >
                Jump to next unlabelled
              </Button>
            ) : null}
            {task === 'segment' ? (
              <p className="field-hint">Click points around a region, then press Enter or click the first point to close it.</p>
            ) : null}
            <div className="keyboard-hints">
              {classes.slice(0, 9).map((cls, classIndex) => (
                <span key={cls.id} className="mono">
                  {classIndex + 1} {cls.name}
                </span>
              ))}
            </div>
          </Panel>

          <Callout tone="info" title={TASK_META[task].labelStep}>
            Annotations are stored as fractions of the original image, so changing a box never alters the photo.
          </Callout>

          <Button variant="primary" className="block" onClick={() => navigate(`/p/${project.id}/train`)}>
            Go to training →
          </Button>
        </aside>
      </div>
    </div>
  )
}
