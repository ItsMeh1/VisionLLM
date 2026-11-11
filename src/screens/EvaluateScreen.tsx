import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BarList, Confusion } from '../components/Charts'
import { Badge, Button, Callout, Empty, Panel, PanelHead, Segmented, Stat } from '../components/ui'
import { useProject } from '../lib/store'
import { useApp } from '../lib/app'
import { evaluateRun, shortEngineError, type EvaluationResponse } from '../lib/engine'
import { blobToBitmap, pct } from '../lib/image'
import { useObjectUrl } from '../lib/hooks'
import { navigate } from '../lib/router'
import type { ClassDef, ImageRecord } from '../lib/types'

type EngineBox = { cls: number; x: number; y: number; w: number; h: number; score?: number }

function BoxedImage({
  image,
  boxes,
  classes,
}: {
  image: ImageRecord
  boxes: { cls: number; x: number; y: number; w: number; h: number; score?: number }[]
  classes: ClassDef[]
}) {
  const url = useObjectUrl(image.blob)
  const ref = useRef<HTMLCanvasElement | null>(null)
  useEffect(() => {
    let cancelled = false
    void (async () => {
      const canvas = ref.current
      if (!canvas || !url) return
      const source = await blobToBitmap(image.blob)
      if (cancelled) {
        if ('close' in source) source.close()
        return
      }
      const max = 460
      const scale = Math.min(1, max / image.width)
      canvas.width = Math.round(image.width * scale)
      canvas.height = Math.round(image.height * scale)
      const context = canvas.getContext('2d')
      if (!context) return
      context.drawImage(source as CanvasImageSource, 0, 0, canvas.width, canvas.height)
      if ('close' in source) source.close()
      context.lineWidth = Math.max(1.5, canvas.width / 320)
      context.font = `${Math.max(9, Math.round(canvas.width / 55))}px 'JetBrains Mono Variable', monospace`
      for (const box of boxes) {
        const color = classes[box.cls]?.color ?? '#FF4A1C'
        const x = box.x * canvas.width
        const y = box.y * canvas.height
        const w = box.w * canvas.width
        const h = box.h * canvas.height
        context.strokeStyle = color
        context.strokeRect(x, y, w, h)
        const label = `${classes[box.cls]?.name ?? '?'} ${box.score !== undefined ? Math.round(box.score * 100) + '%' : ''}`.trim()
        const width = context.measureText(label).width
        context.fillStyle = color
        context.fillRect(x, Math.max(0, y - 14), width + 6, 14)
        context.fillStyle = '#0b0c0b'
        context.fillText(label, x + 3, Math.max(10, y - 3))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [url, boxes, classes, image])
  return <canvas ref={ref} className="result-canvas" aria-label={`Predictions on ${image.name}`} />
}

export function EvaluateScreen() {
  const { project, images, runs, saveRun } = useProject()
  const { online } = useApp()
  const completed = useMemo(() => runs.filter((run) => run.status === 'completed'), [runs])
  const [runId, setRunId] = useState('')
  const [data, setData] = useState<EvaluationResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [show, setShow] = useState<'predictions' | 'truth'>('predictions')

  const activeRun = completed.find((run) => run.id === runId) ?? completed[0]

  const evaluate = useCallback(async () => {
    if (!project || !activeRun) return
    setBusy(true)
    setError(null)
    try {
      const response = await evaluateRun(project.id, activeRun.id)
      setData(response)
      await saveRun({ ...activeRun, metrics: response.metrics })
    } catch (err) {
      setError(shortEngineError(err))
    } finally {
      setBusy(false)
    }
  }, [project, activeRun, saveRun])

  useEffect(() => {
    if (!runId && completed.length) setRunId(completed[0].id)
  }, [completed, runId])

  useEffect(() => {
    if (!online || !activeRun) {
      setData(null)
      return
    }
    void evaluate()
    // Re-evaluate only when the selected run changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeRun?.id, online])

  if (!project) return null

  if (!completed.length) {
    return (
      <div className="screen">
        <Panel>
          <Empty
            title="No trained model to evaluate"
            action={<Button variant="primary" onClick={() => navigate(`/p/${project.id}/workspace`)}>Go to workspace →</Button>}
          >
            Evaluation needs a finished checkpoint. Train a model, then come back; these numbers will come from images it
            never saw.
          </Empty>
        </Panel>
      </div>
    )
  }

  const metrics = (data?.metrics ?? {}) as Record<string, unknown>
  const imageMap = new Map(images.map((image) => [image.id, image]))
  const predictions = (metrics.images as { image_id: string; boxes: EngineBox[] }[] | undefined) ?? []
  const truth = (metrics.ground_truth as { image_id: string; boxes: EngineBox[] }[] | undefined) ?? []
  const truthMap = new Map(truth.map((entry) => [entry.image_id, entry.boxes]))
  const toLocal = (entry: { image_id: string; boxes: { cls: number; x: number; y: number; w: number; h: number; score?: number }[] }) =>
    entry.boxes
      .map((box) => {
        const cls = project.classes[box.cls]
        return cls ? { cls: box.cls, x: box.x, y: box.y, w: box.w, h: box.h, score: box.score } : null
      })
      .filter((box): box is NonNullable<typeof box> => Boolean(box))

  const perClass =
    project.task === 'detect'
      ? project.classes.map((cls, index) => ({
          label: cls.name,
          value: ((metrics.per_class_ap50 as (number | null)[]) ?? [])[index] ?? null,
          color: cls.color,
        }))
      : project.task === 'segment'
        ? project.classes.map((cls, index) => {
            const entry = ((metrics.per_class as { class: number; iou: number | null }[]) ?? []).find((item) => item.class === index + 1)
            return { label: cls.name, value: entry?.iou ?? null, color: cls.color }
          })
        : project.classes.map((cls, index) => {
            const entry = ((metrics.per_class as { f1: number }[]) ?? [])[index]
            return { label: cls.name, value: entry?.f1 ?? null, color: cls.color }
          })

  const headline =
    project.task === 'detect'
      ? { primary: pct(metrics.map50), primaryLabel: 'mAP @ 0.5', secondary: pct(metrics.map5095), secondaryLabel: 'mAP @ .5 : .95' }
      : project.task === 'classify'
        ? { primary: pct(metrics.accuracy), primaryLabel: 'Accuracy', secondary: pct(metrics.macro_f1), secondaryLabel: 'Macro F1' }
        : { primary: pct(metrics.miou), primaryLabel: 'mIoU', secondary: pct(metrics.pixel_accuracy), secondaryLabel: 'Pixel accuracy' }

  return (
    <div className="screen evaluate-screen">
      <Panel>
        <PanelHead
          eyebrow="Evaluate"
          title="Does it actually work?"
          note="Scored on held-out validation images that were excluded from training. Nothing is cached: every number is recomputed from the saved checkpoint."
          actions={
            <div className="button-row">
              <select value={activeRun?.id ?? ''} onChange={(event) => setRunId(event.target.value)} aria-label="Run to evaluate">
                {completed.map((run) => (
                  <option key={run.id} value={run.id}>{run.config.model} · {run.id}</option>
                ))}
              </select>
              <Button size="sm" onClick={() => void evaluate()} busy={busy} disabled={!online}>
                Re-run evaluation
              </Button>
            </div>
          }
        />

        {!online ? <Callout tone="warn" title="Engine offline">Start the engine to score the checkpoint.</Callout> : null}
        {error ? <Callout tone="bad" title="Evaluation failed">{error}</Callout> : null}

        <div className="stat-row">
          <Stat label={headline.primaryLabel} value={busy && !data ? '…' : headline.primary} tone="accent" hint={`${data?.images ?? 0} validation images`} />
          <Stat label={headline.secondaryLabel} value={busy && !data ? '…' : headline.secondary} />
          {project.task === 'detect' ? (
            <>
              <Stat label="Precision" value={pct(metrics.precision)} hint="at 0.25 confidence" />
              <Stat label="Recall" value={pct(metrics.recall)} hint="objects actually found" />
            </>
          ) : project.task === 'classify' ? (
            <>
              <Stat label="Classes" value={project.classes.length} />
              <Stat label="Images" value={data?.images ?? 0} />
            </>
          ) : (
            <>
              <Stat label="Classes" value={project.classes.length} />
              <Stat label="Images" value={data?.images ?? 0} />
            </>
          )}
        </div>

        <div className="eval-grid">
          <div>
            <div className="label">Per class</div>
            <BarList items={perClass} />
          </div>
          <div className="eval-notes">
            <div className="label">Reading this</div>
            {project.task === 'detect' ? (
              <p>
                <strong>mAP@0.5</strong> is the average precision when a box counts as correct if it overlaps the real
                object by at least 50%. <strong>mAP@.5:.95</strong> is stricter — it averages ten overlap thresholds, so
                tight boxes matter.
              </p>
            ) : project.task === 'classify' ? (
              <p>
                Accuracy can hide a lazy model. Check the confusion matrix: if one class is always predicted, that class
                dominates your dataset.
              </p>
            ) : (
              <p>
                <strong>mIoU</strong> measures overlap between predicted and painted regions. Thin structures score
                lower even when the mask looks right at a glance.
              </p>
            )}
            <p className="field-hint">
              These are honest numbers on {data?.images ?? 0} held-out images. A tiny validation set makes them noisy.
            </p>
          </div>
        </div>
      </Panel>

      {project.task === 'detect' && predictions.length ? (
        <Panel>
          <PanelHead
            eyebrow="Reality check"
            title="Look at the misses"
            actions={
              <Segmented
                ariaLabel="Overlay"
                value={show}
                onChange={setShow}
                options={[
                  { value: 'predictions', label: 'Predictions' },
                  { value: 'truth', label: 'Ground truth' },
                ]}
              />
            }
          />
          <div className="result-grid">
            {predictions.slice(0, 9).map((entry) => {
              const image = imageMap.get(entry.image_id)
              if (!image) return null
              const boxes = show === 'truth' ? toLocal({ image_id: entry.image_id, boxes: truthMap.get(entry.image_id) ?? [] }) : toLocal(entry)
              return (
                <figure key={entry.image_id} className="result-figure">
                  <BoxedImage image={image} boxes={boxes} classes={project.classes} />
                  <figcaption>
                    <span className="mono">{image.name}</span>
                    <span className="mono">{boxes.length} {show === 'truth' ? 'real' : 'predicted'}</span>
                  </figcaption>
                </figure>
              )
            })}
          </div>
          {predictions.length > 9 ? <p className="field-hint">Showing 9 of {predictions.length} validation images.</p> : null}
        </Panel>
      ) : null}

      {project.task === 'classify' && metrics.confusion ? (
        <Panel>
          <PanelHead eyebrow="Diagnosis" title="Confusion matrix" note="Rows are the truth, columns are what the model said." />
          <Confusion matrix={metrics.confusion as number[][]} classes={project.classes.map((cls) => cls.name)} />
          <div className="misclassified">
            {(metrics.images as { image_id: string; truth: number; prediction: number; top: { cls: number; score: number }[] }[] | undefined ?? [])
              .filter((entry) => entry.truth !== entry.prediction)
              .slice(0, 8)
              .map((entry) => {
                const image = imageMap.get(entry.image_id)
                return (
                  <div key={entry.image_id} className="miss-row">
                    {image ? <img src={image.thumbnail} alt="" /> : null}
                    <span className="mono">{image?.name ?? entry.image_id}</span>
                    <span>true <b>{project.classes[entry.truth]?.name}</b></span>
                    <span>said <b style={{ color: project.classes[entry.prediction]?.color }}>{project.classes[entry.prediction]?.name}</b></span>
                    <span className="mono">{Math.round((entry.top[0]?.score ?? 0) * 100)}%</span>
                  </div>
                )
              })}
            {!(metrics.images as { truth: number; prediction: number }[]).some((entry) => entry.truth !== entry.prediction) ? (
              <Badge tone="ok">No validation mistakes. Check that the split is not leaking duplicates.</Badge>
            ) : null}
          </div>
        </Panel>
      ) : null}

      <div className="next-row">
        <span className="field-hint">Numbers can look good while the model fails on your actual camera. Test it live next.</span>
        <Button variant="primary" onClick={() => navigate(`/p/${project.id}/workspace`)}>Test on new images →</Button>
      </div>
    </div>
  )
}
