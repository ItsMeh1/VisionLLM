import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { BarList, LineChart } from '../components/Charts'
import { Badge, Button, Callout, Disclosure, Panel, PanelHead, Progress, Stat } from '../components/ui'
import { useProject } from '../lib/store'
import { useApp } from '../lib/app'
import { followJob, cancelJob, shortEngineError, startTraining, type JobSnapshot } from '../lib/engine'
import { defaultConfig, estimateMinutes, INPUT_SIZES, MODEL_INFO, modelOptions } from '../lib/training'
import { formatDuration } from '../lib/image'
import { navigate } from '../lib/router'
import type { EpochRecord, RunRecord, TrainingConfig } from '../lib/types'

export function TrainScreen() {
  const { project, images, runs, saveRun, uploadCurrentDataset, saveImages } = useProject()
  const { online, health, notify } = useApp()
  const [config, setConfig] = useState<TrainingConfig>(() => defaultConfig(project?.task ?? 'detect'))
  const [advanced, setAdvanced] = useState(false)
  const [starting, setStarting] = useState(false)
  const [live, setLive] = useState<RunRecord | null>(null)
  const [log, setLog] = useState<string[]>([])
  const controllerRef = useRef<AbortController | null>(null)
  const epochSaves = useRef(0)

  useEffect(() => {
    if (project) setConfig(defaultConfig(project.task))
  }, [project?.id, project?.task])

  const activeRun = live ?? runs.find((run) => run.status === 'running' || run.status === 'queued') ?? runs[0] ?? null

  const readiness = useMemo(() => {
    if (!project) return { blockers: ['Project loading'], warnings: [] as string[] }
    const blockers: string[] = []
    const warnings: string[] = []
    if (!images.length) blockers.push('Import images first.')
    const trainImages = images.filter((image) => image.split === 'train')
    if (images.length && !trainImages.length) blockers.push('No images are assigned to training.')
    if (project.task === 'classify') {
      const missing = project.classes.filter((cls) => !images.some((image) => image.label === cls.id))
      if (missing.length) blockers.push(`No images labelled ${missing.map((cls) => cls.name).join(', ')}.`)
    } else {
      const empty = project.classes.filter(
        (cls) =>
          !images.some((image) => image.boxes.some((box) => box.cls === cls.id) || image.shapes.some((shape) => shape.cls === cls.id)),
      )
      if (empty.length) blockers.push(`No annotations for ${empty.map((cls) => cls.name).join(', ')}.`)
      const labelledTrain = trainImages.filter((image) => image.boxes.length > 0 || image.shapes.length > 0).length
      if (images.length && labelledTrain < 4) blockers.push(`Only ${labelledTrain} labelled training images.`)
    }
    const val = images.filter((image) => image.split === 'val').length
    if (!val) warnings.push('No validation images; the engine will hold out the last 15%.')
    else if (val < 3) warnings.push(`Only ${val} validation images — metrics will be noisy.`)
    if (images.length < 20) warnings.push('Fewer than 20 images. Add variety for a better model.')
    return { blockers, warnings }
  }, [project, images])

  const follow = useCallback(
    async (run: RunRecord) => {
      if (!run.jobId) return
      controllerRef.current?.abort()
      const controller = new AbortController()
      controllerRef.current = controller
      setLive(run)
      try {
        const snapshot = await followJob(
          run.jobId,
          (update: JobSnapshot) => {
            const epochs = update.events
              .filter((event) => event.type === 'epoch' && event.record)
              .map((event) => event.record as unknown as EpochRecord)
            const messages = update.events
              .filter((event) => event.type === 'stage' && event.message)
              .map((event) => event.message as string)
            if (messages.length) setLog((current) => [...new Set([...current, ...messages])])
            if (epochs.length) {
              const next: RunRecord = {
                ...run,
                status: update.status === 'cancelling' ? 'cancelling' : update.status,
                history: epochs,
              }
              setLive(next)
              epochSaves.current += 1
              if (epochSaves.current % 2 === 0) void saveRun(next)
            }
            if (update.result?.metrics) {
              setLive((current) => (current ? { ...current, metrics: update.result?.metrics } : current))
            }
          },
          controller.signal,
        )
        const epochs = snapshot.events
          .filter((event) => event.type === 'epoch' && event.record)
          .map((event) => event.record as unknown as EpochRecord)
        const final: RunRecord = {
          ...run,
          status: snapshot.status === 'cancelling' ? 'cancelling' : snapshot.status,
          history: epochs,
          metrics: snapshot.result?.metrics as Record<string, unknown> | undefined,
          notes: snapshot.result?.notes as string[] | undefined,
          elapsed: snapshot.result?.elapsed_seconds as number | undefined,
          error: snapshot.error ?? undefined,
        }
        setLive(final)
        await saveRun(final)
        if (snapshot.status === 'completed') {
          notify('Training finished. Evaluate it on held-out images next.', 'ok')
          setLog((current) => [...current, `Finished after ${formatDuration(final.elapsed)}.`])
        }
        if (snapshot.status === 'failed') {
          notify(final.error ?? 'Training failed.', 'error')
        }
      } catch (error) {
        notify(shortEngineError(error), 'error')
      }
    },
    [notify, saveRun],
  )

  useEffect(() => {
    const resumable = runs.find(
      (run) => (run.status === 'running' || run.status === 'queued') && run.jobId,
    )
    if (resumable && !live) void follow(resumable)
    return () => controllerRef.current?.abort()
  }, [runs, live, follow])

  if (!project) return null

  const trainedRuns = runs.filter((run) => run.status === 'completed' || run.status === 'cancelled')
  const history = activeRun?.history ?? []
  const last = history[history.length - 1]
  const running = activeRun?.status === 'running' || activeRun?.status === 'queued' || activeRun?.status === 'cancelling'
  const metrics = activeRun?.metrics ?? {}
  const models = modelOptions(project.task)
  const minutes = estimateMinutes(project.task, config, images.filter((image) => image.split === 'train').length, health?.device ?? 'cpu')

  const begin = async () => {
    if (readiness.blockers.length) {
      notify(readiness.blockers[0], 'error')
      return
    }
    setStarting(true)
    setLog([])
    epochSaves.current = 0
    try {
      if (!images.some((image) => image.split === 'val') && images.length > 3) {
        const { stratifiedSplit } = await import('../lib/analysis')
        await saveImages(stratifiedSplit(project, images, 0.2))
      }
      await uploadCurrentDataset()
      const runId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
      const started = await startTraining(project.id, config, runId)
      const run: RunRecord = {
        id: runId,
        projectId: project.id,
        task: project.task,
        config,
        createdAt: Date.now(),
        status: 'running',
        jobId: started.job_id,
        history: [],
      }
      await saveRun(run)
      setLive(run)
      void follow(run)
    } catch (error) {
      notify(shortEngineError(error), 'error')
    } finally {
      setStarting(false)
    }
  }

  const stop = async () => {
    if (!activeRun?.jobId) return
    try {
      await cancelJob(activeRun.jobId)
      notify('Stopping after the current batch. The checkpoint so far is kept.', 'info')
    } catch (error) {
      notify(shortEngineError(error), 'error')
    }
  }

  const set = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) =>
    setConfig((current) => ({ ...current, [key]: value }))

  return (
    <div className="screen train-screen">
      <Panel className="train-config">
        <PanelHead
          eyebrow="03 · Train"
          title="Choose a detector and press start"
          note="Defaults are chosen for a first model. Everything below runs on your machine."
          actions={<Badge tone={online ? 'ok' : 'bad'}>{online ? `engine · ${health?.device}` : 'engine offline'}</Badge>}
        />

        {!online ? (
          <Callout tone="warn" title="Start the local engine">
            In the project folder run <span className="mono">npm run engine</span>, then come back. Annotation still works
            without it.
          </Callout>
        ) : null}

        <div className="model-grid">
          {models.map((key) => {
            const info = MODEL_INFO[key]
            const active = config.model === key
            return (
              <button
                key={key}
                type="button"
                className={active ? 'model-card active' : 'model-card'}
                onClick={() => set('model', key)}
                aria-pressed={active}
              >
                <span className="model-card-top">
                  <span className="model-name">{info.label}</span>
                  <span className="mono model-licence">{info.licence}</span>
                </span>
                <span className="model-family mono">{info.family}</span>
                <span className="model-blurb">{info.blurb}</span>
              </button>
            )
          })}
        </div>

        <div className="config-row">
          <label className="field">
            <span className="label">Epochs</span>
            <input
              type="number"
              min={1}
              max={500}
              value={config.epochs}
              onChange={(event) => set('epochs', Math.max(1, Math.min(500, Number(event.target.value))))}
            />
            <span className="field-hint">Full passes over the training images. More is not always better.</span>
          </label>
          <label className="field">
            <span className="label">Input size</span>
            <select value={config.imgsz} onChange={(event) => set('imgsz', Number(event.target.value))}>
              {INPUT_SIZES[project.task].map((size) => (
                <option key={size} value={size}>{size} px</option>
              ))}
            </select>
            <span className="field-hint">Longest side is fitted. Aspect ratio never changes.</span>
          </label>
          <label className="field">
            <span className="label">Batch</span>
            <input
              type="number"
              min={1}
              max={64}
              value={config.batch}
              onChange={(event) => set('batch', Math.max(1, Math.min(64, Number(event.target.value))))}
            />
            <span className="field-hint">Lower this if memory runs out.</span>
          </label>
        </div>

        <Disclosure
          summary={<span>Advanced settings</span>}
          open={advanced}
          onToggle={() => setAdvanced(!advanced)}
        >
          <div className="config-row">
            <label className="field">
              <span className="label">Learning rate</span>
              <input
                type="number"
                step="0.0001"
                value={config.lr}
                onChange={(event) => set('lr', Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span className="label">Weight decay</span>
              <input
                type="number"
                step="0.0001"
                value={config.weight_decay}
                onChange={(event) => set('weight_decay', Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span className="label">Early-stop patience</span>
              <input
                type="number"
                min={0}
                value={config.patience}
                onChange={(event) => set('patience', Math.max(0, Number(event.target.value)))}
              />
              <span className="field-hint">0 disables early stopping.</span>
            </label>
          </div>
          <Disclosure summary={<span>Device and reproducibility</span>}>
            <div className="config-row">
              <label className="field">
                <span className="label">Device</span>
                <select value={config.device ?? 'auto'} onChange={(event) => set('device', event.target.value)}>
                  <option value="auto">Auto ({health?.device ?? 'detect'})</option>
                  <option value="mps">Apple GPU (MPS)</option>
                  <option value="cpu">CPU</option>
                </select>
              </label>
              <label className="field">
                <span className="label">Seed</span>
                <input type="number" value={config.seed} onChange={(event) => set('seed', Number(event.target.value))} />
              </label>
            </div>
            <p className="field-hint">
              Augmentation is horizontal flips and small brightness/contrast changes only. Nothing geometric that would
              distort an aspect ratio.
            </p>
          </Disclosure>
        </Disclosure>

        <div className="readiness">
          <div className="readiness-head">
            <span className="label">Readiness</span>
            <span className="mono">
              {images.length} images · {readiness.blockers.length} blockers · {readiness.warnings.length} warnings
            </span>
          </div>
          <ul>
            {readiness.blockers.map((line) => (
              <li key={line} className="check blocker"><span aria-hidden>×</span>{line}</li>
            ))}
            {readiness.warnings.map((line) => (
              <li key={line} className="check warning"><span aria-hidden>!</span>{line}</li>
            ))}
            {!readiness.blockers.length && !readiness.warnings.length ? (
              <li className="check ok"><span aria-hidden>✓</span>Dataset looks ready.</li>
            ) : null}
          </ul>
        </div>

        <div className="start-row">
          <div className="start-estimate">
            <span className="label">Estimated time</span>
            <span className="num">≈ {minutes < 1 ? '<1' : Math.round(minutes)} min</span>
            <span className="field-hint">Rough guide from dataset size, not a promise.</span>
          </div>
          {running ? (
            <Button variant="danger" onClick={() => void stop()}>
              Stop training
            </Button>
          ) : (
            <Button variant="primary" onClick={() => void begin()} disabled={!online || starting} busy={starting}>
              Start training
            </Button>
          )}
        </div>
      </Panel>

      <div className="train-monitor">
        <Panel className="monitor-panel">
          <PanelHead
            eyebrow="Live run"
            title={activeRun ? `${activeRun.config.model} · ${activeRun.config.epochs} epochs` : 'No run yet'}
            actions={
              activeRun ? (
                <Badge tone={running ? 'warn' : activeRun.status === 'completed' ? 'ok' : activeRun.status === 'failed' ? 'bad' : 'neutral'}>
                  {activeRun.status}
                </Badge>
              ) : null
            }
          />
          {activeRun && running ? (
            <div className="progress-block">
              <Progress value={activeRun.history.length ? (activeRun.history.length / activeRun.config.epochs) * 100 : 2} label="Training progress" />
              <div className="progress-meta mono">
                <span>epoch {last?.epoch ?? 0} / {activeRun.config.epochs}</span>
                <span>{formatDuration(last?.elapsed)}</span>
              </div>
            </div>
          ) : null}

          <div className="stat-row">
            <Stat label="Train loss" value={last ? last.train_loss.toFixed(3) : '—'} hint="lower is better" />
            <Stat
              label={project.task === 'detect' ? 'mAP @ 0.5' : project.task === 'classify' ? 'Val accuracy' : 'Val mIoU'}
              value={
                last
                  ? project.task === 'detect'
                    ? `${((last.map50 ?? 0) * 100).toFixed(1)}%`
                    : project.task === 'classify'
                      ? `${((last.val_acc ?? 0) * 100).toFixed(1)}%`
                      : `${((last.miou ?? 0) * 100).toFixed(1)}%`
                  : '—'
              }
              hint="held-out images"
              tone="accent"
            />
            <Stat label="Epoch time" value={last ? `${Math.max(1, Math.round(last.elapsed / Math.max(1, last.epoch)))}s` : '—'} hint="measured" />
          </div>

          {project.task === 'detect' ? (
            <LineChart
              yMax={1}
              yFormat={(value) => `${Math.round(value * 100)}%`}
              series={[
                {
                  key: 'map50',
                  label: 'mAP@0.5',
                  color: 'var(--accent)',
                  points: history.filter((entry) => typeof entry.map50 === 'number').map((entry) => ({ x: entry.epoch, y: entry.map50 ?? 0 })),
                },
                {
                  key: 'map5095',
                  label: 'mAP@.5:.95',
                  color: '#7fb2e5',
                  points: history.filter((entry) => typeof entry.map5095 === 'number').map((entry) => ({ x: entry.epoch, y: entry.map5095 ?? 0 })),
                },
              ]}
              emptyLabel="Metrics appear after the first validation pass"
            />
          ) : project.task === 'classify' ? (
            <LineChart
              yMax={1}
              yFormat={(value) => `${Math.round(value * 100)}%`}
              series={[
                {
                  key: 'val_acc',
                  label: 'validation accuracy',
                  color: 'var(--accent)',
                  points: history.filter((entry) => typeof entry.val_acc === 'number').map((entry) => ({ x: entry.epoch, y: entry.val_acc ?? 0 })),
                },
                {
                  key: 'train_acc',
                  label: 'training accuracy',
                  color: '#8fd48a',
                  points: history.filter((entry) => typeof entry.train_acc === 'number').map((entry) => ({ x: entry.epoch, y: entry.train_acc ?? 0 })),
                },
              ]}
            />
          ) : (
            <LineChart
              yMax={1}
              yFormat={(value) => `${Math.round(value * 100)}%`}
              series={[
                {
                  key: 'miou',
                  label: 'mIoU',
                  color: 'var(--accent)',
                  points: history.filter((entry) => typeof entry.miou === 'number').map((entry) => ({ x: entry.epoch, y: entry.miou ?? 0 })),
                },
                {
                  key: 'pixel_accuracy',
                  label: 'pixel accuracy',
                  color: '#7fb2e5',
                  points: history.filter((entry) => typeof entry.pixel_accuracy === 'number').map((entry) => ({ x: entry.epoch, y: entry.pixel_accuracy ?? 0 })),
                },
              ]}
            />
          )}

          {activeRun && project.task === 'detect' && metrics.per_class_ap50 ? (
            <BarList
              items={project.classes.map((cls, index) => ({
                label: cls.name,
                value: (metrics.per_class_ap50 as (number | null)[])[index] ?? null,
                color: cls.color,
              }))}
              format={(value) => `${(value * 100).toFixed(1)}%`}
            />
          ) : null}

          <div className="log-block">
            <div className="label">Engine log</div>
            <pre className="log" aria-live="polite">
              {log.length ? log.join('\n') : activeRun?.error ? activeRun.error : 'Waiting for a run. The trainer prints real stages and epochs here.'}
            </pre>
          </div>

          {activeRun?.status === 'completed' ? (
            <div className="next-row">
              <span className="field-hint">
                Best epoch {String((activeRun as RunRecord & { bestEpoch?: number }).bestEpoch ?? last?.epoch ?? '—')} ·{' '}
                {formatDuration(activeRun.elapsed)}
              </span>
              <Button variant="primary" size="sm" onClick={() => navigate(`/p/${project.id}/evaluate`)}>
                Evaluate this model →
              </Button>
            </div>
          ) : null}
          {activeRun?.notes?.length ? (
            <ul className="note-list">
              {activeRun.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          ) : null}
        </Panel>

        <Panel>
          <PanelHead eyebrow="History" title="Runs on this machine" />
          {trainedRuns.length ? (
            <ul className="run-list">
              {trainedRuns.map((run) => {
                const runMetrics = run.metrics ?? {}
                return (
                  <li key={run.id}>
                    <button
                      type="button"
                      onClick={() => {
                        setLive(run)
                        setLog([])
                        navigate(`/p/${project.id}/evaluate`)
                      }}
                    >
                      <span className="mono">{run.id}</span>
                      <span className="mono">{run.config.model}</span>
                      <span className="num">
                        {run.task === 'detect'
                          ? `${((runMetrics.map50 as number) ?? 0) * 100 > 0 ? (((runMetrics.map50 as number) ?? 0) * 100).toFixed(1) + '%' : '—'}`
                          : run.task === 'classify'
                            ? `${(((runMetrics.accuracy as number) ?? 0) * 100).toFixed(1)}%`
                            : `${(((runMetrics.miou as number) ?? 0) * 100).toFixed(1)}%`}
                      </span>
                      <span className="mono muted">{formatDuration(run.elapsed)}</span>
                    </button>
                  </li>
                )
              })}
            </ul>
          ) : (
            <p className="field-hint">No completed runs yet. Every run keeps its checkpoint and metrics on disk.</p>
          )}
        </Panel>
      </div>
    </div>
  )
}
