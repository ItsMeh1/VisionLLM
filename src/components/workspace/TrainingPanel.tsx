import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Button, Callout, Disclosure, Progress } from '../ui'
import { useProject } from '../../lib/store'
import { useApp } from '../../lib/app'
import { cancelJob, followJob, shortEngineError, startTraining, type JobSnapshot } from '../../lib/engine'
import { defaultConfig, modelOptions } from '../../lib/training'
import { formatDuration } from '../../lib/image'
import { navigate } from '../../lib/router'
import type { EpochRecord, RunRecord, TrainingConfig } from '../../lib/types'

const MODEL_LABEL: Record<string, { name: string; hint: string }> = {
  'vlyolo-n': { name: 'Nano', hint: 'Fastest' },
  'vlyolo-s': { name: 'Small', hint: 'Balanced' },
  'vlyolo-m': { name: 'Medium', hint: 'Most careful' },
  resnet18: { name: 'ResNet-18', hint: 'Pretrained' },
  unet: { name: 'U-Net', hint: 'Compact' },
}

const metricFor = (task: string) => (task === 'detect' ? 'mAP @ 0.5' : task === 'classify' ? 'Accuracy' : 'mIoU')

export function TrainingPanel({ onPreview }: { onPreview: () => void }) {
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
    const blockers: string[] = []
    const warnings: string[] = []
    const task = project?.task ?? 'detect'
    const classes = project?.classes ?? []
    if (!images.length) blockers.push('Add a few examples first.')
    if (task === 'classify') {
      const missing = classes.filter((cls) => !images.some((image) => image.label === cls.id))
      if (missing.length) blockers.push(`Add examples for ${missing.map((cls) => cls.name).join(', ')}.`)
    } else {
      const empty = classes.filter(
        (cls) =>
          !images.some(
            (image) =>
              image.boxes.some((box) => box.cls === cls.id) || image.shapes.some((shape) => shape.cls === cls.id),
          ),
      )
      if (empty.length) blockers.push(`Add annotations for ${empty.map((cls) => cls.name).join(', ')}.`)
      const labelled = images.filter((image) => image.boxes.length > 0 || image.shapes.length > 0).length
      if (images.length && labelled < 4) blockers.push(`Only ${labelled} annotated images. Four is the minimum.`)
    }
    if (images.length < 12) warnings.push('More examples usually help more than more epochs.')
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
          notify('Model trained. Try it on something new.', 'ok')
        }
        if (snapshot.status === 'failed') notify(final.error ?? 'Training failed.', 'error')
      } catch (error) {
        notify(shortEngineError(error), 'error')
      }
    },
    [notify, saveRun],
  )

  useEffect(() => {
    if (live) return
    const resumable = runs.find((run) => (run.status === 'running' || run.status === 'queued') && run.jobId)
    if (resumable) void follow(resumable)
  }, [runs, live, follow])

  useEffect(() => {
    return () => controllerRef.current?.abort()
  }, [])

  if (!project) return null

  const running = activeRun?.status === 'running' || activeRun?.status === 'queued' || activeRun?.status === 'cancelling'
  const history = activeRun?.history ?? []
  const last = history[history.length - 1]
  const models = modelOptions(project.task)
  const presets =
    project.task === 'detect'
      ? [
          { epochs: 12, label: 'Quick', hint: 'a first look' },
          { epochs: 40, label: 'Standard', hint: 'recommended' },
          { epochs: 80, label: 'Thorough', hint: 'if it has time' },
        ]
      : project.task === 'classify'
        ? [
            { epochs: 8, label: 'Quick', hint: 'a first look' },
            { epochs: 20, label: 'Standard', hint: 'recommended' },
            { epochs: 40, label: 'Thorough', hint: 'if it has time' },
          ]
        : [
            { epochs: 10, label: 'Quick', hint: 'a first look' },
            { epochs: 30, label: 'Standard', hint: 'recommended' },
            { epochs: 60, label: 'Thorough', hint: 'if it has time' },
          ]
  const metric = activeRun?.metrics ?? {}
  const metricValue =
    project.task === 'detect'
      ? (last?.map50 ?? (metric.map50 as number | undefined) ?? 0)
      : project.task === 'classify'
        ? (last?.val_acc ?? (metric.accuracy as number | undefined) ?? 0)
        : (last?.miou ?? (metric.miou as number | undefined) ?? 0)
  const ready = readiness.blockers.length === 0

  const begin = async () => {
    if (readiness.blockers.length) {
      notify(readiness.blockers[0], 'error')
      return
    }
    setStarting(true)
    setLog([])
    epochSaves.current = 0
    try {
      // Keep one image library in the UI. A small holdout is chosen here for an
      // honest score and never needs to be managed by the user.
      if (images.length > 1) {
        const { stratifiedSplit } = await import('../../lib/analysis')
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
      notify('Stopping after this epoch. The checkpoint so far is kept.', 'info')
    } catch (error) {
      notify(shortEngineError(error), 'error')
    }
  }

  const set = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) =>
    setConfig((current) => ({ ...current, [key]: value }))

  return (
    <section className="card train-card">
      <header className="card-head">
        <div>
          <div className="card-title-row">
            <span className="step-badge">2</span>
            <h2>Train</h2>
          </div>
          <p className="card-sub">One button. Real training on this computer.</p>
        </div>
        {activeRun ? (
          <span className={running ? 'run-status working' : activeRun.status === 'completed' ? 'run-status done' : 'run-status'}>
            {activeRun.status === 'completed' ? 'Trained' : activeRun.status === 'failed' ? 'Failed' : activeRun.status === 'cancelled' ? 'Stopped' : 'Training'}
          </span>
        ) : null}
      </header>

      {!online ? (
        <Callout tone="warn" title="Start the training engine">
          <p>
            In the project folder run <code>npm run engine</code>, then come back. Browsing and annotation still work.
          </p>
        </Callout>
      ) : null}

      {models.length > 1 && !running ? (
        <div className="model-picker" role="radiogroup" aria-label="Model size">
          {models.map((key) => {
            const info = MODEL_LABEL[key] ?? { name: key, hint: '' }
            return (
              <button
                key={key}
                type="button"
                role="radio"
                aria-checked={config.model === key}
                className={config.model === key ? 'model-choice active' : 'model-choice'}
                onClick={() => set('model', key)}
              >
                <span className="model-choice-name">{info.name}</span>
                <span className="model-choice-hint">{info.hint}</span>
              </button>
            )
          })}
        </div>
      ) : null}

      {!running ? (
        <div className="train-length">
          <span className="field-label">Training length</span>
          <div className="length-options" role="group" aria-label="Training length">
            {presets.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className={config.epochs === preset.epochs ? 'length-option active' : 'length-option'}
                onClick={() => set('epochs', preset.epochs)}
                aria-pressed={config.epochs === preset.epochs}
              >
                <span>{preset.label}</span>
                <small>{preset.hint}</small>
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="train-readiness">
        <span className={ready ? 'status-dot ok' : 'status-dot warn'} aria-hidden />
        <div>
          <strong>{ready ? 'Ready to train' : 'A few things first'}</strong>
          <span>
            {images.length} example{images.length === 1 ? '' : 's'} · one library
          </span>
        </div>
        {health?.device ? <span className="device-tag">{health.device === 'mps' ? 'Apple GPU' : health.device.toUpperCase()}</span> : null}
      </div>

      {readiness.blockers.length ? (
        <ul className="blocker-list">
          {readiness.blockers.map((line) => (
            <li key={line}>{line}</li>
          ))}
        </ul>
      ) : readiness.warnings.length ? (
        <p className="train-warning">{readiness.warnings[0]}</p>
      ) : null}

      <div className="train-actions">
        <Button
          variant="primary"
          className="train-button"
          onClick={() => void begin()}
          busy={starting}
          disabled={!online || running || !ready}
        >
          {running ? 'Training…' : activeRun?.status === 'completed' ? 'Train again' : 'Train model'}
        </Button>
      </div>

      {running ? (
        <div className="train-progress">
          <Progress
            value={activeRun.history.length ? (activeRun.history.length / activeRun.config.epochs) * 100 : 2}
            label="Training progress"
          />
          <div className="progress-line">
            <span>Epoch {last?.epoch ?? 0} of {activeRun.config.epochs}</span>
            <span>{Math.round(activeRun.history.length ? (activeRun.history.length / activeRun.config.epochs) * 100 : 2)}%</span>
          </div>
          <div className="train-stats live">
            <div>
              <span>{metricFor(project.task)}</span>
              <strong>{last ? `${(metricValue * 100).toFixed(1)}%` : '—'}</strong>
            </div>
            <div>
              <span>Train loss</span>
              <strong>{last ? last.train_loss.toFixed(2) : '—'}</strong>
            </div>
            <div>
              <span>Elapsed</span>
              <strong>{formatDuration(last?.elapsed)}</strong>
            </div>
          </div>
          <div className="train-progress-actions">
            <button type="button" className="text-button danger" onClick={() => void stop()}>
              Stop training
            </button>
            <span className="train-log">{log[log.length - 1] ?? 'Preparing the first epoch…'}</span>
          </div>
        </div>
      ) : null}

      {!running && last ? (
        <div className="train-result">
          <div className="train-stats">
            <div>
              <span>{metricFor(project.task)}</span>
              <strong>{`${(metricValue * 100).toFixed(1)}%`}</strong>
            </div>
            <div>
              <span>Train loss</span>
              <strong>{last.train_loss.toFixed(2)}</strong>
            </div>
            <div>
              <span>Time</span>
              <strong>{formatDuration(activeRun?.elapsed)}</strong>
            </div>
          </div>
          {activeRun?.status === 'completed' ? (
            <div className="train-done">
              <div>
                <strong>Model trained</strong>
                <span>{activeRun.config.model} · {activeRun.history.length} epochs</span>
              </div>
              <div className="train-done-actions">
                <button type="button" className="text-button" onClick={onPreview}>Try it →</button>
                <button type="button" className="text-button" onClick={() => navigate(`/p/${project.id}/evaluate`)}>Evaluate</button>
              </div>
            </div>
          ) : null}
          {activeRun?.error ? <p className="train-warning">{activeRun.error}</p> : null}
        </div>
      ) : null}

      <Disclosure summary={<span>Training settings</span>} open={advanced} onToggle={() => setAdvanced(!advanced)}>
        <div className="config-grid">
          <label className="field">
            <span className="field-label">Epochs</span>
            <input
              type="number"
              min={1}
              max={500}
              value={config.epochs}
              onChange={(event) => set('epochs', Math.max(1, Math.min(500, Number(event.target.value))))}
            />
          </label>
          <label className="field">
            <span className="field-label">Input size</span>
            <select value={config.imgsz} onChange={(event) => set('imgsz', Number(event.target.value))}>
              {[320, 384, 448, 512, 640].map((size) => (
                <option key={size} value={size}>{size} px</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">Batch</span>
            <input
              type="number"
              min={1}
              max={64}
              value={config.batch}
              onChange={(event) => set('batch', Math.max(1, Math.min(64, Number(event.target.value))))}
            />
          </label>
          <label className="field">
            <span className="field-label">Learning rate</span>
            <input
              type="number"
              step="0.0001"
              value={config.lr}
              onChange={(event) => set('lr', Number(event.target.value))}
            />
          </label>
        </div>
        <p className="field-hint">
          Images are letterboxed, never cropped. Augmentation is limited to flips and small brightness changes.
        </p>
      </Disclosure>

      {runs.filter((run) => run.status === 'completed').length > 1 ? (
        <div className="run-history">
          <span className="field-label">Earlier models</span>
          {runs
            .filter((run) => run.status === 'completed')
            .slice(0, 3)
            .map((run) => (
              <button
                key={run.id}
                type="button"
                className={activeRun?.id === run.id ? 'run-row active' : 'run-row'}
                onClick={() => {
                  setLive(run)
                  setLog([])
                }}
              >
                <span>{run.config.model}</span>
                <span>{formatDuration(run.elapsed)}</span>
              </button>
            ))}
        </div>
      ) : null}
    </section>
  )
}
