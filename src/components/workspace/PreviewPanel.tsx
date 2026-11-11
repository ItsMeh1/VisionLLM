import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Callout } from '../ui'
import { IconCamera, IconPlay, IconUpload } from '../Icons'
import { useProject } from '../../lib/store'
import { useApp } from '../../lib/app'
import { disposeOnnxModel, loadOnnxModel, runOnnx, type BrowserPrediction, type LoadedModel } from '../../lib/onnx'
import { engineBase, exportRun, predict, shortEngineError, type Prediction } from '../../lib/engine'
import { canvasToBlob } from '../../lib/image'

type Source = 'image' | 'camera' | 'video' | 'onnx'

const SOURCES: { value: Source; label: string }[] = [
  { value: 'image', label: 'Image' },
  { value: 'camera', label: 'Camera' },
  { value: 'video', label: 'Video' },
  { value: 'onnx', label: 'In browser' },
]

export function PreviewPanel() {
  const { project, runs } = useProject()
  const { online, notify } = useApp()
  const completed = runs.filter((run) => run.status === 'completed')
  const [runId, setRunId] = useState(completed[0]?.id ?? '')
  const [source, setSource] = useState<Source>('image')
  const [conf, setConf] = useState(0.3)
  const [iou] = useState(0.45)
  const [result, setResult] = useState<Prediction | BrowserPrediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [started, setStarted] = useState(false)
  const [onnxModel, setOnnxModel] = useState<LoadedModel | null>(null)
  const [onnxState, setOnnxState] = useState('')

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<number | null>(null)
  const inflight = useRef(false)
  const objectUrl = useRef<string | null>(null)

  const activeRun = completed.find((run) => run.id === runId) ?? completed[0]

  useEffect(() => {
    if (!runId && completed.length) setRunId(completed[0].id)
  }, [completed, runId])

  useEffect(() => {
    setResult(null)
    setStarted(false)
  }, [activeRun?.id])

  useEffect(() => {
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      if (timerRef.current) window.clearInterval(timerRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      void disposeOnnxModel(onnxModel)
    }
    // The ONNX session is released explicitly when replaced; this covers unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const draw = useCallback(
    (sourceElement: CanvasImageSource, width: number, height: number, prediction: Prediction | BrowserPrediction | null) => {
      const canvas = canvasRef.current
      if (!canvas) return
      const max = 1000
      const scale = Math.min(1, max / width)
      canvas.width = Math.round(width * scale)
      canvas.height = Math.round(height * scale)
      const context = canvas.getContext('2d')
      if (!context) return
      context.drawImage(sourceElement, 0, 0, canvas.width, canvas.height)
      if (!prediction) return
      if (prediction.task === 'segment' && 'mask_png' in prediction && prediction.mask_png) {
        const mask = new Image()
        mask.onload = () => context.drawImage(mask, 0, 0, canvas.width, canvas.height)
        mask.src = prediction.mask_png
      }
      if (prediction.task === 'segment' && 'maskDataUrl' in prediction && prediction.maskDataUrl) {
        const mask = new Image()
        mask.onload = () => context.drawImage(mask, 0, 0, canvas.width, canvas.height)
        mask.src = prediction.maskDataUrl
      }
      if (prediction.task === 'detect' && prediction.boxes && project) {
        context.lineWidth = Math.max(2, canvas.width / 420)
        context.font = `600 ${Math.max(12, Math.round(canvas.width / 44))}px 'Figtree Variable', ui-sans-serif, system-ui, sans-serif`
        for (const box of prediction.boxes) {
          const cls = project.classes[box.cls]
          const color = cls?.color ?? '#5B4BE1'
          const x = box.x * canvas.width
          const y = box.y * canvas.height
          const w = box.w * canvas.width
          const h = box.h * canvas.height
          context.strokeStyle = color
          context.strokeRect(x, y, w, h)
          const label = `${cls?.name ?? '?'} ${Math.round(box.score * 100)}%`
          const labelWidth = context.measureText(label).width
          const labelHeight = Math.max(20, Math.round(canvas.width / 34))
          const top = y - labelHeight < 0 ? y : y - labelHeight
          context.fillStyle = color
          context.fillRect(x, top, labelWidth + 12, labelHeight)
          context.fillStyle = '#ffffff'
          context.fillText(label, x + 6, top + labelHeight - 6)
        }
      }
    },
    [project],
  )

  const predictBlob = useCallback(
    async (blob: Blob, sourceElement: CanvasImageSource, width: number, height: number) => {
      if (!project || !activeRun) return
      setBusy(true)
      setError(null)
      try {
        if (source === 'onnx' && onnxModel) {
          const prediction = await runOnnx(onnxModel, sourceElement, width, height, { conf, iou })
          setResult(prediction)
          draw(sourceElement, width, height, prediction)
        } else {
          const response = await predict(project.id, activeRun.id, [blob], { conf, iou })
          const prediction = response.results[0]
          setResult(prediction ?? null)
          draw(sourceElement, width, height, prediction ?? null)
        }
      } catch (err) {
        setError(shortEngineError(err))
      } finally {
        setBusy(false)
      }
    },
    [project, activeRun, source, onnxModel, conf, iou, draw],
  )

  const showImage = useCallback(
    (file: File) => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      objectUrl.current = URL.createObjectURL(file)
      const image = new Image()
      image.onload = () => {
        setStarted(true)
        setResult(null)
        draw(image, image.naturalWidth, image.naturalHeight, null)
        void predictBlob(file, image, image.naturalWidth, image.naturalHeight)
      }
      image.src = objectUrl.current
    },
    [draw, predictBlob],
  )

  const startCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 } }, audio: false })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setStarted(true)
      }
    } catch {
      setCameraError('Camera access was refused or no camera is available.')
    }
  }

  const stopCamera = () => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setPlaying(false)
  }

  const runFrameLoop = useCallback(() => {
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = window.setInterval(async () => {
      const video = videoRef.current
      if (!video || !video.videoWidth || inflight.current) return
      inflight.current = true
      try {
        const canvas = document.createElement('canvas')
        canvas.width = video.videoWidth
        canvas.height = video.videoHeight
        canvas.getContext('2d')?.drawImage(video, 0, 0)
        const blob = await canvasToBlob(canvas, 'image/png')
        await predictBlob(blob, video, video.videoWidth, video.videoHeight)
      } finally {
        inflight.current = false
      }
    }, 750)
  }, [predictBlob])

  useEffect(() => {
    if ((source === 'camera' && streamRef.current) || (source === 'video' && playing)) runFrameLoop()
    if (source !== 'camera' && source !== 'video') {
      if (timerRef.current) window.clearInterval(timerRef.current)
      timerRef.current = null
    }
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [source, playing, runFrameLoop])

  useEffect(() => {
    if (source !== 'camera') stopCamera()
  }, [source])

  const loadBrowserModel = async () => {
    if (!project || !activeRun) return
    setOnnxState('Building export…')
    setError(null)
    try {
      const info = await exportRun(project.id, activeRun.id)
      setOnnxState('Loading ONNX Runtime…')
      const base = engineBase()
      const model = await loadOnnxModel(
        `${base}/projects/${project.id}/runs/${activeRun.id}/export/model.onnx`,
        `${base}/projects/${project.id}/runs/${activeRun.id}/export/labels.json`,
      )
      setOnnxModel(model)
      setOnnxState(`Ready · ${model.backend} · ${(info.onnx_bytes / 1024 / 1024).toFixed(1)} MB`)
      notify('Exported model loaded into this browser.', 'ok')
    } catch (error) {
      setOnnxState('')
      setError(shortEngineError(error))
    }
  }

  if (!project) return null

  const detections = result?.task === 'detect' ? result.boxes ?? [] : []
  const scores = result?.task === 'classify' ? result.predictions ?? [] : []
  const coverage = result?.task === 'segment' ? result.present ?? [] : []

  return (
    <section className="card preview-card">
      <header className="card-head">
        <div>
          <div className="card-title-row">
            <span className="step-badge">3</span>
            <h2>Preview</h2>
          </div>
          <p className="card-sub">Try the model on something it has never seen.</p>
        </div>
        {completed.length > 1 ? (
          <select className="mini-select" value={activeRun?.id ?? ''} onChange={(event) => setRunId(event.target.value)} aria-label="Model version">
            {completed.map((run) => (
              <option key={run.id} value={run.id}>{run.config.model} · {run.id}</option>
            ))}
          </select>
        ) : activeRun ? (
          <span className="model-tag">{activeRun.config.model}</span>
        ) : null}
      </header>

      {!completed.length ? (
        <div className="preview-locked">
          <div className="preview-art" aria-hidden>
            <span className="preview-frame" />
            <span className="preview-chip" />
            <span className="preview-dot" />
          </div>
          <strong>Preview appears after training</strong>
          <p>Train a model once and you can point it at an image or your camera here.</p>
        </div>
      ) : (
        <>
          <div className="preview-tabs" role="tablist" aria-label="Preview source">
            {SOURCES.map((entry) => (
              <button
                key={entry.value}
                type="button"
                role="tab"
                aria-selected={source === entry.value}
                className={source === entry.value ? 'preview-tab active' : 'preview-tab'}
                onClick={() => {
                  setSource(entry.value)
                  setResult(null)
                  setStarted(false)
                  setError(null)
                }}
              >
                {entry.label}
              </button>
            ))}
          </div>

          <div
            className={started ? 'preview-stage has-content' : 'preview-stage'}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault()
              const file = event.dataTransfer.files?.[0]
              if (file && source === 'image') showImage(file)
            }}
          >
            <canvas ref={canvasRef} className={started ? 'preview-canvas has-content' : 'preview-canvas'} />
            {!started ? (
              <div className="preview-empty">
                <div className="preview-art" aria-hidden>
                  <span className="preview-frame" />
                  <span className="preview-chip" />
                  <span className="preview-dot" />
                </div>
                <strong>Drop an image to see it think</strong>
                <p>
                  {source === 'camera'
                    ? 'Start the camera below.'
                    : source === 'video'
                      ? 'Choose a video below.'
                      : source === 'onnx'
                        ? 'Prepare the exported model, then choose an image.'
                        : 'The model runs immediately and draws what it finds.'}
                </p>
              </div>
            ) : null}
            {busy ? <span className="preview-busy">Looking…</span> : null}
            {source === 'image' ? (
              <label className="preview-drop" title="Choose an image">
                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (file) showImage(file)
                    event.target.value = ''
                  }}
                />
                {started ? (
                  <span className="preview-change">
                    <IconUpload size={14} /> Change image
                  </span>
                ) : null}
              </label>
            ) : null}
            <video ref={videoRef} muted playsInline className="hidden-video" />
          </div>

          <div className="preview-controls">
            {source === 'image' ? (
              <label className="btn btn-secondary btn-sm">
                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (file) showImage(file)
                    event.target.value = ''
                  }}
                />
                <span>Choose image</span>
              </label>
            ) : null}

            {source === 'camera' ? (
              streamRef.current ? (
                <Button size="sm" variant="danger" onClick={stopCamera}>
                  Stop camera
                </Button>
              ) : (
                <Button size="sm" variant="primary" onClick={() => void startCamera()}>
                  <IconCamera size={16} /> Start camera
                </Button>
              )
            ) : null}

            {source === 'video' ? (
              <>
                <label className="btn btn-secondary btn-sm">
                  <input
                    type="file"
                    accept="video/*"
                    hidden
                    onChange={(event) => {
                      const file = event.target.files?.[0]
                      if (!file || !videoRef.current) return
                      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
                      objectUrl.current = URL.createObjectURL(file)
                      videoRef.current.src = objectUrl.current
                      videoRef.current.load()
                      setStarted(true)
                      setResult(null)
                      event.target.value = ''
                    }}
                  />
                  <span>Choose video</span>
                </label>
                <Button
                  size="sm"
                  variant="primary"
                  onClick={() => (playing ? videoRef.current?.pause() : void videoRef.current?.play())}
                >
                  <IconPlay size={14} /> {playing ? 'Pause' : 'Play + detect'}
                </Button>
                <Button
                  size="sm"
                  onClick={async () => {
                    const video = videoRef.current
                    if (!video || !video.videoWidth) return
                    const canvas = document.createElement('canvas')
                    canvas.width = video.videoWidth
                    canvas.height = video.videoHeight
                    canvas.getContext('2d')?.drawImage(video, 0, 0)
                    await predictBlob(await canvasToBlob(canvas, 'image/png'), video, video.videoWidth, video.videoHeight)
                  }}
                >
                  Detect a frame
                </Button>
              </>
            ) : null}

            {source === 'onnx' ? (
              <>
                <Button
                  size="sm"
                  variant={onnxModel ? 'secondary' : 'primary'}
                  onClick={() => void loadBrowserModel()}
                  busy={onnxState.startsWith('Building') || onnxState.startsWith('Loading')}
                >
                  {onnxModel ? 'Reload exported model' : 'Prepare browser model'}
                </Button>
                <label className="btn btn-secondary btn-sm">
                  <input
                    type="file"
                    accept="image/*"
                    hidden
                    onChange={(event) => {
                      const file = event.target.files?.[0]
                      if (!file) return
                      if (!onnxModel) {
                        notify('Prepare the browser model first.', 'error')
                        return
                      }
                      const url = URL.createObjectURL(file)
                      const image = new Image()
                      image.onload = () => {
                        setStarted(true)
                        void (async () => {
                          setBusy(true)
                          try {
                            const prediction = await runOnnx(onnxModel, image, image.naturalWidth, image.naturalHeight, { conf, iou })
                            setResult(prediction)
                            draw(image, image.naturalWidth, image.naturalHeight, prediction)
                          } finally {
                            setBusy(false)
                            URL.revokeObjectURL(url)
                          }
                        })()
                      }
                      image.src = url
                      event.target.value = ''
                    }}
                  />
                  <span>Choose image</span>
                </label>
                <span className="field-hint">{onnxState || 'Uses the exported ONNX file, not Python.'}</span>
              </>
            ) : null}
          </div>

          {project.task === 'detect' ? (
            <label className="preview-confidence">
              <span>Confidence</span>
              <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(event) => setConf(Number(event.target.value))} />
              <span className="confidence-value">{conf.toFixed(2)}</span>
            </label>
          ) : null}

          {cameraError ? <Callout tone="warn">{cameraError}</Callout> : null}
          {error ? <Callout tone="bad" title="Prediction failed">{error}</Callout> : null}

          <div className="preview-results">
            <div className="preview-results-head">
              <span>{result?.task === 'classify' ? 'Scores' : result?.task === 'segment' ? 'Coverage' : 'Detections'}</span>
              <span>{result ? `${Math.round(result.ms)} ms` : started && busy ? 'running…' : '—'}</span>
            </div>

            {result?.task === 'detect' ? (
              detections.length ? (
                <ul className="detection-list">
                  {detections.slice(0, 8).map((box, index) => (
                    <li key={index}>
                      <span className="class-dot" style={{ background: project.classes[box.cls]?.color }} aria-hidden />
                      <span>{project.classes[box.cls]?.name ?? `class ${box.cls}`}</span>
                      <span className="detection-score">{Math.round(box.score * 100)}%</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="preview-note">Nothing above {conf.toFixed(2)}. Try lowering confidence, or add more examples of this view.</p>
              )
            ) : null}

            {result?.task === 'classify' ? (
              <ul className="detection-list">
                {scores.map((entry) => (
                  <li key={entry.cls}>
                    <span className="class-dot" style={{ background: project.classes[entry.cls]?.color }} aria-hidden />
                    <span>{project.classes[entry.cls]?.name ?? `class ${entry.cls}`}</span>
                    <span className="detection-score">{(entry.score * 100).toFixed(1)}%</span>
                  </li>
                ))}
              </ul>
            ) : null}

            {result?.task === 'segment' ? (
              coverage.length ? (
                <ul className="detection-list">
                  {coverage.map((entry) => (
                    <li key={entry.cls}>
                      <span className="class-dot" style={{ background: project.classes[entry.cls - 1]?.color }} aria-hidden />
                      <span>{project.classes[entry.cls - 1]?.name ?? `class ${entry.cls}`}</span>
                      <span className="detection-score">{(entry.fraction * 100).toFixed(0)}%</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="preview-note">No region covered a meaningful part of this image.</p>
              )
            ) : null}

            {!result ? <p className="preview-note">Predictions appear here, with real latency from this machine.</p> : null}
          </div>

          <p className="preview-foot">
            {source === 'onnx' ? 'Running the exported ONNX graph in WebAssembly/WebGPU.' : online ? 'Inference runs on the local engine.' : 'Engine offline: start it to run predictions.'}
          </p>
        </>
      )}
    </section>
  )
}
