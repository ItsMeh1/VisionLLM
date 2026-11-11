import { useCallback, useEffect, useRef, useState } from 'react'
import { Badge, Button, Callout, Panel, PanelHead, Segmented, Stat } from '../components/ui'
import { useProject } from '../lib/store'
import { useApp } from '../lib/app'
import { disposeOnnxModel, loadOnnxModel, runOnnx, type BrowserPrediction, type LoadedModel } from '../lib/onnx'
import {
  engineBase,
  exportRun,
  predict,
  shortEngineError,
  type Prediction,
} from '../lib/engine'
import { canvasToBlob } from '../lib/image'
import { navigate } from '../lib/router'

type SourceKind = 'image' | 'video' | 'webcam' | 'onnx'

export function TestScreen() {
  const { project, runs } = useProject()
  const { online, notify } = useApp()
  const completed = runs.filter((run) => run.status === 'completed')
  const [runId, setRunId] = useState(completed[0]?.id ?? '')
  const [source, setSource] = useState<SourceKind>('image')
  const [conf, setConf] = useState(0.3)
  const [iou, setIou] = useState(0.45)
  const [result, setResult] = useState<Prediction | BrowserPrediction | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [onnxModel, setOnnxModel] = useState<LoadedModel | null>(null)
  const [onnxState, setOnnxState] = useState<string>('')
  const [live, setLive] = useState(false)

  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
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
    return () => {
      if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
      if (timerRef.current) window.clearInterval(timerRef.current)
      streamRef.current?.getTracks().forEach((track) => track.stop())
      void disposeOnnxModel(onnxModel)
    }
    // ONNX session is released explicitly when replaced; this cleanup covers unmount.
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
        context.lineWidth = Math.max(2, canvas.width / 320)
        context.font = `${Math.max(11, Math.round(canvas.width / 45))}px 'JetBrains Mono Variable', monospace`
        for (const box of prediction.boxes) {
          const cls = project.classes[box.cls]
          const color = cls?.color ?? '#FF4A1C'
          const x = box.x * canvas.width
          const y = box.y * canvas.height
          const w = box.w * canvas.width
          const h = box.h * canvas.height
          context.strokeStyle = color
          context.strokeRect(x, y, w, h)
          const label = `${cls?.name ?? '?'} ${Math.round(box.score * 100)}%`
          const widthPx = context.measureText(label).width
          const top = y - 20 < 0 ? y : y - 20
          context.fillStyle = color
          context.fillRect(x, top, widthPx + 8, 20)
          context.fillStyle = '#0b0c0b'
          context.fillText(label, x + 4, top + 15)
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

  const pickImage = (file: File) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
    objectUrl.current = URL.createObjectURL(file)
    const image = new Image()
    image.onload = () => {
      imageRef.current = image
      const canvas = canvasRef.current
      if (canvas) {
        canvas.width = image.naturalWidth
        canvas.height = image.naturalHeight
        const context = canvas.getContext('2d')
        context?.drawImage(image, 0, 0)
      }
      setResult(null)
      void predictBlob(file, image, image.naturalWidth, image.naturalHeight)
    }
    image.src = objectUrl.current
  }

  const startCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 } }, audio: false })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setLive(true)
      }
    } catch {
      setCameraError('Camera access was refused or no camera is available.')
    }
  }

  const stopCamera = () => {
    setLive(false)
    if (timerRef.current) window.clearInterval(timerRef.current)
    timerRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
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
    }, 700)
  }, [predictBlob])

  useEffect(() => {
    if ((source === 'webcam' && live) || (source === 'video' && playing)) runFrameLoop()
    if (source !== 'webcam' && source !== 'video') {
      if (timerRef.current) window.clearInterval(timerRef.current)
      timerRef.current = null
    }
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current)
    }
  }, [source, live, runFrameLoop])

  useEffect(() => {
    if (source !== 'webcam') stopCamera()
  }, [source])

  const loadBrowserModel = async () => {
    if (!project || !activeRun) return
    setOnnxState('building export…')
    setError(null)
    try {
      const info = await exportRun(project.id, activeRun.id)
      setOnnxState('loading ONNX Runtime…')
      const base = engineBase()
      const model = await loadOnnxModel(
        `${base}/projects/${project.id}/runs/${activeRun.id}/export/model.onnx`,
        `${base}/projects/${project.id}/runs/${activeRun.id}/export/labels.json`,
      )
      setOnnxModel(model)
      setOnnxState(`ready · ${model.backend} · ${(info.onnx_bytes / 1024 / 1024).toFixed(1)} MB`)
      notify('ONNX model loaded into the browser.', 'ok')
    } catch (error) {
      setOnnxState('')
      setError(shortEngineError(error))
    }
  }

  if (!project) return null

  if (!completed.length) {
    return (
      <div className="screen">
        <Panel>
          <div className="empty-block">
            <h3>No trained model yet</h3>
            <p className="empty-copy">Test uses a finished checkpoint. Train a model first, then point a camera at it.</p>
            <Button variant="primary" onClick={() => navigate(`/p/${project.id}/train`)}>Go to training →</Button>
          </div>
        </Panel>
      </div>
    )
  }

  return (
    <div className="screen test-screen">
      <Panel>
        <PanelHead
          eyebrow="05 · Test"
          title="Point it at something new"
          note="Images, video frames and webcam frames are letterboxed exactly like training data. Nothing is saved unless you ask."
          actions={
            <div className="button-row">
              <select value={activeRun?.id ?? ''} onChange={(event) => setRunId(event.target.value)} aria-label="Model run">
                {completed.map((run) => (
                  <option key={run.id} value={run.id}>{run.config.model} · {run.id}</option>
                ))}
              </select>
              <Badge tone={online ? 'ok' : 'bad'}>{online ? 'engine' : 'engine offline'}</Badge>
            </div>
          }
        />

        <div className="test-toolbar">
          <Segmented
            ariaLabel="Input source"
            value={source}
            onChange={(next) => {
              setSource(next)
              setResult(null)
            }}
            options={[
              { value: 'image', label: 'Image' },
              { value: 'video', label: 'Video' },
              { value: 'webcam', label: 'Webcam' },
              { value: 'onnx', label: 'In browser' },
            ]}
          />
          <div className="slider-group">
            <label>
              <span className="label">Confidence <span className="num">{conf.toFixed(2)}</span></span>
              <input type="range" min={0.05} max={0.95} step={0.05} value={conf} onChange={(event) => setConf(Number(event.target.value))} />
            </label>
            <label>
              <span className="label">NMS IoU <span className="num">{iou.toFixed(2)}</span></span>
              <input type="range" min={0.1} max={0.9} step={0.05} value={iou} onChange={(event) => setIou(Number(event.target.value))} />
            </label>
          </div>
        </div>

        {source === 'onnx' ? (
          <Callout tone="info" title="Run the exported ONNX file in this browser">
            <p>This executes the actual exported artifact with ONNX Runtime Web, not the Python model.</p>
            <div className="button-row">
              <Button size="sm" variant="primary" onClick={() => void loadBrowserModel()} busy={onnxState.startsWith('build') || onnxState.startsWith('loading')}>
                Build + load ONNX
              </Button>
              <span className="mono">{onnxState || 'not loaded'}</span>
            </div>
          </Callout>
        ) : null}

        {error ? <Callout tone="bad" title="Prediction failed">{error}</Callout> : null}

        <div className="test-layout">
          <div className="test-stage">
            {source === 'image' ? (
              <label className="test-drop">
                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (file) pickImage(file)
                    event.target.value = ''
                  }}
                />
                <canvas ref={canvasRef} className={result ? 'test-canvas has-result' : 'test-canvas'} />
                {!result ? (
                  <span className="test-drop-hint">
                    <strong>Drop an image or click to choose</strong>
                    <span className="mono">the model will run immediately</span>
                  </span>
                ) : null}
              </label>
            ) : null}

            {source === 'video' ? (
              <div className="test-video">
                <input
                  type="file"
                  accept="video/*"
                  onChange={(event) => {
                    const file = event.target.files?.[0]
                    if (!file || !videoRef.current) return
                    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current)
                    objectUrl.current = URL.createObjectURL(file)
                    videoRef.current.src = objectUrl.current
                    videoRef.current.load()
                    setResult(null)
                    event.target.value = ''
                  }}
                />
                <canvas ref={canvasRef} className="test-canvas has-result" />
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  className="hidden-video"
                  onPlay={() => setPlaying(true)}
                  onPause={() => setPlaying(false)}
                  onLoadedData={() => {
                    const video = videoRef.current
                    if (video) {
                      const canvas = canvasRef.current
                      if (canvas) {
                        canvas.width = video.videoWidth
                        canvas.height = video.videoHeight
                        canvas.getContext('2d')?.drawImage(video, 0, 0)
                      }
                    }
                  }}
                />
                <div className="video-controls">
                  <Button size="sm" variant="primary" onClick={() => (playing ? videoRef.current?.pause() : void videoRef.current?.play())}>
                    {playing ? 'Pause' : 'Play + detect'}
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
                    Detect this frame
                  </Button>
                  <span className="mono field-hint">sampling every 0.7s while playing</span>
                </div>
              </div>
            ) : null}

            {source === 'webcam' || source === 'onnx' ? (
              <div className="test-video">
                <canvas ref={canvasRef} className="test-canvas has-result" />
                <video ref={videoRef} muted playsInline className="hidden-video" />
                <div className="video-controls">
                  {source === 'webcam' ? (
                    live ? (
                      <Button size="sm" variant="danger" onClick={stopCamera}>Stop camera</Button>
                    ) : (
                      <Button size="sm" variant="primary" onClick={() => void startCamera()}>Start camera</Button>
                    )
                  ) : null}
                  {source === 'onnx' ? (
                    <>
                      <label className="file-button">
                        Choose image
                        <input
                          type="file"
                          accept="image/*"
                          hidden
                          onChange={(event) => {
                            const file = event.target.files?.[0]
                            if (!file) return
                            const url = URL.createObjectURL(file)
                            const image = new Image()
                            image.onload = () => {
                              void (async () => {
                                const canvas = document.createElement('canvas')
                                canvas.width = image.naturalWidth
                                canvas.height = image.naturalHeight
                                canvas.getContext('2d')?.drawImage(image, 0, 0)
                                if (!onnxModel) {
                                  notify('Load the ONNX model first.', 'error')
                                  return
                                }
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
                      </label>
                      <span className="mono field-hint">{onnxModel ? 'session ready' : 'build + load to begin'}</span>
                    </>
                  ) : null}
                </div>
                {cameraError ? <Callout tone="warn">{cameraError}</Callout> : null}
              </div>
            ) : null}
          </div>

          <aside className="test-side">
            <div className="stat-row tight">
              <Stat label="Latency" value={result ? `${Math.round(result.ms)}ms` : '—'} hint="last frame" />
              <Stat label="Backend" value={result && 'backend' in result ? result.backend : 'pytorch'} />
            </div>
            {result?.task === 'detect' ? (
              <>
                <div className="label">Detections</div>
                <ul className="detection-list">
                  {(result.boxes ?? []).slice(0, 20).map((box, index) => (
                    <li key={index}>
                      <span className="swatch" style={{ background: project.classes[box.cls]?.color }} />
                      <span>{project.classes[box.cls]?.name ?? `class ${box.cls}`}</span>
                      <span className="mono">{Math.round(box.score * 100)}%</span>
                      <span className="mono muted">{Math.round(box.w)}×{Math.round(box.h)}</span>
                    </li>
                  ))}
                  {!result.boxes?.length ? <li className="mono muted">nothing above {conf.toFixed(2)}</li> : null}
                </ul>
              </>
            ) : null}
            {result?.task === 'classify' ? (
              <>
                <div className="label">Scores</div>
                <ul className="detection-list">
                  {(result.predictions ?? []).map((entry) => (
                    <li key={entry.cls}>
                      <span className="swatch" style={{ background: project.classes[entry.cls]?.color }} />
                      <span>{project.classes[entry.cls]?.name ?? `class ${entry.cls}`}</span>
                      <span className="mono">{(entry.score * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
            {result?.task === 'segment' ? (
              <>
                <div className="label">Coverage</div>
                <ul className="detection-list">
                  {(result.present ?? []).map((entry) => (
                    <li key={entry.cls}>
                      <span className="swatch" style={{ background: project.classes[entry.cls]?.color }} />
                      <span>{project.classes[entry.cls]?.name}</span>
                      <span className="mono">{(entry.fraction * 100).toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </>
            ) : null}
            {busy ? <span className="mono field-hint">running inference…</span> : null}
            <Callout tone="info" title="Test data is disposable">
              Images tested here are never added to the training set. If you want a frame in the dataset, add it through the
              Dataset step first.
            </Callout>
          </aside>
        </div>
      </Panel>
    </div>
  )
}
