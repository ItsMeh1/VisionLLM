import { useEffect, useState } from 'react'
import { Badge, Button, Callout, Panel, PanelHead, Stat } from '../components/ui'
import { useProject } from '../lib/store'
import { useApp } from '../lib/app'
import { engineBase, exportRun, shortEngineError, type ExportInfo } from '../lib/engine'
import { copyText } from '../lib/hooks'
import { navigate } from '../lib/router'

export function ExportScreen() {
  const { project, runs } = useProject()
  const { notify } = useApp()
  const completed = runs.filter((run) => run.status === 'completed')
  const [runId, setRunId] = useState(completed[0]?.id ?? '')
  const [info, setInfo] = useState<ExportInfo | null>(null)
  const [busy, setBusy] = useState(false)
  const [snippet, setSnippet] = useState('')
  const [card, setCard] = useState('')
  const activeRun = completed.find((run) => run.id === runId) ?? completed[0]

  useEffect(() => {
    if (!runId && completed.length) setRunId(completed[0].id)
  }, [completed, runId])

  const build = async () => {
    if (!project || !activeRun) return
    setBusy(true)
    setSnippet('')
    setCard('')
    try {
      const response = await exportRun(project.id, activeRun.id)
      setInfo(response)
      const base = engineBase()
      const jsUrl = `${base}${response.downloads['inference.js']}`
      const cardUrl = `${base}${response.downloads['model-card.json']}`
      setSnippet(await fetch(jsUrl).then((res) => res.text()))
      setCard(await fetch(cardUrl).then((res) => res.text()))
      notify('Export built and verified with ONNX Runtime.', 'ok')
    } catch (error) {
      notify(shortEngineError(error), 'error')
    } finally {
      setBusy(false)
    }
  }

  if (!project) return null

  if (!completed.length) {
    return (
      <div className="screen">
        <Panel>
          <div className="empty-block">
            <h3>Nothing trained yet</h3>
            <p className="empty-copy">Export takes a finished checkpoint and produces ONNX, labels, a model card and a JS example.</p>
            <Button variant="primary" onClick={() => navigate(`/p/${project.id}/workspace`)}>Go to workspace →</Button>
          </div>
        </Panel>
      </div>
    )
  }

  return (
    <div className="screen export-screen">
      <Panel>
        <PanelHead
          eyebrow="Export"
          title="A model file you can actually use"
          note="ONNX is exported at the input size the model was trained at, then loaded into ONNX Runtime to prove the graph runs."
          actions={
            <div className="button-row">
              <select value={activeRun?.id ?? ''} onChange={(event) => setRunId(event.target.value)} aria-label="Run to export">
                {completed.map((run) => (
                  <option key={run.id} value={run.id}>{run.config.model} · {run.id}</option>
                ))}
              </select>
              <Button variant="primary" size="sm" onClick={() => void build()} busy={busy}>
                Build export
              </Button>
            </div>
          }
        />

        <div className="export-plan">
          <div>
            <span className="label">Format</span>
            <span className="mono">ONNX · opset 17</span>
          </div>
          <div>
            <span className="label">Input</span>
            <span className="mono">{activeRun?.config.imgsz}×{activeRun?.config.imgsz} · NCHW · 0..1</span>
          </div>
          <div>
            <span className="label">Preprocessing</span>
            <span className="mono">letterbox · pad 114</span>
          </div>
          <div>
            <span className="label">Licence</span>
            <span className="mono">Apache-2.0</span>
          </div>
        </div>

        {!info ? (
          <Callout tone="info" title="What gets built">
            <p>
              <span className="mono">model.onnx</span>, a <span className="mono">labels.json</span> describing classes and
              preprocessing, a working <span className="mono">inference.js</span>, a model card with the real evaluation
              numbers, and a YOLO-format copy of your annotations.
            </p>
          </Callout>
        ) : (
          <>
            <div className="stat-row">
              <Stat label="ONNX size" value={`${(info.onnx_bytes / 1024 / 1024).toFixed(1)} MB`} />
              <Stat
                label="Graph verification"
                value={info.verification.ran ? 'passed' : 'failed'}
                tone={info.verification.ran ? 'ok' : 'bad'}
                hint={info.verification.ran ? `max Δ ${info.verification.max_abs_diff?.toExponential(1)}` : info.verification.error}
              />
              <Stat label="Output" value={`[${info.verification.output_shape?.join(', ') ?? '—'}]`} hint="same as PyTorch" />
              <Stat label="Files" value={info.files.length} hint="in the bundle" />
            </div>

            <div className="export-files">
              {info.files.map((file) => (
                <a key={file} className="file-row" href={`${engineBase()}${info.downloads[file]}`} download>
                  <span className="mono file-name">{file}</span>
                  <span className="mono file-action">download ↓</span>
                </a>
              ))}
            </div>

            <div className="button-row">
              <a className="btn btn-primary btn-sm" href={`${engineBase()}${info.downloads[info.zip]}`} download>
                <span>Download bundle (.zip)</span>
              </a>
              <Button
                size="sm"
                variant="ghost"
                onClick={async () => {
                  const ok = await copyText(snippet)
                  notify(ok ? 'inference.js copied.' : 'The browser blocked clipboard access.', ok ? 'ok' : 'error')
                }}
              >
                Copy inference.js
              </Button>
              <a className="text-link" href={`#/p/${project.id}/workspace`}>Test in browser →</a>
            </div>
          </>
        )}
      </Panel>

      {snippet ? (
        <Panel>
          <PanelHead
            eyebrow="Integration"
            title="inference.js, exactly as generated"
            note="This is the same file inside the bundle. It needs onnxruntime-web and a plain <img> or <video> element."
          />
          <pre className="code-block">{snippet}</pre>
        </Panel>
      ) : null}

      {card ? (
        <Panel>
          <PanelHead eyebrow="Provenance" title="Model card" note="Written from the run's real configuration and metrics." />
          <pre className="code-block small">{card}</pre>
        </Panel>
      ) : null}

      {project.task === 'detect' ? (
        <Panel>
          <PanelHead eyebrow="Dataset" title="Your annotations, in YOLO text format" />
          <p className="field-hint">
            One <span className="mono">.txt</span> per image plus <span className="mono">data.yaml</span>, with the
            untouched images. Useful for training elsewhere or auditing the labels.
          </p>
          <a className="btn btn-secondary btn-sm" href={`${engineBase()}/projects/${project.id}/dataset.zip`} download>
            <span>Download dataset-yolo.zip</span>
          </a>
        </Panel>
      ) : null}

      <Panel>
        <PanelHead eyebrow="Licensing" title="Why this stack" />
        <div className="licence-grid">
          <div>
            <Badge tone="ok">Apache-2.0</Badge>
            <p>VLYOLO architecture and the VLYOLO/U-Net training code in this project. Models you train are yours to use.</p>
          </div>
          <div>
            <Badge tone="ok">BSD-3-Clause</Badge>
            <p>PyTorch, torchvision, and the COCO-pretrained detectors used for annotation suggestions and ResNet-18.</p>
          </div>
          <div>
            <Badge tone="ok">MIT</Badge>
            <p>ONNX Runtime Web, which runs the exported graph in the browser.</p>
          </div>
          <div>
            <Badge tone="warn">Deliberately not used</Badge>
            <p>
              Ultralytics YOLOv5/v8/YOLO11 are AGPL-3.0, and Ultralytics states that the licence extends to models trained
              with their code. Wrapping it would put that obligation on your model.
            </p>
          </div>
        </div>
      </Panel>
    </div>
  )
}
