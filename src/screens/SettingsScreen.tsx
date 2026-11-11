import { useEffect, useState } from 'react'
import { Badge, Button, Callout, Panel, PanelHead } from '../components/ui'
import { useApp } from '../lib/app'
import { clearStore, estimateStorage, getAll } from '../lib/db'
import { formatBytes } from '../lib/image'
import { navigate } from '../lib/router'
import type { ImageRecord } from '../lib/types'

export function SettingsScreen() {
  const { health, online, checking, refreshHealth, notify } = useApp()
  const [storage, setStorage] = useState<{ usage: number; quota: number } | null>(null)
  const [images, setImages] = useState<ImageRecord[]>([])

  useEffect(() => {
    void estimateStorage().then(setStorage)
    void getAll<ImageRecord>('images').then(setImages)
  }, [])

  const totalBytes = images.reduce((sum, image) => sum + image.bytes, 0)

  const clearEverything = async () => {
    if (!window.confirm('Delete every VisionLLM project, image and run stored in this browser? This cannot be undone; exported files and models in ~/.visionllm are separate.')) return
    await Promise.all(['projects', 'images', 'runs', 'agentRuns'].map((store) => clearStore(store as 'projects')))
    notify('Local browser data cleared.', 'ok')
    navigate('/')
  }

  return (
    <main className="page settings-page">
      <header className="page-head">
        <div>
          <div className="label">Settings</div>
          <h1>How VisionLLM runs</h1>
        </div>
        <a className="text-link" href="#/">← Projects</a>
      </header>

      <div className="settings-grid">
        <Panel>
          <PanelHead
            eyebrow="Engine"
            title="Training companion"
            note="A small Python service on 127.0.0.1:8765. It only listens on loopback."
            actions={<Badge tone={online ? 'ok' : 'bad'}>{online ? 'online' : 'offline'}</Badge>}
          />
          {health ? (
            <dl className="meta-list wide">
              <div><dt>Device</dt><dd className="mono">{health.device}</dd></div>
              <div><dt>PyTorch</dt><dd className="mono">{health.torch}</dd></div>
              <div><dt>torchvision</dt><dd className="mono">{health.torchvision}</dd></div>
              <div><dt>Python</dt><dd className="mono">{health.python}</dd></div>
              <div><dt>Apple GPU</dt><dd className="mono">{health.mps ? 'available' : 'no'}</dd></div>
              <div><dt>CUDA</dt><dd className="mono">{health.cuda ? 'available' : 'no'}</dd></div>
              <div><dt>Model folder</dt><dd className="mono">{health.storage}</dd></div>
            </dl>
          ) : (
            <Callout tone="warn" title="Not reachable">
              Open a terminal in the project folder and run <span className="mono">npm run engine</span>. Then press
              refresh.
            </Callout>
          )}
          <div className="button-row">
            <Button size="sm" onClick={() => void refreshHealth()} busy={checking}>Check again</Button>
          </div>
          <div className="start-command">
            <span className="label">Start it</span>
            <code className="mono">npm run engine</code>
          </div>
        </Panel>

        <Panel>
          <PanelHead eyebrow="Storage" title="What lives where" />
          <dl className="meta-list wide">
            <div><dt>Browser</dt><dd>{images.length} images · {formatBytes(totalBytes)} of originals</dd></div>
            <div><dt>Browser quota</dt><dd className="mono">{storage ? `${formatBytes(storage.usage)} / ${formatBytes(storage.quota)}` : 'unknown'}</dd></div>
            <div><dt>Engine</dt><dd className="mono">{health?.storage ?? '~/.visionllm'}</dd></div>
          </dl>
          <p className="field-hint">
            Trainers receive a copy of the dataset inside the engine folder. Deleting a project in the app removes the
            browser copy; the engine copy can be removed by deleting that project's folder.
          </p>
          <Button size="sm" variant="danger" onClick={() => void clearEverything()}>
            Clear all browser data
          </Button>
        </Panel>

        <Panel className="span-2">
          <PanelHead eyebrow="Licences" title="Open source, deliberately" />
          <div className="licence-grid">
            <div><Badge tone="ok">Apache-2.0</Badge><p>VisionLLM application and the VLYOLO detector implementation.</p></div>
            <div><Badge tone="ok">BSD-3-Clause</Badge><p>PyTorch, torchvision and the pretrained COCO detectors.</p></div>
            <div><Badge tone="ok">MIT</Badge><p>ONNX Runtime Web and React.</p></div>
            <div><Badge tone="neutral">No accounts</Badge><p>There is no server-side component of VisionLLM. No telemetry is collected.</p></div>
          </div>
        </Panel>
      </div>
    </main>
  )
}
