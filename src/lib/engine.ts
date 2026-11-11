/** Client for the local VisionLLM training engine (Python, loopback only). */

import type { ImageRecord, Project, RunRecord, Task, TrainingConfig } from './types'

const isDev = typeof location !== 'undefined' && location.port === '5173'
const BASE = (import.meta.env.VITE_ENGINE_URL as string | undefined) || (isDev ? '/engine' : 'http://127.0.0.1:8765')

export class EngineError extends Error {
  status: number
  constructor(message: string, status = 0) {
    super(message)
    this.status = status
    this.name = 'EngineError'
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 30000): Promise<T> {
  const controller = new AbortController()
  const timer = window.setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal })
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`
      try {
        const body = await response.json()
        detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
      } catch {
        /* keep the status text */
      }
      throw new EngineError(detail, response.status)
    }
    return (await response.json()) as T
  } catch (error) {
    if (error instanceof EngineError) throw error
    if ((error as Error).name === 'AbortError') throw new EngineError('The engine did not respond in time.', 0)
    throw new EngineError(
      'Cannot reach the local engine on port 8765. Start it with `npm run engine` in the project folder.',
      0,
    )
  } finally {
    window.clearTimeout(timer)
  }
}

export function engineBase(): string {
  return BASE
}

export interface Health {
  ok: boolean
  engine: string
  python: string
  platform: string
  torch: string
  torchvision: string
  device: string
  mps: boolean
  cuda: boolean
  storage: string
  models: Record<string, { name: string; head: number; description: string }>
  pretrained: Record<string, { label: string; note: string; license: string; loaded: boolean }>
  licenses: Record<string, string>
}

export function health(): Promise<Health> {
  return request<Health>('/health', undefined, 6000)
}

const EXT: Record<string, string> = {
  'image/jpeg': 'jpg',
  'image/png': 'png',
  'image/webp': 'webp',
  'image/gif': 'gif',
  'image/bmp': 'bmp',
  'image/tiff': 'tif',
}

function extFor(image: ImageRecord): string {
  return EXT[image.mime] ?? 'png'
}

export interface UploadResult {
  stored: number
  skipped: { id: string; reason: string }[]
  counts: { train: number; val: number }
  storage: string
}

export async function uploadDataset(project: Project, images: ImageRecord[]): Promise<UploadResult> {
  const form = new FormData()
  const manifest = {
    task: project.task,
    classes: project.classes.map((c) => c.name),
    images: images.map((image) => {
      const boxIndex = (cls: string) => project.classes.findIndex((c) => c.id === cls)
      return {
        id: image.id,
        file: `${image.id}.${extFor(image)}`,
        split: image.split,
        originalName: image.name,
        label: image.label ? boxIndex(image.label) : undefined,
        boxes: image.boxes
          .map((box) => ({ cls: boxIndex(box.cls), x: box.x, y: box.y, w: box.w, h: box.h }))
          .filter((box) => box.cls >= 0),
        shapes: image.shapes
          .map((shape) => ({ cls: boxIndex(shape.cls), points: shape.points }))
          .filter((shape) => shape.cls >= 0),
      }
    }),
  }
  form.append('manifest', JSON.stringify(manifest))
  for (const image of images) form.append('files', image.blob, `${image.id}.${extFor(image)}`)
  return request<UploadResult>(`/projects/${project.id}/dataset`, { method: 'POST', body: form }, 180000)
}

export interface JobSnapshot {
  id: string
  kind: string
  run_id: string
  status: 'queued' | 'running' | 'cancelling' | 'completed' | 'failed' | 'cancelled'
  progress: { epoch: number; epochs: number; percent: number }
  events: { t: number; type: string; message?: string; record?: Record<string, number> }[]
  result: {
    status?: string
    metrics?: Record<string, unknown>
    classes?: string[]
    epochs_run?: number
    best_epoch?: number
    elapsed_seconds?: number
    notes?: string[]
    error?: string
  } | null
  error: string | null
}

export function startTraining(
  projectId: string,
  config: TrainingConfig,
  runId: string,
): Promise<{ job_id: string; run_id: string; task: Task }> {
  return request(`/projects/${projectId}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...config, run_id: runId }),
  }, 20000)
}

export function jobStatus(jobId: string): Promise<JobSnapshot> {
  return request<JobSnapshot>(`/jobs/${jobId}`, undefined, 15000)
}

export function cancelJob(jobId: string): Promise<{ ok: boolean }> {
  return request(`/jobs/${jobId}/cancel`, { method: 'POST' })
}

export async function followJob(
  jobId: string,
  onUpdate: (snapshot: JobSnapshot) => void,
  signal?: AbortSignal,
): Promise<JobSnapshot> {
  let snapshot = await jobStatus(jobId)
  onUpdate(snapshot)
  while (!['completed', 'failed', 'cancelled'].includes(snapshot.status)) {
    if (signal?.aborted) return snapshot
    await new Promise((resolve) => window.setTimeout(resolve, 900))
    if (signal?.aborted) return snapshot
    snapshot = await jobStatus(jobId)
    onUpdate(snapshot)
  }
  return snapshot
}

export interface DetectionBox {
  cls: number
  score: number
  x: number
  y: number
  w: number
  h: number
  xn: number
  yn: number
  wn: number
  hn: number
}

export interface Prediction {
  task: Task
  width: number
  height: number
  ms: number
  boxes?: DetectionBox[]
  predictions?: { cls: number; score: number }[]
  present?: { cls: number; fraction: number }[]
  mask_png?: string
}

export async function predict(
  projectId: string,
  runId: string,
  blobs: Blob[],
  options: { conf: number; iou: number },
): Promise<{ results: Prediction[]; classes: string[] }> {
  const form = new FormData()
  blobs.slice(0, 8).forEach((blob, index) => form.append('files', blob, `frame-${index}.png`))
  form.append('conf', String(options.conf))
  form.append('iou', String(options.iou))
  return request(`/projects/${projectId}/runs/${runId}/predict`, { method: 'POST', body: form }, 120000)
}

export interface AutolabelBox {
  cls: number
  score: number
  x: number
  y: number
  w: number
  h: number
  source: string
}

export interface AutolabelResult {
  file: string
  width: number
  height: number
  boxes: AutolabelBox[]
  matched: number
  detected_categories: Record<string, number>
  model: string
  license: string
}

export async function autolabel(
  blobs: { name: string; blob: Blob }[],
  classes: string[],
  model: string,
  conf: number,
): Promise<{ results: AutolabelResult[] }> {
  const form = new FormData()
  blobs.forEach((entry) => form.append('files', entry.blob, entry.name))
  form.append('classes', JSON.stringify(classes))
  form.append('model', model)
  form.append('conf', String(conf))
  return request(`/autolabel`, { method: 'POST', body: form }, 300000)
}

export interface EvaluationResponse {
  task: Task
  classes: string[]
  images: number
  metrics: Record<string, unknown>
  result: Record<string, unknown>
}

export function evaluateRun(projectId: string, runId: string): Promise<EvaluationResponse> {
  return request(`/projects/${projectId}/runs/${runId}/evaluate`, { method: 'POST' }, 180000)
}

export interface ExportInfo {
  task: Task
  classes: string[]
  input_size: number
  files: string[]
  onnx_bytes: number
  verification: { ran: boolean; max_abs_diff?: number; output_shape?: number[]; onnxruntime?: string; error?: string }
  zip: string
  downloads: Record<string, string>
}

export function exportRun(projectId: string, runId: string): Promise<ExportInfo> {
  return request(`/projects/${projectId}/runs/${runId}/export`, { method: 'POST' }, 180000)
}

export function downloadUrl(projectId: string, runId: string, file: string): string {
  return `${BASE}/projects/${projectId}/runs/${runId}/export/${encodeURIComponent(file)}`
}

export function datasetDownloadUrl(projectId: string): string {
  return `${BASE}/projects/${projectId}/dataset.zip`
}

export async function deleteRemoteProject(projectId: string): Promise<void> {
  try {
    await request(`/projects/${projectId}`, { method: 'DELETE' })
  } catch {
    /* the engine may not be running; local data removal still succeeds */
  }
}

export function shortEngineError(error: unknown): string {
  if (error instanceof EngineError) return error.message
  return error instanceof Error ? error.message : String(error)
}

export function configFromRun(run: RunRecord): TrainingConfig {
  return run.config
}
