/** Browser inference on the exported ONNX model, via onnxruntime-web.
 *  Mirrors the generated inference.js so the shipped example is the same math. */

import type { InferenceSession, Tensor } from 'onnxruntime-web'
import { letterboxGeometry } from './image'
import type { Task } from './types'

export interface BrowserBox {
  cls: number
  label: string
  score: number
  x: number
  y: number
  w: number
  h: number
}

export interface BrowserPrediction {
  task: Task
  width: number
  height: number
  ms: number
  backend: string
  boxes?: BrowserBox[]
  predictions?: { cls: number; label: string; score: number }[]
  maskDataUrl?: string
  present?: { cls: number; label: string; fraction: number }[]
}

let ortModule: typeof import('onnxruntime-web') | null = null

async function ort() {
  if (!ortModule) {
    ortModule = await import('onnxruntime-web')
    // The wasm assets ship in node_modules; Vite serves them from the app origin.
    ortModule.env.wasm.wasmPaths = `${location.origin}/ort/`
    ortModule.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2)
  }
  return ortModule
}

export interface LoadedModel {
  session: InferenceSession
  backend: string
  labels: string[]
  size: number
  task: Task
}

export async function loadOnnxModel(url: string, metaUrl: string): Promise<LoadedModel> {
  const module = await ort()
  const meta = await fetch(metaUrl).then((response) => response.json())
  let backend = 'wasm'
  let session: InferenceSession
  try {
    session = await module.InferenceSession.create(url, { executionProviders: ['webgpu'] })
    backend = 'webgpu'
  } catch {
    session = await module.InferenceSession.create(url, { executionProviders: ['wasm'] })
    backend = 'wasm'
  }
  return {
    session,
    backend,
    labels: meta.classes ?? [],
    size: meta.input_size ?? 512,
    task: (meta.task ?? 'detect') as Task,
  }
}

export async function disposeOnnxModel(model: LoadedModel | null): Promise<void> {
  if (!model) return
  try {
    await model.session.release()
  } catch {
    /* session may already be released */
  }
}

function toTensorData(source: CanvasImageSource, width: number, height: number): Float32Array {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('Canvas is unavailable')
  context.drawImage(source, 0, 0, width, height)
  const { data } = context.getImageData(0, 0, width, height)
  const area = width * height
  const out = new Float32Array(area * 3)
  for (let i = 0; i < area; i++) {
    out[i] = data[i * 4] / 255
    out[area + i] = data[i * 4 + 1] / 255
    out[2 * area + i] = data[i * 4 + 2] / 255
  }
  return out
}

function iou(a: BrowserBox, b: BrowserBox): number {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w)
  const y2 = Math.min(a.y + a.h, b.y + b.h)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  return inter / Math.max(a.w * a.h + b.w * b.h - inter, 1e-9)
}

function nms(boxes: BrowserBox[], threshold: number, max: number): BrowserBox[] {
  const kept: BrowserBox[] = []
  const groups = new Map<number, BrowserBox[]>()
  for (const box of boxes) {
    if (!groups.has(box.cls)) groups.set(box.cls, [])
    groups.get(box.cls)?.push(box)
  }
  for (const group of groups.values()) {
    group.sort((a, b) => b.score - a.score)
    const taken: BrowserBox[] = []
    for (const candidate of group) {
      if (taken.every((other) => iou(candidate, other) <= threshold)) taken.push(candidate)
    }
    kept.push(...taken)
  }
  return kept.sort((a, b) => b.score - a.score).slice(0, max)
}

export async function runOnnx(
  model: LoadedModel,
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  options: { conf: number; iou: number },
): Promise<BrowserPrediction> {
  const module = await ort()
  const geometry = letterboxGeometry(sourceWidth, sourceHeight, model.size)
  const canvas = document.createElement('canvas')
  canvas.width = model.size
  canvas.height = model.size
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas is unavailable')
  context.fillStyle = '#727272'
  context.fillRect(0, 0, model.size, model.size)
  context.drawImage(source, geometry.padX, geometry.padY, geometry.newW, geometry.newH)
  const input = new module.Tensor('float32', toTensorData(canvas, model.size, model.size), [
    1,
    3,
    model.size,
    model.size,
  ])
  const started = performance.now()
  const outputs = await model.session.run({ images: input as Tensor })
  const elapsed = performance.now() - started
  const output = (outputs.detections ?? outputs.probabilities ?? outputs.masks ?? Object.values(outputs)[0]) as Tensor

  if (model.task === 'detect') {
    const [, rows, row] = output.dims as number[]
    const classes = row - 4
    const data = output.data as Float32Array
    const raw: BrowserBox[] = []
    for (let i = 0; i < rows; i++) {
      const offset = i * row
      let best = 0
      let bestClass = 0
      for (let c = 0; c < classes; c++) {
        const score = data[offset + 4 + c]
        if (score > best) {
          best = score
          bestClass = c
        }
      }
      if (best < options.conf) continue
      const cx = data[offset] * model.size
      const cy = data[offset + 1] * model.size
      const w = data[offset + 2] * model.size
      const h = data[offset + 3] * model.size
      raw.push({
        cls: bestClass,
        label: model.labels[bestClass] ?? String(bestClass),
        score: best,
        x: (cx - w / 2 - geometry.padX) / geometry.ratio,
        y: (cy - h / 2 - geometry.padY) / geometry.ratio,
        w: w / geometry.ratio,
        h: h / geometry.ratio,
      })
    }
    return {
      task: 'detect',
      width: sourceWidth,
      height: sourceHeight,
      ms: elapsed,
      backend: model.backend,
      boxes: nms(raw, options.iou, 100),
    }
  }

  if (model.task === 'classify') {
    const data = Array.from(output.data as Float32Array)
    const ranked = data
      .map((score, cls) => ({ cls, label: model.labels[cls] ?? String(cls), score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 5)
    return {
      task: 'classify',
      width: sourceWidth,
      height: sourceHeight,
      ms: elapsed,
      backend: model.backend,
      predictions: ranked,
    }
  }

  const [, channels, height, width] = output.dims as number[]
  const data = output.data as Float32Array
  const mask = document.createElement('canvas')
  mask.width = width
  mask.height = height
  const maskContext = mask.getContext('2d')
  if (!maskContext) throw new Error('Canvas is unavailable')
  const rgba = maskContext.createImageData(width, height)
  const colors = ['#000000', '#FF4A1C', '#4EA8DE', '#F2C14E', '#8AC926', '#C77DFF', '#FF8FA3', '#4ECDC4']
  const counts = new Array(channels).fill(0)
  for (let p = 0; p < width * height; p++) {
    let best = 0
    let bestScore = -Infinity
    for (let c = 0; c < channels; c++) {
      const value = data[c * width * height + p]
      if (value > bestScore) {
        bestScore = value
        best = c
      }
    }
    counts[best]++
    const hex = colors[best] ?? '#000000'
    rgba.data[p * 4] = parseInt(hex.slice(1, 3), 16)
    rgba.data[p * 4 + 1] = parseInt(hex.slice(3, 5), 16)
    rgba.data[p * 4 + 2] = parseInt(hex.slice(5, 7), 16)
    rgba.data[p * 4 + 3] = best === 0 ? 0 : 140
  }
  maskContext.putImageData(rgba, 0, 0)
  const full = document.createElement('canvas')
  full.width = sourceWidth
  full.height = sourceHeight
  const fullContext = full.getContext('2d')
  if (!fullContext) throw new Error('Canvas is unavailable')
  fullContext.imageSmoothingEnabled = false
  fullContext.drawImage(
    mask,
    geometry.padX,
    geometry.padY,
    Math.round(sourceWidth * geometry.ratio),
    Math.round(sourceHeight * geometry.ratio),
    0,
    0,
    sourceWidth,
    sourceHeight,
  )
  const present = counts
    .map((count, cls) => ({
      cls,
      label: cls === 0 ? 'background' : model.labels[cls - 1] ?? String(cls),
      fraction: count / (width * height),
    }))
    .filter((entry) => entry.cls > 0 && entry.fraction > 0.001)
    .sort((a, b) => b.fraction - a.fraction)
  return {
    task: 'segment',
    width: sourceWidth,
    height: sourceHeight,
    ms: elapsed,
    backend: model.backend,
    maskDataUrl: full.toDataURL('image/png'),
    present,
  }
}
