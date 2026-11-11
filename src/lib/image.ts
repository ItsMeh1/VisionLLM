/** Image decoding and the exact letterbox geometry used by the engine. */

import type { Box } from './types'

export interface Geometry {
  width: number
  height: number
  size: number
  newW: number
  newH: number
  padX: number
  padY: number
  ratio: number
}

/** Mirrors engine/letterbox.py: round the fit, floor the pad. */
export function letterboxGeometry(width: number, height: number, size: number): Geometry {
  const ratio = Math.min(size / width, size / height)
  const newW = Math.max(1, Math.min(size, Math.round(width * ratio)))
  const newH = Math.max(1, Math.min(size, Math.round(height * ratio)))
  return {
    width,
    height,
    size,
    newW,
    newH,
    padX: Math.floor((size - newW) / 2),
    padY: Math.floor((size - newH) / 2),
    ratio,
  }
}

export function loadImageElement(blob: Blob): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob)
    const image = new Image()
    image.onload = () => {
      URL.revokeObjectURL(url)
      resolve(image)
    }
    image.onerror = () => {
      URL.revokeObjectURL(url)
      reject(new Error('This file could not be decoded as an image.'))
    }
    image.src = url
  })
}

export async function blobToBitmap(blob: Blob): Promise<ImageBitmap | HTMLImageElement> {
  if ('createImageBitmap' in window) {
    try {
      return await createImageBitmap(blob)
    } catch {
      /* fall through to an <img> */
    }
  }
  return loadImageElement(blob)
}

export function sourceSize(source: ImageBitmap | HTMLImageElement): { width: number; height: number } {
  if (source instanceof HTMLImageElement) {
    return { width: source.naturalWidth, height: source.naturalHeight }
  }
  return { width: source.width, height: source.height }
}

export function drawSource(
  context: CanvasRenderingContext2D,
  source: ImageBitmap | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement,
  x: number,
  y: number,
  width: number,
  height: number,
): void {
  context.drawImage(source, x, y, width, height)
}

/** A UI thumbnail. The original blob is never re-encoded. */
export async function makeThumbnail(blob: Blob, max = 420): Promise<string> {
  const source = await blobToBitmap(blob)
  const { width, height } = sourceSize(source)
  const scale = Math.min(1, max / Math.max(width, height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.max(1, Math.round(width * scale))
  canvas.height = Math.max(1, Math.round(height * scale))
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas is unavailable')
  context.imageSmoothingQuality = 'high'
  drawSource(context, source, 0, 0, canvas.width, canvas.height)
  if ('close' in source) source.close()
  return canvas.toDataURL('image/jpeg', 0.86)
}

/** Render exactly what the model receives: letterboxed, padded, uncropped. */
export async function renderModelInput(blob: Blob, size: number): Promise<{ dataUrl: string; geometry: Geometry }> {
  const source = await blobToBitmap(blob)
  const { width, height } = sourceSize(source)
  const geometry = letterboxGeometry(width, height, size)
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext('2d')
  if (!context) throw new Error('Canvas is unavailable')
  context.fillStyle = '#727272'
  context.fillRect(0, 0, size, size)
  context.imageSmoothingQuality = 'high'
  drawSource(context, source, geometry.padX, geometry.padY, geometry.newW, geometry.newH)
  if ('close' in source) source.close()
  return { dataUrl: canvas.toDataURL('image/png'), geometry }
}

export async function canvasToBlob(canvas: HTMLCanvasElement, type = 'image/png', quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error('Could not create an image'))),
      type,
      quality,
    )
  })
}

/** Draw boxes in normalised coordinates onto a canvas of the given size. */
export function drawBoxes(
  context: CanvasRenderingContext2D,
  boxes: Box[],
  width: number,
  height: number,
  options: { color: (box: Box) => string; lineWidth?: number; dashed?: boolean; labels?: (box: Box) => string },
): void {
  const lineWidth = options.lineWidth ?? Math.max(2, width / 320)
  context.lineWidth = lineWidth
  context.font = `${Math.max(11, Math.round(width / 52))}px 'JetBrains Mono Variable', ui-monospace, monospace`
  for (const box of boxes) {
    const x = box.x * width
    const y = box.y * height
    const w = box.w * width
    const h = box.h * height
    const color = options.color(box)
    context.strokeStyle = color
    context.fillStyle = color
    context.setLineDash(options.dashed ? [6, 4] : [])
    context.strokeRect(x, y, w, h)
    context.setLineDash([])
    const label = options.labels?.(box) ?? ''
    if (label) {
      const textWidth = context.measureText(label).width
      const boxHeight = Math.max(14, Math.round(width / 60))
      const top = y - boxHeight - 2 < 0 ? y : y - boxHeight - 2
      context.fillRect(x, top, textWidth + 8, boxHeight)
      context.fillStyle = '#0b0c0b'
      context.fillText(label, x + 4, top + boxHeight - 4)
    }
  }
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export function formatDate(value: number): string {
  return new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export function formatDuration(seconds?: number): string {
  if (!seconds || seconds <= 0) return '—'
  if (seconds < 60) return `${seconds.toFixed(0)}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${Math.round(seconds % 60)}s`
}

export function pct(value: unknown, digits = 1): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}
