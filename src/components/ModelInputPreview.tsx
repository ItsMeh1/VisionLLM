import { useEffect, useRef } from 'react'
import { blobToBitmap, letterboxGeometry, sourceSize } from '../lib/image'
import type { Box, ClassDef } from '../lib/types'

/** Shows the exact tensor the model receives, next to the untouched original. */
export function ModelInputPreview({
  blob,
  size,
  boxes = [],
  classes,
  caption = 'Model input',
}: {
  blob: Blob | null
  size: number
  boxes?: Box[]
  classes: ClassDef[]
  caption?: string
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      const canvas = canvasRef.current
      if (!canvas || !blob) return
      const source = await blobToBitmap(blob)
      if (cancelled) {
        if ('close' in source) source.close()
        return
      }
      const { width, height } = sourceSize(source)
      const geometry = letterboxGeometry(width, height, size)
      const display = 320
      const scale = display / size
      canvas.width = display
      canvas.height = display
      const context = canvas.getContext('2d')
      if (!context) return
      context.fillStyle = '#727272'
      context.fillRect(0, 0, display, display)
      context.drawImage(
        source as CanvasImageSource,
        geometry.padX * scale,
        geometry.padY * scale,
        geometry.newW * scale,
        geometry.newH * scale,
      )
      if ('close' in source) source.close()

      context.lineWidth = 1.5
      context.font = "10px 'JetBrains Mono Variable', monospace"
      for (const box of boxes) {
        const x = (box.x * width * geometry.ratio + geometry.padX) * scale
        const y = (box.y * height * geometry.ratio + geometry.padY) * scale
        const w = box.w * width * geometry.ratio * scale
        const h = box.h * height * geometry.ratio * scale
        const color = classes.find((cls) => cls.id === box.cls)?.color ?? '#FF4A1C'
        context.strokeStyle = color
        context.strokeRect(x, y, w, h)
      }
      // Padding markers: the grey bands are what keeps the aspect ratio intact.
      context.fillStyle = 'rgba(11,12,11,0.55)'
      const top = geometry.padY * scale
      const bottom = display - (geometry.padY + geometry.newH) * scale
      if (top > 8) context.fillRect(0, 0, display, top)
      if (bottom > 8) context.fillRect(0, display - bottom, display, bottom)
      if (top > 8) {
        context.fillStyle = '#e9e7df'
        context.fillText('PAD 114 · NOT CROPPED', 8, Math.min(top - 6, 44))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [blob, size, boxes, classes])

  return (
    <figure className="model-input">
      <canvas ref={canvasRef} aria-label={`${caption}: ${size} pixel letterboxed preview`} />
      <figcaption>
        <span className="label">{caption}</span>
        <span className="mono">
          {size}×{size} · letterbox
        </span>
      </figcaption>
    </figure>
  )
}
