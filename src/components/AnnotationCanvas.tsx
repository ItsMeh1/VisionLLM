import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import { useObjectUrl } from '../lib/hooks'
import type { Box, ClassDef, ImageRecord, Shape, Task } from '../lib/types'

type Handle = 'nw' | 'ne' | 'sw' | 'se' | null

interface DragState {
  mode: 'draw' | 'move' | 'resize'
  startX: number
  startY: number
  original: Box
  handle: Handle
}

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)
const clamp01 = (value: number) => Math.max(0, Math.min(1, value))

export function AnnotationCanvas({
  image,
  classes,
  task,
  tool,
  activeClass,
  selectedId,
  onSelect,
  onChange,
  proposals = [],
  showProposals = false,
}: {
  image: ImageRecord
  classes: ClassDef[]
  task: Task
  tool: 'select' | 'draw'
  activeClass: string
  selectedId: string | null
  onSelect: (id: string | null) => void
  onChange: (patch: { boxes?: Box[]; shapes?: Shape[] }) => void
  proposals?: Box[]
  showProposals?: boolean
}) {
  const url = useObjectUrl(image.blob)
  const wrapRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [display, setDisplay] = useState({ width: 0, height: 0 })
  const [draftBox, setDraftBox] = useState<Box | null>(null)
  const [draftPoints, setDraftPoints] = useState<[number, number][]>([])
  const [cursor, setCursor] = useState<[number, number] | null>(null)
  const dragRef = useRef<DragState | null>(null)
  const boxes = image.boxes
  const shapes = image.shapes
  const selected = boxes.find((box) => box.id === selectedId) ?? null

  const colorOf = useCallback(
    (cls: string) => classes.find((entry) => entry.id === cls)?.color ?? '#FF4A1C',
    [classes],
  )
  const nameOf = useCallback(
    (cls: string) => classes.find((entry) => entry.id === cls)?.name ?? 'unnamed',
    [classes],
  )

  useEffect(() => {
    const element = wrapRef.current
    const stage = element?.parentElement
    if (!stage) return
    const measure = () => {
      const style = getComputedStyle(stage)
      const padX = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight)
      const padY = parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)
      const availableW = Math.max(1, stage.clientWidth - padX)
      const availableH = Math.max(1, stage.clientHeight - padY)
      const scale = Math.min(availableW / image.width, availableH / image.height)
      setDisplay({ width: Math.max(1, image.width * scale), height: Math.max(1, image.height * scale) })
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(stage)
    return () => observer.disconnect()
  }, [image.width, image.height])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !display.width) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(display.width * dpr)
    canvas.height = Math.round(display.height * dpr)
    canvas.style.width = `${display.width}px`
    canvas.style.height = `${display.height}px`
    const context = canvas.getContext('2d')
    if (!context) return
    context.setTransform(dpr, 0, 0, dpr, 0, 0)
    context.clearRect(0, 0, display.width, display.height)
    const w = display.width
    const h = display.height
    const line = Math.max(1.5, w / 500)
    const fontSize = Math.max(10, Math.round(w / 60))
    context.font = `${fontSize}px 'JetBrains Mono Variable', ui-monospace, monospace`

    const drawLabel = (box: Box, color: string, text: string) => {
      const width = context.measureText(text).width
      const height = fontSize + 6
      const x = box.x * w
      const y = box.y * h
      const top = y - height - 1 < 0 ? y : y - height - 1
      context.fillStyle = color
      context.fillRect(x, top, width + 8, height)
      context.fillStyle = '#0b0c0b'
      context.fillText(text, x + 4, top + fontSize)
    }

    for (const shape of shapes) {
      context.beginPath()
      shape.points.forEach(([x, y], index) => {
        if (index === 0) context.moveTo(x * w, y * h)
        else context.lineTo(x * w, y * h)
      })
      context.closePath()
      const color = colorOf(shape.cls)
      context.fillStyle = `${color}33`
      context.fill()
      context.strokeStyle = color
      context.lineWidth = line
      context.stroke()
    }

    for (const box of boxes) {
      const color = colorOf(box.cls)
      const active = box.id === selectedId
      context.strokeStyle = color
      context.lineWidth = active ? line * 2 : line
      context.setLineDash([])
      context.strokeRect(box.x * w, box.y * h, box.w * w, box.h * h)
      drawLabel(box, color, `${nameOf(box.cls)}${box.score ? ` ${(box.score * 100).toFixed(0)}%` : ''}`)
    }

    if (showProposals) {
      for (const proposal of proposals) {
        const color = colorOf(proposal.cls)
        context.strokeStyle = color
        context.globalAlpha = 0.75
        context.setLineDash([5, 4])
        context.lineWidth = line
        context.strokeRect(proposal.x * w, proposal.y * h, proposal.w * w, proposal.h * h)
        context.setLineDash([])
        context.globalAlpha = 1
        drawLabel(proposal, color, `proposal ${(proposal.score ?? 0).toFixed(2)}`)
      }
    }

    if (draftBox) {
      context.strokeStyle = '#FF4A1C'
      context.setLineDash([6, 3])
      context.lineWidth = line * 1.5
      context.strokeRect(draftBox.x * w, draftBox.y * h, draftBox.w * w, draftBox.h * h)
      context.setLineDash([])
    }

    if (draftPoints.length) {
      const color = colorOf(activeClass)
      context.beginPath()
      draftPoints.forEach(([x, y], index) => {
        if (index === 0) context.moveTo(x * w, y * h)
        else context.lineTo(x * w, y * h)
      })
      if (cursor) context.lineTo(cursor[0] * w, cursor[1] * h)
      context.strokeStyle = color
      context.lineWidth = line * 1.6
      context.stroke()
      context.fillStyle = 'rgba(255,74,28,0.14)'
      if (draftPoints.length >= 3) {
        context.beginPath()
        draftPoints.forEach(([x, y], index) => {
          if (index === 0) context.moveTo(x * w, y * h)
          else context.lineTo(x * w, y * h)
        })
        context.closePath()
        context.fill()
      }
      for (const [x, y] of draftPoints) {
        context.beginPath()
        context.arc(x * w, y * h, 3.5, 0, Math.PI * 2)
        context.fillStyle = '#0b0c0b'
        context.fill()
        context.strokeStyle = color
        context.lineWidth = 1.5
        context.stroke()
      }
    }

    if (selected && task !== 'segment') {
      const handles: [number, number][] = [
        [selected.x, selected.y],
        [selected.x + selected.w, selected.y],
        [selected.x, selected.y + selected.h],
        [selected.x + selected.w, selected.y + selected.h],
      ]
      context.fillStyle = '#0b0c0b'
      for (const [x, y] of handles) {
        context.beginPath()
        context.rect(x * w - 4, y * h - 4, 8, 8)
        context.fill()
        context.strokeStyle = '#e9e7df'
        context.lineWidth = 1.5
        context.stroke()
      }
    }
  }, [display, boxes, shapes, selectedId, selected, proposals, showProposals, draftBox, draftPoints, cursor, classes, task, activeClass, colorOf, nameOf])

  const pointer = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect()
    return {
      x: clamp01((event.clientX - rect.left) / rect.width),
      y: clamp01((event.clientY - rect.top) / rect.height),
    }
  }

  const hitTest = (x: number, y: number): Box | null => {
    for (let i = boxes.length - 1; i >= 0; i--) {
      const box = boxes[i]
      if (x >= box.x - 0.004 && x <= box.x + box.w + 0.004 && y >= box.y - 0.004 && y <= box.y + box.h + 0.004) {
        return box
      }
    }
    return null
  }

  const handleAt = (box: Box, x: number, y: number): Handle => {
    const tolerance = 0.015
    const corners: [Handle, number, number][] = [
      ['nw', box.x, box.y],
      ['ne', box.x + box.w, box.y],
      ['sw', box.x, box.y + box.h],
      ['se', box.x + box.w, box.y + box.h],
    ]
    for (const [handle, hx, hy] of corners) {
      if (Math.abs(x - hx) < tolerance && Math.abs(y - hy) < tolerance) return handle
    }
    return null
  }

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const { x, y } = pointer(event)
    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      // Pointer may already be released (or synthetic); drawing still works without capture.
    }

    if (task === 'segment') {
      if (tool !== 'draw') {
        const hit = [...shapes].reverse().find((shape) => {
          const inside = shape.points.reduce((acc, [px, py], index) => {
            const [qx, qy] = shape.points[(index + shape.points.length - 1) % shape.points.length]
            const intersects = py > y !== qy > y && x < ((qx - px) * (y - py)) / (qy - py + Number.EPSILON) + px
            return intersects ? !acc : acc
          }, false)
          return inside
        })
        if (hit) onSelect(hit.id)
        else onSelect(null)
        return
      }
      if (draftPoints.length >= 3) {
        const [fx, fy] = draftPoints[0]
        if (Math.hypot(x - fx, y - fy) * display.width < 12) {
          commitDraft()
          return
        }
      }
      setDraftPoints((points) => [...points, [x, y]])
      return
    }

    if (tool === 'draw') {
      const next: Box = { id: 'draft', cls: activeClass, x, y, w: 0, h: 0 }
      dragRef.current = { mode: 'draw', startX: x, startY: y, original: next, handle: null }
      setDraftBox(next)
      onSelect(null)
      return
    }

    const currentBox = selectedId ? boxes.find((box) => box.id === selectedId) : null
    if (currentBox) {
      const handle = handleAt(currentBox, x, y)
      if (handle) {
        dragRef.current = { mode: 'resize', startX: x, startY: y, original: { ...currentBox }, handle }
        return
      }
    }
    const hit = hitTest(x, y)
    if (hit) {
      onSelect(hit.id)
      dragRef.current = { mode: 'move', startX: x, startY: y, original: { ...hit }, handle: null }
    } else {
      onSelect(null)
    }
  }

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const { x, y } = pointer(event)
    setCursor([x, y])
    const drag = dragRef.current
    if (!drag) return
    if (drag.mode === 'draw') {
      setDraftBox({
        id: 'draft',
        cls: activeClass,
        x: Math.min(drag.startX, x),
        y: Math.min(drag.startY, y),
        w: Math.abs(x - drag.startX),
        h: Math.abs(y - drag.startY),
      })
      return
    }
    if (drag.mode === 'move') {
      const dx = x - drag.startX
      const dy = y - drag.startY
      const next = {
        ...drag.original,
        x: Math.max(0, Math.min(1 - drag.original.w, drag.original.x + dx)),
        y: Math.max(0, Math.min(1 - drag.original.h, drag.original.y + dy)),
      }
      onChange({ boxes: boxes.map((box) => (box.id === next.id ? next : box)) })
      return
    }
    const original = drag.original
    let left = original.x
    let top = original.y
    let right = original.x + original.w
    let bottom = original.y + original.h
    if (drag.handle === 'nw' || drag.handle === 'sw') left = Math.min(x, right - 0.004)
    if (drag.handle === 'ne' || drag.handle === 'se') right = Math.max(x, left + 0.004)
    if (drag.handle === 'nw' || drag.handle === 'ne') top = Math.min(y, bottom - 0.004)
    if (drag.handle === 'sw' || drag.handle === 'se') bottom = Math.max(y, top + 0.004)
    const next = {
      ...original,
      x: Math.max(0, left),
      y: Math.max(0, top),
      w: Math.min(1, right) - Math.max(0, left),
      h: Math.min(1, bottom) - Math.max(0, top),
    }
    onChange({ boxes: boxes.map((box) => (box.id === next.id ? next : box)) })
  }

  const onPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current
    dragRef.current = null
    if (!drag) return
    if (drag.mode === 'draw') {
      const { x, y } = pointer(event)
      const next = {
        id: uid(),
        cls: activeClass,
        x: Math.min(drag.startX, x),
        y: Math.min(drag.startY, y),
        w: Math.abs(x - drag.startX),
        h: Math.abs(y - drag.startY),
      }
      setDraftBox(null)
      if (next.w > 0.006 && next.h > 0.006) {
        onChange({ boxes: [...boxes, next] })
        onSelect(next.id)
      }
    }
  }

  const commitDraft = useCallback(() => {
    if (draftPoints.length < 3) return
    const shape: Shape = { id: uid(), cls: activeClass, points: draftPoints }
    onChange({ shapes: [...shapes, shape] })
    onSelect(shape.id)
    setDraftPoints([])
  }, [draftPoints, activeClass, shapes, onChange, onSelect])

  useEffect(() => {
    if (!draftPoints.length) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Enter') {
        event.preventDefault()
        commitDraft()
      }
      if (event.key === 'Escape') setDraftPoints([])
      if (event.key === 'Backspace') {
        event.preventDefault()
        setDraftPoints((points) => points.slice(0, -1))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [draftPoints.length, commitDraft])

  return (
    <div className="annotator" ref={wrapRef}>
      {url ? <img className="annotator-image" src={url} alt={image.name} draggable={false} style={{ width: display.width, height: display.height }} /> : null}
      <canvas
        ref={canvasRef}
        className={`annotator-canvas tool-${tool}`}
        style={{ width: display.width, height: display.height }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => setCursor(null)}
        onDoubleClick={() => task === 'segment' && commitDraft()}
        role="application"
        aria-label="Annotation surface. Draw boxes by dragging. Select a box to move or resize it."
      />
      {task === 'segment' && draftPoints.length > 0 ? (
        <div className="annotator-hint mono">
          {draftPoints.length} points · Enter to close · Backspace to undo · Esc to cancel
        </div>
      ) : null}
      {task === 'segment' && tool === 'draw' && draftPoints.length === 0 ? (
        <div className="annotator-hint mono">Click to plant points · double-click or Enter to close the shape</div>
      ) : null}
    </div>
  )
}
