/** Real, local dataset analysis: no network calls, no invented numbers. */

import { blobToBitmap, sourceSize } from './image'
import type { ImageRecord, Project } from './types'

function hashBits(canvas: HTMLCanvasElement): { bits: string; luminance: number } {
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) return { bits: '', luminance: 0 }
  const { width, height } = canvas
  const data = context.getImageData(0, 0, width, height).data
  const gray: number[] = []
  for (let i = 0; i < data.length; i += 4) {
    gray.push(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2])
  }
  const mean = gray.reduce((sum, value) => sum + value, 0) / Math.max(gray.length, 1)
  return { bits: gray.map((value) => (value >= mean ? '1' : '0')).join(''), luminance: mean }
}

export interface ImageQuality {
  id: string
  hash: string
  blur: number
  luminance: number
  dupeGroup?: number
}

const qualityCache = new Map<string, ImageQuality>()

export async function analyseImage(image: ImageRecord): Promise<ImageQuality> {
  const cached = qualityCache.get(image.id)
  if (cached) return cached
  const source = await blobToBitmap(image.blob)
  const { width, height } = sourceSize(source)

  const hashCanvas = document.createElement('canvas')
  hashCanvas.width = 8
  hashCanvas.height = 8
  const hashContext = hashCanvas.getContext('2d', { willReadFrequently: true })
  hashContext?.drawImage(source as CanvasImageSource, 0, 0, 8, 8)
  const { bits, luminance } = hashBits(hashCanvas)

  const blurCanvas = document.createElement('canvas')
  const scale = 96 / Math.max(width, height)
  blurCanvas.width = Math.max(16, Math.round(width * scale))
  blurCanvas.height = Math.max(16, Math.round(height * scale))
  const blurContext = blurCanvas.getContext('2d', { willReadFrequently: true })
  blurContext?.drawImage(source as CanvasImageSource, 0, 0, blurCanvas.width, blurCanvas.height)
  let blur = 0
  if (blurContext) {
    const { width: bw, height: bh } = blurCanvas
    const pixels = blurContext.getImageData(0, 0, bw, bh).data
    const gray = new Float32Array(bw * bh)
    for (let i = 0; i < bw * bh; i++) {
      gray[i] = 0.299 * pixels[i * 4] + 0.587 * pixels[i * 4 + 1] + 0.114 * pixels[i * 4 + 2]
    }
    let sum = 0
    let sumSq = 0
    let count = 0
    for (let y = 1; y < bh - 1; y++) {
      for (let x = 1; x < bw - 1; x++) {
        const i = y * bw + x
        const lap = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - bw] - gray[i + bw]
        sum += lap
        sumSq += lap * lap
        count++
      }
    }
    const mean = sum / Math.max(count, 1)
    blur = sumSq / Math.max(count, 1) - mean * mean
  }
  if ('close' in source) source.close()

  const quality: ImageQuality = {
    id: image.id,
    hash: bits,
    blur: Number.isFinite(blur) ? blur : 0,
    luminance,
  }
  qualityCache.set(image.id, quality)
  return quality
}

export function hamming(a: string, b: string): number {
  let distance = 0
  const length = Math.min(a.length, b.length)
  for (let i = 0; i < length; i++) if (a[i] !== b[i]) distance++
  return distance + Math.abs(a.length - b.length)
}

export type AuditIssueKind =
  | 'duplicate'
  | 'blurry'
  | 'tiny'
  | 'unlabelled'
  | 'class-imbalance'
  | 'no-validation'
  | 'empty-class'
  | 'box-on-border'

export interface AuditIssue {
  kind: AuditIssueKind
  severity: 'high' | 'medium' | 'low'
  imageIds: string[]
  detail: string
}

export interface ClassCount {
  id: string
  name: string
  color: string
  images: number
  objects: number
}

export interface Audit {
  images: number
  train: number
  val: number
  labelled: number
  unlabelled: number
  totalObjects: number
  classes: ClassCount[]
  issues: AuditIssue[]
  duplicates: { imageIds: string[]; distance: number }[]
  blurry: { id: string; blur: number }[]
  tiny: { id: string; width: number; height: number }[]
  imbalance: number
  perImage: Record<string, ImageQuality>
}

export async function auditDataset(project: Project, images: ImageRecord[]): Promise<Audit> {
  const perImage: Record<string, ImageQuality> = {}
  for (const image of images) {
    try {
      perImage[image.id] = await analyseImage(image)
    } catch {
      /* unreadable images are reported by the importer instead */
    }
  }

  const counts = new Map<string, { images: number; objects: number }>()
  project.classes.forEach((cls) => counts.set(cls.id, { images: 0, objects: 0 }))
  for (const image of images) {
    if (project.task === 'classify') {
      if (image.label) {
        const entry = counts.get(image.label)
        if (entry) {
          entry.images += 1
          entry.objects += 1
        }
      }
      continue
    }
    const seen = new Set<string>()
    for (const box of image.boxes) {
      const entry = counts.get(box.cls)
      if (!entry) continue
      entry.objects += 1
      if (!seen.has(box.cls)) {
        entry.images += 1
        seen.add(box.cls)
      }
    }
    for (const shape of image.shapes) {
      const entry = counts.get(shape.cls)
      if (!entry) continue
      entry.objects += 1
      if (!seen.has(shape.cls)) {
        entry.images += 1
        seen.add(shape.cls)
      }
    }
  }

  const labelled = images.filter((image) =>
    project.task === 'classify'
      ? Boolean(image.label)
      : image.boxes.length > 0 || image.shapes.length > 0,
  ).length

  const ids = Object.keys(perImage)
  const handled = new Set<string>()
  const duplicates: { imageIds: string[]; distance: number }[] = []
  for (let i = 0; i < ids.length; i++) {
    if (handled.has(ids[i])) continue
    const group = [ids[i]]
    let worst = 0
    for (let j = i + 1; j < ids.length; j++) {
      if (handled.has(ids[j])) continue
      const distance = hamming(perImage[ids[i]].hash, perImage[ids[j]].hash)
      if (distance <= 8) {
        group.push(ids[j])
        handled.add(ids[j])
        worst = Math.max(worst, distance)
      }
    }
    if (group.length > 1) {
      handled.add(ids[i])
      duplicates.push({ imageIds: group, distance: worst })
    }
  }

  const blurry = images
    .filter((image) => (perImage[image.id]?.blur ?? 999) < 55)
    .map((image) => ({ id: image.id, blur: Math.round(perImage[image.id]?.blur ?? 0) }))
  const tiny = images
    .filter((image) => Math.min(image.width, image.height) < 200)
    .map((image) => ({ id: image.id, width: image.width, height: image.height }))
  const unlabelledImages = images.filter((image) =>
    project.task === 'classify' ? !image.label : image.boxes.length === 0 && image.shapes.length === 0,
  )

  const classCounts: ClassCount[] = project.classes.map((cls) => ({
    id: cls.id,
    name: cls.name,
    color: cls.color,
    images: counts.get(cls.id)?.images ?? 0,
    objects: counts.get(cls.id)?.objects ?? 0,
  }))
  const objects = classCounts.map((entry) => entry.objects)
  const nonZero = objects.filter((value) => value > 0)
  const imbalance = nonZero.length && Math.min(...nonZero) > 0 ? Math.max(...nonZero) / Math.min(...nonZero) : 0

  const issues: AuditIssue[] = []
  if (unlabelledImages.length) {
    issues.push({
      kind: 'unlabelled',
      severity: unlabelledImages.length > images.length * 0.25 ? 'high' : 'medium',
      imageIds: unlabelledImages.map((image) => image.id),
      detail: `${unlabelledImages.length} of ${images.length} images have no ${project.task === 'classify' ? 'label' : 'annotations'}. They add no training signal.`,
    })
  }
  if (duplicates.length) {
    issues.push({
      kind: 'duplicate',
      severity: 'medium',
      imageIds: duplicates.flatMap((group) => group.imageIds),
      detail: `${duplicates.length} group${duplicates.length === 1 ? '' : 's'} of near-identical images. Duplicates inflate one scene and skew validation.`,
    })
  }
  if (blurry.length) {
    issues.push({
      kind: 'blurry',
      severity: 'low',
      imageIds: blurry.map((entry) => entry.id),
      detail: `${blurry.length} image${blurry.length === 1 ? '' : 's'} look soft or low-detail (Laplacian variance below 55).`,
    })
  }
  if (tiny.length) {
    issues.push({
      kind: 'tiny',
      severity: 'low',
      imageIds: tiny.map((entry) => entry.id),
      detail: `${tiny.length} image${tiny.length === 1 ? '' : 's'} are smaller than 200px on a side. Details may vanish after letterboxing.`,
    })
  }
  if (imbalance > 4) {
    issues.push({
      kind: 'class-imbalance',
      severity: imbalance > 8 ? 'high' : 'medium',
      imageIds: [],
      detail: `The most frequent class has ${imbalance.toFixed(1)}x the objects of the rarest. Add examples of the rare class.`,
    })
  }
  if (images.length > 1 && !images.some((image) => image.split === 'val')) {
    issues.push({
      kind: 'no-validation',
      severity: 'high',
      imageIds: [],
      detail: 'No validation images are assigned, so accuracy cannot be measured honestly.',
    })
  }
  const emptyClasses = classCounts.filter((entry) => entry.objects === 0)
  if (emptyClasses.length) {
    issues.push({
      kind: 'empty-class',
      severity: 'high',
      imageIds: [],
      detail: `No annotations exist for: ${emptyClasses.map((entry) => entry.name).join(', ')}. Those classes cannot be learned.`,
    })
  }
  const borderBoxes = images.flatMap((image) =>
    image.boxes
      .filter((box) => box.x < 0.002 || box.y < 0.002 || box.x + box.w > 0.998 || box.y + box.h > 0.998)
      .map(() => image.id),
  )
  if (borderBoxes.length) {
    issues.push({
      kind: 'box-on-border',
      severity: 'low',
      imageIds: [...new Set(borderBoxes)],
      detail: `${borderBoxes.length} box${borderBoxes.length === 1 ? '' : 'es'} touch the image edge. Check they are not truncated.`,
    })
  }

  return {
    images: images.length,
    train: images.filter((image) => image.split === 'train').length,
    val: images.filter((image) => image.split === 'val').length,
    labelled,
    unlabelled: unlabelledImages.length,
    totalObjects: objects.reduce((sum, value) => sum + value, 0),
    classes: classCounts,
    issues,
    duplicates,
    blurry,
    tiny,
    imbalance,
    perImage,
  }
}

/** Deterministic, class-aware split. Shuffles with a seeded PRNG for repeatability. */
export function stratifiedSplit(project: Project, images: ImageRecord[], valRatio = 0.2): ImageRecord[] {
  let seed = 20261111
  const random = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed / 2147483648
  }
  const shuffle = <T,>(items: T[]): T[] => {
    const copy = [...items]
    for (let i = copy.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1))
      ;[copy[i], copy[j]] = [copy[j], copy[i]]
    }
    return copy
  }

  const groups = new Map<string, ImageRecord[]>()
  for (const image of images) {
    const key =
      project.task === 'classify'
        ? image.label ?? 'unlabelled'
        : image.boxes[0]?.cls ?? image.shapes[0]?.cls ?? 'unlabelled'
    if (!groups.has(key)) groups.set(key, [])
    groups.get(key)?.push(image)
  }

  const valIds = new Set<string>()
  for (const [, group] of groups) {
    const shuffled = shuffle(group)
    const count = Math.max(1, Math.round(shuffled.length * valRatio))
    if (shuffled.length <= 2) continue
    shuffled.slice(0, count).forEach((image) => valIds.add(image.id))
  }
  if (valIds.size === 0 && images.length > 1) valIds.add(images[images.length - 1].id)
  return images.map((image) => ({ ...image, split: valIds.has(image.id) ? 'val' : 'train' }))
}
