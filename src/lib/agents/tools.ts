/** Tools the agents can actually call. Every one reads or writes real app state. */

import { auditDataset, stratifiedSplit, type Audit } from '../analysis'
import {
  autolabel,
  evaluateRun,
  exportRun,
  followJob,
  EngineError,
  engineBase,
  shortEngineError,
  startTraining,
  uploadDataset,
  type ExportInfo,
} from '../engine'
import { defaultConfig } from '../training'
import type { ImageRecord, Project, RunRecord, TrainingConfig } from '../types'

export interface ToolContext {
  project: Project
  getImages: () => ImageRecord[]
  online: boolean
  device: string
  saveImages: (images: ImageRecord[]) => Promise<void>
  saveRun: (run: RunRecord) => Promise<void>
  getRuns: () => RunRecord[]
  ensureUploaded: () => Promise<void>
}

export interface ToolResult {
  ok: boolean
  summary: string
  detail?: unknown
  changed?: boolean
}

export interface ToolDef {
  name: string
  label: string
  description: string
  params: string
  run: (ctx: ToolContext, args: Record<string, unknown>) => Promise<ToolResult>
}

const newId = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

function requireEngine(ctx: ToolContext): void {
  if (!ctx.online) throw new EngineError('The local training engine is not running.')
}

function iou(a: { x: number; y: number; w: number; h: number }, b: { x: number; y: number; w: number; h: number }) {
  const x1 = Math.max(a.x, b.x)
  const y1 = Math.max(a.y, b.y)
  const x2 = Math.min(a.x + a.w, b.x + b.w)
  const y2 = Math.min(a.y + a.h, b.y + b.h)
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
  return inter / Math.max(a.w * a.h + b.w * b.h - inter, 1e-9)
}

export function labellableCount(project: Project, images: ImageRecord[]): number {
  return images.filter((image) =>
    project.task === 'classify' ? !image.label : image.boxes.length === 0 && image.shapes.length === 0,
  ).length
}

const auditTool: ToolDef = {
  name: 'dataset.audit',
  label: 'Audit dataset',
  description: 'Measure class balance, duplicates, blur, missing labels and split quality on this machine.',
  params: 'none',
  run: async (ctx) => {
    const audit = await auditDataset(ctx.project, ctx.getImages())
    const top = audit.issues[0]
    return {
      ok: true,
      summary: `${audit.images} images · ${audit.train} train / ${audit.val} val · ${audit.labelled} labelled · ${audit.issues.length} issues`,
      detail: { ...audit, perImage: undefined, topIssue: top?.detail ?? null },
    }
  },
}

const splitTool: ToolDef = {
  name: 'dataset.split',
  label: 'Split train / validation',
  description: 'Assign a stratified validation split so evaluation is measured on images the model never trained on.',
  params: 'valRatio (0.1–0.4, default 0.2)',
  run: async (ctx, args) => {
    const ratio = typeof args.valRatio === 'number' ? Math.min(0.4, Math.max(0.1, args.valRatio)) : 0.2
    const images = ctx.getImages()
    if (images.length < 3) return { ok: false, summary: 'Need at least 3 images to make a meaningful split.' }
    const split = stratifiedSplit(ctx.project, images, ratio)
    await ctx.saveImages(split)
    const val = split.filter((image) => image.split === 'val').length
    return {
      ok: true,
      changed: true,
      summary: `Held out ${val} of ${split.length} images for validation (${Math.round(ratio * 100)}% target, stratified by class).`,
      detail: { val, train: split.length - val },
    }
  },
}

const suggestTool: ToolDef = {
  name: 'label.suggest',
  label: 'Propose boxes with a pretrained detector',
  description:
    'Run a COCO-pretrained detector on this machine and map matching classes onto your project. Only matching class names are added.',
  params: 'scope ("unlabelled" | "all"), conf (0–1), replace (boolean)',
  run: async (ctx, args) => {
    if (ctx.project.task !== 'detect') {
      return { ok: false, summary: 'Box suggestions only apply to object detection projects.' }
    }
    requireEngine(ctx)
    const scope = args.scope === 'all' ? 'all' : 'unlabelled'
    const conf = typeof args.conf === 'number' ? args.conf : ctx.project.autolabelConf
    const replace = args.replace === true
    const images = ctx.getImages()
    const candidates = images.filter((image) =>
      scope === 'all' ? true : image.boxes.length === 0,
    )
    if (!candidates.length) {
      return { ok: true, summary: 'Every image already has boxes; nothing was sent to the pretrained detector.' }
    }
    const byId = new Map(candidates.map((image) => [image.id, image]))
    let boxesAdded = 0
    let withMatches = 0
    let scanned = 0
    const unmatched = new Map<string, number>()
    const failures: string[] = []
    for (let offset = 0; offset < candidates.length; offset += 8) {
      const batch = candidates.slice(offset, offset + 8)
      try {
        const response = await autolabel(
          batch.map((image) => ({ name: `${image.id}.png`, blob: image.blob })),
          ctx.project.classes.map((cls) => cls.name),
          ctx.project.autolabelModel,
          conf,
        )
        for (const result of response.results) {
          scanned++
          const id = result.file.replace(/\.png$/, '')
          const image = byId.get(id)
          if (!image) continue
          for (const [name, count] of Object.entries(result.detected_categories)) {
            const known = ctx.project.classes.some((cls) => cls.name.toLowerCase() === name.toLowerCase())
            if (!known) unmatched.set(name, (unmatched.get(name) ?? 0) + count)
          }
          const additions = result.boxes
            .map((box) => {
              const cls = ctx.project.classes[box.cls]
              if (!cls) return null
              return {
                id: newId(),
                cls: cls.id,
                x: box.x,
                y: box.y,
                w: box.w,
                h: box.h,
                score: box.score,
                source: `pretrained:${box.source}`,
              }
            })
            .filter((box): box is NonNullable<typeof box> => Boolean(box))
          if (!additions.length) continue
          withMatches++
          let boxes = replace ? [] : [...image.boxes]
          for (const box of additions) {
            if (boxes.some((existing) => existing.cls === box.cls && iou(existing, box) > 0.8)) continue
            boxes.push(box)
            boxesAdded++
          }
          image.boxes = boxes
        }
        await ctx.saveImages(candidates.filter((image) => image.boxes.length > 0))
      } catch (error) {
        failures.push(shortEngineError(error))
        break
      }
    }
    if (failures.length) {
      return {
        ok: false,
        summary: failures[0],
        detail: { scanned, boxesAdded, unmatched: Object.fromEntries(unmatched) },
      }
    }
    const topUnmatched = [...unmatched.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)
    return {
      ok: true,
      changed: boxesAdded > 0,
      summary:
        boxesAdded > 0
          ? `Reviewed ${scanned} images, added ${boxesAdded} boxes across ${withMatches} of them. Review before training.`
          : `Reviewed ${scanned} images. The pretrained detector found no objects matching your class names.`,
      detail: {
        scanned,
        boxesAdded,
        withMatches,
        unmatched: topUnmatched.map(([name, count]) => `${name} (${count})`),
        note: 'Detections are proposals, not ground truth. They are stored as normal boxes and can be edited.',
      },
    }
  },
}

const cleanupTool: ToolDef = {
  name: 'label.cleanup',
  label: 'Clean annotations',
  description: 'Clamp boxes into the frame, discard degenerate boxes and merge near-identical duplicates.',
  params: 'none',
  run: async (ctx) => {
    const images = ctx.getImages()
    let clamped = 0
    let dropped = 0
    let merged = 0
    const changed: ImageRecord[] = []
    for (const image of images) {
      let touched = false
      const kept: ImageRecord['boxes'] = []
      for (const box of image.boxes) {
        const x = Math.max(0, Math.min(1, box.x))
        const y = Math.max(0, Math.min(1, box.y))
        const w = Math.min(1 - x, Math.max(0, box.w))
        const h = Math.min(1 - y, Math.max(0, box.h))
        if (x !== box.x || y !== box.y || w !== box.w || h !== box.h) {
          touched = true
          clamped++
        }
        if (w < 0.002 || h < 0.002) {
          touched = true
          dropped++
          continue
        }
        const next = { ...box, x, y, w, h }
        if (kept.some((other) => other.cls === next.cls && iou(other, next) > 0.9)) {
          touched = true
          merged++
          continue
        }
        kept.push(next)
      }
      if (touched) {
        changed.push({ ...image, boxes: kept })
      }
    }
    if (changed.length) await ctx.saveImages(changed)
    return {
      ok: true,
      changed: changed.length > 0,
      summary: `Checked ${images.length} images: ${clamped} boxes clamped, ${dropped} degenerate boxes removed, ${merged} duplicates merged.`,
      detail: { imagesChanged: changed.length },
    }
  },
}

const readinessTool: ToolDef = {
  name: 'train.readiness',
  label: 'Check training readiness',
  description: 'Verify there is enough labelled data and a validation set before spending time on training.',
  params: 'none',
  run: async (ctx) => {
    const images = ctx.getImages()
    const audit = await auditDataset(ctx.project, images)
    const blockers: string[] = []
    const warnings: string[] = []
    if (images.length === 0) blockers.push('No images imported.')
    if (ctx.project.task === 'classify') {
      const missing = ctx.project.classes.filter((cls) => !images.some((image) => image.label === cls.id))
      if (missing.length) blockers.push(`No images labelled: ${missing.map((cls) => cls.name).join(', ')}.`)
    } else {
      const empty = audit.classes.filter((entry) => entry.objects === 0)
      if (empty.length) blockers.push(`No annotations for: ${empty.map((entry) => entry.name).join(', ')}.`)
      const trainLabelled = images.filter(
        (image) => image.split === 'train' && (image.boxes.length > 0 || image.shapes.length > 0),
      ).length
      if (trainLabelled < 4) blockers.push(`Only ${trainLabelled} labelled training images. Draw at least 4.`)
    }
    if (audit.val === 0) warnings.push('No validation images assigned; the engine will hold out the last 15% of training data.')
    if (audit.val > 0 && audit.val < 3) warnings.push(`Only ${audit.val} validation images, so metrics will be noisy.`)
    if (audit.images < 20) warnings.push('Fewer than 20 images. Expect a rough first model; add more variety for a better one.')
    if (audit.imbalance > 4) warnings.push(`Class imbalance is ${audit.imbalance.toFixed(1)}x.`)
    const config = defaultConfig(ctx.project.task)
    return {
      ok: blockers.length === 0,
      summary: blockers.length ? `Not ready: ${blockers[0]}` : `Ready to train on ${images.length} images${warnings.length ? ` (${warnings.length} warnings)` : ''}.`,
      detail: { blockers, warnings, config, device: ctx.device, val: audit.val, labelled: audit.labelled },
    }
  },
}

const startTool: ToolDef = {
  name: 'train.start',
  label: 'Start real training',
  description: 'Upload the dataset to the local engine and launch a real training run.',
  params: 'model, epochs, imgsz, batch, lr (all optional)',
  run: async (ctx, args) => {
    requireEngine(ctx)
    const images = ctx.getImages()
    const readiness = await readinessTool.run(ctx, {})
    const detail = readiness.detail as { blockers: string[] }
    if (!readiness.ok) return { ok: false, summary: readiness.summary, detail: readiness.detail }
    let working = images
    if (!working.some((image) => image.split === 'val') && working.length > 3) {
      working = stratifiedSplit(ctx.project, working, 0.2)
      await ctx.saveImages(working)
    }
    const config: TrainingConfig = { ...defaultConfig(ctx.project.task) }
    if (typeof args.model === 'string') config.model = args.model
    if (typeof args.epochs === 'number') config.epochs = Math.max(1, Math.round(args.epochs))
    if (typeof args.imgsz === 'number') config.imgsz = args.imgsz
    if (typeof args.batch === 'number') config.batch = Math.max(1, Math.round(args.batch))
    if (typeof args.lr === 'number') config.lr = args.lr
    await ctx.ensureUploaded()
    const runId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`
    const started = await startTraining(ctx.project.id, config, runId)
    const run: RunRecord = {
      id: runId,
      projectId: ctx.project.id,
      task: ctx.project.task,
      config,
      createdAt: Date.now(),
      status: 'running',
      jobId: started.job_id,
      history: [],
      agent: 'training',
    }
    await ctx.saveRun(run)
    return {
      ok: true,
      changed: true,
      summary: `Started ${config.model} for ${config.epochs} epochs at ${config.imgsz}px on ${working.filter((i) => i.split === 'train').length} training images.`,
      detail: { runId, jobId: started.job_id, config, val: working.filter((i) => i.split === 'val').length, warnings: detail.blockers },
    }
  },
}

const waitTool: ToolDef = {
  name: 'train.wait',
  label: 'Watch training',
  description: 'Follow the real training job and record each epoch as it finishes.',
  params: 'runId, seconds (default 45)',
  run: async (ctx, args) => {
    requireEngine(ctx)
    const runs = ctx.getRuns()
    const run =
      (typeof args.runId === 'string' ? runs.find((entry) => entry.id === args.runId) : undefined) ??
      runs.find((entry) => entry.status === 'running') ??
      runs[0]
    if (!run?.jobId) return { ok: false, summary: 'No training job to watch.' }
    const seconds = typeof args.seconds === 'number' ? Math.min(180, args.seconds) : 45
    const deadline = Date.now() + seconds * 1000
    let latest = run
    let snapshot = await followJob(
      run.jobId,
      (update) => {
        const epochs = update.events
          .filter((event) => event.type === 'epoch' && event.record)
          .map((event) => event.record as unknown as RunRecord['history'][number])
        latest = {
          ...run,
          status: update.status === 'cancelling' ? 'cancelling' : update.status,
          history: epochs,
        }
        if (update.result?.metrics) {
          latest.metrics = update.result.metrics
          latest.notes = update.result.notes
          latest.elapsed = update.result.elapsed_seconds
        }
      },
      undefined,
    )
    if (snapshot.status === 'running' || snapshot.status === 'queued') {
      const start = Date.now()
      while (Date.now() - start < seconds * 1000) {
        await new Promise((resolve) => window.setTimeout(resolve, 1200))
        snapshot = await (await import('../engine')).jobStatus(run.jobId)
        if (Date.now() > deadline) break
        if (['completed', 'failed', 'cancelled'].includes(snapshot.status)) break
      }
      const epochs = snapshot.events
        .filter((event) => event.type === 'epoch' && event.record)
        .map((event) => event.record as unknown as RunRecord['history'][number])
      latest = {
        ...run,
        status: snapshot.status === 'cancelling' ? 'cancelling' : snapshot.status,
        history: epochs,
        jobId: run.jobId,
      }
      if (snapshot.result?.metrics) latest.metrics = snapshot.result.metrics
      if (snapshot.error) latest.error = snapshot.error
      await ctx.saveRun(latest)
    }
    await ctx.saveRun(latest)
    const last = latest.history[latest.history.length - 1]
    return {
      ok: latest.status !== 'failed',
      summary:
        latest.status === 'failed'
          ? `Training failed: ${latest.error ?? 'unknown error'}`
          : `Epoch ${last?.epoch ?? 0}/${run.config.epochs} · loss ${last?.train_loss?.toFixed(3) ?? '—'}${
              typeof last?.map50 === 'number' ? ` · mAP@0.5 ${(last.map50 * 100).toFixed(1)}%` : ''
            } · status ${latest.status}`,
      detail: { runId: latest.id, status: latest.status, lastEpoch: last, historyLength: latest.history.length },
    }
  },
}

function weakestMetric(task: string, metrics: Record<string, unknown>, classes: string[]): string {
  if (task === 'detect') {
    const perClass = (metrics.per_class_ap50 as (number | null)[]) ?? []
    let worst = { name: classes[0], value: Infinity }
    perClass.forEach((value, index) => {
      if (typeof value === 'number' && value < worst.value) worst = { name: classes[index], value }
    })
    if (Number.isFinite(worst.value)) return `${worst.name} (AP@0.5 ${(worst.value * 100).toFixed(1)}%)`
  }
  if (task === 'classify') {
    const perClass = (metrics.per_class as { precision: number; recall: number; f1: number }[]) ?? []
    let worst = { name: classes[0], value: 2 }
    perClass.forEach((entry, index) => {
      if (entry && entry.f1 < worst.value) worst = { name: classes[index], value: entry.f1 }
    })
    if (worst.value < 2) return `${worst.name} (F1 ${(worst.value * 100).toFixed(1)}%)`
  }
  if (task === 'segment') {
    const perClass = (metrics.per_class as { class: number; iou: number | null }[]) ?? []
    let worst = { name: classes[0], value: Infinity }
    perClass.forEach((entry) => {
      if (entry.class > 0 && typeof entry.iou === 'number' && entry.iou < worst.value) {
        worst = { name: classes[entry.class - 1], value: entry.iou }
      }
    })
    if (Number.isFinite(worst.value)) return `${worst.name} (IoU ${(worst.value * 100).toFixed(1)}%)`
  }
  return 'not enough per-class data'
}

const evaluateTool: ToolDef = {
  name: 'eval.run',
  label: 'Evaluate saved model',
  description: 'Score the trained model on held-out validation images and report the weakest class.',
  params: 'runId (optional)',
  run: async (ctx, args) => {
    requireEngine(ctx)
    const runs = ctx.getRuns()
    const run =
      (typeof args.runId === 'string' ? runs.find((entry) => entry.id === args.runId) : undefined) ??
      runs.find((entry) => entry.status === 'completed')
    if (!run) return { ok: false, summary: 'No completed run to evaluate. Train a model first.' }
    const response = await evaluateRun(ctx.project.id, run.id)
    const metrics = response.metrics
    const weakest = weakestMetric(ctx.project.task, metrics, ctx.project.classes.map((cls) => cls.name))
    const headline =
      ctx.project.task === 'detect'
        ? `mAP@0.5 ${((metrics.map50 as number) * 100).toFixed(1)}% · mAP@.5:.95 ${((metrics.map5095 as number) * 100).toFixed(1)}% · precision ${((metrics.precision as number) * 100).toFixed(1)}% · recall ${((metrics.recall as number) * 100).toFixed(1)}%`
        : ctx.project.task === 'classify'
          ? `accuracy ${((metrics.accuracy as number) * 100).toFixed(1)}% · macro F1 ${((metrics.macro_f1 as number) * 100).toFixed(1)}%`
          : `mIoU ${((metrics.miou as number) * 100).toFixed(1)}% · pixel accuracy ${((metrics.pixel_accuracy as number) * 100).toFixed(1)}%`
    await ctx.saveRun({ ...run, metrics, history: run.history, status: run.status })
    return {
      ok: true,
      changed: true,
      summary: `${headline}. Weakest: ${weakest}`,
      detail: { runId: run.id, metrics: { ...metrics, images: undefined, ground_truth: undefined }, weakest },
    }
  },
}

const exportTool: ToolDef = {
  name: 'export.build',
  label: 'Export ONNX bundle',
  description: 'Export the trained checkpoint to ONNX, verify it with ONNX Runtime and package a JS example.',
  params: 'runId (optional)',
  run: async (ctx, args) => {
    requireEngine(ctx)
    const runs = ctx.getRuns()
    const run =
      (typeof args.runId === 'string' ? runs.find((entry) => entry.id === args.runId) : undefined) ??
      runs.find((entry) => entry.status === 'completed')
    if (!run) return { ok: false, summary: 'Train a model before exporting.' }
    const info: ExportInfo = await exportRun(ctx.project.id, run.id)
    return {
      ok: true,
      summary: `Exported ${info.files.length} files (${(info.onnx_bytes / 1024 / 1024).toFixed(1)} MB ONNX). ONNX Runtime verification: ${
        info.verification.ran ? `passed, max diff ${info.verification.max_abs_diff?.toExponential(1)}` : 'failed'
      }.`,
      detail: { ...info, downloadBase: engineBase() },
    }
  },
}

const summaryTool: ToolDef = {
  name: 'project.summary',
  label: 'Summarise project',
  description: 'Read the current dataset and last run without changing anything.',
  params: 'none',
  run: async (ctx) => {
    const images = ctx.getImages()
    const runs = ctx.getRuns()
    const last = runs[0]
    const val = images.filter((image) => image.split === 'val').length
    const labelled = images.filter((image) =>
      ctx.project.task === 'classify' ? image.label : image.boxes.length || image.shapes.length,
    ).length
    return {
      ok: true,
      summary: `${images.length} images (${labelled} labelled, ${val} validation) · ${ctx.project.classes.length} classes · ${
        last ? `last run ${last.status}` : 'no training runs yet'
      }`,
      detail: {
        images: images.length,
        labelled,
        val,
        classes: ctx.project.classes.map((cls) => cls.name),
        lastRun: last ? { id: last.id, status: last.status, metrics: last.metrics } : null,
      },
    }
  },
}

export const TOOLS: Record<string, ToolDef> = {
  'dataset.audit': auditTool,
  'dataset.split': splitTool,
  'label.suggest': suggestTool,
  'label.cleanup': cleanupTool,
  'train.readiness': readinessTool,
  'train.start': startTool,
  'train.wait': waitTool,
  'eval.run': evaluateTool,
  'export.build': exportTool,
  'project.summary': summaryTool,
}

export type { Audit }
export { uploadDataset }
