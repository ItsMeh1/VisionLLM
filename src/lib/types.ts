export type Task = 'detect' | 'classify' | 'segment'
export type Split = 'train' | 'val'

export interface ClassDef {
  id: string
  name: string
  color: string
}

/** Normalised to the original image, 0..1. Never stored in model-input space. */
export interface Box {
  id: string
  cls: string
  x: number
  y: number
  w: number
  h: number
  score?: number
  source?: string
}

export interface Shape {
  id: string
  cls: string
  points: [number, number][]
  score?: number
  source?: string
}

export interface ImageRecord {
  id: string
  projectId: string
  name: string
  blob: Blob
  thumbnail: string
  width: number
  height: number
  bytes: number
  mime: string
  addedAt: number
  split: Split
  boxes: Box[]
  shapes: Shape[]
  label?: string
  origin: 'upload' | 'folder' | 'webcam'
}

export interface Project {
  id: string
  name: string
  task: Task
  classes: ClassDef[]
  createdAt: number
  updatedAt: number
  autolabelModel: string
  autolabelConf: number
  imageCount?: number
  llm?: LlmSettings
}

export interface EpochRecord {
  epoch: number
  train_loss: number
  cls_loss?: number
  box_loss?: number
  ctr_loss?: number
  val_loss?: number
  map50?: number
  map5095?: number
  precision?: number
  recall?: number
  val_acc?: number
  train_acc?: number
  macro_f1?: number
  miou?: number
  pixel_accuracy?: number
  lr?: number
  elapsed: number
}

export interface TrainingConfig {
  task: Task
  model: string
  epochs: number
  batch: number
  imgsz: number
  lr: number
  patience: number
  weight_decay: number
  seed: number
  conf: number
  pretrained?: boolean
  device?: string
}

export interface RunRecord {
  id: string
  projectId: string
  task: Task
  config: TrainingConfig
  createdAt: number
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'cancelling'
  jobId?: string
  history: EpochRecord[]
  metrics?: Record<string, unknown>
  notes?: string[]
  elapsed?: number
  error?: string
  agent?: string
}

export interface LlmSettings {
  enabled: boolean
  baseUrl: string
  model: string
  apiKey: string
}

export interface AgentStep {
  id: string
  tool: string
  args?: Record<string, unknown>
  status: 'running' | 'ok' | 'error' | 'skipped'
  summary?: string
  detail?: unknown
  startedAt?: number
  finishedAt?: number
}

export interface AgentRun {
  id: string
  projectId: string
  agent: 'labeling' | 'dataset' | 'training' | 'evaluation' | 'assistant'
  instruction: string
  status: 'running' | 'done' | 'error'
  steps: AgentStep[]
  report: string[]
  startedAt: number
  finishedAt?: number
  planner: 'local' | 'llm'
}

export const CLASS_COLORS = [
  '#FF4A1C',
  '#4EA8DE',
  '#F2C14E',
  '#8AC926',
  '#C77DFF',
  '#FF8FA3',
  '#4ECDC4',
  '#D4A373',
  '#A3C4F3',
  '#F4A259',
]

export const TASK_META: Record<Task, { label: string; short: string; blurb: string; labelStep: string }> = {
  detect: {
    label: 'Object detection',
    short: 'Detect',
    blurb: 'Find and outline several things in one image. Boxes, counts, positions.',
    labelStep: 'Draw boxes',
  },
  classify: {
    label: 'Image classification',
    short: 'Classify',
    blurb: 'Decide what an image shows, one label per image. Fastest to train.',
    labelStep: 'Assign labels',
  },
  segment: {
    label: 'Image segmentation',
    short: 'Segment',
    blurb: 'Paint the exact shape of each region, pixel by pixel.',
    labelStep: 'Paint masks',
  },
}
