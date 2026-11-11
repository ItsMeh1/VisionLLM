import type { Task, TrainingConfig } from './types'

export const MODEL_INFO: Record<string, { label: string; family: string; blurb: string; licence: string }> = {
  'vlyolo-n': {
    label: 'VLYOLO-N',
    family: 'YOLO · anchor-free',
    blurb: 'Nano. Trains in minutes on a laptop. Best first choice for a few hundred images.',
    licence: 'Apache-2.0',
  },
  'vlyolo-s': {
    label: 'VLYOLO-S',
    family: 'YOLO · anchor-free',
    blurb: 'Small. More capacity for crowded scenes and small objects. Still comfortable on Apple silicon.',
    licence: 'Apache-2.0',
  },
  'vlyolo-m': {
    label: 'VLYOLO-M',
    family: 'YOLO · anchor-free',
    blurb: 'Medium. Use when you have thousands of images and want the best accuracy this app offers.',
    licence: 'Apache-2.0',
  },
  resnet18: {
    label: 'ResNet-18',
    family: 'CNN classifier',
    blurb: 'ImageNet-pretrained backbone, fine-tuned on your classes. Fast and reliable.',
    licence: 'BSD-3-Clause',
  },
  unet: {
    label: 'U-Net',
    family: 'Encoder–decoder',
    blurb: 'Compact segmentation network trained from scratch on your painted masks.',
    licence: 'Apache-2.0',
  },
}

export const INPUT_SIZES: Record<Task, number[]> = {
  detect: [320, 384, 448, 512, 640],
  classify: [192, 224, 256, 320],
  segment: [256, 320, 384, 448],
}

export function defaultConfig(task: Task): TrainingConfig {
  if (task === 'classify') {
    return {
      task,
      model: 'resnet18',
      epochs: 20,
      batch: 16,
      imgsz: 224,
      lr: 0.0003,
      patience: 0,
      weight_decay: 0.01,
      seed: 0,
      conf: 0.25,
      pretrained: true,
      device: 'auto',
    }
  }
  if (task === 'segment') {
    return {
      task,
      model: 'unet',
      epochs: 30,
      batch: 4,
      imgsz: 384,
      lr: 0.001,
      patience: 0,
      weight_decay: 0.0005,
      seed: 0,
      conf: 0.25,
      device: 'auto',
    }
  }
  return {
    task,
    model: 'vlyolo-s',
    epochs: 40,
    batch: 8,
    imgsz: 512,
    lr: 0.001,
    patience: 12,
    weight_decay: 0.0005,
    seed: 0,
    conf: 0.25,
    device: 'auto',
  }
}

export function modelOptions(task: Task): string[] {
  if (task === 'detect') return ['vlyolo-n', 'vlyolo-s', 'vlyolo-m']
  if (task === 'classify') return ['resnet18']
  return ['unet']
}

/** Beginner-facing guess, clearly labelled as an estimate, from the real dataset size. */
export function estimateMinutes(task: Task, config: TrainingConfig, images: number, device: string): number {
  const perImage = task === 'detect' ? 0.09 : task === 'classify' ? 0.012 : 0.06
  const modelScale = config.model === 'vlyolo-n' ? 0.65 : config.model === 'vlyolo-m' ? 2.4 : 1
  const sizeScale = Math.pow(config.imgsz / (task === 'detect' ? 512 : 320), 2)
  const deviceScale = device === 'cpu' ? 1.8 : 0.65
  const epochs = config.epochs
  const minutes = (images * perImage * modelScale * sizeScale * deviceScale * epochs) / 60
  return Math.max(0.2, minutes)
}
