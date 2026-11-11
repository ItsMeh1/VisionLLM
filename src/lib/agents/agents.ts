/** The agent workspace. Agents are local planners over real tools; an LLM only
 *  helps parse an instruction when the user explicitly enables one. */

import { TOOLS, type ToolContext, type ToolResult } from './tools'
import type { AgentStep, LlmSettings } from '../types'

export type AgentId = 'labeling' | 'dataset' | 'training' | 'evaluation' | 'assistant'

export interface AgentMeta {
  id: AgentId
  name: string
  role: string
  description: string
  quick: { label: string; instruction: string }[]
}

export const AGENTS: AgentMeta[] = [
  {
    id: 'assistant',
    name: 'Assistant',
    role: 'Coordinates the others',
    description:
      'Takes a plain-language instruction, decides which specialist should act, and reports what each one actually did. Local intent routing by default; an external model is optional.',
    quick: [
      { label: 'Prepare and train', instruction: 'Clean up the dataset, label anything unlabelled, then train a model.' },
      { label: 'What should I do next?', instruction: 'Give me a summary of this project and tell me what to fix next.' },
    ],
  },
  {
    id: 'dataset',
    name: 'Dataset Agent',
    role: 'Quality and balance',
    description:
      'Measures duplicates, soft images, tiny images, class imbalance, missing labels and validation splits. All analysis runs in your browser.',
    quick: [
      { label: 'Audit everything', instruction: 'Audit my dataset and list the most important problems.' },
      { label: 'Make a validation split', instruction: 'Create a stratified validation split.' },
    ],
  },
  {
    id: 'labeling',
    name: 'Labeling Agent',
    role: 'Proposes boxes',
    description:
      'Runs a COCO-pretrained detector on this machine and converts matching detections into editable boxes. Nothing is accepted blindly: every box stays reviewable.',
    quick: [
      { label: 'Label unlabelled images', instruction: 'Find images without boxes and propose annotations.' },
      { label: 'Propose for all images', instruction: 'Propose boxes for all images.' },
    ],
  },
  {
    id: 'training',
    name: 'Training Agent',
    role: 'Runs the trainer',
    description:
      'Checks readiness, uploads the dataset to the local engine, starts a real run with beginner-safe defaults and records every epoch.',
    quick: [
      { label: 'Check readiness', instruction: 'Is my dataset ready to train?' },
      { label: 'Start training', instruction: 'Train a model with the default settings.' },
      { label: 'Quick run', instruction: 'Train a nano model for 12 epochs.' },
    ],
  },
  {
    id: 'evaluation',
    name: 'Evaluation Agent',
    role: 'Reads the numbers',
    description:
      'Scores the saved checkpoint on held-out images, finds the weakest class and explains what the metrics mean in plain language.',
    quick: [
      { label: 'Evaluate latest', instruction: 'Evaluate my latest model and explain the weakest class.' },
    ],
  },
]

type Flow = 'dataset' | 'labeling' | 'training' | 'evaluation' | 'export'

interface Plan {
  flows: Flow[]
  args: Record<string, unknown>
  planner: 'local' | 'llm'
}

const FLOW_ORDER: Flow[] = ['dataset', 'labeling', 'training', 'evaluation', 'export']

const PATTERNS: Record<Flow, RegExp> = {
  dataset: /\b(audit|clean|duplicate|duplicates|quality|split|balance|imbalance|dataset|data set|messy)\b/i,
  labeling: /\b(label|labels|labeling|labelling|annotate|annotation|annotations|boxes|bounding|suggest|auto-?label)\b/i,
  training: /\b(train|training|fit|epoch|epochs|model)\b/i,
  evaluation: /\b(evaluate|evaluation|eval|accuracy|map|score|performance|weak|weakest|how good)\b/i,
  export: /\b(export|onnx|download|deploy|integration|integrate|ship)\b/i,
}

function parseArgs(text: string): Record<string, unknown> {
  const args: Record<string, unknown> = {}
  const epochs = text.match(/(\d{1,3})\s*(?:epochs?|rounds?)/i)
  if (epochs) args.epochs = Math.max(1, Math.min(500, Number(epochs[1])))
  const confidence = text.match(/(?:confidence|threshold)\D{0,8}(0?\.\d+)/i)
  if (confidence) args.conf = Math.min(0.95, Math.max(0.05, Number(confidence[1])))
  if (/\bnano\b|\bvlyolo-n\b/i.test(text)) args.model = 'vlyolo-n'
  if (/\bsmall\b|\bvlyolo-s\b/i.test(text)) args.model = 'vlyolo-s'
  if (/\bmedium\b|\bvlyolo-m\b/i.test(text)) args.model = 'vlyolo-m'
  if (/\ball images\b|\beverything\b|\bevery image\b/i.test(text)) args.scope = 'all'
  return args
}

export function localPlan(instruction: string): Plan {
  const flows = FLOW_ORDER.filter((flow) => PATTERNS[flow].test(instruction))
  return { flows, args: parseArgs(instruction), planner: 'local' }
}

async function llmPlan(
  settings: LlmSettings,
  instruction: string,
  ctx: ToolContext,
): Promise<Plan | null> {
  const base = settings.baseUrl.replace(/\/+$/, '')
  const prompt = [
    'You route instructions for a local computer-vision app. Choose zero or more flows.',
    'Flows: dataset (audit/clean/split), labeling (propose boxes), training (start training), evaluation (score model), export (build ONNX).',
    'Reply with compact JSON only: {"flows":["training"],"args":{"epochs":12,"model":"vlyolo-n"}}',
    'Available classes: ' + ctx.project.classes.map((cls) => cls.name).join(', '),
    'Task: ' + ctx.project.task,
    'Instruction: ' + instruction,
  ].join('\n')
  try {
    const response = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${settings.apiKey}` },
      body: JSON.stringify({
        model: settings.model,
        temperature: 0,
        messages: [
          { role: 'system', content: 'You output only valid JSON. You never invent results.' },
          { role: 'user', content: prompt },
        ],
      }),
    })
    if (!response.ok) return null
    const body = await response.json()
    const text: string = body?.choices?.[0]?.message?.content ?? ''
    const match = text.match(/\{[\s\S]*\}/)
    if (!match) return null
    const parsed = JSON.parse(match[0]) as { flows?: unknown; args?: unknown }
    const flows = (Array.isArray(parsed.flows) ? parsed.flows : []).filter((flow): flow is Flow =>
      FLOW_ORDER.includes(flow as Flow),
    )
    return {
      flows: flows.length ? flows : localPlan(instruction).flows,
      args: (parsed.args as Record<string, unknown>) ?? parseArgs(instruction),
      planner: 'llm',
    }
  } catch {
    return null
  }
}

export interface AgentCallbacks {
  onStep: (step: AgentStep) => void
  onReport: (line: string) => void
  signal?: AbortSignal
}

async function callTool(
  ctx: ToolContext,
  name: string,
  args: Record<string, unknown>,
  callbacks: AgentCallbacks,
): Promise<ToolResult> {
  const tool = TOOLS[name]
  const step: AgentStep = {
    id: crypto.randomUUID(),
    tool: name,
    args,
    status: 'running',
    startedAt: Date.now(),
  }
  callbacks.onStep(step)
  if (callbacks.signal?.aborted) {
    const summary = 'Stopped before this step ran.'
    callbacks.onStep({ ...step, status: 'skipped', summary, finishedAt: Date.now() })
    return { ok: false, summary }
  }
  try {
    const result = await tool.run(ctx, args)
    callbacks.onStep({
      ...step,
      status: result.ok ? 'ok' : 'error',
      summary: result.summary,
      detail: result.detail,
      finishedAt: Date.now(),
    })
    return result
  } catch (error) {
    const summary = error instanceof Error ? error.message : String(error)
    callbacks.onStep({ ...step, status: 'error', summary, finishedAt: Date.now() })
    return { ok: false, summary }
  }
}

function weakClassLine(task: string, classes: string[], metrics: Record<string, unknown>): string | null {
  if (task === 'detect') {
    const values = (metrics.per_class_ap50 as (number | null)[]) ?? []
    const scored = values
      .map((value, index) => ({ value, name: classes[index] }))
      .filter((entry): entry is { value: number; name: string } => typeof entry.value === 'number')
      .sort((a, b) => a.value - b.value)
    if (scored.length) return scored[0].name
  }
  if (task === 'classify') {
    const values = (metrics.per_class as { f1: number }[]) ?? []
    const scored = values
      .map((value, index) => ({ value: value?.f1 ?? 1, name: classes[index] }))
      .sort((a, b) => a.value - b.value)
    if (scored.length) return scored[0].name
  }
  return null
}

async function runDatasetFlow(ctx: ToolContext, args: Record<string, unknown>, callbacks: AgentCallbacks) {
  const first = await callTool(ctx, 'dataset.audit', {}, callbacks)
  if (!first.ok) {
    callbacks.onReport(`The audit could not run: ${first.summary}`)
    return false
  }
  const audit = first.detail as {
    images: number
    train: number
    val: number
    labelled: number
    issues: { detail: string; severity: string }[]
    imbalance: number
  }
  callbacks.onReport(
    `Measured ${audit.images} images: ${audit.train} in training, ${audit.val} held out, ${audit.labelled} labelled.`,
  )
  if (audit.val === 0 && audit.images > 2 && args.splitIfNeeded !== false) {
    await callTool(ctx, 'dataset.split', { valRatio: 0.2 }, callbacks)
  }
  for (const issue of audit.issues.slice(0, 5)) {
    callbacks.onReport(`${issue.severity === 'high' ? 'Blocking' : issue.severity === 'medium' ? 'Worth fixing' : 'Minor'}: ${issue.detail}`)
  }
  if (!audit.issues.length) callbacks.onReport('No structural problems found in this dataset.')
  return true
}

async function runLabelingFlow(ctx: ToolContext, args: Record<string, unknown>, callbacks: AgentCallbacks) {
  if (ctx.project.task !== 'detect') {
    callbacks.onReport('Box suggestions only exist for object detection. This project is ' + ctx.project.task + '.')
    return false
  }
  if (!ctx.online) {
    callbacks.onReport('The local engine is offline, so the pretrained detector cannot run. Start it with `npm run engine`.')
    return false
  }
  const result = await callTool(
    ctx,
    'label.suggest',
    { scope: args.scope ?? 'unlabelled', conf: args.conf, replace: false },
    callbacks,
  )
  const detail = result.detail as { scanned?: number; boxesAdded?: number; withMatches?: number; unmatched?: string[] } | undefined
  if (!result.ok) {
    callbacks.onReport(`Annotation suggestions failed: ${result.summary}`)
    return false
  }
  if (detail?.boxesAdded) {
    callbacks.onReport(
      `${detail.boxesAdded} boxes were added to ${detail.withMatches} of ${detail.scanned} images. They are ordinary annotations now — check them in the Annotate step before training.`,
    )
  } else {
    callbacks.onReport('No boxes were added. Either every image is already labelled, or the pretrained detector found nothing that matches your class names.')
  }
  if (detail?.unmatched?.length) {
    callbacks.onReport(
      `The detector also saw classes you have not defined: ${detail.unmatched.slice(0, 5).join(', ')}. Add matching class names to use them.`,
    )
  }
  return Boolean(detail?.boxesAdded)
}

async function runTrainingFlow(ctx: ToolContext, args: Record<string, unknown>, callbacks: AgentCallbacks) {
  const readiness = await callTool(ctx, 'train.readiness', {}, callbacks)
  const detail = readiness.detail as { blockers?: string[]; warnings?: string[] } | undefined
  for (const warning of detail?.warnings ?? []) callbacks.onReport(`Note: ${warning}`)
  if (!readiness.ok) {
    for (const blocker of detail?.blockers ?? []) callbacks.onReport(`Blocked: ${blocker}`)
    return false
  }
  if (!ctx.online) {
    callbacks.onReport('The dataset is ready, but the local engine is offline. Start it with `npm run engine`, then ask me again.')
    return false
  }
  const start = await callTool(
    ctx,
    'train.start',
    { epochs: args.epochs, model: args.model, imgsz: args.imgsz },
    callbacks,
  )
  if (!start.ok) {
    callbacks.onReport(`Training did not start: ${start.summary}`)
    return false
  }
  const startedDetail = start.detail as { runId: string; config: { epochs: number; model: string } }
  callbacks.onReport(`Run ${startedDetail.runId} is live in the engine: ${startedDetail.config.epochs} epochs of ${startedDetail.config.model}.`)
  const wait = await callTool(ctx, 'train.wait', { runId: startedDetail.runId, seconds: 50 }, callbacks)
  const waited = wait.detail as { status?: string; historyLength?: number } | undefined
  if (waited?.status === 'running') {
    callbacks.onReport('It is still training. Open the Train step to watch the epoch chart; the engine keeps going when this panel closes.')
  } else if (waited?.status === 'completed') {
    callbacks.onReport('Training finished. The next honest check is evaluation on the held-out images.')
  } else if (waited?.status === 'failed') {
    callbacks.onReport('Training failed. The engine log above has the exact error.')
  }
  return true
}

async function runEvaluationFlow(ctx: ToolContext, args: Record<string, unknown>, callbacks: AgentCallbacks) {
  if (!ctx.online) {
    callbacks.onReport('The engine is offline, so the saved checkpoint cannot be scored right now.')
    return false
  }
  const result = await callTool(ctx, 'eval.run', { runId: args.runId }, callbacks)
  if (!result.ok) {
    callbacks.onReport(result.summary)
    return false
  }
  const detail = result.detail as { metrics: Record<string, unknown>; weakest: string }
  const weak = weakClassLine(ctx.project.task, ctx.project.classes.map((cls) => cls.name), detail.metrics)
  if (weak) {
    callbacks.onReport(
      `${weak} is the weakest class. Collect more varied examples of it, make sure its boxes are tight, and check that training images show it at different scales.`,
    )
  }
  callbacks.onReport('These numbers come from images the model never trained on. Treat them as the honest score.')
  return true
}

async function runExportFlow(ctx: ToolContext, args: Record<string, unknown>, callbacks: AgentCallbacks) {
  if (!ctx.online) {
    callbacks.onReport('The engine is offline, so the ONNX bundle cannot be built.')
    return false
  }
  const result = await callTool(ctx, 'export.build', { runId: args.runId }, callbacks)
  if (!result.ok) {
    callbacks.onReport(result.summary)
    return false
  }
  const detail = result.detail as { files?: string[]; verification?: { ran: boolean } }
  callbacks.onReport(
    `Bundle contains: ${(detail.files ?? []).join(', ')}. The ONNX model was ${
      detail.verification?.ran ? 'loaded and compared against PyTorch successfully' : 'NOT verified — check the export log'
    }.`,
  )
  return true
}

const FLOW_LABEL: Record<Flow, string> = {
  dataset: 'Dataset Agent',
  labeling: 'Labeling Agent',
  training: 'Training Agent',
  evaluation: 'Evaluation Agent',
  export: 'Export (via Assistant)',
}

export async function executePlan(
  plan: Plan,
  ctx: ToolContext,
  callbacks: AgentCallbacks,
): Promise<boolean> {
  let anySuccess = false
  for (const flow of plan.flows) {
    callbacks.onReport(`Handing this to the ${FLOW_LABEL[flow]}.`)
    if (flow === 'dataset') anySuccess = (await runDatasetFlow(ctx, plan.args, callbacks)) || anySuccess
    if (flow === 'labeling') anySuccess = (await runLabelingFlow(ctx, plan.args, callbacks)) || anySuccess
    if (flow === 'training') anySuccess = (await runTrainingFlow(ctx, plan.args, callbacks)) || anySuccess
    if (flow === 'evaluation') anySuccess = (await runEvaluationFlow(ctx, plan.args, callbacks)) || anySuccess
    if (flow === 'export') anySuccess = (await runExportFlow(ctx, plan.args, callbacks)) || anySuccess
  }
  return anySuccess
}

export async function runAgent(
  agent: AgentId,
  instruction: string,
  ctx: ToolContext,
  settings: LlmSettings,
  callbacks: AgentCallbacks,
): Promise<{ ok: boolean; planner: 'local' | 'llm' }> {
  if (agent === 'assistant') {
    let plan = localPlan(instruction)
    let planner: 'local' | 'llm' = 'local'
    if (settings.enabled && settings.apiKey && settings.baseUrl && settings.model) {
      const remote = await llmPlan(settings, instruction, ctx)
      if (remote) {
        plan = remote
        planner = 'llm'
        callbacks.onReport(`Planned with ${settings.model}.`)
      } else {
        callbacks.onReport('The external model did not answer usefully, so the local planner took over. No result was invented.')
      }
    }
    if (!plan.flows.length) {
      await callTool(ctx, 'project.summary', {}, callbacks)
      callbacks.onReport('Tell me what you want to do: clean the dataset, label images, train, evaluate or export.')
      return { ok: false, planner }
    }
    const ok = await executePlan(plan, ctx, callbacks)
    return { ok, planner }
  }

  if (agent === 'dataset') {
    const ok = await runDatasetFlow(ctx, parseArgs(instruction), callbacks)
    return { ok, planner: 'local' }
  }
  if (agent === 'labeling') {
    const ok = await runLabelingFlow(ctx, parseArgs(instruction), callbacks)
    return { ok, planner: 'local' }
  }
  if (agent === 'training') {
    const ok = await runTrainingFlow(ctx, parseArgs(instruction), callbacks)
    return { ok, planner: 'local' }
  }
  const ok = await runEvaluationFlow(ctx, parseArgs(instruction), callbacks)
  return { ok, planner: 'local' }
}
