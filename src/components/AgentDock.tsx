import { useRef, useState } from 'react'
import { Badge, Button } from './ui'
import { useProject } from '../lib/store'
import { useApp } from '../lib/app'
import { AGENTS, runAgent, type AgentId } from '../lib/agents/agents'
import { TOOLS } from '../lib/agents/tools'
import { put } from '../lib/db'
import type { AgentRun, AgentStep } from '../lib/types'

export function AgentDock({ onClose }: { onClose: () => void }) {
  const { project, toolContext } = useProject()
  const { settings } = useApp()
  const [active, setActive] = useState<AgentId>('assistant')
  const [instruction, setInstruction] = useState('')
  const [steps, setSteps] = useState<AgentStep[]>([])
  const [report, setReport] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [planner, setPlanner] = useState<'local' | 'llm'>('local')
  const abortRef = useRef<AbortController | null>(null)
  const meta = AGENTS.find((agent) => agent.id === active) ?? AGENTS[0]

  const run = async (text: string) => {
    if (!project || busy) return
    const message = text.trim()
    if (!message) return
    setBusy(true)
    setSteps([])
    setReport([])
    const controller = new AbortController()
    abortRef.current = controller
    const startedAt = Date.now()
    const runId = crypto.randomUUID()
    let finalStatus: AgentRun['status'] = 'done'
    try {
      const result = await runAgent(active, message, toolContext(), settings, {
        onStep: (step) =>
          setSteps((current) => {
            const index = current.findIndex((entry) => entry.id === step.id)
            if (index === -1) return [...current, step]
            const next = [...current]
            next[index] = step
            return next
          }),
        onReport: (line) => setReport((current) => [...current, line]),
        signal: controller.signal,
      })
      setPlanner(result.planner)
      if (!result.ok) finalStatus = 'error'
    } catch (error) {
      finalStatus = 'error'
      setReport((current) => [...current, error instanceof Error ? error.message : String(error)])
    } finally {
      setBusy(false)
      abortRef.current = null
      const record: AgentRun = {
        id: runId,
        projectId: project.id,
        agent: active,
        instruction: message,
        status: finalStatus,
        steps,
        report,
        startedAt,
        finishedAt: Date.now(),
        planner,
      }
      void put('agentRuns', record)
    }
  }

  return (
    <aside className="agent-dock" aria-label="Agent workspace">
      <header className="agent-head">
        <div>
          <div className="label">Agent workspace</div>
          <h2>Local specialists</h2>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close agents">×</button>
      </header>

      <div className="agent-tabs" role="tablist" aria-label="Agents">
        {AGENTS.map((agent) => (
          <button
            key={agent.id}
            type="button"
            role="tab"
            aria-selected={agent.id === active}
            className={agent.id === active ? 'agent-tab active' : 'agent-tab'}
            onClick={() => {
              setActive(agent.id)
              setInstruction('')
            }}
          >
            <span className="agent-tab-name">{agent.name}</span>
            <span className="agent-tab-role mono">{agent.role}</span>
          </button>
        ))}
      </div>

      <div className="agent-body">
        <p className="agent-description">{meta.description}</p>

        <div className="agent-quick">
          {meta.quick.map((action) => (
            <button key={action.label} type="button" className="quick-action" onClick={() => void run(action.instruction)} disabled={busy}>
              {action.label}
              <span aria-hidden>↗</span>
            </button>
          ))}
        </div>

        <label className="field">
          <span className="label">Instruction</span>
          <textarea
            rows={3}
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            placeholder={
              active === 'assistant'
                ? 'e.g. clean the dataset, label what is missing, then train a nano model for 15 epochs'
                : 'Tell this agent what to do…'
            }
            onKeyDown={(event) => {
              if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) void run(instruction)
            }}
          />
        </label>
        <div className="agent-actions">
          <Button size="sm" variant="primary" onClick={() => void run(instruction)} busy={busy} disabled={!instruction.trim()}>
            Run
          </Button>
          {busy ? (
            <Button size="sm" variant="ghost" onClick={() => abortRef.current?.abort()}>
              Stop
            </Button>
          ) : null}
          <span className="mono field-hint">
            {settings.enabled && settings.apiKey ? 'llm parsing available' : 'local planner'}
          </span>
        </div>

        <div className="agent-transcript" aria-live="polite">
          {!steps.length && !report.length ? (
            <p className="field-hint">
              Agents call real tools on this project: audits, pretrained detection, training and evaluation. They report
              what happened, not what might have happened.
            </p>
          ) : null}
          {steps.map((step) => (
            <div key={step.id} className={`step step-${step.status}`}>
              <div className="step-head">
                <span className={`step-dot ${step.status}`} aria-hidden />
                <span className="step-tool">{TOOLS[step.tool]?.label ?? step.tool}</span>
                <span className="mono step-status">{step.status}</span>
              </div>
              {step.args && Object.keys(step.args).length ? (
                <div className="mono step-args">
                  {Object.entries(step.args)
                    .filter(([, value]) => value !== undefined)
                    .map(([key, value]) => `${key}=${String(value)}`)
                    .join(' · ')}
                </div>
              ) : null}
              {step.summary ? <div className="step-summary">{step.summary}</div> : null}
              {step.detail ? (
                <details className="step-detail">
                  <summary className="mono">raw result</summary>
                  <pre>{JSON.stringify(step.detail, null, 2)}</pre>
                </details>
              ) : null}
            </div>
          ))}
          {report.length ? (
            <div className="agent-report">
              <div className="label">Report</div>
              {report.map((line, index) => (
                <p key={`${index}-${line.slice(0, 12)}`}>{line}</p>
              ))}
              <Badge tone={busy ? 'warn' : 'ok'}>{busy ? 'working' : 'finished'}</Badge>
            </div>
          ) : null}
        </div>
      </div>

      <footer className="agent-foot">
        <span className="mono">{planner === 'llm' ? `planned with ${settings.model}` : 'local planner'}</span>
        <span className="mono">tools run on device</span>
      </footer>
    </aside>
  )
}
