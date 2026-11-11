import { Wordmark } from '../components/Signature'
import { useProject } from '../lib/store'
import { navigate } from '../lib/router'
import { TASK_META } from '../lib/types'
import { WorkspaceScreen } from './WorkspaceScreen'
import { EvaluateScreen } from './EvaluateScreen'
import { ExportScreen } from './ExportScreen'

const TABS = [
  { id: 'workspace', label: 'Workspace' },
  { id: 'evaluate', label: 'Evaluate' },
  { id: 'export', label: 'Export' },
] as const

type TabId = (typeof TABS)[number]['id']

export function ProjectLayout({ step }: { step: string }) {
  const { project, loading } = useProject()

  if (loading) {
    return (
      <div className="workspace-loading">
        <span className="loading-dot" aria-hidden />
        Opening project…
      </div>
    )
  }

  if (!project) {
    return (
      <main className="page">
        <div className="empty-block">
          <h3>This project is not in the local database.</h3>
          <p className="empty-copy">It may have been deleted, or it lives in a different browser profile.</p>
          <a className="text-link" href="#/">Back to projects</a>
        </div>
      </main>
    )
  }

  const active: TabId = step === 'evaluate' || step === 'export' ? step : 'workspace'

  return (
    <div className="project-shell">
      <header className="vl-header project-header">
        <div className="vl-header-inner">
          <a className="brand" href="#/" aria-label="VisionLLM home">
            <Wordmark />
          </a>
          <span className="header-divider" aria-hidden />
          <div className="project-identity">
            <h1>{project.name}</h1>
            <span className="task-chip">{TASK_META[project.task].label}</span>
          </div>
          <nav className="project-tabs" aria-label="Project views">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                type="button"
                className={active === tab.id ? 'project-tab active' : 'project-tab'}
                onClick={() => navigate(`/p/${project.id}/${tab.id}`)}
                aria-current={active === tab.id ? 'page' : undefined}
              >
                {tab.label}
              </button>
            ))}
          </nav>
          <div className="header-actions">
            <a className="icon-link" href="#/settings">
              Settings
            </a>
          </div>
        </div>
      </header>

      <main className={active === 'workspace' ? 'project-main workspace-main' : 'project-main'}>
        {active === 'workspace' ? <WorkspaceScreen initialAnnotate={step === 'annotate'} /> : null}
        {active === 'evaluate' ? <EvaluateScreen /> : null}
        {active === 'export' ? <ExportScreen /> : null}
      </main>

    </div>
  )
}
