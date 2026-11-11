import { useEffect, useState } from 'react'
import { TaskGlyph } from '../components/TaskGlyph'
import { Button, Dialog, Field } from '../components/ui'
import { deleteProjectData, loadAllProjects } from '../lib/store'
import { deleteRemoteProject } from '../lib/engine'
import { useApp } from '../lib/app'
import { put } from '../lib/db'
import { navigate } from '../lib/router'
import { TASK_META, type Project, type Task } from '../lib/types'
import { formatDate } from '../lib/image'

export function ProjectsScreen() {
  const { notify } = useApp()
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState('')
  const [renameTarget, setRenameTarget] = useState<Project | null>(null)
  const [renameValue, setRenameValue] = useState('')
  const [renaming, setRenaming] = useState(false)
  const [deleteTarget, setDeleteTarget] = useState<Project | null>(null)
  const [deleting, setDeleting] = useState(false)

  useEffect(() => {
    void loadAllProjects().then((all) => {
      setProjects(all)
      setLoading(false)
    })
  }, [])

  const needle = query.trim().toLowerCase()
  const visibleProjects = needle
    ? projects.filter((project) =>
        `${project.name} ${project.classes.map((cls) => cls.name).join(' ')}`.toLowerCase().includes(needle),
      )
    : projects

  const openProject = (project: Project) => navigate(`/p/${project.id}/workspace`)

  const startRename = (project: Project) => {
    setRenameTarget(project)
    setRenameValue(project.name)
  }

  const saveRename = async () => {
    if (!renameTarget) return
    const name = renameValue.trim()
    if (!name) return
    if (name === renameTarget.name) {
      setRenameTarget(null)
      return
    }
    setRenaming(true)
    try {
      const next = { ...renameTarget, name, updatedAt: Date.now() }
      await put('projects', next)
      setProjects((items) =>
        items.map((item) => (item.id === next.id ? next : item)).sort((a, b) => b.updatedAt - a.updatedAt),
      )
      setRenameTarget(null)
      notify('Project renamed.', 'ok')
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), 'error')
    } finally {
      setRenaming(false)
    }
  }

  const removeProject = async () => {
    if (!deleteTarget) return
    setDeleting(true)
    try {
      const name = deleteTarget.name
      await deleteProjectData(deleteTarget.id)
      setProjects((items) => items.filter((item) => item.id !== deleteTarget.id))
      setDeleteTarget(null)
      notify(`“${name}” deleted from this browser.`, 'ok')
      await deleteRemoteProject(deleteTarget.id)
    } catch (error) {
      notify(error instanceof Error ? error.message : String(error), 'error')
    } finally {
      setDeleting(false)
    }
  }

  return (
    <main className="home">
      <section className="home-hero">
        <div className="home-hero-copy">
          <span className="pill">No account · No cloud · Free</span>
          <h1>
            Teach a computer
            <br />
            to see what you see.
          </h1>
          <p>
            Add a handful of examples, press train, and try a real vision model. It runs on your computer and keeps
            your photos right where they are.
          </p>
          <div className="home-hero-actions">
            <Button variant="primary" onClick={() => navigate('/new')}>
              Start a project
            </Button>
            <a className="text-link" href="#how">
              See how it works
            </a>
          </div>
          <ul className="home-facts">
            <li>Runs locally</li>
            <li>Your images stay private</li>
            <li>Export to ONNX</li>
          </ul>
        </div>

        <div className="home-demo" aria-hidden>
          <div className="demo-board">
            <div className="demo-card demo-examples">
              <div className="demo-title">
                <span className="step-badge">1</span>
                Examples
              </div>
              <div className="demo-classes">
                <span><i style={{ background: '#E8623C' }} />Bolt</span>
                <span><i style={{ background: '#1F9D8B' }} />Nut</span>
              </div>
              <div className="demo-thumbs">
                <span className="demo-thumb"><b className="shape-square" /></span>
                <span className="demo-thumb"><b className="shape-circle" /></span>
                <span className="demo-thumb"><b className="shape-square small" /></span>
              </div>
              <div className="demo-hint">Drop photos here</div>
            </div>
            <div className="demo-card demo-train">
              <div className="demo-title">
                <span className="step-badge">2</span>
                Train
              </div>
              <div className="demo-progress"><span /></div>
              <div className="demo-meta"><span>Learning…</span><strong>64%</strong></div>
              <div className="demo-button">Train model</div>
            </div>
            <div className="demo-card demo-preview">
              <div className="demo-title">
                <span className="step-badge">3</span>
                Preview
              </div>
              <div className="demo-photo">
                <span className="demo-photo-object" />
                <span className="demo-detection" />
                <span className="demo-detection-label">bolt</span>
              </div>
              <div className="demo-meta"><span>Found 1 object</span><strong>18 ms</strong></div>
            </div>
          </div>
        </div>
      </section>

      <section className="home-projects">
        <header className="section-head">
          <div>
            <h2>
              Your projects
              {projects.length ? <span className="project-count">{projects.length}</span> : null}
            </h2>
            <p>Pick up where you left off, or start something new.</p>
          </div>
          {projects.length ? (
            <div className="project-tools">
              <input
                className="project-search"
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search projects"
                aria-label="Search projects"
              />
              <Button size="sm" onClick={() => navigate('/new')}>
                New project
              </Button>
            </div>
          ) : null}
        </header>

        {loading ? <div className="grid-empty">Opening your projects…</div> : null}

        {!loading && !projects.length ? (
          <div className="task-cards">
            {(Object.keys(TASK_META) as Task[]).map((task) => (
              <button key={task} type="button" className="task-card" onClick={() => navigate(`/new?task=${task}`)}>
                <span className={`task-card-art ${task}`}>
                  <TaskGlyph task={task} />
                </span>
                <span className="task-card-copy">
                  <strong>{TASK_META[task].label}</strong>
                  <span>{TASK_META[task].blurb}</span>
                </span>
                <span className="task-card-cta">Start →</span>
              </button>
            ))}
          </div>
        ) : null}

        {projects.length && visibleProjects.length ? (
          <div className="project-cards">
            {visibleProjects.map((project) => (
              <article
                key={project.id}
                className="project-card"
                role="link"
                tabIndex={0}
                aria-label={`Open ${project.name}`}
                onClick={() => openProject(project)}
                onKeyDown={(event) => {
                  if (event.target !== event.currentTarget) return
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    openProject(project)
                  }
                }}
              >
                <div className="project-card-top">
                  <span className="project-card-mark" style={{ background: project.classes[0]?.color ?? '#5B4BE1' }}>
                    <TaskGlyph task={project.task} size={26} />
                  </span>
                  <span className="project-card-task">{TASK_META[project.task].short}</span>
                </div>
                <h3>{project.name}</h3>
                <div className="project-card-classes">
                  {project.classes.slice(0, 3).map((cls) => (
                    <span key={cls.id} className="class-pill">
                      <i style={{ background: cls.color }} />
                      {cls.name}
                    </span>
                  ))}
                  {project.classes.length > 3 ? <span className="class-pill more">+{project.classes.length - 3}</span> : null}
                </div>
                <div className="project-card-meta">
                  <span>{project.imageCount ?? 0} images</span>
                  <span>Edited {formatDate(project.updatedAt)}</span>
                  <span className="project-open">Open →</span>
                </div>
                <div className="project-card-actions">
                  <button
                    type="button"
                    className="text-button"
                    onClick={(event) => {
                      event.stopPropagation()
                      startRename(project)
                    }}
                  >
                    Rename
                  </button>
                  <button
                    type="button"
                    className="text-button danger"
                    onClick={(event) => {
                      event.stopPropagation()
                      setDeleteTarget(project)
                    }}
                  >
                    Delete
                  </button>
                </div>
              </article>
            ))}
            <button type="button" className="project-card new" onClick={() => navigate('/new')}>
              <span className="new-mark">+</span>
              <strong>New project</strong>
              <span>Train another model</span>
            </button>
          </div>
        ) : null}

        {projects.length && !visibleProjects.length ? (
          <div className="grid-empty">No projects match “{query.trim()}”.</div>
        ) : null}
      </section>

      <section className="how" id="how">
        <div className="how-head">
          <h2>Three steps, no setup</h2>
          <p>Everything below happens on this computer. There is no account and no upload.</p>
        </div>
        <ol className="how-steps">
          <li>
            <span className="how-num">1</span>
            <h3>Show examples</h3>
            <p>Drop in photos and draw a box around each thing you want to find.</p>
          </li>
          <li>
            <span className="how-num">2</span>
            <h3>Train a model</h3>
            <p>A local trainer learns from your examples. You watch real progress, not a spinner.</p>
          </li>
          <li>
            <span className="how-num">3</span>
            <h3>Try it out</h3>
            <p>Point it at a new image or your camera, then export the model for the web.</p>
          </li>
        </ol>
      </section>

      <footer className="home-foot">
        <span>VisionLLM</span>
        <span>Free and local by design</span>
        <span>Models you train are yours · Apache-2.0</span>
      </footer>

      <Dialog
        open={Boolean(renameTarget)}
        onClose={() => {
          if (!renaming) setRenameTarget(null)
        }}
        title="Rename project"
      >
        {renameTarget ? (
          <form
            className="dialog-form"
            onSubmit={(event) => {
              event.preventDefault()
              void saveRename()
            }}
          >
            <Field label="Project name">
              <input
                className="text-input"
                value={renameValue}
                onChange={(event) => setRenameValue(event.target.value)}
                maxLength={80}
                autoFocus
              />
            </Field>
            <div className="button-row">
              <Button type="submit" variant="primary" busy={renaming} disabled={!renameValue.trim()}>
                Save name
              </Button>
              <Button type="button" variant="ghost" onClick={() => setRenameTarget(null)} disabled={renaming}>
                Cancel
              </Button>
            </div>
          </form>
        ) : null}
      </Dialog>

      <Dialog
        open={Boolean(deleteTarget)}
        onClose={() => {
          if (!deleting) setDeleteTarget(null)
        }}
        title="Delete project"
      >
        {deleteTarget ? (
          <div className="dialog-form">
            <p>
              Delete <strong>{deleteTarget.name}</strong>?
            </p>
            <p className="field-hint">
              This removes {deleteTarget.imageCount ?? 0} browser image
              {(deleteTarget.imageCount ?? 0) === 1 ? '' : 's'}, their annotations and runs, and asks the local engine
              to delete its training copy. Original files on disk are untouched.
            </p>
            <div className="button-row">
              <Button variant="danger" busy={deleting} onClick={() => void removeProject()}>
                Delete project
              </Button>
              <Button variant="ghost" onClick={() => setDeleteTarget(null)} disabled={deleting}>
                Cancel
              </Button>
            </div>
          </div>
        ) : null}
      </Dialog>
    </main>
  )
}
