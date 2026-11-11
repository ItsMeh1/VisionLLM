import { useEffect, useState } from 'react'
import { TaskGlyph } from '../components/TaskGlyph'
import { Button } from '../components/ui'
import { loadAllProjects } from '../lib/store'
import { navigate } from '../lib/router'
import { TASK_META, type Project, type Task } from '../lib/types'
import { formatDate } from '../lib/image'

export function ProjectsScreen() {
  const [projects, setProjects] = useState<Project[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    void loadAllProjects().then((all) => {
      setProjects(all)
      setLoading(false)
    })
  }, [])

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
            <h2>Your projects</h2>
            <p>Pick up where you left off, or start something new.</p>
          </div>
          {projects.length ? (
            <Button size="sm" onClick={() => navigate('/new')}>
              New project
            </Button>
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

        {projects.length ? (
          <div className="project-cards">
            {projects.map((project) => (
              <a key={project.id} className="project-card" href={`#/p/${project.id}/workspace`}>
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
              </a>
            ))}
            <button type="button" className="project-card new" onClick={() => navigate('/new')}>
              <span className="new-mark">+</span>
              <strong>New project</strong>
              <span>Train another model</span>
            </button>
          </div>
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
    </main>
  )
}
