import { useEffect, useState } from 'react'
import { ClassEditor } from '../components/ClassEditor'
import { TaskGlyph } from '../components/TaskGlyph'
import { Button } from '../components/ui'
import { navigate, useRoute } from '../lib/router'
import { put } from '../lib/db'
import { TASK_META, type ClassDef, type Project, type Task } from '../lib/types'
import { CLASS_COLORS } from '../lib/types'

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

const DEFAULT_CLASSES: Record<Task, string[]> = {
  detect: ['object'],
  classify: ['class one', 'class two'],
  segment: ['region'],
}

export function NewProjectScreen() {
  const { query } = useRoute()
  const initialTask = (query.get('task') as Task) || 'detect'
  const [task, setTask] = useState<Task>(TASK_META[initialTask] ? initialTask : 'detect')
  const [name, setName] = useState('')
  const [classes, setClasses] = useState<ClassDef[]>(() =>
    DEFAULT_CLASSES[(TASK_META[initialTask] ? initialTask : 'detect') as Task].map((label, index) => ({
      id: uid(),
      name: label,
      color: CLASS_COLORS[index % CLASS_COLORS.length],
    })),
  )
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setClasses((current) =>
      current.length === DEFAULT_CLASSES[task].length &&
      current.every((cls, index) => cls.name === DEFAULT_CLASSES[task][index])
        ? current
        : DEFAULT_CLASSES[task].map((label, index) => ({
            id: uid(),
            name: label,
            color: CLASS_COLORS[index % CLASS_COLORS.length],
          })),
    )
  }, [task])

  const create = async () => {
    const cleanName = name.trim() || 'Untitled project'
    const cleanClasses = classes
      .map((cls, index) => ({ ...cls, name: cls.name.trim() || `class ${index + 1}` }))
      .filter((cls) => cls.name.length > 0)
    if (!cleanClasses.length) {
      setError('Add at least one class.')
      return
    }
    setBusy(true)
    setError(null)
    try {
      const project: Project = {
        id: uid(),
        name: cleanName,
        task,
        classes: cleanClasses,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        autolabelModel: 'fasterrcnn-mobilenet',
        autolabelConf: 0.3,
        imageCount: 0,
      }
      await put('projects', project)
      navigate(`/p/${project.id}/workspace`)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      setBusy(false)
    }
  }

  return (
    <main className="page create-page">
      <header className="create-head">
        <a className="back-link" href="#/">← All projects</a>
        <h1>What should your model learn?</h1>
        <p>Choose a starting point, give it a name, and list the things it should recognise.</p>
      </header>

      <div className="create-grid">
        <div className="create-main">
          <section className="create-section">
            <div className="create-section-head">
              <span className="section-number">1</span>
              <div>
                <h2>Choose a task</h2>
                <p>Object detection is the best place to start for most real projects.</p>
              </div>
            </div>
            <div className="task-picker">
              {(Object.keys(TASK_META) as Task[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  className={task === option ? 'task-option active' : 'task-option'}
                  onClick={() => setTask(option)}
                  aria-pressed={task === option}
                >
                  <span className={`task-option-icon ${option}`}>
                    <TaskGlyph task={option} size={34} />
                  </span>
                  <span className="task-option-copy">
                    <span className="task-option-title">
                      {TASK_META[option].label}
                      {option === 'detect' ? <span className="soft-tag">Most popular</span> : null}
                    </span>
                    <span className="task-option-blurb">{TASK_META[option].blurb}</span>
                  </span>
                  <span className="task-option-radio" aria-hidden />
                </button>
              ))}
            </div>
          </section>

          <section className="create-section">
            <div className="create-section-head">
              <span className="section-number">2</span>
              <div>
                <h2>Name it and add classes</h2>
                <p>
                  {task === 'detect'
                    ? 'One class per kind of thing you want boxes around. Background is automatic.'
                    : task === 'classify'
                      ? 'Each image gets one of these labels.'
                      : 'Each painted region belongs to one of these classes.'}
                </p>
              </div>
            </div>
            <label className="field">
              <span className="field-label">Project name</span>
              <input
                className="text-input"
                value={name}
                onChange={(event) => setName(event.target.value)}
                placeholder={task === 'detect' ? 'Workshop parts' : task === 'classify' ? 'Mushroom species' : 'Floor defects'}
                autoFocus
              />
            </label>
            <div className="field">
              <span className="field-label">Classes</span>
              <ClassEditor classes={classes} onChange={setClasses} />
            </div>
          </section>
        </div>

        <aside className="create-aside">
          <div className="card create-summary">
            <span className="summary-label">Your project</span>
            <h2>{name.trim() || 'Untitled project'}</h2>
            <dl className="summary-list">
              <div>
                <dt>Task</dt>
                <dd>{TASK_META[task].label}</dd>
              </div>
              <div>
                <dt>Classes</dt>
                <dd>{classes.length}</dd>
              </div>
              <div>
                <dt>First step</dt>
                <dd>{TASK_META[task].labelStep}</dd>
              </div>
            </dl>
            {error ? <p className="form-error">{error}</p> : null}
            <Button variant="primary" className="block" onClick={() => void create()} busy={busy}>
              Create project
            </Button>
            <p className="field-hint">Nothing is uploaded. The project lives in this browser.</p>
          </div>
          <div className="create-tip">
            <strong>A good first class</strong>
            <p>Pick something you can photograph 20–50 times in different lighting. “Mug” beats “objects”.</p>
          </div>
        </aside>
      </div>
    </main>
  )
}
