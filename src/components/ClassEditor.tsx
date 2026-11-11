import { useState } from 'react'
import { CLASS_COLORS, type ClassDef } from '../lib/types'
import { Button } from './ui'

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

export function ClassEditor({
  classes,
  onChange,
  compact = false,
}: {
  classes: ClassDef[]
  onChange: (classes: ClassDef[]) => void
  compact?: boolean
}) {
  const [draft, setDraft] = useState('')

  const add = () => {
    const name = draft.trim()
    if (!name) return
    if (classes.some((cls) => cls.name.toLowerCase() === name.toLowerCase())) {
      setDraft('')
      return
    }
    onChange([...classes, { id: uid(), name, color: CLASS_COLORS[classes.length % CLASS_COLORS.length] }])
    setDraft('')
  }

  const rename = (id: string, name: string) => {
    onChange(classes.map((cls) => (cls.id === id ? { ...cls, name } : cls)))
  }

  const cycleColor = (id: string) => {
    onChange(
      classes.map((cls) => {
        if (cls.id !== id) return cls
        const index = CLASS_COLORS.indexOf(cls.color)
        return { ...cls, color: CLASS_COLORS[(index + 1) % CLASS_COLORS.length] }
      }),
    )
  }

  return (
    <div className={compact ? 'class-editor compact' : 'class-editor'}>
      <ul className="class-list">
        {classes.map((cls, index) => (
          <li key={cls.id} className="class-row">
            <button
              type="button"
              className="class-swatch"
              style={{ background: cls.color }}
              onClick={() => cycleColor(cls.id)}
              title="Change colour"
              aria-label={`Change colour for ${cls.name}`}
            />
            <span className="class-index num">{String(index + 1).padStart(2, '0')}</span>
            <input
              className="class-name"
              value={cls.name}
              onChange={(event) => rename(cls.id, event.target.value)}
              aria-label={`Class ${index + 1} name`}
              spellCheck={false}
            />
            <button
              type="button"
              className="class-remove"
              onClick={() => onChange(classes.filter((entry) => entry.id !== cls.id))}
              aria-label={`Remove ${cls.name}`}
              disabled={classes.length <= 1}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
      <div className="class-add">
        <input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault()
              add()
            }
          }}
          placeholder="Add a class…"
          aria-label="New class name"
          spellCheck={false}
        />
        <Button size="sm" onClick={add} disabled={!draft.trim()}>
          Add class
        </Button>
      </div>
      <p className="field-hint">
        Name classes exactly as you would describe them. Short, concrete nouns work best — “mug”, “bolt”, “crack”.
      </p>
    </div>
  )
}
