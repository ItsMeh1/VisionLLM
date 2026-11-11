import { useEffect, useId, useState, type ButtonHTMLAttributes, type ReactNode } from 'react'

export function Button({
  children,
  variant = 'secondary',
  size = 'md',
  busy,
  className = '',
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
  busy?: boolean
}) {
  return (
    <button
      {...rest}
      className={`btn btn-${variant} btn-${size} ${className}`}
      disabled={rest.disabled || busy}
      aria-busy={busy || undefined}
    >
      {busy ? <span className="spinner" aria-hidden /> : null}
      <span>{children}</span>
    </button>
  )
}

export function Panel({
  children,
  className = '',
  as: Tag = 'section',
}: {
  children: ReactNode
  className?: string
  as?: 'section' | 'div' | 'aside'
}) {
  return <Tag className={`panel ${className}`}>{children}</Tag>
}

export function PanelHead({
  eyebrow,
  title,
  actions,
  note,
}: {
  eyebrow?: string
  title: ReactNode
  actions?: ReactNode
  note?: ReactNode
}) {
  return (
    <header className="panel-head">
      <div>
        {eyebrow ? <div className="label">{eyebrow}</div> : null}
        <h2 className="panel-title">{title}</h2>
        {note ? <p className="panel-note">{note}</p> : null}
      </div>
      {actions ? <div className="panel-actions">{actions}</div> : null}
    </header>
  )
}

export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string
  value: ReactNode
  hint?: ReactNode
  tone?: 'accent' | 'ok' | 'warn' | 'bad'
}) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ''}`}>
      <div className="label">{label}</div>
      <div className="stat-value num">{value}</div>
      {hint ? <div className="stat-hint">{hint}</div> : null}
    </div>
  )
}

export function Badge({
  children,
  tone = 'neutral',
  title,
}: {
  children: ReactNode
  tone?: 'neutral' | 'accent' | 'ok' | 'warn' | 'bad' | 'info'
  title?: string
}) {
  return (
    <span className={`badge badge-${tone}`} title={title}>
      {children}
    </span>
  )
}

export function Callout({
  tone = 'info',
  title,
  children,
  action,
}: {
  tone?: 'info' | 'ok' | 'warn' | 'bad'
  title?: string
  children: ReactNode
  action?: ReactNode
}) {
  return (
    <div className={`callout callout-${tone}`}>
      <div className="callout-bar" aria-hidden />
      <div className="callout-body">
        {title ? <strong>{title}</strong> : null}
        <div>{children}</div>
      </div>
      {action ? <div className="callout-action">{action}</div> : null}
    </div>
  )
}

export function Field({
  label,
  hint,
  children,
  htmlFor,
}: {
  label: string
  hint?: ReactNode
  children: ReactNode
  htmlFor?: string
}) {
  return (
    <label className="field" htmlFor={htmlFor}>
      <span className="label">{label}</span>
      {children}
      {hint ? <span className="field-hint">{hint}</span> : null}
    </label>
  )
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: T
  options: { value: T; label: string; disabled?: boolean; title?: string }[]
  onChange: (value: T) => void
  ariaLabel: string
}) {
  return (
    <div className="segmented" role="group" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className={option.value === value ? 'seg active' : 'seg'}
          onClick={() => onChange(option.value)}
          aria-pressed={option.value === value}
          disabled={option.disabled}
          title={option.title}
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
  hint,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  hint?: ReactNode
}) {
  const id = useId()
  return (
    <div className="toggle-row">
      <div>
        <label htmlFor={id} className="toggle-label">
          {label}
        </label>
        {hint ? <div className="field-hint">{hint}</div> : null}
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        className={checked ? 'switch on' : 'switch'}
        onClick={() => onChange(!checked)}
      >
        <span className="switch-knob" />
      </button>
    </div>
  )
}

export function Dialog({
  open,
  onClose,
  title,
  children,
  width = 620,
}: {
  open: boolean
  onClose: () => void
  title: string
  children: ReactNode
  width?: number
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, onClose])
  if (!open) return null
  return (
    <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="dialog" role="dialog" aria-modal="true" aria-label={title} style={{ maxWidth: width }}>
        <header className="dialog-head">
          <h2>{title}</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            ×
          </button>
        </header>
        <div className="dialog-body">{children}</div>
      </div>
    </div>
  )
}

export function Disclosure({ summary, children, open, onToggle }: { summary: ReactNode; children: ReactNode; open?: boolean; onToggle?: () => void }) {
  const [internal, setInternal] = useState(false)
  const isOpen = open ?? internal
  return (
    <div className={isOpen ? 'disclosure open' : 'disclosure'}>
      <button
        type="button"
        className="disclosure-head"
        aria-expanded={isOpen}
        onClick={() => {
          if (onToggle) onToggle()
          else setInternal(!internal)
        }}
      >
        <span className="disclosure-chevron" aria-hidden>
          {isOpen ? '−' : '+'}
        </span>
        {summary}
      </button>
      {isOpen ? <div className="disclosure-body">{children}</div> : null}
    </div>
  )
}

export function Empty({
  title,
  children,
  action,
}: {
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="empty-block">
      <h3>{title}</h3>
      {children ? <div className="empty-copy">{children}</div> : null}
      {action ? <div className="empty-action">{action}</div> : null}
    </div>
  )
}

export function Progress({ value, label }: { value: number; label?: string }) {
  const clamped = Math.max(0, Math.min(100, value))
  return (
    <div className="progress" role="progressbar" aria-valuenow={clamped} aria-valuemin={0} aria-valuemax={100} aria-label={label ?? 'Progress'}>
      <div className="progress-fill" style={{ width: `${clamped}%` }} />
    </div>
  )
}
