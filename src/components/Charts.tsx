import type { ReactNode } from 'react'

interface Series {
  key: string
  label: string
  color: string
  points: { x: number; y: number }[]
}

export function LineChart({
  series,
  height = 180,
  yFormat = (value: number) => value.toFixed(2),
  xLabel = 'epoch',
  yMax,
  emptyLabel = 'No data yet',
}: {
  series: Series[]
  height?: number
  yFormat?: (value: number) => string
  xLabel?: string
  yMax?: number
  emptyLabel?: string
}) {
  const all = series.flatMap((entry) => entry.points)
  const hasData = all.length > 1
  const width = 640
  const padding = { left: 42, right: 14, top: 14, bottom: 26 }
  const innerW = width - padding.left - padding.right
  const innerH = height - padding.top - padding.bottom
  const maxX = Math.max(1, ...all.map((point) => point.x))
  const maxY = yMax ?? Math.max(0.0001, ...all.map((point) => point.y)) * 1.15
  const x = (value: number) => padding.left + (value / maxX) * innerW
  const y = (value: number) => padding.top + innerH - (value / maxY) * innerH
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => maxY * fraction)

  if (!hasData) {
    return (
      <div className="chart-empty" style={{ height }}>
        <span className="mono">{emptyLabel}</span>
      </div>
    )
  }

  return (
    <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training metrics over epochs">
      {ticks.map((tick) => (
        <g key={tick}>
          <line
            x1={padding.left}
            x2={width - padding.right}
            y1={y(tick)}
            y2={y(tick)}
            stroke="var(--line)"
            strokeDasharray="2 4"
          />
          <text x={padding.left - 8} y={y(tick) + 3} textAnchor="end" className="chart-tick">
            {yFormat(tick)}
          </text>
        </g>
      ))}
      <line x1={padding.left} x2={width - padding.right} y1={padding.top + innerH} y2={padding.top + innerH} stroke="var(--line-strong)" />
      {series.map((entry) => {
        if (entry.points.length < 2) return null
        const path = entry.points
          .map((point, index) => `${index === 0 ? 'M' : 'L'}${x(point.x).toFixed(1)},${y(point.y).toFixed(1)}`)
          .join(' ')
        const last = entry.points[entry.points.length - 1]
        return (
          <g key={entry.key}>
            <path d={path} fill="none" stroke={entry.color} strokeWidth="1.8" strokeLinejoin="round" />
            <circle cx={x(last.x)} cy={y(last.y)} r="2.6" fill={entry.color} />
          </g>
        )
      })}
      <text x={width - padding.right} y={height - 6} textAnchor="end" className="chart-tick">
        {xLabel} {maxX}
      </text>
      {series.map((entry, index) => (
        <g key={entry.key} transform={`translate(${padding.left + index * 132}, ${height - 6})`}>
          <line x1="0" x2="12" y1="-4" y2="-4" stroke={entry.color} strokeWidth="2" />
          <text x="17" y="0" className="chart-legend">
            {entry.label}
          </text>
        </g>
      ))}
    </svg>
  )
}

export function BarList({
  items,
  format = (value) => `${(value * 100).toFixed(1)}%`,
  max,
  tone = 'accent',
}: {
  items: { label: string; value: number | null; color?: string; hint?: ReactNode }[]
  format?: (value: number) => string
  max?: number
  tone?: 'accent' | 'ok'
}) {
  const ceiling = max ?? Math.max(0.0001, ...items.map((item) => item.value ?? 0))
  return (
    <div className="bar-list">
      {items.map((item) => (
        <div className="bar-row" key={item.label}>
          <div className="bar-label">
            {item.color ? <span className="swatch" style={{ background: item.color }} /> : null}
            <span>{item.label}</span>
          </div>
          <div className="bar-track">
            <div
              className={`bar-fill ${tone}`}
              style={{
                width: `${Math.max(1.5, ((item.value ?? 0) / ceiling) * 100)}%`,
                background: item.value === null ? 'var(--line)' : undefined,
              }}
            />
          </div>
          <div className="bar-value num">{item.value === null ? 'n/a' : format(item.value)}</div>
        </div>
      ))}
    </div>
  )
}

export function Confusion({
  matrix,
  classes,
}: {
  matrix: number[][]
  classes: string[]
}) {
  if (!matrix.length) return null
  const max = Math.max(1, ...matrix.flat())
  return (
    <div className="confusion-wrap">
      <table className="confusion">
        <thead>
          <tr>
            <th className="corner">true ↓ / pred →</th>
            {classes.map((name) => (
              <th key={name} title={name}>
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, rowIndex) => (
            <tr key={classes[rowIndex]}>
              <th scope="row">{classes[rowIndex]}</th>
              {row.map((value, columnIndex) => {
                const correct = rowIndex === columnIndex
                const intensity = value / max
                return (
                  <td
                    key={columnIndex}
                    className={correct ? 'correct' : value > 0 ? 'wrong' : ''}
                    style={{
                      background: value
                        ? correct
                          ? `rgba(143, 212, 138, ${0.12 + intensity * 0.6})`
                          : `rgba(255, 107, 107, ${0.1 + intensity * 0.5})`
                        : undefined,
                    }}
                    title={`${classes[rowIndex]} → ${classes[columnIndex]}: ${value}`}
                  >
                    {value || ''}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Sparkline({
  values,
  color = 'var(--accent)',
  height = 34,
}: {
  values: number[]
  color?: string
  height?: number
}) {
  if (values.length < 2) return <div className="spark-empty" style={{ height }} />
  const max = Math.max(...values)
  const min = Math.min(...values)
  const range = max - min || 1
  const points = values
    .map((value, index) => `${(index / (values.length - 1)) * 100},${height - ((value - min) / range) * (height - 4) - 2}`)
    .join(' ')
  return (
    <svg className="sparkline" viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" style={{ height }} aria-hidden>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}
