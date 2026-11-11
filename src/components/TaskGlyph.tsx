import type { Task } from '../lib/types'

export function TaskGlyph({ task, size = 44 }: { task: Task; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="4" y="8" width="40" height="32" rx="7" opacity="0.4" />
      {task === 'detect' ? (
        <>
          <rect x="11" y="16" width="13" height="10" rx="2.5" />
          <rect x="26" y="24" width="11" height="9" rx="2.5" />
          <circle cx="11" cy="16" r="1.7" fill="currentColor" stroke="none" />
          <circle cx="37" cy="33" r="1.7" fill="currentColor" stroke="none" />
        </>
      ) : null}
      {task === 'classify' ? (
        <>
          <path d="M13 18h13M13 24h20M13 30h9" />
          <path d="M32 17.5l2.5 2.5 4.5-5" />
        </>
      ) : null}
      {task === 'segment' ? (
        <>
          <path
            d="M14 33.5c-3-6 1-13.5 8-15.5 8-2 13.5 2 13.5 8 0 6-6 8-11.5 8-4 0-8 3-10-.5z"
            fill="currentColor"
            fillOpacity="0.12"
          />
          <circle cx="21" cy="25" r="1.5" fill="currentColor" stroke="none" />
        </>
      ) : null}
    </svg>
  )
}
