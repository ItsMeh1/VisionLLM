import type { ReactNode } from 'react'

type IconProps = { size?: number; className?: string }

function Svg({ size = 18, className, children }: IconProps & { children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

export function IconPlus(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 5v14M5 12h14" />
    </Svg>
  )
}

export function IconCamera(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M4 8h3l1.4-2h7.2L17 8h3v11H4z" />
      <circle cx="12" cy="13.2" r="3.6" />
    </Svg>
  )
}

export function IconSpark(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 3.5l1.7 4.6 4.8 1.7-4.8 1.7L12 16.1l-1.7-4.6L5.5 9.8l4.8-1.7z" />
      <path d="M18.2 15.5l.8 2.1 2.1.8-2.1.8-.8 2.1-.8-2.1-2.1-.8 2.1-.8z" />
    </Svg>
  )
}

export function IconX(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M6 6l12 12M18 6L6 18" />
    </Svg>
  )
}

export function IconUpload(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M12 16V4M7 9l5-5 5 5" />
      <path d="M4 16v3.5h16V16" />
    </Svg>
  )
}

export function IconPlay(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M8 5l11 7-11 7z" />
    </Svg>
  )
}

export function IconStop(props: IconProps) {
  return (
    <Svg {...props}>
      <rect x="6" y="6" width="12" height="12" rx="2" />
    </Svg>
  )
}

export function IconChevron(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M9 5l7 7-7 7" />
    </Svg>
  )
}

export function IconCheck(props: IconProps) {
  return (
    <Svg {...props}>
      <path d="M5 12.5l4.2 4.2L19 7" />
    </Svg>
  )
}
