import { useEffect, useState } from 'react'

/** Object URL created from a blob and revoked when it changes or unmounts.
 *  Creating it inside the effect keeps StrictMode's setup/cleanup cycle safe:
 *  the cleanup revokes only the URL that setup created. */
export function useObjectUrl(blob: Blob | null | undefined): string | null {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!blob) {
      setUrl(null)
      return
    }
    const next = URL.createObjectURL(blob)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return url
}

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => (typeof window === 'undefined' ? false : window.matchMedia(query).matches))
  useEffect(() => {
    const list = window.matchMedia(query)
    const handler = () => setMatches(list.matches)
    handler()
    list.addEventListener('change', handler)
    return () => list.removeEventListener('change', handler)
  }, [query])
  return matches
}

export function useNow(intervalMs = 1000): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])
  return now
}

export async function copyText(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}
