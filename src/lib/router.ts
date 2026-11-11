import { useEffect, useState } from 'react'

export function currentHash(): string {
  return location.hash.replace(/^#/, '') || '/'
}

export function navigate(path: string): void {
  location.hash = path
}

export function useRoute(): { hash: string; parts: string[]; query: URLSearchParams } {
  const [hash, setHash] = useState(currentHash)
  useEffect(() => {
    const onChange = () => setHash(currentHash())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  const [path, search] = hash.split('?')
  return {
    hash,
    parts: path.split('/').filter(Boolean),
    query: new URLSearchParams(search ?? ''),
  }
}
