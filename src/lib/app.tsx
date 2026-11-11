import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { get, put } from './db'
import { health, type Health } from './engine'
import type { LlmSettings } from './types'

export interface Toast {
  id: number
  message: string
  tone: 'info' | 'ok' | 'error'
}

interface AppState {
  health: Health | null
  online: boolean
  checking: boolean
  refreshHealth: () => Promise<void>
  settings: LlmSettings
  saveSettings: (next: LlmSettings) => Promise<void>
  toasts: Toast[]
  notify: (message: string, tone?: Toast['tone']) => void
  dismissToast: (id: number) => void
}

const DEFAULT_LLM: LlmSettings = { enabled: false, baseUrl: 'https://api.openai.com/v1', model: '', apiKey: '' }

const AppContext = createContext<AppState | null>(null)

export function AppProvider({ children }: { children: ReactNode }) {
  const [healthState, setHealthState] = useState<Health | null>(null)
  const [checking, setChecking] = useState(true)
  const [settings, setSettings] = useState<LlmSettings>(DEFAULT_LLM)
  const [toasts, setToasts] = useState<Toast[]>([])
  const toastId = useRef(1)

  const refreshHealth = useCallback(async () => {
    setChecking(true)
    try {
      const next = await health()
      setHealthState(next)
    } catch {
      setHealthState(null)
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => {
    void refreshHealth()
    const timer = window.setInterval(() => void refreshHealth(), 20000)
    return () => window.clearInterval(timer)
  }, [refreshHealth])

  useEffect(() => {
    void (async () => {
      const stored = await get<{ key: string; value: LlmSettings }>('settings', 'llm')
      if (stored?.value) setSettings({ ...DEFAULT_LLM, ...stored.value })
    })()
  }, [])

  const saveSettings = useCallback(async (next: LlmSettings) => {
    setSettings(next)
    await put('settings', { key: 'llm', value: next })
  }, [])

  const dismissToast = useCallback((id: number) => {
    setToasts((current) => current.filter((toast) => toast.id !== id))
  }, [])

  const notify = useCallback(
    (message: string, tone: Toast['tone'] = 'info') => {
      const id = toastId.current++
      setToasts((current) => [...current.slice(-2), { id, message, tone }])
      window.setTimeout(() => dismissToast(id), tone === 'error' ? 9000 : 5000)
    },
    [dismissToast],
  )

  const value = useMemo<AppState>(
    () => ({
      health: healthState,
      online: Boolean(healthState?.ok),
      checking,
      refreshHealth,
      settings,
      saveSettings,
      toasts,
      notify,
      dismissToast,
    }),
    [healthState, checking, refreshHealth, settings, saveSettings, toasts, notify, dismissToast],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp(): AppState {
  const context = useContext(AppContext)
  if (!context) throw new Error('useApp must be used inside AppProvider')
  return context
}
