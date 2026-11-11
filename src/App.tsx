import { useEffect } from 'react'
import { TopBar } from './components/TopBar'
import { Toasts } from './components/Toasts'
import { AppProvider } from './lib/app'
import { ProjectProvider } from './lib/store'
import { useRoute } from './lib/router'
import { ProjectsScreen } from './screens/ProjectsScreen'
import { NewProjectScreen } from './screens/NewProjectScreen'
import { ProjectLayout } from './screens/ProjectLayout'
import { SettingsScreen } from './screens/SettingsScreen'

function Router() {
  const { parts } = useRoute()
  const projectId = parts[0] === 'p' ? parts[1] : undefined
  const step = parts[2] ?? 'workspace'
  const routeKey = parts.join('/')

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [routeKey])

  return (
    <div className="app">
      {!projectId ? <TopBar /> : null}
      {parts[0] === 'new' ? <NewProjectScreen /> : null}
      {parts[0] === 'settings' ? <SettingsScreen /> : null}
      {projectId ? (
        <ProjectProvider projectId={projectId} key={projectId}>
          <ProjectLayout step={step} />
        </ProjectProvider>
      ) : null}
      {!projectId && parts[0] !== 'new' && parts[0] !== 'settings' ? <ProjectsScreen /> : null}
      <Toasts />
    </div>
  )
}

export default function App() {
  return (
    <AppProvider>
      <Router />
    </AppProvider>
  )
}
