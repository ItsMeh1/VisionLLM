import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import { del, delMany, getAll, getAllByProject, get, put, putMany } from './db'
import { uploadDataset, type UploadResult } from './engine'
import { loadImageElement, makeThumbnail } from './image'
import { stratifiedSplit } from './analysis'
import type { ToolContext } from './agents/tools'
import type { AgentRun, ImageRecord, Project, RunRecord } from './types'
import { useApp } from './app'

export interface ImportReport {
  added: number
  skipped: string[]
  records: ImageRecord[]
}

interface ProjectState {
  project: Project | null
  images: ImageRecord[]
  runs: RunRecord[]
  loading: boolean
  saveProject: (patch: Partial<Project>) => Promise<Project | null>
  saveImages: (images: ImageRecord[]) => Promise<void>
  addFiles: (files: File[], origin?: ImageRecord['origin']) => Promise<ImportReport>
  addBlob: (blob: Blob, name: string, origin?: ImageRecord['origin']) => Promise<ImageRecord | null>
  removeImages: (ids: string[]) => Promise<void>
  saveRun: (run: RunRecord) => Promise<void>
  refreshRuns: () => Promise<void>
  refreshImages: () => Promise<void>
  uploadCurrentDataset: () => Promise<UploadResult>
  toolContext: () => ToolContext
  clearProject: () => Promise<void>
}

const ProjectContext = createContext<ProjectState | null>(null)

const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`)

const IMAGE_EXT = /\.(jpe?g|png|webp|gif|bmp|tiff?)$/i

function isImageFile(file: File): boolean {
  return file.type.startsWith('image/') || IMAGE_EXT.test(file.name)
}

function sortImages(images: ImageRecord[]): ImageRecord[] {
  return [...images].sort((a, b) => a.addedAt - b.addedAt)
}

export function ProjectProvider({ projectId, children }: { projectId: string; children: ReactNode }) {
  const { online, health, notify } = useApp()
  const [project, setProject] = useState<Project | null>(null)
  const [images, setImages] = useState<ImageRecord[]>([])
  const [runs, setRuns] = useState<RunRecord[]>([])
  const [loading, setLoading] = useState(true)
  const imagesRef = useRef<ImageRecord[]>([])
  const projectRef = useRef<Project | null>(null)
  const runsRef = useRef<RunRecord[]>([])

  useEffect(() => {
    imagesRef.current = images
  }, [images])
  useEffect(() => {
    projectRef.current = project
  }, [project])
  useEffect(() => {
    runsRef.current = runs
  }, [runs])

  useEffect(() => {
    let active = true
    setLoading(true)
    void (async () => {
      const [storedProject, storedImages, storedRuns] = await Promise.all([
        get<Project>('projects', projectId),
        getAllByProject<ImageRecord>('images', projectId),
        getAllByProject<RunRecord>('runs', projectId),
      ])
      if (!active) return
      setProject(storedProject ?? null)
      const sorted = sortImages(storedImages)
      imagesRef.current = sorted
      setImages(sorted)
      const orderedRuns = [...storedRuns].sort((a, b) => b.createdAt - a.createdAt)
      runsRef.current = orderedRuns
      setRuns(orderedRuns)
      setLoading(false)
    })()
    return () => {
      active = false
    }
  }, [projectId])

  const saveProject = useCallback(async (patch: Partial<Project>) => {
    const current = projectRef.current
    if (!current) return null
    const next: Project = { ...current, ...patch, id: current.id, updatedAt: Date.now() }
    projectRef.current = next
    setProject(next)
    await put('projects', next)
    return next
  }, [])

  useEffect(() => {
    if (!project || loading) return
    if (project.imageCount !== images.length) {
      void saveProject({ imageCount: images.length })
    }
  }, [images.length, project, loading, saveProject])

  const saveImages = useCallback(async (changed: ImageRecord[]) => {
    if (!changed.length) return
    // Update the UI before persistence so back-to-back annotation edits see the
    // latest boxes instead of racing the IndexedDB write.
    const map = new Map(imagesRef.current.map((image) => [image.id, image]))
    for (const image of changed) map.set(image.id, image)
    const next = sortImages([...map.values()])
    imagesRef.current = next
    setImages(next)
    await putMany('images', changed)
  }, [])

  const addRecords = useCallback(
    async (records: ImageRecord[]) => {
      if (!records.length) return
      await putMany('images', records)
      let next = sortImages([...imagesRef.current, ...records])
      const project = projectRef.current
      if (project && next.length > 2 && !next.some((image) => image.split === 'val')) {
        next = sortImages(stratifiedSplit(project, next, 0.2))
        await putMany('images', next)
      }
      imagesRef.current = next
      setImages(next)
    },
    [],
  )

  const addFiles = useCallback(
    async (files: File[], origin: ImageRecord['origin'] = 'upload'): Promise<ImportReport> => {
      const project = projectRef.current
      if (!project) return { added: 0, skipped: ['Project is still loading.'], records: [] }
      const skipped: string[] = []
      const records: ImageRecord[] = []
      const existing = new Set(imagesRef.current.map((image) => `${image.name}:${image.bytes}`))
      for (const file of files) {
        if (!isImageFile(file)) {
          skipped.push(`${file.name} is not an image`)
          continue
        }
        if (existing.has(`${file.name}:${file.size}`)) {
          skipped.push(`${file.name} was already imported`)
          continue
        }
        try {
          const image = await loadImageElement(file)
          const thumbnail = await makeThumbnail(file)
          const record: ImageRecord = {
            id: uid(),
            projectId: project.id,
            name: file.name || `image-${records.length + 1}`,
            blob: file,
            thumbnail,
            width: image.naturalWidth,
            height: image.naturalHeight,
            bytes: file.size,
            mime: file.type || 'image/png',
            addedAt: Date.now() + records.length,
            split: 'train',
            boxes: [],
            shapes: [],
            origin,
          }
          if (project.task === 'classify' && project.classes.length === 1) record.label = project.classes[0].id
          records.push(record)
          existing.add(`${file.name}:${file.size}`)
        } catch {
          skipped.push(`${file.name} could not be decoded`)
        }
      }
      await addRecords(records)
      return { added: records.length, skipped, records }
    },
    [addRecords],
  )

  const addBlob = useCallback(
    async (blob: Blob, name: string, origin: ImageRecord['origin'] = 'webcam') => {
      const project = projectRef.current
      if (!project) return null
      const file = new File([blob], name, { type: blob.type || 'image/png' })
      const image = await loadImageElement(file)
      const thumbnail = await makeThumbnail(file)
      const record: ImageRecord = {
        id: uid(),
        projectId: project.id,
        name,
        blob: file,
        thumbnail,
        width: image.naturalWidth,
        height: image.naturalHeight,
        bytes: file.size,
        mime: file.type || 'image/png',
        addedAt: Date.now(),
        split: 'train',
        boxes: [],
        shapes: [],
        origin,
      }
      await addRecords([record])
      return record
    },
    [addRecords],
  )

  const removeImages = useCallback(async (ids: string[]) => {
    await delMany('images', ids)
    const next = imagesRef.current.filter((image) => !ids.includes(image.id))
    imagesRef.current = next
    setImages(next)
  }, [])

  const saveRun = useCallback(async (run: RunRecord) => {
    await put('runs', run)
    const map = new Map(runsRef.current.map((entry) => [entry.id, entry]))
    map.set(run.id, run)
    const next = [...map.values()].sort((a, b) => b.createdAt - a.createdAt)
    runsRef.current = next
    setRuns(next)
  }, [])

  const refreshRuns = useCallback(async () => {
    const stored = await getAllByProject<RunRecord>('runs', projectId)
    const ordered = stored.sort((a, b) => b.createdAt - a.createdAt)
    runsRef.current = ordered
    setRuns(ordered)
  }, [projectId])

  const refreshImages = useCallback(async () => {
    const stored = await getAllByProject<ImageRecord>('images', projectId)
    const sorted = sortImages(stored)
    imagesRef.current = sorted
    setImages(sorted)
  }, [projectId])

  const uploadCurrentDataset = useCallback(async () => {
    const project = projectRef.current
    if (!project) throw new Error('Project not loaded')
    const current = imagesRef.current
    if (!current.length) throw new Error('There are no images to upload.')
    const response = await uploadDataset(project, current)
    if (response.skipped.length) {
      notify(`${response.skipped.length} images were rejected by the engine. Check the dataset log.`, 'error')
    }
    return response
  }, [notify])

  const toolContext = useCallback<() => ToolContext>(
    () => ({
      project: projectRef.current as Project,
      getImages: () => imagesRef.current,
      online,
      device: health?.device ?? 'unknown',
      saveImages,
      saveRun,
      getRuns: () => runsRef.current,
      ensureUploaded: async () => {
        await uploadCurrentDataset()
      },
    }),
    [online, health?.device, saveImages, saveRun, uploadCurrentDataset],
  )

  const clearProject = useCallback(async () => {
    const current = projectRef.current
    if (!current) return
    await delMany('images', imagesRef.current.map((image) => image.id))
    await delMany('runs', runsRef.current.map((run) => run.id))
    await del('projects', current.id)
    imagesRef.current = []
    runsRef.current = []
    setImages([])
    setRuns([])
    setProject(null)
  }, [])

  const value = useMemo<ProjectState>(
    () => ({
      project,
      images,
      runs,
      loading,
      saveProject,
      saveImages,
      addFiles,
      addBlob,
      removeImages,
      saveRun,
      refreshRuns,
      refreshImages,
      uploadCurrentDataset,
      toolContext,
      clearProject,
    }),
    [
      project,
      images,
      runs,
      loading,
      saveProject,
      saveImages,
      addFiles,
      addBlob,
      removeImages,
      saveRun,
      refreshRuns,
      refreshImages,
      uploadCurrentDataset,
      toolContext,
      clearProject,
    ],
  )

  return <ProjectContext.Provider value={value}>{children}</ProjectContext.Provider>
}

export function useProject(): ProjectState {
  const context = useContext(ProjectContext)
  if (!context) throw new Error('useProject must be used inside ProjectProvider')
  return context
}

export async function loadAllProjects(): Promise<Project[]> {
  const projects = await getAll<Project>('projects')
  return projects.sort((a, b) => b.updatedAt - a.updatedAt)
}

/** Remove a project and everything the browser stores for it. Engine cleanup is separate. */
export async function deleteProjectData(projectId: string): Promise<void> {
  const [images, runs, agentRuns] = await Promise.all([
    getAllByProject<ImageRecord>('images', projectId),
    getAllByProject<RunRecord>('runs', projectId),
    getAllByProject<AgentRun>('agentRuns', projectId),
  ])
  await Promise.all([
    delMany('images', images.map((image) => image.id)),
    delMany('runs', runs.map((run) => run.id)),
    delMany('agentRuns', agentRuns.map((run) => run.id)),
  ])
  await del('projects', projectId)
}
