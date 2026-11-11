import { useEffect, useRef, useState } from 'react'
import { ExamplesPanel } from '../components/workspace/ExamplesPanel'
import { TrainingPanel } from '../components/workspace/TrainingPanel'
import { PreviewPanel } from '../components/workspace/PreviewPanel'
import { AnnotatorDialog } from '../components/workspace/AnnotatorDialog'
import { useProject } from '../lib/store'

export function WorkspaceScreen({ initialAnnotate = false }: { initialAnnotate?: boolean }) {
  const { images } = useProject()
  const [annotate, setAnnotate] = useState<{ imageId?: string; classId?: string } | null>(null)
  const opened = useRef(false)
  const previewRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (!initialAnnotate || opened.current || !images.length) return
    opened.current = true
    setAnnotate({})
  }, [initialAnnotate, images.length])

  return (
    <div className="workspace">
      <div className="workspace-grid">
        <ExamplesPanel onAnnotate={(imageId, classId) => setAnnotate({ imageId, classId })} />
        <TrainingPanel onPreview={() => previewRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })} />
        <div className="preview-column" ref={previewRef}>
          <PreviewPanel />
        </div>
      </div>
      {annotate ? (
        <AnnotatorDialog
          initialImageId={annotate.imageId}
          initialClassId={annotate.classId}
          onClose={() => setAnnotate(null)}
        />
      ) : null}
    </div>
  )
}
