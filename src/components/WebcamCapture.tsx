import { useCallback, useEffect, useRef, useState } from 'react'
import { Button, Dialog } from './ui'
import { canvasToBlob } from '../lib/image'

export function WebcamCapture({
  open,
  onClose,
  onCapture,
}: {
  open: boolean
  onClose: () => void
  onCapture: (blob: Blob) => Promise<void>
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [ready, setReady] = useState(false)
  const [busy, setBusy] = useState(false)
  const [lastShot, setLastShot] = useState<string | null>(null)

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    setReady(false)
  }, [])

  useEffect(() => {
    if (!open) {
      stop()
      return
    }
    let cancelled = false
    void (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        })
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play().catch(() => undefined)
          setReady(true)
        }
      } catch (err) {
        setError(
          err instanceof Error && err.name === 'NotAllowedError'
            ? 'Camera permission was refused. Allow camera access in the browser, then reopen this panel.'
            : 'No camera is available. You can still upload images or a folder.',
        )
      }
    })()
    return () => {
      cancelled = true
      stop()
    }
  }, [open, stop])

  const capture = async () => {
    const video = videoRef.current
    if (!video) return
    setBusy(true)
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const context = canvas.getContext('2d')
    if (context) {
      context.drawImage(video, 0, 0)
      const blob = await canvasToBlob(canvas, 'image/png')
      await onCapture(blob)
      setLastShot(URL.createObjectURL(blob))
    }
    setBusy(false)
  }

  return (
    <Dialog open={open} onClose={onClose} title="Capture with camera" width={720}>
      {error ? (
        <p className="callout callout-warn">{error}</p>
      ) : (
        <div className="webcam-stage">
          <video ref={videoRef} muted playsInline className="webcam-video" />
          {!ready ? <div className="webcam-waiting mono">starting camera…</div> : null}
        </div>
      )}
      <div className="webcam-actions">
        <Button variant="primary" onClick={() => void capture()} disabled={!ready || busy} busy={busy}>
          Capture frame
        </Button>
        <Button variant="ghost" onClick={onClose}>
          Done
        </Button>
        <span className="field-hint">Frames are captured as PNG at the camera's native resolution.</span>
      </div>
      {lastShot ? (
        <div className="webcam-last">
          <img src={lastShot} alt="Last captured frame" />
          <span className="mono">added to the dataset</span>
        </div>
      ) : null}
    </Dialog>
  )
}
