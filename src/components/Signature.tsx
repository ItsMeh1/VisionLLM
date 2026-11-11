/** VisionLLM's mark: a detection frame with a found point. */
export function Mark({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true" className="brand-mark">
      <rect x="0" y="0" width="32" height="32" rx="9.5" fill="currentColor" />
      <rect x="7.2" y="7.2" width="13.2" height="13.2" rx="3.4" fill="none" stroke="#fff" strokeWidth="2" />
      <circle cx="22.4" cy="21.6" r="2.8" fill="#fff" />
    </svg>
  )
}

export function Wordmark() {
  return (
    <span className="wordmark">
      <Mark />
      <span className="wordmark-text">
        Vision<span className="wordmark-accent">LLM</span>
      </span>
    </span>
  )
}

/** Three aspect ratios inside one square input: the product's core promise. */
export function LetterboxDiagram() {
  const frames = [
    { label: '16:9', w: 98, h: 55 },
    { label: '4:3', w: 90, h: 68 },
    { label: '1:1', w: 76, h: 76 },
  ]
  return (
    <figure className="letterbox-diagram" aria-label="Three aspect ratios fitted into one square input with padding instead of cropping">
      <div className="letterbox-board">
        {frames.map((frame) => (
          <div key={frame.label} className="letterbox-sample">
            <div className="letterbox-square">
              <div className="letterbox-pad" style={{ width: frame.w, height: frame.h }}>
                <div className="letterbox-photo" style={{ width: frame.w, height: frame.h }}>
                  <span className="letterbox-box" />
                </div>
              </div>
            </div>
            <span className="letterbox-ratio">{frame.label}</span>
          </div>
        ))}
      </div>
      <figcaption>
        <strong>Letterboxed, never cropped.</strong>
        <span>Every shape arrives intact. Padding fills the square.</span>
      </figcaption>
    </figure>
  )
}
