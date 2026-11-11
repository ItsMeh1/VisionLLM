import { Wordmark } from './Signature'

export function TopBar() {
  return (
    <header className="vl-header">
      <div className="vl-header-inner">
        <a className="brand" href="#/" aria-label="VisionLLM home">
          <Wordmark />
        </a>
        <nav className="vl-nav" aria-label="Main">
          <a href="#/">Projects</a>
        </nav>
        <div className="vl-header-actions">
          <span className="privacy-pill">
            <span className="privacy-dot" aria-hidden />
            Local &amp; private
          </span>
          <a className="icon-link" href="#/settings">
            Settings
          </a>
        </div>
      </div>
    </header>
  )
}
