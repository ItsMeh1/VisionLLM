# VisionLLM website

The static, single-page showcase for VisionLLM. This branch contains only the site, so GitHub Pages can serve the
repository root directly. No framework, build step or TypeScript.

## Preview locally

From the repository root:

```bash
python3 -m http.server 4173
```

Open <http://127.0.0.1:4173>.

## GitHub Pages

Publish this branch from the repository root (`/`) in Settings → Pages. `.nojekyll` keeps the files exactly as
committed. There is no build command.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure and inline interface illustration |
| `styles.css` | Responsive layout and brand styling |
| `main.js` | Copy-to-clipboard enhancement |
| `assets/` | Logo, web fonts and font licences |
| `.nojekyll` | Tells GitHub Pages to serve the static files without Jekyll |

The app and its source live on `main`; this branch is intentionally only the showcase. The example parts in the
workbench illustration are synthetic SVG drawings, not user images.
