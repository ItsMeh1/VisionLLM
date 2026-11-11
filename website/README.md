# VisionLLM website

A static, single-page showcase for VisionLLM. No framework and no build step.

## Preview

From the repository root:

```bash
python3 -m http.server 4173 --directory website
```

Open <http://127.0.0.1:4173>.

The page uses the local Figtree and JetBrains Mono font files in `assets/fonts/`.
The example parts in the workbench illustration are synthetic SVG drawings, not
user images.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure and inline interface illustration |
| `styles.css` | Responsive layout and brand styling |
| `main.js` | Copy-to-clipboard enhancement |
| `assets/` | Logo, web fonts and font licences |

This branch is intentionally separate from the Vite app. The product source lives
on `main`.
