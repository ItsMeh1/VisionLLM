# AGENTS.md — VisionLLM

VisionLLM is a local-first computer-vision trainer. The UI, originals and annotations live in the browser; a loopback
Python engine trains real models; exported ONNX models run in the browser. There are no accounts, no cloud uploads and
no mock training.

This file is the shared instruction set for external coding agents (Codex, Claude Code, Freebuff, …). `CLAUDE.md`
points Claude Code here. Read it before changing code. For a shorter onboarding map, start with
[docs/agents.md](docs/agents.md); the contract below is binding.

## Product contract

- **Free, local, accountless.** Do not add authentication, telemetry, hosted storage, analytics or automatic uploads.
- **Real functionality only.** Do not add placeholder buttons, simulated progress, fake metrics or canned results.
- **Never modify originals.** Imported bytes are stored exactly as received; annotations are separate normalised
  coordinates stored beside the image. Never paint annotations onto the original.
- **Preserve aspect ratio.** Every model input is letterboxed (`min(size/w, size/h)`, pad 114, centred padding).
  Never center-crop or squash an image into a square, in Python or in the browser.
- **The UI is a light, friendly workbench.** `src/styles/tokens.css` holds the palette, type and radii. Reuse the
  existing components, CSS variables and layout patterns instead of introducing a second design system.
- **No in-app chat or agent dock.** External coding agents are the LLM integration. There is intentionally no Ask
  button, embedded assistant or API-key panel; do not add one back.
- **One shared image library.** Adding images must put them in the library, then let the user open an image, draw
  boxes or labels, and assign classes. Class cards and counts are derived automatically from annotations. Do not add
  per-class import paths that pre-label images. Do not add train/validation set controls; keep one image library and
  choose a small scoring holdout automatically when training starts.
- **Deleting app data never deletes originals.** Project deletion clears browser records and asks the engine to
  remove its training copy; the user's source image files must never be touched.

## Run it

```bash
npm install
npm run engine     # terminal 1 — FastAPI on http://127.0.0.1:8765
npm run dev        # terminal 2 — Vite on http://127.0.0.1:5173
```

The engine uses `engine/.venv`. Recreate it if necessary:

```bash
cd engine
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

The engine only binds to loopback. Its data lives in `~/.visionllm` (`VISIONLLM_HOME` overrides it); the port is
`VISIONLLM_PORT` (default 8765).

## Where things live

| Path | Responsibility |
| --- | --- |
| `docs/` | Setup, architecture and the short agent guide |
| `src/lib/store.tsx` | IndexedDB project/image/run state, imports, dataset upload |
| `src/lib/engine.ts` | Typed client for the loopback engine |
| `src/lib/letterbox.ts` | Browser-side letterbox geometry (must match Python) |
| `src/components/AnnotationCanvas.tsx` | Real canvas boxes, polygons, handles, proposals |
| `src/components/workspace/` | Examples library, training card, preview card, annotation dialog |
| `src/screens/WorkspaceScreen.tsx` | Composes the three-part workspace |
| `src/screens/ProjectLayout.tsx` | Project header, Workspace/Evaluate/Export tabs |
| `src/screens/` | Projects, new project, project layout, workspace, evaluate, export, settings |
| `src/styles/app.css` | All layout and component styles |
| `engine/server.py` | FastAPI routes, dataset storage, job orchestration, inference |
| `engine/vlyolo.py` | Original Apache-2.0 detector: model, losses, decode, ONNX wrapper |
| `engine/train_detect.py`, `train_cls.py`, `train_seg.py` | Real training and evaluation loops |
| `engine/exporter.py` | ONNX export, ONNX Runtime verification, JS template, zip |
| `engine/selftest.py` | Synthetic end-to-end training/ONNX test (no HTTP) |
| `engine/apitest.py` | Full HTTP API exercise: upload → train → evaluate → predict → export |

The browser is the source of truth for projects and originals. Before training, the UI uploads a dataset copy to
`~/.visionllm/projects/<project-id>/`; the engine never reads the browser database itself.

## Engine API

Base URL: `http://127.0.0.1:8765`. IDs match `[A-Za-z0-9_-]{1,64}`. The CORS policy allows only loopback origins.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Versions, device, storage path, model specs, licences |
| GET | `/health/simple` | Liveness only |
| POST | `/projects/{id}/dataset` | Multipart upload: `manifest` JSON + `files`; replaces the dataset copy |
| GET | `/projects/{id}/dataset` | Dataset summary |
| GET | `/projects/{id}/dataset.zip` | YOLO-format dataset zip |
| POST | `/projects/{id}/train` | Start a real job; returns `job_id`, `run_id` |
| GET | `/jobs/{job_id}` | Status, epoch events, metrics, errors |
| POST | `/jobs/{job_id}/cancel` | Request cancellation |
| GET | `/projects/{id}/runs` | Completed runs and summaries |
| POST | `/projects/{id}/runs/{run}/evaluate` | Recompute metrics from the checkpoint on validation images |
| POST | `/projects/{id}/runs/{run}/predict` | Multipart images; form fields `conf`, `iou` |
| POST | `/projects/{id}/runs/{run}/export` | Build ONNX + labels + inference.js + model card + zip |
| GET | `/projects/{id}/runs/{run}/export/{name}` | Download an exported artifact |
| POST | `/autolabel` | Multipart images, `classes` JSON, `model`, `conf`; COCO proposals |
| DELETE | `/projects/{id}` | Remove that project's engine copy |

Dataset manifest shape:

```json
{
  "task": "detect",
  "classes": ["square", "circle"],
  "images": [
    {
      "id": "bench-01",
      "file": "bench-01.png",
      "split": "train",
      "width": 320,
      "height": 240,
      "boxes": [{ "cls": 0, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4 }]
    }
  ]
}
```

`task` is `detect`, `classify` or `segment`. Detection images use normalised `boxes`; segmentation uses normalised
`shapes` (polygons); classification uses an integer `label`. Every multipart file must be sent under the field name
`files` with a filename matching the manifest's `file`.

Training config: `{ "model": "vlyolo-s", "epochs": 40, "batch": 8, "imgsz": 512, "lr": 0.001 }` (detection model
names come from `/health`; classification and segmentation ignore `model`). Poll `/jobs/{job_id}` until `status` is
`completed`, `failed` or `cancelled`.

## Verification

Run these before handing off:

```bash
npx tsc --noEmit
npm run build
engine/.venv/bin/python -m engine.selftest
# with the engine already listening:
engine/.venv/bin/python -m engine.apitest
```

`selftest.py` trains a synthetic detector and checks the ONNX graph against PyTorch. `apitest.py` exercises the HTTP
path end to end with real images. UI changes should also be inspected at 1440, 1024, 768 and 390 px widths.

## Working rules

- Read the component and its callers before changing shared state; most training logic is already real and should be
  preserved.
- Keep the engine contract and the browser letterbox geometry in sync. Changing one without the other is a bug.
- Use the existing index/run formats; do not silently rename stored fields.
- Do not use Ultralytics YOLO (AGPL-3.0). VLYOLO is original, Apache-2.0 code.
- Do not commit `node_modules`, `dist`, `engine/.venv`, caches, `.DS_Store`, screenshots or secrets; `.gitignore`
  already covers them.
- Update this file, `README.md` and the relevant `docs/` page when commands, the API or the UX contract changes.
- Keep changes focused. Do not replace working training, annotation, dataset or export paths with mockups.

## Website

The standalone showcase lives on the `website` branch under `website/`. It is plain
HTML, CSS and JavaScript with no build step and is not part of the Vite app. Keep the
claims factual and use synthetic example artwork, never private user images or a fake
live-training result.
