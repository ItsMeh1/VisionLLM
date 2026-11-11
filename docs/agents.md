# Agent guide

Read [AGENTS.md](../AGENTS.md) first. It is the binding contract. This page is the
short map for getting oriented without reading the whole engine.

## Mental model

VisionLLM is a local-first vision trainer with three real parts:

1. A React/Vite app in the browser owns projects, original images, annotations and runs.
2. A loopback FastAPI engine receives a dataset copy, trains with PyTorch and writes
   checkpoints and exports under `~/.visionllm`.
3. Exported ONNX models run in the browser through ONNX Runtime Web.

There are no accounts, telemetry, cloud storage or hosted inference.

## Non-negotiables

- Keep it local and accountless. Do not add authentication, analytics or uploads.
- Do not add fake buttons, canned metrics or simulated training progress.
- Never paint annotations onto originals. Store normalised coordinates separately.
- Letterbox every model input; never crop or squash an aspect ratio.
- Keep one image library. No train/validation set management in the UI; the scoring
  holdout is automatic.
- Do not add an in-app chat, Ask button or agent dock. External agents are the LLM
  integration.
- Do not use Ultralytics YOLO. VLYOLO is original Apache-2.0 code.

## Run it

```bash
npm install
npm run engine   # terminal 1
npm run dev      # terminal 2
```

The page is at <http://127.0.0.1:5173>; the engine health check is at
<http://127.0.0.1:8765/health>. Details and venv setup are in [Setup](setup.md).

## Where to work

| Area | Files |
| --- | --- |
| Browser state and imports | `src/lib/store.tsx` |
| Engine client | `src/lib/engine.ts` |
| Annotation canvas | `src/components/AnnotationCanvas.tsx` |
| Workspace panels | `src/components/workspace/` |
| Screens and navigation | `src/screens/` |
| Design tokens and layout | `src/styles/tokens.css`, `src/styles/app.css` |
| Dataset storage and API | `engine/server.py`, `engine/data.py` |
| Training and evaluation | `engine/train_detect.py`, `train_cls.py`, `train_seg.py` |
| ONNX export | `engine/exporter.py` |
| Tests | `engine/selftest.py`, `engine/apitest.py` |

## Engine contract

Base URL: `http://127.0.0.1:8765`. IDs match `[A-Za-z0-9_-]{1,64}`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Versions, device, storage, model specs |
| POST | `/projects/{id}/dataset` | Upload dataset copy |
| GET | `/projects/{id}/dataset` | Dataset summary |
| POST | `/projects/{id}/train` | Start a real training job |
| GET | `/jobs/{job_id}` | Poll status and epoch events |
| POST | `/jobs/{job_id}/cancel` | Request cancellation |
| POST | `/projects/{id}/runs/{run}/evaluate` | Recompute metrics |
| POST | `/projects/{id}/runs/{run}/predict` | Run a checkpoint |
| POST | `/projects/{id}/runs/{run}/export` | Build ONNX bundle |
| DELETE | `/projects/{id}` | Remove the engine copy |

The full manifest shape and route details are in [AGENTS.md](../AGENTS.md#engine-api)
and [Architecture](architecture.md).

## Verify before handing off

```bash
npx tsc --noEmit
npm run build
engine/.venv/bin/python -m engine.selftest
```

With the engine running:

```bash
engine/.venv/bin/python -m engine.apitest
```

For UI work, inspect at 1440, 1024, 768 and 390 px. Use real state and real engine
results when testing; do not replace a working path with a mockup.

## Website branch

The standalone showcase lives on the `website` branch under `website/`. It is plain
HTML, CSS and JavaScript with no build step. It is separate from the Vite app.

Keep the site factual, use synthetic example artwork instead of private user images,
and do not imply that the page is a live training demo.

## Definition of done

- The app still typechecks and builds.
- Training, evaluation, annotation and export paths remain real.
- Originals and stored record formats are unchanged unless the change requires it.
- Docs are updated when commands, the API or the UX contract changes.
- No generated files, private screenshots, virtualenvs or secrets are committed.
