# AGENTS.md — VisionLLM model playbook

This is the agent guide for making a computer-vision model with VisionLLM. It is written for an AI agent — Claude
Code, Codex, Freebuff or another external agent — working with a person on that person's images.

It is **not** a guide to building VisionLLM. Do not change the app to get a result. Use the workbench and the loopback
engine. If the request is to modify the product itself, that is a separate job: use `README.md`, `docs/setup.md` and
`docs/architecture.md` instead.

## What success means

The agent's job is to help the user go from their own images to a real, evaluated model:

1. Choose the smallest task that answers their goal.
2. Define consistent classes and help build a clean labelled dataset.
3. Train it locally with the Python engine.
4. Read metrics from the held-out images, not from the training loop.
5. Export the verified ONNX bundle and explain clearly what it can and cannot do.

A finished run without an honest evaluation is not finished.

## Non-negotiables

- **Everything stays local.** The engine binds to loopback. Do not add accounts, telemetry, cloud storage, hosted
  inference or automatic uploads.
- **Never modify an original.** Imported bytes are stored exactly as received. Annotations are separate, normalised
  coordinates. Do not paint labels into an image.
- **Never invent data.** Do not fabricate images, labels, boxes, polygons, metrics, training progress or a finished
  export. Annotation suggestions are proposals; a person verifies them.
- **Use held-out evidence.** Training loss is not validation accuracy. Report the metrics the engine produces on the
  scoring holdout, with the number of images used.
- **Keep one image library.** Do not organise per-class import paths. Do not add train/validation set controls; the
  app chooses a small holdout automatically.
- **Respect aspect ratio.** The engine letterboxes every input. Do not pre-crop or squash images to make them square.
- **Report limitations.** Say what the data does not cover and where the model is likely to fail.

## 1. Start the local workbench

From the repository root:

```bash
npm install
npm run engine   # terminal 1 — FastAPI on http://127.0.0.1:8765
npm run dev      # terminal 2 — Vite on http://127.0.0.1:5173
```

The first run needs the engine environment. Follow [docs/setup.md](docs/setup.md) for the one-time Python setup.

Check that the engine is alive and read its capabilities before training:

```bash
curl -s http://127.0.0.1:8765/health
```

`/health` returns the device, storage path and the model specs that are actually available. Use the model names it
returns. Do not assume a different checkpoint is installed.

Work in the browser UI when the user is present. Use the loopback HTTP API when you need to automate a batch or
build a repeatable dataset upload.

## 2. Choose the task and the classes

Start with the user's question, not with a model.

| The user wants to… | Task | Labels |
| --- | --- | --- |
| Find, locate or count separate objects | `detect` | One box per instance, with a class |
| Say what is in the image | `classify` | One class for the whole image |
| Outline the exact area of an object | `segment` | A polygon per region, with a class |

Use the simplest task that answers the question. A detector is not automatically better than a classifier, and
segmentation is unnecessary when a box is enough.

Define classes with the user before bulk annotation:

- Use one stable spelling per class, with no near-duplicates (“bolt” and “bolts” are two classes).
- Choose classes a person can identify consistently from the image alone.
- Keep the first set small enough to label well. Add a class only when the user has examples of it.
- Do not add an “other” or “unknown” class unless the user can actually label those examples.
- Do not use a per-image class folder to pre-label a shared library.

If the goal or class boundaries are ambiguous, ask one focused question. Do not guess at ground truth.

## 3. Build the image library and annotate

Import the user's images into the single project library. Duplicates are skipped.

For a useful first pass, aim for several dozen varied examples per class, not just several dozen images total. More
variation in background, lighting, angle, scale and object appearance is more valuable than more near-duplicates.
There is no magic number: coverage of the real setting is the test. Never invent data to reach a count.

When annotating:

- **Detection:** draw a tight box around each object instance; include partially visible objects consistently.
- **Segmentation:** draw a polygon that follows the visible boundary; use more points for fine structures.
- **Classification:** assign one class to the image based on the subject the user cares about.
- Keep class counts reasonably balanced. If one class dominates, say so and ask for more examples rather than
  quietly discarding the user's data.
- Review the whole library before training. Inconsistent edges or missed instances teach the model the wrong thing.

The engine's annotation suggestions use COCO classes and are **proposals only**. A human must review every accepted
box or polygon. Never treat a model's guess as a label.

## 4. Train

Choose the model from `/health`. The usual mappings are:

| Task | Model |
| --- | --- |
| Detection | `vlyolo-s` |
| Classification | ResNet-18 |
| Segmentation | Compact U-Net |

Start with the defaults unless the user has a reason to change them:

```json
{ "model": "vlyolo-s", "epochs": 40, "batch": 8, "imgsz": 512, "lr": 0.001 }
```

Classification and segmentation ignore `model`; detection uses the name returned by `/health`.

Start one real job and poll it until its status is `completed`, `failed` or `cancelled`. Report progress at useful
intervals, not every event. Do not describe a job as finished while it is still running. If the user asks to stop,
use the cancel endpoint and wait for the engine to acknowledge it.

If the run fails because of memory, device or dependency problems, report the actual error. Do not replace it with a
fake result or a canned model.

## 5. Evaluate before claiming anything

Evaluate the saved checkpoint on the scoring holdout. The app chooses that holdout automatically; if you are
building a dataset manifest yourself, reserve a small, representative `val` split and keep it fixed.

Report the task's real metrics:

- **Detection:** mAP@0.5, mAP@0.5:0.95, per-class AP, precision and recall.
- **Classification:** accuracy, macro F1, per-class precision/recall/F1 and the confusion matrix.
- **Segmentation:** mIoU, per-class IoU and pixel accuracy.

Include the number of held-out images. Do not quote training accuracy as validation quality. Do not cherry-pick one
good run or one threshold without saying so.

If the results are weak, inspect the data before tuning:

- Are classes defined consistently?
- Are boxes or polygons tight, complete and present on every relevant image?
- Does the library cover the conditions where the model will be used?
- Is one class much rarer than another?

Improve the data and retrain. Never hide a poor result or write a performance claim the evaluation does not support.

## 6. Export and hand off

Export only after evaluation. The export contains the ONNX model, labels, a working `inference.js`, a model card and
the dataset, and the engine verifies the ONNX graph with ONNX Runtime. If verification fails, report it; do not hand
over a broken artifact.

A good handoff includes:

```text
Task:        detection
Classes:     bracket, gear
Images:      84 in the library; 12 held out for scoring
Model:       vlyolo-s, 40 epochs, 512 px
Evaluation:  mAP@0.5 0.87, mAP@0.5:0.95 0.62 on 12 held-out images
Artifact:    model.onnx + labels.json + inference.js + model card
Limits:      one workshop bench; glare and very small parts are underrepresented
```

Tell the user how to run the export, what input size it expects, and which conditions are not represented. Do not
call a small local model state-of-the-art.

## Driving the engine directly

Base URL: `http://127.0.0.1:8765`. IDs match `[A-Za-z0-9_-]{1,64}`. The engine accepts only loopback origins.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/health` | Versions, device, storage path, model specs and licences |
| GET | `/health/simple` | Liveness only |
| POST | `/projects/{id}/dataset` | Upload a dataset copy: `manifest` JSON plus `files` |
| GET | `/projects/{id}/dataset` | Dataset summary |
| GET | `/projects/{id}/dataset.zip` | Download the dataset in YOLO format |
| POST | `/projects/{id}/train` | Start a real job; returns `job_id` and `run_id` |
| GET | `/jobs/{job_id}` | Status, epoch events, metrics and errors |
| POST | `/jobs/{job_id}/cancel` | Request cancellation |
| GET | `/projects/{id}/runs` | Completed runs and summaries |
| POST | `/projects/{id}/runs/{run}/evaluate` | Recompute metrics from the checkpoint on validation images |
| POST | `/projects/{id}/runs/{run}/predict` | Run a checkpoint on multipart images; fields `conf`, `iou` |
| POST | `/projects/{id}/runs/{run}/export` | Build ONNX, labels, `inference.js`, model card and zip |
| GET | `/projects/{id}/runs/{run}/export/{name}` | Download an exported artifact |
| POST | `/autolabel` | Propose COCO boxes for images; fields `classes`, `model`, `conf` |
| DELETE | `/projects/{id}` | Remove that project's engine copy; never the user's originals |

Dataset manifest shape for detection:

```json
{
  "task": "detect",
  "classes": ["bracket", "gear"],
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

The browser is the normal source of truth for projects and originals. The engine holds only a training copy under
`~/.visionllm/projects/<project-id>/`. Deleting that copy is not permission to delete the user's source images.

## Stop and ask when

- The task or class boundaries are genuinely ambiguous.
- The user has too few varied examples for the class they want.
- The engine cannot start or the run fails on their hardware.
- The annotations look inconsistent enough that training would teach the wrong lesson.
- The user asks for a public or cloud deployment of their images.

Do not ask the user to choose a detail you can safely default. Explain the default and move on.

## Reference

- [docs/agents.md](docs/agents.md) — the short workflow and handoff checklist.
- [docs/setup.md](docs/setup.md) — local install, engine environment and ports.
- [docs/architecture.md](docs/architecture.md) — models, metrics, image handling and limits.

For maintaining VisionLLM itself, use the repository docs rather than this playbook.
