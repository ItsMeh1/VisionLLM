# Agent guide: making a vision model

This is the short version for an AI agent helping a person turn their own images into a working vision model with
VisionLLM. [AGENTS.md](../AGENTS.md) is the full playbook, including the local engine API. Use the repository docs
instead if the task is to maintain VisionLLM itself.

## The job

Choose the simplest task, define clean classes, help build a labelled library, train locally, read held-out metrics,
then export and explain the result. A run without an honest evaluation is not done.

## Before you start

- Confirm the local engine is listening at <http://127.0.0.1:8765/health>.
- Start it with `npm run engine` if needed; the app is `npm run dev`. First-time Python setup is in [Setup](setup.md).
- Work on the user's actual images. Everything stays on their machine; there is no account or cloud step.
- Do not change the VisionLLM source to get a model. This guide is for using the product.

## Choose the task

| Goal | Task | Annotation |
| --- | --- | --- |
| Find, locate or count separate objects | `detect` | A tight box per instance |
| Label the main subject of an image | `classify` | One class per image |
| Outline the exact area of an object | `segment` | A polygon per region |

Pick the least complex task that answers the question. Agree the class names and their boundaries with the user
before bulk annotation. Use one spelling per class and do not create classes the user cannot label consistently.

## Workflow

1. **Build one library.** Import the user's images into the project. Do not pre-label through folder names or use a
   per-class import path. A few dozen varied examples per class is a reasonable starting point, not a guarantee.
2. **Annotate carefully.** Boxes should be tight; polygons should follow the visible boundary. Include partial
   objects consistently. Review the whole set before training.
3. **Verify suggestions.** `POST /autolabel` and the UI's suggestions are proposals. A person checks every accepted
   annotation. Never present a prediction as ground truth.
4. **Train one real job.** Use the model names from `/health` and the default configuration unless there is a reason
   to change it. Wait for a terminal job status; do not report a running job as complete.
5. **Evaluate the checkpoint.** Use the held-out scoring images. Report the task's metrics with the number of
   images: detection mAP/precision/recall, classification accuracy/F1, or segmentation mIoU/pixel accuracy.
6. **Export and hand off.** Generate the ONNX bundle, confirm verification succeeded, and explain the input size,
   expected usage and data limits.

## Rules that matter

- Never modify original images. Annotations stay separate.
- Never invent labels, data, metrics or progress.
- Training loss is not validation quality. Use the holdout.
- If results are weak, improve the data and retrain. Do not hide or cherry-pick the result.
- Keep the user's files local. Do not upload them anywhere.
- Do not add train/validation controls to the app; the holdout is automatic. If you build a manifest directly,
  reserve a separate, representative `val` split and keep it fixed.

## Handoff checklist

Include the task, classes, image counts, held-out image count, model and configuration, exact metrics, artifact
names, and the main limitations. Tell the user what the model has not seen and what could make it fail.

## Go deeper

- [AGENTS.md](../AGENTS.md) — full model-making playbook and engine API.
- [Architecture](architecture.md) — models, metrics, image guarantees and limits.
- [Setup](setup.md) — local install and run commands.
