<p align="center">
  <img src="assets/visionllm-mark.svg" width="76" height="76" alt="VisionLLM logo" />
</p>

<h1 align="center">VisionLLM</h1>

<p align="center"><strong>Train a computer vision model on your own images, on your own machine.</strong></p>

<p align="center">No account. No cloud upload. No mock training.</p>

VisionLLM is a browser workbench backed by a small Python engine. Bring a folder of
images, draw boxes or masks, train a real model locally, and export an ONNX model
that runs in the browser.

It supports **object detection**, **image classification** and **image segmentation**.

## Quick start

**Requirements:** Node.js 20+ and Python 3.12.

```bash
npm start
```

This starts the local engine and browser app, creates missing dependencies if
needed, and opens <http://127.0.0.1:5173>. Leave it running while you train.
Existing services are left untouched, so it is safe to run while a training job
is already in progress. Press Ctrl-C to stop the services this command started.

Use the app from `main`. The [`website` branch](https://github.com/ItsMeh1/VisionLLM/tree/website)
is a static deployment branch; it does not contain the app or training environment.

## What you get

- **One image library.** Import images, camera frames or folders. Annotate them in
  place. There are no training/validation sets to organise.
- **Real training.** VLYOLO detection, ResNet-18 classification and a compact U-Net
  run through PyTorch on your machine.
- **Honest evaluation.** A small scoring holdout is chosen automatically. Metrics are
  recomputed from the checkpoint, not cached.
- **Browser-ready export.** Get `model.onnx`, labels, a working `inference.js`, a model
  card and the dataset after ONNX Runtime verification.
- **Local by design.** The engine binds to loopback. Originals are kept byte-for-byte;
  annotations are separate coordinates.

## Documentation

- [Setup](docs/setup.md) — install, run and configure the app and engine.
- [Agent guide](docs/agents.md) — how an AI agent helps you go from images to a trained model.
- [Architecture](docs/architecture.md) — models, licensing, image guarantees and limits.
- [AGENTS.md](AGENTS.md) — the model-making playbook for agents, including the local engine API.

The standalone showcase lives on the
[`website` branch](https://github.com/ItsMeh1/VisionLLM/tree/website), a static site prepared for GitHub Pages.
It is deployment-only; app and training work stays on `main`.

## Licence

Original code in this repository is Apache-2.0. VisionLLM does not use Ultralytics
YOLO. See [Architecture](docs/architecture.md#licensing) for the third-party details.
