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
npm install
npm run engine   # terminal 1
npm run dev      # terminal 2
```

Open <http://127.0.0.1:5173>.

The first run needs the Python engine environment. The
[setup guide](docs/setup.md) has the one-time commands.

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
- [Agent guide](docs/agents.md) — a short map for coding agents and contributors.
- [Architecture](docs/architecture.md) — models, licensing, image guarantees and limits.
- [AGENTS.md](AGENTS.md) — the binding product and engineering contract.

The standalone showcase lives on the
[`website` branch](https://github.com/ItsMeh1/VisionLLM/tree/website/website).

## Licence

Original code in this repository is Apache-2.0. VisionLLM does not use Ultralytics
YOLO. See [Architecture](docs/architecture.md#licensing) for the third-party details.
