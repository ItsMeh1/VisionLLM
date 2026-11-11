# Setup

VisionLLM has two local processes: a Vite app in the browser and a Python training
engine on loopback.

## Requirements

- Node.js 20+ and npm
- Python 3.12
- `uv` is optional; a standard `venv` works too

## 1. Install the app

From the repository root:

```bash
npm install
```

## 2. Create the engine environment

```bash
cd engine
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cd ..
```

Without `uv`:

```bash
cd engine
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd ..
```

`engine/.venv` is local and ignored by git.

## 3. Run it

Use two terminals from the repository root:

```bash
npm run engine   # FastAPI on http://127.0.0.1:8765
```

```bash
npm run dev      # Vite on http://127.0.0.1:5173
```

Open <http://127.0.0.1:5173>. Browsing and annotation work without the engine;
training and evaluation need it running.

## Checks

```bash
npx tsc --noEmit
npm run build
engine/.venv/bin/python -m engine.selftest
```

With the engine listening, the full HTTP path can be checked separately:

```bash
engine/.venv/bin/python -m engine.apitest
```

## Where data lives

- Projects, originals, annotations and runs: browser IndexedDB (`visionllm`).
- Training copy, checkpoints, metrics and exports: `~/.visionllm/projects/<id>/`.

Set `VISIONLLM_HOME` to use a different engine data directory. The engine listens on
port `8765` and only accepts loopback origins.
