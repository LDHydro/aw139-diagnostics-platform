# AW139 Diagnostics — Offline iPad App

A native **SwiftUI iPadOS** app that runs the AW139 Smart Troubleshooting
pipeline **fully on-device**, with no server and no internet required at query
time. It uses Apple's [**MLX**](https://github.com/ml-explore/mlx-swift-examples)
framework to run a quantized LLM and an embedding model directly on the iPad's
GPU. The document corpus and model files are loaded by **manual file transfer**
(Finder / Files app) — that's how you push content updates, no rebuild needed.

> This is the offline reimplementation of the web platform's diagnostic flow
> (`rag_api.py`). It reproduces the same query classification, the five
> senior-specialist system prompts, DMC citation, and cosine-similarity
> retrieval — but the cloud GPT‑4 + OpenAI embeddings are replaced with
> on-device MLX models.

---

## ⚠️ Honest expectations

- **Requires an Apple-silicon iPad** (M1/M2/M3/M4 or A17 Pro+). MLX needs a
  modern GPU and several GB of RAM. It will **not** run on the iOS Simulator.
- An on-device 3B-class model is **not** GPT‑4. Answers are good for a
  reference assistant but **must always be verified against the approved
  AMM/IETP**. This app does not replace approved maintenance data.
- You will finish the build on a **Mac with Xcode 16+** (signing, provisioning,
  deploying to the iPad). That part can't be done from Linux/CI.

---

## What's in here

```
ios/
├── AW139Diagnostics.xcodeproj/        # Open this in Xcode
├── AW139Diagnostics/
│   ├── AW139DiagnosticsApp.swift      # App entry
│   ├── Info.plist                     # iPad-only, file sharing enabled
│   ├── Assets.xcassets/
│   ├── Domain/                        # ATA codes, models, DMC + prompt logic
│   │   ├── ATACodes.swift
│   │   ├── DiagnosticModels.swift
│   │   ├── DMC.swift
│   │   └── PromptBuilder.swift        # faithful port of rag_api.py prompts
│   ├── Services/
│   │   ├── AppPaths.swift             # where transferred files live
│   │   ├── CorpusStore.swift          # loads corpus.json
│   │   ├── VectorSearch.swift         # cosine top-k (Accelerate)
│   │   ├── Inference.swift            # protocols + stub fallback
│   │   ├── MLXGenerator.swift         # on-device LLM
│   │   ├── MLXEmbedder.swift          # on-device embeddings
│   │   ├── ModelManager.swift         # load state
│   │   └── DiagnosticEngine.swift     # the pipeline
│   ├── ViewModels/DiagnosticViewModel.swift
│   └── Views/                         # Form, Result, Setup, ContentView
├── tools/build_offline_corpus.py      # re-embed docs for offline use
├── project.yml                        # optional: regenerate project with XcodeGen
└── README.md
```

---

## Setup

### 1. Open the project

```bash
open ios/AW139Diagnostics.xcodeproj
```

On first open, Xcode resolves the Swift Package dependency
(`mlx-swift-examples`, which pulls in `MLXLLM`, `MLXLMCommon`, `MLXEmbedders`).
Let it finish (File ▸ Packages ▸ Resolve Package Versions if needed).

> If the committed project ever fails to open, regenerate it with
> [XcodeGen](https://github.com/yonaskolb/XcodeGen):
> `brew install xcodegen && cd ios && xcodegen generate`.

Set your **Signing Team** under the target ▸ Signing & Capabilities, then build
to a connected Apple-silicon iPad (not the Simulator).

### 2. Build the offline corpus (once, on your Mac/PC)

The on-device app needs documents embedded with the **same model** it uses at
query time (default `BAAI/bge-small-en-v1.5`, 384-dim). Re-embed the existing
server corpus:

```bash
pip install sentence-transformers
cd ios/tools
python build_offline_corpus.py \
    --input ../../embeddings.json \
    --output corpus.json \
    --model BAAI/bge-small-en-v1.5
```

This produces `corpus.json` (text + 384-dim normalized embeddings).

### 3. Get the MLX model files

Download MLX-format (quantized) models on your Mac, e.g. with the `huggingface-cli`:

```bash
# LLM (≈1.8 GB, 4-bit)
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir llm

# Embedding model
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir embedder
```

Each model folder must contain its `config.json`, weight files
(`*.safetensors`), and tokenizer files.

> Alternatively, leave the model folders empty and the app will try a **one-time**
> Hugging Face download on first run (requires internet that once). For strictly
> offline operation, transfer the files as below and set
> `llmFallbackHubID = nil` in `ModelManager.swift`.

### 4. Transfer files onto the iPad

1. Connect the iPad to a Mac. In **Finder**, select the iPad → **Files** tab →
   **AW139 Diagnostics**.
2. Drag in:
   - `corpus.json`
   - a `models` folder containing `llm/` and `embedder/` subfolders
3. In the app, open **Setup (⚙)** and tap the **Load** buttons.

Resulting on-device layout:

```
<App>/Documents/
├── corpus.json
└── models/
    ├── llm/         (config.json, *.safetensors, tokenizer.json, …)
    └── embedder/    (config.json, *.safetensors, tokenizer.json, …)
```

**To update content later**, just replace these files — no rebuild required.
That's the "manual file transfer" update path.

---

## How it works (parity with the server)

| Step | Server (`rag_api.py`) | This app |
|------|------------------------|----------|
| Query embedding | OpenAI `text-embedding-3-small` | MLX `bge-small-en-v1.5` |
| Retrieval | cosine similarity, top‑5 | `VectorSearch` (Accelerate), top‑5 |
| Query classification | fault / calibration / procedure / electrical / general | `PromptBuilder.classify` (identical rules) |
| System prompts | 5 senior-specialist prompts | `PromptBuilder` (verbatim) |
| DMC citation | regex on doc path | `DMC.swift` (same regex) |
| Generation | GPT‑4‑turbo, temp 0 | MLX LLM, temp 0 |

---

## Notes for maintainers

- The two MLX call sites (`MLXGenerator.swift`, `MLXEmbedder.swift`) target the
  `mlx-swift-examples` `main` API. MLX Swift evolves; if compilation fails after
  package resolution, those are the only spots likely to need a minor signature
  tweak. The app still builds and runs with the `StubGenerator` fallback while
  you wire up models.
- Embedding model parity is critical: corpus and query embeddings must come from
  the same model, or retrieval breaks. The app guards against dimension
  mismatch and surfaces a clear error.
- This app reproduces the **diagnostic** flow only. Fleet management, history,
  auth, expert booking, and inventory were server/Postgres features and are not
  part of the offline app.
