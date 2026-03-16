# Voicer

Translate text (Ollama + translategemma) and clone voice with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS). Desktop app (CustomTkinter) and CLI.

**Author:** Marat Zimnurov (zimtir@mail.ru)  
**License:** [LICENSE](LICENSE) (EN), [LICENSE-RU.md](LICENSE-RU.md) (RU). Commercial use requires attribution. No warranty.

![demo](./examples/app-1.png)
![demo](./examples/app-2.png)
![demo](./examples/app-3.png)

- [Voice demo 1](./examples/output-translated-clone.wav) · [2](./examples/001.wav) · [3](./examples/002.wav)

---

## Requirements

- Python 3.12, Poetry
- Ollama with `translategemma:27b`
- PyTorch (MPS on Apple, CUDA on NVIDIA)
- Qwen3-TTS in repo root: `git submodule update --init` or clone into `Qwen3-TTS/`

## Install

```bash
git clone https://github.com/the-homeless-god/voicer.git && cd voicer
git submodule update --init --recursive
poetry install
```

## Run

| Action | Command |
|--------|---------|
| Desktop app | `make app` |
| Translate (3 steps) | `make translate` — input: `src/python/reference_text_to_translate.txt`, output: `src/python/translated.txt` |
| Clone to one WAV | `make clone` — uses `translated.txt`, writes `src/python/output-translated-clone.wav` |
| Clone to chunks | `make clone-chunks` — writes `src/python/chunks/*.wav` |

Custom paths:

```bash
INPUT=in.txt OUTPUT=out.txt make translate
poetry run python src/python/clone_chunks.py in.txt -o out/ --ref-audio ref.wav --ref-text ref.txt --device mps:0 --language Russian
```

Env for `make clone`: `VOICER_DEVICE`, `VOICER_ATTN`, `VOICER_LANGUAGE`.

Check env: `poetry run python src/python/env_check.py`

## MLflow

Terminal 1: `make mlflow` (port 5001).  
Terminal 2: `MLFLOW_TRACKING_URI=http://localhost:5001 make translate` or `make clone-chunks`.  
Open http://localhost:5001 → experiment `voicer` → Runs / Traces.

## Tests and lint

- `make test` — pytest
- `make lint` — black --check
- `make format` — black

CI runs these on push/PR ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## Build app

`poetry add --group dev pyinstaller && poetry install` once, then:

```bash
make build-app
```

Result: `dist/Voicer.app` (macOS) or `dist/Voicer/` (Windows/Linux). Release workflow builds for all three ([.github/workflows/release.yml](.github/workflows/release.yml)).

## License

Use allowed with attribution. Commercial use: credit "Voicer by Marat Zimnurov". Optional: support via [digitable.ru](https://digitable.ru). No warranty — see [LICENSE](LICENSE).
