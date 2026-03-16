# Voicer — корень репозитория (папка python)
# Запуск: make translate | make clone | make app

PY_SRC     := src/python
POETRY_RUN := poetry run python

INPUT  ?= $(PY_SRC)/reference_text_to_translate.txt
OUTPUT ?= $(PY_SRC)/translated.txt
MLFLOW_PORT ?= 5001

CHUNKS_DIR ?= $(PY_SRC)/chunks
CLONE_BATCH_SIZE ?= 5
CLONE_MAX_NEW_TOKENS ?= 1024

.PHONY: translate clone clone-chunks mlflow help app build-app test lint format

help:
	@echo "Voicer Makefile (репозиторий = эта папка)"
	@echo ""
	@echo "  make translate     — трёхшаговый перевод EN→RU (Ollama translategemma:27b)"
	@echo "  make clone         — клонирование в один файл: $(PY_SRC)/output-translated-clone.wav"
	@echo "  make clone-chunks  — клонирование по предложениям в $(CHUNKS_DIR)/"
	@echo "  make mlflow        — MLflow UI на порту $(MLFLOW_PORT)"
	@echo "  make app           — запуск десктопного приложения"
	@echo "  make build-app     — сборка приложения (PyInstaller): dist/Voicer/ или Voicer.app"
	@echo "  make test          — юнит-тесты (pytest)"
	@echo "  make lint          — проверка форматирования (black --check)"
	@echo "  make format        — форматирование кода (black)"
	@echo ""

translate:
	$(POETRY_RUN) $(PY_SRC)/translate_with_gemma.py "$(INPUT)" -o "$(OUTPUT)"
	@echo "Перевод записан в $(OUTPUT)"

clone:
	poetry run python $(PY_SRC)/__init__.py
	@echo "Готово. Файл: $(PY_SRC)/output-translated-clone.wav"

clone-chunks:
	poetry run python $(PY_SRC)/clone_chunks.py '$(PY_SRC)/translated.txt' -o '$(CHUNKS_DIR)' --batch-size $(CLONE_BATCH_SIZE) --max-new-tokens $(CLONE_MAX_NEW_TOKENS)
	@echo "Готово. Чанки в $(CHUNKS_DIR)/"

mlflow:
	poetry run python -m mlflow server --port $(MLFLOW_PORT)

app:
	poetry run python $(PY_SRC)/voicer_app.py

build-app:
	poetry run python build_app.py

test:
	poetry run pytest tests/ -v --tb=short

lint:
	poetry run black --check src/python tests

format:
	poetry run black src/python tests
