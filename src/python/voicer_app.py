#!/usr/bin/env python3
"""
Voicer — desktop app: raw text → translation (3 steps) → voice cloning by sentences.
Десктопное приложение: сырой текст → перевод (3 шага) → клонирование голоса по предложениям.
Language, references, stress dictionary, output folder, model, logs; device/attention/language options.
"""
from __future__ import annotations

import json
import os
import platform
import queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import customtkinter as ctk

def _get_script_dir() -> Path:
    """Scripts directory; when built as .app (PyInstaller) use _MEIPASS. Каталог скриптов: при сборке в .app — из _MEIPASS."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
    return Path(__file__).resolve().parent


SCRIPT_DIR = _get_script_dir()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Импорты после добавления пути
from stress_utils import apply_stress_overrides

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_REF_AUDIO = SCRIPT_DIR / "reference_audio.wav"
DEFAULT_REF_TEXT = SCRIPT_DIR / "reference_voice.txt"
DEFAULT_STRESS_FILE = SCRIPT_DIR / "stress_overrides.txt"
PROMPTS_DIR = SCRIPT_DIR / "translation_prompts"
SETTINGS_FILE = SCRIPT_DIR / "voicer_settings.json"

# Высота полей по режиму (компактно / нормально / большой)
FIELD_HEIGHTS = {
    "compact": {"input": 90, "step1": 50, "step2": 50, "step3": 55, "prompt": 50, "stress": 45, "log": 140},
    "normal": {"input": 130, "step1": 70, "step2": 70, "step3": 85, "prompt": 100, "stress": 65, "log": 200},
    "large": {"input": 180, "step1": 100, "step2": 100, "step3": 120, "prompt": 150, "stress": 90, "log": 280},
}


class LogQueue:
    """Queue for log lines from background threads to the GUI. Очередь логов из фоновых потоков для вывода в GUI."""
    def __init__(self, text_widget: ctk.CTkTextbox):
        self._queue: queue.Queue[str] = queue.Queue()
        self._widget = text_widget

    def write(self, msg: str) -> None:
        self._queue.put(msg)

    def flush(self) -> None:
        pass

    def poll(self) -> None:
        while True:
            try:
                line = self._queue.get_nowait()
                self._widget.insert("end", line)
                self._widget.see("end")
            except queue.Empty:
                break


class VoicerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Voicer — перевод и клонирование голоса")
        self.geometry("1000x750")
        self.minsize(800, 600)

        self.raw_text = ""
        self.translate_result: dict[str, str] | None = None  # step1, step2, step3, final
        self.output_dir = Path(tempfile.gettempdir()) / "voicer_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ref_audio_path = DEFAULT_REF_AUDIO
        self.ref_text_path = DEFAULT_REF_TEXT
        self.stress_path = DEFAULT_STRESS_FILE
        self.log_queue: LogQueue | None = None

        self._build_ui()
        self._load_settings()
        self._poll_logs()

    def _build_ui(self):
        # Две колонки: слева — основная область, справа — логи
        content_pane = ctk.CTkFrame(self, fg_color="transparent")
        content_pane.pack(fill="both", expand=True, padx=10, pady=10)

        # Слева: вкладки «Основное» и «Промпты для перевода»
        left_pane = ctk.CTkFrame(content_pane, fg_color="transparent")
        left_pane.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.tabview = ctk.CTkTabview(left_pane, width=600)
        self.tabview.pack(fill="both", expand=True)
        self.tabview.add("Основное")
        self.tabview.add("Промпты для перевода")
        self.tabview.add("Словарь ударений")
        self.scroll = ctk.CTkScrollableFrame(self.tabview.tab("Основное"), width=580)
        self.scroll.pack(fill="both", expand=True)

        log_pane = ctk.CTkFrame(content_pane, fg_color="transparent", width=340)
        log_pane.pack(side="right", fill="y", padx=(8, 0))
        log_pane.pack_propagate(False)

        # --- Настройки (сверху) ---
        f_top = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f_top.pack(fill="x", padx=5, pady=(0, 10))
        ctk.CTkLabel(f_top, text="Настройки:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        ctk.CTkButton(f_top, text="Сохранить настройки", width=140, command=self._save_settings).pack(side="left", padx=2)
        ctk.CTkButton(f_top, text="Загрузить настройки", width=140, command=self._load_settings).pack(side="left", padx=2)
        ctk.CTkButton(f_top, text="Сбросить всё", width=100, command=self._reset_all, fg_color="gray").pack(side="left", padx=(15, 2))
        ctk.CTkButton(f_top, text="Проверить окружение", width=160, command=self._check_environment).pack(side="left", padx=2)
        ctk.CTkLabel(f_top, text="  Размер полей:").pack(side="left", padx=(15, 5))
        self.field_scale_var = ctk.StringVar(value="Нормально")
        scale_combo = ctk.CTkComboBox(
            f_top, values=["Компактно", "Нормально", "Большой"],
            variable=self.field_scale_var, width=110,
            command=self._on_field_scale_change,
        )
        scale_combo.pack(side="left", padx=2)
        self._field_scale_map = {"Компактно": "compact", "Нормально": "normal", "Большой": "large"}

        # --- Режим: с переводом или только озвучка ---
        f_mode = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f_mode.pack(fill="x", padx=5, pady=(5, 2))
        ctk.CTkLabel(f_mode, text="Режим:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=(0, 10))
        self.mode_var = ctk.StringVar(value="with_translate")
        ctk.CTkRadioButton(f_mode, text="С переводом (EN→RU и т.д.)", variable=self.mode_var, value="with_translate", command=self._on_mode_change).pack(side="left", padx=(0, 15))
        ctk.CTkRadioButton(f_mode, text="Только озвучка (текст уже готов)", variable=self.mode_var, value="voice_only", command=self._on_mode_change).pack(side="left", padx=0)

        # --- Ввод текста ---
        self.label_input = ctk.CTkLabel(self.scroll, text="Исходный текст (сырой транскрипт):", font=ctk.CTkFont(weight="bold"))
        self.label_input.pack(anchor="w", padx=5, pady=(10, 2))
        h = FIELD_HEIGHTS["normal"]
        self.text_input = ctk.CTkTextbox(self.scroll, height=h["input"], font=ctk.CTkFont(size=13))
        self.text_input.pack(fill="x", padx=5, pady=5)
        f_inp = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f_inp.pack(fill="x", padx=5, pady=5)
        self.lang_label = ctk.CTkLabel(f_inp, text="Язык:")
        self.lang_label.pack(side="left", padx=(0, 5))
        self.lang_var = ctk.StringVar(value="en")
        self.lang_combo = ctk.CTkComboBox(f_inp, values=["en", "de", "fr"], variable=self.lang_var, width=80)
        self.lang_combo.pack(side="left", padx=5)
        ctk.CTkLabel(f_inp, text="Модель перевода:").pack(side="left", padx=(15, 5))
        self.translate_model_var = ctk.StringVar(value="translategemma:27b")
        ctk.CTkEntry(f_inp, textvariable=self.translate_model_var, width=180).pack(side="left", padx=5)
        self.btn_translate = ctk.CTkButton(f_inp, text="Выполнить перевод", command=self._on_translate, width=180)
        self.btn_translate.pack(side="left", padx=5)
        self.btn_use_voice = ctk.CTkButton(f_inp, text="Использовать для озвучки", command=self._on_use_for_voice, width=180)
        # btn_use_voice показывается только в режиме «Только озвучка» (_on_mode_change)
        self.translate_status = ctk.CTkLabel(f_inp, text="", text_color="gray")
        self.translate_status.pack(side="left", padx=10)
        self._on_mode_change()  # показать нужную кнопку по режиму

        # --- Результаты перевода ---
        self.frame_steps = ctk.CTkFrame(self.scroll, fg_color="transparent")
        self.frame_steps.pack(fill="x", padx=5, pady=(15, 2))
        ctk.CTkLabel(self.frame_steps, text="Результаты перевода:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(0, 2))
        self.step1_text = ctk.CTkTextbox(self.frame_steps, height=h["step1"], font=ctk.CTkFont(size=11))
        self.step1_text.pack(fill="x", padx=5, pady=(0, 2))
        self.step2_text = ctk.CTkTextbox(self.frame_steps, height=h["step2"], font=ctk.CTkFont(size=11))
        self.step2_text.pack(fill="x", padx=5, pady=(0, 2))
        self.step3_text = ctk.CTkTextbox(self.frame_steps, height=h["step3"], font=ctk.CTkFont(size=11))
        self.step3_text.pack(fill="x", padx=5, pady=(0, 8))
        for w in (self.step1_text, self.step2_text, self.step3_text):
            self._make_readonly_copyable(w)

        # --- Клонирование ---
        ctk.CTkLabel(self.scroll, text="Клонирование голоса:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(10, 2))
        f1 = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f1.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f1, text="Выходная папка:").pack(side="left", padx=(0, 5))
        self.out_dir_var = ctk.StringVar(value=str(self.output_dir))
        ctk.CTkEntry(f1, textvariable=self.out_dir_var, width=320).pack(side="left", padx=5)
        ctk.CTkButton(f1, text="Обзор...", width=70, command=self._browse_output).pack(side="left", padx=(0, 5))
        ctk.CTkButton(f1, text="Открыть папку", width=100, command=self._open_output_folder).pack(side="left")

        f2 = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f2.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f2, text="Батч:").pack(side="left", padx=(0, 5))
        self.batch_var = ctk.StringVar(value="5")
        ctk.CTkEntry(f2, textvariable=self.batch_var, width=50).pack(side="left", padx=5)
        ctk.CTkLabel(f2, text="max_new_tokens:").pack(side="left", padx=(10, 5))
        self.max_tokens_var = ctk.StringVar(value="1024")
        ctk.CTkEntry(f2, textvariable=self.max_tokens_var, width=70).pack(side="left", padx=5)
        ctk.CTkLabel(f2, text="Модель TTS:").pack(side="left", padx=(15, 5))
        self.model_var = ctk.StringVar(value=DEFAULT_MODEL)
        ctk.CTkEntry(f2, textvariable=self.model_var, width=280).pack(side="left", padx=5)

        f2b = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f2b.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f2b, text="Устройство:").pack(side="left", padx=(0, 5))
        self.device_var = ctk.StringVar(value="mps:0")
        ctk.CTkComboBox(f2b, values=["mps:0", "cuda:0", "cpu"], variable=self.device_var, width=90).pack(side="left", padx=5)
        ctk.CTkLabel(f2b, text="Attention:").pack(side="left", padx=(10, 5))
        self.attn_var = ctk.StringVar(value="sdpa")
        ctk.CTkComboBox(f2b, values=["sdpa", "eager"], variable=self.attn_var, width=80).pack(side="left", padx=5)
        ctk.CTkLabel(f2b, text="Язык референса:").pack(side="left", padx=(10, 5))
        self.ref_language_var = ctk.StringVar(value="Russian")
        ctk.CTkEntry(f2b, textvariable=self.ref_language_var, width=120).pack(side="left", padx=5)

        f_ref_audio = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f_ref_audio.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f_ref_audio, text="Референс аудио:").pack(anchor="w", padx=(0, 5))
        self.ref_audio_var = ctk.StringVar(value=str(DEFAULT_REF_AUDIO))
        self.entry_ref_audio = ctk.CTkEntry(f_ref_audio, textvariable=self.ref_audio_var)
        self.entry_ref_audio.pack(fill="x", padx=(0, 5), pady=(2, 5))
        ctk.CTkButton(f_ref_audio, text="Файл...", width=80, command=self._browse_ref_audio).pack(anchor="w", padx=0, pady=(0, 5))
        f_ref_text = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f_ref_text.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f_ref_text, text="Текст референса:").pack(anchor="w", padx=(0, 5))
        self.ref_text_var = ctk.StringVar(value=str(DEFAULT_REF_TEXT))
        self.entry_ref_text = ctk.CTkEntry(f_ref_text, textvariable=self.ref_text_var)
        self.entry_ref_text.pack(fill="x", padx=(0, 5), pady=(2, 5))
        ctk.CTkButton(f_ref_text, text="Файл...", width=80, command=self._browse_ref_text).pack(anchor="w", padx=0, pady=(0, 5))

        # Словарь ударений — во вкладке «Словарь ударений»

        f_btn = ctk.CTkFrame(self.scroll, fg_color="transparent")
        f_btn.pack(fill="x", padx=5, pady=5)
        self.btn_clone = ctk.CTkButton(f_btn, text="Клонировать голос", command=self._on_clone, width=200)
        self.btn_clone.pack(side="left", padx=(0, 10))
        self.clone_status = ctk.CTkLabel(f_btn, text="", text_color="gray")
        self.clone_status.pack(side="left")

        # --- Вкладка «Промпты для перевода» ---
        prom_tab = self.tabview.tab("Промпты для перевода")
        ctk.CTkLabel(prom_tab, text="Шаг 1 — Очистка транскрипта:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(10, 2))
        self.prompt_clean_text = ctk.CTkTextbox(prom_tab, height=h["prompt"], font=ctk.CTkFont(size=11))
        self.prompt_clean_text.pack(fill="x", padx=5, pady=(0, 8))
        ctk.CTkLabel(prom_tab, text="Шаг 2 — Перевод EN→RU:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(5, 2))
        self.prompt_translate_text = ctk.CTkTextbox(prom_tab, height=h["prompt"], font=ctk.CTkFont(size=11))
        self.prompt_translate_text.pack(fill="x", padx=5, pady=(0, 8))
        ctk.CTkLabel(prom_tab, text="Шаг 3 — Финальная правка под нарацию:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(5, 2))
        self.prompt_final_text = ctk.CTkTextbox(prom_tab, height=h["prompt"], font=ctk.CTkFont(size=11))
        self.prompt_final_text.pack(fill="both", expand=True, padx=5, pady=(0, 10))
        self._load_prompts_from_files()

        # --- Вкладка «Словарь ударений» ---
        stress_tab = self.tabview.tab("Словарь ударений")
        ctk.CTkLabel(stress_tab, text="Словарь ударений (одно слово в строке, ударная буква — заглавная, напр. судОку):", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(10, 2))
        self.stress_text = ctk.CTkTextbox(stress_tab, height=h["stress"], font=ctk.CTkFont(size=11))
        self.stress_text.pack(fill="both", expand=True, padx=5, pady=(0, 10))
        if DEFAULT_STRESS_FILE.exists():
            self.stress_text.insert("1.0", DEFAULT_STRESS_FILE.read_text(encoding="utf-8"))

        # --- Логи (правая колонка) ---
        ctk.CTkLabel(log_pane, text="Логи", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=0, pady=(0, 4))
        self.log_text = ctk.CTkTextbox(log_pane, height=h["log"], font=ctk.CTkFont(family="Monaco", size=11))
        self.log_text.pack(fill="both", expand=True, padx=0, pady=0)
        self._make_readonly_copyable(self.log_text)
        self.log_queue = LogQueue(self.log_text)

    def _log(self, msg: str) -> None:
        if self.log_queue:
            self.log_queue.write(msg + "\n")

    def _poll_logs(self) -> None:
        if self.log_queue:
            self.log_queue.poll()
        self.after(200, self._poll_logs)

    def _browse_output(self) -> None:
        path = ctk.filedialog.askdirectory(title="Папка для чанков")
        if path:
            self.out_dir_var.set(path)

    def _browse_ref_audio(self) -> None:
        path = ctk.filedialog.askopenfilename(title="Референсное аудио", filetypes=[("WAV", "*.wav"), ("Все", "*")])
        if path:
            self.ref_audio_var.set(path)

    def _browse_ref_text(self) -> None:
        path = ctk.filedialog.askopenfilename(title="Текст референса", filetypes=[("Текст", "*.txt"), ("Все", "*")])
        if path:
            self.ref_text_var.set(path)

    def _on_mode_change(self) -> None:
        """Show/hide buttons by mode (with translation / voice only). Показать/скрыть кнопки по режиму."""
        is_voice_only = self.mode_var.get() == "voice_only"
        if is_voice_only:
            self.label_input.configure(text="Текст для озвучки (уже переведённый или на целевом языке):")
            self.btn_translate.pack_forget()
            self.btn_use_voice.pack(side="left", padx=5, before=self.translate_status)
            self.lang_label.pack_forget()
            self.lang_combo.pack_forget()
        else:
            self.label_input.configure(text="Исходный текст (сырой транскрипт):")
            self.btn_use_voice.pack_forget()
            self.lang_label.pack(side="left", padx=(0, 5), before=self.lang_combo)
            self.lang_combo.pack(side="left", padx=5, before=self.btn_translate)
            self.btn_translate.pack(side="left", padx=5, before=self.translate_status)

    def _on_use_for_voice(self) -> None:
        """Voice-only mode: take text from field as ready for cloning. Режим «только озвучка»: текст из поля для клонирования."""
        raw = self.text_input.get("1.0", "end").strip()
        if not raw:
            self._log("Введите текст для озвучки.")
            return
        self.translate_result = {"step1": "", "step2": "", "step3": "", "final": raw}
        self.translate_status.configure(text="Текст принят. Можно нажимать «Клонировать голос».", text_color="gray")
        self._log("Текст для озвучки принят. Переходите к клонированию.")

    def _on_field_scale_change(self, choice: str) -> None:
        """Изменить высоту всех текстовых полей (Компактно / Нормально / Большой)."""
        scale = self._field_scale_map.get(choice, "normal")
        self._apply_field_scale(scale)

    def _apply_field_scale(self, scale: str) -> None:
        h = FIELD_HEIGHTS.get(scale, FIELD_HEIGHTS["normal"])
        self.text_input.configure(height=h["input"])
        self.step1_text.configure(height=h["step1"])
        self.step2_text.configure(height=h["step2"])
        self.step3_text.configure(height=h["step3"])
        self.prompt_clean_text.configure(height=h["prompt"])
        self.prompt_translate_text.configure(height=h["prompt"])
        self.prompt_final_text.configure(height=h["prompt"])
        if hasattr(self, "stress_text"):
            self.stress_text.configure(height=h["stress"])
        self.log_text.configure(height=h["log"])

    def _make_readonly_copyable(self, text_widget: ctk.CTkTextbox) -> None:
        """Только чтение + копирование: контекстное меню «Копировать» и Ctrl+C в буфер."""
        def copy_selection() -> None:
            try:
                tw = getattr(text_widget, "_textbox", None)
                if tw is None:
                    return
                try:
                    sel = tw.get("sel.first", "sel.last")
                except Exception:
                    return
                if sel:
                    self.clipboard_clear()
                    self.clipboard_append(sel)
            except Exception:
                pass

        def on_key(event):
            try:
                mod = getattr(event, "state", 0) or 0
                key = getattr(event, "keysym", "") or ""
                if (mod & 0x4) and key.lower() == "c":  # Ctrl+C — копируем в буфер
                    copy_selection()
                    return "break"
                if (mod & 0x4) and key.lower() == "a":
                    return
            except Exception:
                pass
            return "break"

        def on_right_click(event):
            menu = ctk.CTkToplevel(self)
            menu.wm_overrideredirect(True)
            menu.wm_geometry(f"+{event.x_root + 5}+{event.y_root + 5}")
            btn = ctk.CTkButton(menu, text="Копировать", width=120, command=lambda: (copy_selection(), menu.destroy()))
            btn.pack(padx=2, pady=2)
            def close_menu(_e):
                menu.destroy()
            menu.bind("<FocusOut>", close_menu)
            menu.after(200, btn.focus_set)

        text_widget.bind("<Key>", on_key)
        text_widget.bind("<Button-2>", on_right_click)
        text_widget.bind("<Button-3>", on_right_click)
        # Привязка к внутреннему tk-виджету для надёжного Ctrl+C
        tk_w = getattr(text_widget, "_textbox", None)
        if tk_w is not None:
            tk_w.bind("<Control-c>", lambda e: (copy_selection(), "break"))

    def _open_output_folder(self) -> None:
        """Открыть выходную папку в проводнике системы."""
        path = Path(self.out_dir_var.get()).resolve()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        try:
            sys_name = platform.system()
            if sys_name == "Darwin":
                subprocess.run(["open", str(path)], check=False)
            elif sys_name == "Windows":
                os.startfile(str(path))
            else:
                subprocess.run(["xdg-open", str(path)], check=False)
        except Exception as e:
            self._log(f"Не удалось открыть папку: {e}")

    def _check_environment(self) -> None:
        """Проверить: Python, Ollama, модель перевода, модель TTS. Результаты в лог."""
        self._log("Проверка окружения...")
        tts_model = self.model_var.get().strip() or DEFAULT_MODEL

        def do_check():
            try:
                from env_check import run_all_checks, format_checks_for_log
                checks = run_all_checks(tts_model_id=tts_model)
                lines = format_checks_for_log(checks)
                for line in lines:
                    self.after(0, lambda l=line: self._log(l))
            except Exception as e:
                self.after(0, lambda: self._log(f"Ошибка проверки: {e}"))

        threading.Thread(target=do_check, daemon=True).start()

    def _reset_all(self) -> None:
        """Сбросить текст, результаты перевода, статусы и логи."""
        self.text_input.delete("1.0", "end")
        for w in (self.step1_text, self.step2_text, self.step3_text):
            w.delete("1.0", "end")
        self.translate_result = None
        self.translate_status.configure(text="", text_color="gray")
        self.clone_status.configure(text="", text_color="gray")
        self.log_text.delete("1.0", "end")
        while True:
            try:
                self.log_queue._queue.get_nowait()
            except queue.Empty:
                break
        self._load_prompts_from_files()
        self._log("Всё сброшено.")

    def _load_prompts_from_files(self) -> None:
        """Загрузить промпты из translation_prompts/ в поля вкладки Промпты."""
        for name, widget in [
            ("prompt_clean_sub.txt", self.prompt_clean_text),
            ("prompt_translate.txt", self.prompt_translate_text),
            ("prompt_final.txt", self.prompt_final_text),
        ]:
            path = PROMPTS_DIR / name
            widget.delete("1.0", "end")
            if path.exists():
                widget.insert("1.0", path.read_text(encoding="utf-8"))

    def _get_prompts_dict(self) -> dict[str, str]:
        return {
            "clean": self.prompt_clean_text.get("1.0", "end").strip(),
            "translate": self.prompt_translate_text.get("1.0", "end").strip(),
            "final": self.prompt_final_text.get("1.0", "end").strip(),
        }

    def _save_settings(self) -> None:
        data = {
            "out_dir": self.out_dir_var.get(),
            "batch_size": self.batch_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "ref_audio": self.ref_audio_var.get(),
            "ref_text": self.ref_text_var.get(),
            "model": self.model_var.get(),
            "device": self.device_var.get(),
            "attn": self.attn_var.get(),
            "ref_language": self.ref_language_var.get(),
            "translate_model": self.translate_model_var.get(),
            "lang": self.lang_var.get(),
            "mode": self.mode_var.get(),
            "field_scale": self.field_scale_var.get(),
            "prompt_clean": self.prompt_clean_text.get("1.0", "end"),
            "prompt_translate": self.prompt_translate_text.get("1.0", "end"),
            "prompt_final": self.prompt_final_text.get("1.0", "end"),
            "stress": self.stress_text.get("1.0", "end"),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            self._log("Настройки сохранены.")
        except Exception as e:
            self._log(f"Ошибка сохранения настроек: {e}")

    def _load_settings(self) -> None:
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            if data.get("out_dir"):
                self.out_dir_var.set(data["out_dir"])
            if data.get("batch_size"):
                self.batch_var.set(str(data["batch_size"]))
            if data.get("max_tokens"):
                self.max_tokens_var.set(str(data["max_tokens"]))
            if data.get("ref_audio"):
                self.ref_audio_var.set(data["ref_audio"])
            if data.get("ref_text"):
                self.ref_text_var.set(data["ref_text"])
            if data.get("model"):
                self.model_var.set(data["model"])
            if data.get("device"):
                self.device_var.set(data["device"])
            if data.get("attn"):
                self.attn_var.set(data["attn"])
            if data.get("ref_language"):
                self.ref_language_var.set(data["ref_language"])
            if data.get("translate_model"):
                self.translate_model_var.set(data["translate_model"])
            if data.get("lang"):
                self.lang_var.set(data["lang"])
            if data.get("prompt_clean") is not None:
                self.prompt_clean_text.delete("1.0", "end")
                self.prompt_clean_text.insert("1.0", data["prompt_clean"])
            if data.get("prompt_translate") is not None:
                self.prompt_translate_text.delete("1.0", "end")
                self.prompt_translate_text.insert("1.0", data["prompt_translate"])
            if data.get("prompt_final") is not None:
                self.prompt_final_text.delete("1.0", "end")
                self.prompt_final_text.insert("1.0", data["prompt_final"])
            if data.get("stress") is not None:
                self.stress_text.delete("1.0", "end")
                self.stress_text.insert("1.0", data["stress"])
            if data.get("mode") in ("with_translate", "voice_only"):
                self.mode_var.set(data["mode"])
                self._on_mode_change()
            if data.get("field_scale") in ("Компактно", "Нормально", "Большой"):
                self.field_scale_var.set(data["field_scale"])
                self._apply_field_scale(self._field_scale_map.get(data["field_scale"], "normal"))
        except Exception as e:
            if self.log_queue:
                self._log(f"Ошибка загрузки настроек: {e}")

    @staticmethod
    def _count_sentences(text: str) -> int:
        """Приблизительный подсчёт предложений (как в clone_chunks)."""
        text = text.strip()
        if not text:
            return 0
        parts = re.split(r"(?<=[.!?])\s+", text)
        return len([p for p in parts if p.strip()])

    def _on_translate(self) -> None:
        raw = self.text_input.get("1.0", "end").strip()
        if not raw:
            self._log("Ошибка: введите исходный текст.")
            return
        self.btn_translate.configure(state="disabled")
        self.translate_status.configure(text="Идёт перевод… (ожидайте, не нажимайте кнопку)", text_color="orange")
        self._log("Запуск перевода (Ollama translategemma)...")

        prompts = self._get_prompts_dict()
        def do_translate():
            def cb(msg: str) -> None:
                self.after(0, lambda m=msg: self._log(m))
                # Обновляем статус по шагам, чтобы было видно прогресс
                if "Step 1" in msg or "Шаг 1" in msg:
                    self.after(0, lambda: self.translate_status.configure(text="Шаг 1/3 — очистка…", text_color="orange"))
                elif "Step 2" in msg or "Шаг 2" in msg:
                    self.after(0, lambda: self.translate_status.configure(text="Шаг 2/3 — перевод…", text_color="orange"))
                elif "Step 3" in msg or "Шаг 3" in msg:
                    self.after(0, lambda: self.translate_status.configure(text="Шаг 3/3 — финальная правка…", text_color="orange"))
            try:
                from translate_with_gemma import run_translate
                model = self.translate_model_var.get().strip() or "translategemma:27b"
                result = run_translate(raw, log_callback=cb, prompts=prompts, model=model)
                self.after(0, lambda: self._apply_translate_result(result))
            except Exception as e:
                self.after(0, lambda: self._log(f"Ошибка перевода: {e}"))
                self.after(0, lambda: self.translate_status.configure(text="Ошибка", text_color="red"))
            finally:
                self.after(0, lambda: self.btn_translate.configure(state="normal"))

        threading.Thread(target=do_translate, daemon=True).start()

    def _apply_translate_result(self, result: dict[str, str]) -> None:
        self.translate_result = result
        for name, widget in [("step1", self.step1_text), ("step2", self.step2_text), ("step3", self.step3_text)]:
            widget.delete("1.0", "end")
            widget.insert("1.0", result.get(name, ""))
        self.translate_status.configure(text="Готово", text_color="gray")
        self._log("Перевод завершён.")

    def _on_clone(self) -> None:
        # Text to clone: translation result or, if none, input field (voice-only or raw). Текст для клонирования: перевод или поле ввода.
        if self.translate_result:
            final_text = self.translate_result.get("final", "").strip()
        else:
            final_text = self.text_input.get("1.0", "end").strip()
        if not final_text:
            self._log("Введите текст для озвучки или сначала выполните перевод.")
            return

        out_dir = Path(self.out_dir_var.get())
        ref_audio = Path(self.ref_audio_var.get())
        ref_text = Path(self.ref_text_var.get())
        batch_size_s = self.batch_var.get().strip() or "5"
        batch_size = int(batch_size_s)
        max_tokens = self.max_tokens_var.get().strip() or "1024"
        model = self.model_var.get().strip() or DEFAULT_MODEL
        device = self.device_var.get().strip() or "mps:0"
        attn_impl = self.attn_var.get().strip() or "sdpa"
        ref_language = self.ref_language_var.get().strip() or "Russian"
        stress_content = self.stress_text.get("1.0", "end")
        stress_file = SCRIPT_DIR / "stress_overrides.txt"
        stress_file.write_text(stress_content, encoding="utf-8")

        out_dir.mkdir(parents=True, exist_ok=True)
        text_for_file = apply_stress_overrides(final_text, stress_file)
        tmp_input = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
        tmp_input.write(text_for_file)
        tmp_input.close()
        input_path = tmp_input.name

        self.btn_clone.configure(state="disabled", text="Клонирование...")
        self.clone_status.configure(text="Запуск...")

        env = os.environ.copy()
        frozen = getattr(sys, "frozen", False)
        clone_extra = [
            "--device", device,
            "--attn-implementation", attn_impl,
            "--language", ref_language,
        ]
        if frozen:
            python_dir = Path(getattr(sys, "_MEIPASS", os.path.dirname(sys.executable)))
            cmd = [
                sys.executable, "--run-clone-chunks", input_path, "-o", str(out_dir),
                "--batch-size", batch_size_s, "--max-new-tokens", max_tokens,
                "--ref-audio", str(ref_audio), "--ref-text", str(ref_text),
                "--model", model,
                *clone_extra,
            ]
        else:
            python_dir = SCRIPT_DIR.parent.parent
            if not (python_dir / "pyproject.toml").exists():
                python_dir = Path.cwd()
            clone_script = python_dir / "src/python/clone_chunks.py"
            if not clone_script.exists():
                clone_script = SCRIPT_DIR / "clone_chunks.py"
            cmd = [
                sys.executable, str(clone_script), input_path, "-o", str(out_dir),
                "--batch-size", batch_size_s, "--max-new-tokens", max_tokens,
                "--ref-audio", str(ref_audio), "--ref-text", str(ref_text),
                "--model", model,
                *clone_extra,
            ]

        def do_clone():
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(python_dir),
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                )
                for line in proc.stdout:
                    self.after(0, lambda l=line: self._log(l.rstrip()))
                proc.wait()
                self.after(0, lambda: self._clone_done(proc.returncode, out_dir))
            except Exception as e:
                self.after(0, lambda: self._log(f"Ошибка клонирования: {e}"))
                self.after(0, lambda: self._clone_done(1, out_dir))
            finally:
                try:
                    os.unlink(input_path)
                except Exception:
                    pass
                self.after(0, lambda: (
                    self.btn_clone.configure(state="normal", text="Клонировать голос"),
                    self.clone_status.configure(text=""),
                ))

        threading.Thread(target=do_clone, daemon=True).start()

    def _clone_done(self, returncode: int, out_dir: Path) -> None:
        if returncode != 0:
            self.clone_status.configure(text="Завершилось с ошибкой", text_color="orange")
            return
        self.clone_status.configure(text=f"Готово. Чанки в {out_dir}")
        self._log(f"Чанки сохранены в {out_dir}")


def main() -> None:
    app = VoicerApp()
    app.mainloop()


if __name__ == "__main__":
    # Режим сборки: тот же exe вызывается для клонирования (subprocess из GUI)
    if "--run-clone-chunks" in sys.argv:
        idx = sys.argv.index("--run-clone-chunks")
        sys.argv = ["clone_chunks"] + sys.argv[idx + 1 :]
        import clone_chunks  # noqa: E402
        clone_chunks.main()
        sys.exit(0)
    main()
