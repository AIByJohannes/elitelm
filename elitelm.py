import argparse
from dataclasses import dataclass
from typing import Callable, Literal, Optional
import json
import os
import sys
import time
from pathlib import Path
import weakref

import numpy as np
import onnxruntime_genai as og
import yaml
from pydantic import BaseModel, Field

_DLL_HANDLES: list[object] = []
_REGISTERED_DLL_PATHS: set[str] = set()

_CONFIG_PROVIDER_OPTIONS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def _ensure_config_provider_helpers() -> None:
    if getattr(og.Config, "_elitelm_provider_patch", False):
        return

    original_clear = og.Config.clear_providers
    original_set = og.Config.set_provider_option

    def clear_providers(self) -> None:
        _CONFIG_PROVIDER_OPTIONS.pop(self, None)
        return original_clear(self)

    def set_provider_option(self, provider: str, key: str, value: str) -> None:
        provider_map = _CONFIG_PROVIDER_OPTIONS.setdefault(self, {})
        provider_map.setdefault(provider, {})[key] = value
        return original_set(self, provider, key, value)

    def get_provider_option(self, provider: str, key: str) -> str | None:
        return _CONFIG_PROVIDER_OPTIONS.get(self, {}).get(provider, {}).get(key)

    og.Config.clear_providers = clear_providers
    og.Config.set_provider_option = set_provider_option
    og.Config.get_provider_option = get_provider_option
    setattr(og.Config, "_elitelm_provider_patch", True)


_ensure_config_provider_helpers()


class _DummyConfig:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self._providers: list[str] = []
        self._options: dict[str, dict[str, str]] = {}

    def clear_providers(self) -> None:
        self._providers.clear()
        self._options.clear()

    def append_provider(self, provider: str) -> None:
        if provider not in self._providers:
            self._providers.append(provider)

    def set_provider_option(self, provider: str, key: str, value: str) -> None:
        self._options.setdefault(provider, {})[key] = value

    def get_provider_option(self, provider: str, key: str) -> str | None:
        return self._options.get(provider, {}).get(key)

    def __repr__(self) -> str:
        return f"_DummyConfig(model_dir={self.model_dir!s})"


def _add_dll_dir(path: Path) -> None:
    if not path.exists():
        return
    path = path.resolve()
    path_str = str(path)
    if path_str in _REGISTERED_DLL_PATHS:
        return
    _DLL_HANDLES.append(os.add_dll_directory(path_str))
    current_path = os.environ.get("PATH")
    os.environ["PATH"] = f"{path_str};{current_path}" if current_path else path_str
    _REGISTERED_DLL_PATHS.add(path_str)


def _resolve_qnn_sdk_root(sdk_arg: str | None) -> Path:
    if sdk_arg:
        candidate = Path(sdk_arg).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Provided QNN SDK path does not exist: {candidate}")
        return candidate

    env_sdk = os.environ.get("QNN_SDK_ROOT")
    if env_sdk:
        candidate = Path(env_sdk)
        if candidate.exists():
            return candidate

    repo_root = Path(__file__).resolve().parent
    qairt_root = repo_root / "qairt"
    if qairt_root.exists():
        versions = sorted((p for p in qairt_root.iterdir() if p.is_dir()), reverse=True)
        if versions:
            return versions[0].resolve()

    raise FileNotFoundError(
        "Unable to locate the QNN SDK. Provide --qnn-sdk or set QNN_SDK_ROOT."
    )


def _default_backend_path(sdk_root: Path) -> tuple[Path, str]:
    is_windows = sys.platform.startswith("win")
    lib_name = "QnnHtp.dll" if is_windows else "libQnnHtp.so"
    lib_root = sdk_root / "lib"
    if not lib_root.exists():
        raise FileNotFoundError(f"Unable to locate {lib_name} in the QNN SDK lib directory")

    priorities = ["aarch64", "arm64x", "x86_64"] if is_windows else ["aarch64", "x86_64"]
    best_candidate = None

    for arch_dir in sorted(p for p in lib_root.iterdir() if p.is_dir()):
        candidate = arch_dir / lib_name
        if not candidate.exists():
            continue
        arch_lower = arch_dir.name.lower()
        try:
            rank = next(idx for idx, token in enumerate(priorities) if token in arch_lower)
        except StopIteration:
            rank = len(priorities)
        if best_candidate is None or rank < best_candidate[0]:
            best_candidate = (rank, arch_dir.name, candidate.resolve())

    if best_candidate:
        _, arch_name, backend_path = best_candidate
        return backend_path, arch_name

    raise FileNotFoundError(f"Unable to locate {lib_name} in the QNN SDK lib directory")


def _configure_qnn_provider(model_dir: Path, sdk_arg: str | None, backend_arg: str | None) -> og.Config:
    if not og.is_qnn_available():
        raise RuntimeError("onnxruntime-genai was built without QNN support on this platform")

    _ensure_config_provider_helpers()

    sdk_root = _resolve_qnn_sdk_root(sdk_arg).resolve()
    config_path = model_dir / "genai_config.json"

    if config_path.exists():
        config = og.Config(str(model_dir))
    else:
        config = _DummyConfig(model_dir)

    config.clear_providers()
    config.append_provider("QNNExecutionProvider")

    if backend_arg:
        backend_path = Path(backend_arg).expanduser()
        if not backend_path.exists():
            raise FileNotFoundError(f"QNN backend DLL not found at: {backend_path}")
        backend_path = backend_path.resolve()
        arch = backend_path.parent.name
    else:
        backend_path, arch = _default_backend_path(sdk_root)

    if not backend_path.exists():
        raise FileNotFoundError(f"QNN backend DLL not found at: {backend_path}")

    backend_dir = backend_path.parent

    os.environ["QNN_SDK_ROOT"] = str(sdk_root)
    _add_dll_dir(backend_dir)
    _add_dll_dir(sdk_root / "lib" / arch)
    _add_dll_dir(sdk_root / "bin" / arch)

    config.set_provider_option("QNNExecutionProvider", "backend_path", str(backend_path))
    config.set_provider_option("QNNExecutionProvider", "qnn_sdk_root", str(sdk_root))
    return config


class QnnConfig(BaseModel):
    sdk_root: Optional[str] = None
    backend: Optional[str] = None


class GenerationConfig(BaseModel):
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    do_sample: bool = False


class RuntimeConfig(BaseModel):
    verbose: bool = False
    timings: bool = False


class AppConfig(BaseModel):
    model: str
    device: Literal['cpu', 'qnn'] = 'cpu'
    qnn: QnnConfig = Field(default_factory=QnnConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    config_path: Optional[str] = None

    @property
    def qnn_sdk(self) -> str | None:
        return self.qnn.sdk_root

    @property
    def qnn_backend(self) -> str | None:
        return self.qnn.backend

    @property
    def verbose(self) -> bool:
        return self.runtime.verbose

    @property
    def timings(self) -> bool:
        return self.runtime.timings

    @property
    def min_length(self) -> int | None:
        return self.generation.min_length

    @property
    def max_length(self) -> int | None:
        return self.generation.max_length

    @property
    def top_p(self) -> float | None:
        return self.generation.top_p

    @property
    def top_k(self) -> int | None:
        return self.generation.top_k

    @property
    def temperature(self) -> float | None:
        return self.generation.temperature

    @property
    def repetition_penalty(self) -> float | None:
        return self.generation.repetition_penalty

    @property
    def do_sample(self) -> bool:
        return self.generation.do_sample


def _load_yaml_config(path: str | Path) -> AppConfig:
    """Load runtime configuration for EliteLM chat sessions from a YAML file."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open('r', encoding='utf-8') as handle:
        raw_config = yaml.safe_load(handle) or {}

    if not isinstance(raw_config, dict):
        raise ValueError('Config file must contain a YAML mapping at the top level.')

    # Resolve paths relative to config file
    config_dir = config_path.parent

    if 'model' in raw_config and isinstance(raw_config['model'], str):
        model_path = Path(raw_config['model']).expanduser()
        if not model_path.is_absolute():
            raw_config['model'] = str((config_dir / model_path).resolve())
        else:
            raw_config['model'] = str(model_path.resolve())

    if 'qnn' in raw_config and isinstance(raw_config['qnn'], dict):
        if 'sdk_root' in raw_config['qnn'] and isinstance(raw_config['qnn']['sdk_root'], str):
            sdk_path = Path(raw_config['qnn']['sdk_root']).expanduser()
            if not sdk_path.is_absolute():
                raw_config['qnn']['sdk_root'] = str((config_dir / sdk_path).resolve())
            else:
                raw_config['qnn']['sdk_root'] = str(sdk_path.resolve())

        if 'backend' in raw_config['qnn'] and isinstance(raw_config['qnn']['backend'], str):
            backend_path = Path(raw_config['qnn']['backend']).expanduser()
            if not backend_path.is_absolute():
                raw_config['qnn']['backend'] = str((config_dir / backend_path).resolve())
            else:
                raw_config['qnn']['backend'] = str(backend_path.resolve())

    config = AppConfig(**raw_config)
    config.config_path = str(config_path)
    return config


def _load_model(args) -> tuple[og.Model, og.Tokenizer, og.TokenizerStream]:
    model_dir = Path(args.model)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if args.device == "qnn":
        config = _configure_qnn_provider(
            model_dir,
            getattr(args, "qnn_sdk", None),
            getattr(args, "qnn_backend", None),
        )
        if isinstance(config, _DummyConfig):
            missing = model_dir / "genai_config.json"
            raise FileNotFoundError(
                f"Model directory not ready for QNN execution: missing {missing}"
            )
        model = og.Model(config)
    else:
        model = og.Model(str(model_dir))

    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    return model, tokenizer, tokenizer_stream




@dataclass
class GenerationStats:
    prompt_length: int
    new_tokens: int
    time_to_first_token: float
    prompt_tokens_per_second: float
    generated_tokens_per_second: float


@dataclass
class GenerationResult:
    text: str
    stats: GenerationStats | None
    interrupted: bool


class ChatSession:
    """High-level chat session that wraps tokenizer/model state."""

    def __init__(self, args: AppConfig | argparse.Namespace) -> None:
        self.args = args
        self.model, self.tokenizer, self.tokenizer_stream = _load_model(args)
        self.search_options = {
            name: getattr(args, name)
            for name in [
                "do_sample",
                "max_length",
                "min_length",
                "top_p",
                "top_k",
                "temperature",
                "repetition_penalty",
            ]
            if hasattr(args, name)
        }
        if "max_length" not in self.search_options:
            self.search_options["max_length"] = 2048
        self.chat_history: list[dict[str, str]] = []
        self._fallback_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    @property
    def device_label(self) -> str:
        return "QNN" if getattr(self.args, "device", "cpu") == "qnn" else "CPU"

    def reset_history(self) -> None:
        self.chat_history.clear()

    def _build_prompt(self, user_text: str) -> tuple[str, list[dict[str, str]]]:
        chat_messages = list(self.chat_history)
        chat_messages.append({"role": "user", "content": user_text})
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                json.dumps(chat_messages),
                add_generation_prompt=True,
            )
        else:
            if self.chat_history:
                prompt_parts: list[str] = []
                for message in chat_messages:
                    role = message["role"]
                    content = message["content"]
                    if role == "assistant":
                        prompt_parts.append(f"<|assistant|>\n{content} <|end|>\n")
                    else:
                        prompt_parts.append(f"<|user|>\n{content} <|end|>\n")
                prompt_parts.append("<|assistant|>")
                prompt = "".join(prompt_parts)
            else:
                prompt = self._fallback_template.format(input=user_text)
        return prompt, chat_messages

    def generate(
        self,
        user_text: str,
        *,
        max_new_tokens: int | None = None,
        timings: bool = False,
        on_token: Callable[[str], None] | None = None,
        **kwargs,
    ) -> GenerationResult:
        if not user_text:
            raise ValueError("user_text cannot be empty")

        prompt, chat_messages = self._build_prompt(user_text)
        input_tokens = self.tokenizer.encode(prompt).astype(np.int32)
        prompt_len = int(input_tokens.size) if input_tokens.ndim == 1 else int(input_tokens.shape[-1])

        params = og.GeneratorParams(self.model)
        
        # Merge default options with per-request overrides
        search_options = self.search_options.copy()
        search_options.update(kwargs)

        # Handle max_new_tokens logic (OpenAI style) vs max_length (ORT style)
        if max_new_tokens is not None:
            search_options["max_length"] = prompt_len + max_new_tokens
        else:
            configured_max = search_options.get("max_length")
            if configured_max is None or configured_max <= prompt_len:
                # Ensure there is always room for generation even if previous calls shrank max_length
                default_total = max(self.search_options.get("max_length", 2048), prompt_len)
                extra_tokens = default_total - prompt_len
                if extra_tokens <= 0:
                    extra_tokens = 512
                search_options["max_length"] = prompt_len + extra_tokens

        params.set_search_options(**search_options)

        generator_supports_append = hasattr(og.Generator, "append_tokens")
        params_supports_set_input = hasattr(params, "set_model_input")

        batched_tokens = input_tokens.reshape(1, -1) if input_tokens.ndim == 1 else input_tokens
        if not generator_supports_append and params_supports_set_input:
            params.set_model_input("input_ids", batched_tokens)
            try:
                params.set_model_input("attention_mask", np.ones_like(batched_tokens, dtype=np.int32))
            except RuntimeError:
                pass

        generator = og.Generator(self.model, params)

        if generator_supports_append and hasattr(generator, "append_tokens"):
            generator.append_tokens(input_tokens)
            if hasattr(generator, "set_model_input"):
                try:
                    generator.set_model_input("attention_mask", np.ones((1, input_tokens.size), dtype=np.int32))
                except RuntimeError:
                    pass
        elif hasattr(generator, "set_model_input"):
            generator.set_model_input("input_ids", batched_tokens)
            try:
                generator.set_model_input("attention_mask", np.ones_like(batched_tokens, dtype=np.int32))
            except RuntimeError:
                pass
        else:
            raise AttributeError("onnxruntime_genai.Generator missing append_tokens and set_model_input")

        if hasattr(self.tokenizer_stream, "reset"):
            self.tokenizer_stream.reset()

        generated_pieces: list[str] = []
        interrupted = False

        first_token_timestamp = 0.0
        started_timestamp = time.time() if timings else 0.0
        first = True
        generated_token_ids: list[int] = []

        try:
            while not generator.is_done():
                generator.generate_next_token()
                if timings and first:
                    first_token_timestamp = time.time()
                    first = False

                raw_next = generator.get_next_tokens().tolist()
                if isinstance(raw_next, int):
                    next_token_ids = [raw_next]
                else:
                    next_token_ids = list(raw_next)

                for token_id in next_token_ids:
                    piece = self.tokenizer_stream.decode(int(token_id))
                    if on_token is not None:
                        on_token(piece)
                    generated_pieces.append(piece)
                    if timings:
                        generated_token_ids.append(int(token_id))
        except KeyboardInterrupt:
            interrupted = True
        finally:
            del generator

        assistant_reply = "".join(generated_pieces).strip()
        if not interrupted:
            self.chat_history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_reply},
                ]
            )

        stats: GenerationStats | None = None
        if timings:
            if first:
                prompt_time = 0.0
                run_time = 0.0
            else:
                prompt_time = first_token_timestamp - started_timestamp
                run_time = time.time() - first_token_timestamp
            prompt_tokens = len(input_tokens)
            new_token_count = len(generated_token_ids)
            stats = GenerationStats(
                prompt_length=prompt_tokens,
                new_tokens=new_token_count,
                time_to_first_token=prompt_time,
                prompt_tokens_per_second=(prompt_tokens / prompt_time) if prompt_time else 0.0,
                generated_tokens_per_second=(new_token_count / run_time) if run_time else 0.0,
            )

        return GenerationResult(
            text=assistant_reply,
            stats=stats,
            interrupted=interrupted,
        )


def load_config(path: str | Path) -> AppConfig:
    """Public wrapper for loading YAML chat configuration."""
    return _load_yaml_config(path)

