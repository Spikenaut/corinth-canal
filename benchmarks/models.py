"""Inference backends used by the Python benchmark harness.

All third-party runtimes are optional and imported only when an adapter is
loaded.  Importing this module is therefore safe in the CPU-only CI profile.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import importlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


class BackendUnavailableError(RuntimeError):
    """Raised when a selected optional inference runtime is unavailable."""


def _field(manifest: Mapping[str, Any] | object, name: str, default: Any = None) -> Any:
    if isinstance(manifest, Mapping):
        return manifest.get(name, default)
    return getattr(manifest, name, default)


def _model_path(manifest: Mapping[str, Any] | object) -> str:
    for field in ("path", "model_path", "artifact_path", "artifact", "model_id"):
        value = _field(manifest, field)
        if value:
            return str(value)
    raise ValueError(
        "model manifest must define path, model_path, artifact_path, artifact, or model_id"
    )


class ModelAdapter(ABC):
    """Small common interface implemented by every benchmark backend."""

    backend: str

    def __init__(self, manifest: Mapping[str, Any] | object, **options: Any) -> None:
        self.manifest = manifest
        self.model_path = _model_path(manifest)
        self.options = options
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model artifact and its runtime."""

    @abstractmethod
    def generate(self, prompt: str, **generation: Any) -> str:
        """Generate text for one prompt."""

    def predict(self, prompt: str, **generation: Any) -> str:
        """Compatibility alias used by benchmark evaluators."""
        return self.generate(prompt, **generation)


class MockModelAdapter(ModelAdapter):
    """Dependency-free adapter retained for smoke tests and development."""

    backend = "mock"

    def __init__(self, manifest: Mapping[str, Any] | object | None = None, **options: Any) -> None:
        super().__init__(manifest or {"path": "mock"}, **options)

    def load(self) -> None:
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        return str(generation.get("response", self.options.get("response", prompt)))


class LlamaCppAdapter(ModelAdapter):
    """GGUF inference through llama-cpp-python or the llama CLI."""

    backend = "llama.cpp"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            module = importlib.import_module("llama_cpp")
        except ImportError:
            module = None
        if module is not None:
            kwargs = {
                "n_ctx": self.options.get("n_ctx", 2048),
                "n_gpu_layers": self.options.get("n_gpu_layers", 0),
                "verbose": self.options.get("verbose", False),
            }
            self._model = module.Llama(model_path=self.model_path, **kwargs)
            self._mode = "python"
            self._loaded = True
            return

        executable = self.options.get("executable")
        executable = shutil.which(executable) if executable else (
            shutil.which("llama-cli") or shutil.which("main")
        )
        if not executable:
            raise BackendUnavailableError(
                "llama.cpp backend is unavailable: install llama-cpp-python or put "
                "llama-cli on PATH (an executable may also be passed explicitly)"
            )
        self._executable = executable
        self._mode = "cli"
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        max_tokens = int(generation.get("max_tokens", 32))
        temperature = float(generation.get("temperature", 0.0))
        if self._mode == "python":
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            return str(result["choices"][0]["text"])
        command = [
            self._executable,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--no-display-prompt",
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        return completed.stdout.strip()


class VllmAdapter(ModelAdapter):
    """High-throughput inference for Hugging Face/Safetensors artifacts."""

    backend = "vllm"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            module = importlib.import_module("vllm")
        except ImportError as error:
            raise BackendUnavailableError(
                "vLLM backend is unavailable: install the optional 'vllm' package"
            ) from error
        kwargs = dict(self.options)
        kwargs.pop("executable", None)
        self._sampling_params = module.SamplingParams
        self._model = module.LLM(model=self.model_path, **kwargs)
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        params = self._sampling_params(
            max_tokens=int(generation.get("max_tokens", 32)),
            temperature=float(generation.get("temperature", 0.0)),
            top_p=float(generation.get("top_p", 1.0)),
        )
        result = self._model.generate([prompt], params, use_tqdm=False)
        return str(result[0].outputs[0].text)


class HuggingFaceAdapter(ModelAdapter):
    """Transformers fallback for standard causal language models."""

    backend = "huggingface"

    def load(self) -> None:
        if self._loaded:
            return
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise BackendUnavailableError(
                "Hugging Face backend is unavailable: install the optional "
                "'transformers' package (and a supported tensor runtime)"
            ) from error
        trust_remote_code = bool(self.options.get("trust_remote_code", False))
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code
        )
        model_options = {
            key: value for key, value in self.options.items()
            if key not in {"trust_remote_code", "executable"}
        }
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code, **model_options
        )
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        encoded = self._tokenizer(prompt, return_tensors="pt")
        output = self._model.generate(
            **encoded,
            max_new_tokens=int(generation.get("max_tokens", 32)),
            do_sample=float(generation.get("temperature", 0.0)) > 0,
            temperature=float(generation.get("temperature", 1.0)),
            top_p=float(generation.get("top_p", 1.0)),
        )
        prompt_tokens = encoded["input_ids"].shape[-1]
        return self._tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)


_GGUF_FORMATS = {"gguf", "q8_0", "q6_k", "q5_k_m", "iq4_nl", "iq3_m"}
_HF_FORMATS = {"safetensors", "hf", "huggingface", "bf16", "fp16", "fp8", "int8", "bnb-4bit", "bnb_4bit"}


def detect_backend(manifest: Mapping[str, Any] | object) -> str:
    """Choose a backend from explicit runtime and artifact format metadata."""
    runtime = str(_field(manifest, "runtime_format", "") or "").lower().replace("-", "_")
    explicit = {
        "llama.cpp": "llama.cpp", "llama_cpp": "llama.cpp",
        "llamacpp": "llama.cpp", "gguf": "llama.cpp",
        "vllm": "vllm", "hf": "huggingface", "huggingface": "huggingface",
        "hf_transformers": "huggingface", "transformers": "huggingface", "mock": "mock",
    }
    if runtime:
        if runtime not in explicit:
            raise ValueError(f"unsupported runtime_format {runtime!r}")
        return explicit[runtime]
    source = str(_field(manifest, "source_format", "") or "").lower()
    path = str(_field(manifest, "path", "") or "").lower()
    if source in _GGUF_FORMATS or path.endswith(".gguf"):
        return "llama.cpp"
    if source in _HF_FORMATS or source.startswith("safetensors"):
        return "vllm"
    raise ValueError(
        "cannot detect inference backend; set manifest runtime_format or source_format"
    )


def adapter_for_manifest(
    manifest: Mapping[str, Any] | object, *, load: bool = False, **options: Any
) -> ModelAdapter:
    """Construct (and optionally load) the adapter selected by a manifest."""
    adapters = {
        "llama.cpp": LlamaCppAdapter,
        "vllm": VllmAdapter,
        "huggingface": HuggingFaceAdapter,
        "mock": MockModelAdapter,
    }
    adapter = adapters[detect_backend(manifest)](manifest, **options)
    if load:
        adapter.load()
    return adapter


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON model manifest for simple runner integrations."""
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("model manifest must be a JSON object")
    return value
