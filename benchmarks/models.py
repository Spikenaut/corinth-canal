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
import warnings
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


def _validate_generation(generation: dict[str, Any]) -> tuple[int, float, float]:
    """Validate and normalize generation options."""
    max_tokens = generation.get("max_tokens", 32)
    try:
        max_tokens = int(max_tokens)
    except (TypeError, ValueError) as e:
        raise ValueError(f"max_tokens must be an integer, got {max_tokens!r}") from e
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
    if max_tokens > 4096:
        raise ValueError(f"max_tokens {max_tokens} exceeds limit 4096")
    temperature = generation.get("temperature", 0.0)
    try:
        temperature = float(temperature)
    except (TypeError, ValueError) as e:
        raise ValueError(f"temperature must be numeric, got {temperature!r}") from e
    if temperature < 0.0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    top_p = generation.get("top_p", 1.0)
    try:
        top_p = float(top_p)
    except (TypeError, ValueError) as e:
        raise ValueError(f"top_p must be numeric, got {top_p!r}") from e
    if not 0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    return max_tokens, temperature, top_p


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

    def __init__(self, manifest: Mapping[str, Any] | object, **options: Any) -> None:
        super().__init__(manifest, **options)
        self._mode: str | None = None
        self._executable: str | None = None
        self._model: Any | None = None

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
        if executable:
            # Explicit path: verify it exists and is executable, do not search PATH
            if not Path(executable).is_file():
                raise BackendUnavailableError(
                    f"llama.cpp executable not found: {executable!r}"
                )
        else:
            # Only search for the known llama.cpp CLI, not generic "main"
            executable = shutil.which("llama-cli")
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
        if self._mode is None:
            raise RuntimeError("LlamaCppAdapter not loaded correctly")
        max_tokens, temperature, _ = _validate_generation(generation)
        if self._mode == "python":
            assert self._model is not None
            result = self._model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
            )
            return str(result["choices"][0]["text"])
        assert self._executable is not None
        command = [
            self._executable,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "--no-display-prompt",
        ]
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
                input="",
            )
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llama-cli timed out after 30s: {e}") from e
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llama-cli failed: {e.stderr}") from e
        text = completed.stdout.strip()
        if not text:
            raise RuntimeError("llama-cli returned empty output")
        return text


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
        # Filter options to known safe keys; do not blanket-forward trust_remote_code
        allowed_keys = {"dtype", "tensor_parallel_size", "gpu_memory_utilization", "max_model_len", "enforce_eager"}
        kwargs = {k: v for k, v in self.options.items() if k in allowed_keys}
        if "trust_remote_code" in self.options and self.options["trust_remote_code"]:
            warnings.warn("trust_remote_code=True is not supported for vLLM adapter; ignoring")
        kwargs.pop("executable", None)
        self._sampling_params = module.SamplingParams
        self._model = module.LLM(model=self.model_path, **kwargs)
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        max_tokens, temperature, top_p = _validate_generation(generation)
        params = self._sampling_params(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
        if trust_remote_code:
            warnings.warn(
                "trust_remote_code=True allows arbitrary code execution from model files; "
                "only enable for trusted models",
                UserWarning,
                stacklevel=2,
            )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code
        )
        model_options = {
            key: value for key, value in self.options.items()
            if key not in {"trust_remote_code", "executable"}
        }
        # Only allow known safe HF model kwargs
        allowed_hf_keys = {"torch_dtype", "device_map", "low_cpu_mem_usage", "attn_implementation"}
        filtered_options = {k: v for k, v in model_options.items() if k in allowed_hf_keys}
        if len(filtered_options) != len(model_options):
            dropped = set(model_options) - set(filtered_options)
            warnings.warn(f"Ignoring unsupported HF options: {dropped}")
        self._model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=trust_remote_code, **filtered_options
        )
        # Determine device for later tensor placement
        try:
            self._device = next(self._model.parameters()).device
        except StopIteration:
            self._device = None
        self._loaded = True

    def generate(self, prompt: str, **generation: Any) -> str:
        if not self._loaded:
            self.load()
        max_tokens, temperature, top_p = _validate_generation(generation)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        # Move tokenizer tensors to model device to avoid GPU crash
        if getattr(self, "_device", None) is not None:
            try:
                import torch
                encoded = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in encoded.items()}
            except ImportError:
                pass
        # Consistent temperature handling: single default
        do_sample = temperature > 0
        output = self._model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p,
        )
        prompt_tokens = encoded["input_ids"].shape[-1]
        return self._tokenizer.decode(output[0][prompt_tokens:], skip_special_tokens=True)


_GGUF_FORMATS = {"gguf", "q4_k_m", "q4_0", "q4_k_s", "q5_k_m", "q5_0", "q6_k", "q8_0", "iq4_nl", "iq3_m", "q2_k", "q3_k_m"}
_HF_FORMATS = {"safetensors", "hf", "huggingface", "bf16", "fp16", "fp8", "int8", "bnb-4bit", "bnb_4bit", "gptq", "awq", "pt", "bin"}


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
    # Use _model_path to handle alias fields (path, model_path, artifact_path, etc.)
    try:
        model_path_str = _model_path(manifest).lower()
    except ValueError:
        model_path_str = str(_field(manifest, "path", "") or "").lower()
    if source in _GGUF_FORMATS or model_path_str.endswith(".gguf"):
        return "llama.cpp"
    if source in _HF_FORMATS or source.startswith("safetensors") or model_path_str.endswith(".safetensors"):
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
