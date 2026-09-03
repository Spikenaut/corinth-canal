from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from benchmarks.models import (
    BackendUnavailableError,
    HuggingFaceAdapter,
    LlamaCppAdapter,
    VllmAdapter,
    adapter_for_manifest,
    detect_backend,
)


class BackendDetectionTests(unittest.TestCase):
    def test_detects_gguf_and_safetensors(self) -> None:
        self.assertEqual(detect_backend({"path": "tiny.gguf", "source_format": "GGUF"}), "llama.cpp")
        self.assertEqual(detect_backend({"path": "org/model", "source_format": "BF16"}), "vllm")

    def test_explicit_runtime_wins(self) -> None:
        manifest = {"path": "org/model", "source_format": "safetensors", "runtime_format": "transformers"}
        self.assertIsInstance(adapter_for_manifest(manifest), HuggingFaceAdapter)

    def test_unknown_metadata_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime_format or source_format"):
            detect_backend({"path": "model.bin"})


class OptionalBackendTests(unittest.TestCase):
    def test_llama_python_backend_generates_text(self) -> None:
        class FakeLlama:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __call__(self, prompt, **kwargs):
                return {"choices": [{"text": " real answer"}]}

        module = types.SimpleNamespace(Llama=FakeLlama)
        with mock.patch.dict(sys.modules, {"llama_cpp": module}):
            adapter = LlamaCppAdapter({"path": "tiny.gguf"})
            self.assertEqual(adapter.generate("question"), " real answer")

    def test_vllm_backend_generates_text(self) -> None:
        class FakeLLM:
            def __init__(self, **kwargs):
                pass

            def generate(self, prompts, params, use_tqdm):
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="answer")])]

        module = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kwargs: kwargs)
        with mock.patch.dict(sys.modules, {"vllm": module}):
            adapter = VllmAdapter({"path": "tiny-model"})
            self.assertEqual(adapter.predict("question"), "answer")

    def test_missing_dependency_is_wrapped(self) -> None:
        with mock.patch("benchmarks.models.shutil.which", return_value=None):
            with mock.patch("benchmarks.models.importlib.import_module", side_effect=ImportError):
                with self.assertRaisesRegex(BackendUnavailableError, "llama-cpp-python"):
                    LlamaCppAdapter({"path": "tiny.gguf"}).load()


if __name__ == "__main__":
    unittest.main()
