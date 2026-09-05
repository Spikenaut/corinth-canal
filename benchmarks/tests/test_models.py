from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmarks import load_manifest as exported_load_manifest
from benchmarks.models import (
    BackendUnavailableError,
    HuggingFaceAdapter,
    LlamaCppAdapter,
    MockModelAdapter,
    VllmAdapter,
    _validate_generation,
    adapter_for_manifest,
    detect_backend,
    load_manifest,
)


class BackendDetectionTests(unittest.TestCase):
    def test_detects_gguf_and_safetensors(self) -> None:
        self.assertEqual(detect_backend({"path": "tiny.gguf", "source_format": "GGUF"}), "llama.cpp")
        self.assertEqual(detect_backend({"path": "org/model", "source_format": "BF16"}), "vllm")

    def test_detects_suffixes_on_path_aliases(self) -> None:
        self.assertEqual(detect_backend({"model_path": "tiny.gguf"}), "llama.cpp")
        self.assertEqual(detect_backend({"artifact_path": "model.safetensors"}), "vllm")
        self.assertEqual(detect_backend({"model_id_or_path": "weights.gguf"}), "llama.cpp")
        self.assertEqual(detect_backend({"model_id_or_local_path": "shard.safetensors"}), "vllm")
        self.assertEqual(detect_backend({"checkpoint_path": "ckpt.safetensors"}), "vllm")

    def test_explicit_runtime_wins(self) -> None:
        manifest = {"path": "org/model", "source_format": "safetensors", "runtime_format": "transformers"}
        self.assertIsInstance(adapter_for_manifest(manifest), HuggingFaceAdapter)

    def test_unknown_metadata_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "runtime_format or source_format"):
            detect_backend({"path": "model.bin"})

    def test_pathless_explicit_mock_manifest(self) -> None:
        adapter = adapter_for_manifest({"runtime_format": "mock"})
        self.assertIsInstance(adapter, MockModelAdapter)
        self.assertEqual(adapter.generate("hello"), "hello")


class PathAliasTests(unittest.TestCase):
    def test_adapter_accepts_canonical_path_fields(self) -> None:
        cases = {
            "model_id_or_path": "from-run-entry.gguf",
            "model_id_or_local_path": "from-adapter-config.gguf",
            "checkpoint_path": "from-experiment.gguf",
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                adapter = LlamaCppAdapter({field: value})
                self.assertEqual(adapter.model_path, value)


class GenerationValidationTests(unittest.TestCase):
    def test_rejects_fractional_and_boolean_max_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer"):
            _validate_generation({"max_tokens": 31.9})
        with self.assertRaisesRegex(ValueError, "integer"):
            _validate_generation({"max_tokens": True})

    def test_accepts_integer_max_tokens(self) -> None:
        max_tokens, _, _ = _validate_generation({"max_tokens": 64})
        self.assertEqual(max_tokens, 64)
        max_tokens, _, _ = _validate_generation({"max_tokens": "16"})
        self.assertEqual(max_tokens, 16)

    def test_rejects_non_finite_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_generation({"temperature": math.nan})
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_generation({"temperature": math.inf})
        with self.assertRaisesRegex(ValueError, "finite"):
            _validate_generation({"temperature": -math.inf})


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

    def test_llama_python_forwards_top_p(self) -> None:
        calls: list[dict] = []

        class FakeLlama:
            def __init__(self, **kwargs):
                pass

            def __call__(self, prompt, **kwargs):
                calls.append(kwargs)
                return {"choices": [{"text": "sampled"}]}

        module = types.SimpleNamespace(Llama=FakeLlama)
        with mock.patch.dict(sys.modules, {"llama_cpp": module}):
            adapter = LlamaCppAdapter({"path": "tiny.gguf"})
            self.assertEqual(adapter.generate("question", top_p=0.7, temperature=0.8), "sampled")
        self.assertEqual(calls[0]["top_p"], 0.7)
        self.assertEqual(calls[0]["temperature"], 0.8)

    def test_llama_cli_forwards_top_p_and_honors_timeout(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "cli"
        adapter._executable = "/opt/llama-cli"
        completed = mock.Mock(stdout="cli answer\n")
        with mock.patch("benchmarks.models.subprocess.run", return_value=completed) as run:
            text = adapter.generate("prompt", top_p=0.55, timeout=12)
        self.assertEqual(text, "cli answer")
        command = run.call_args.args[0]
        self.assertEqual(command[0], "/opt/llama-cli")
        self.assertIn("--top-p", command)
        self.assertEqual(command[command.index("--top-p") + 1], "0.55")
        self.assertIn("-no-cnv", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 12)
        self.assertIs(run.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertFalse(run.call_args.kwargs["shell"])

    def test_llama_cli_empty_stdout_is_not_success(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "cli"
        adapter._executable = "/opt/llama-cli"
        with mock.patch("benchmarks.models.subprocess.run", return_value=mock.Mock(stdout="  \n")):
            with self.assertRaisesRegex(RuntimeError, "empty output"):
                adapter.generate("prompt")

    def test_llama_cli_permission_error_is_unavailable(self) -> None:
        adapter = LlamaCppAdapter({"path": "tiny.gguf"})
        adapter._loaded = True
        adapter._mode = "cli"
        adapter._executable = "/opt/llama-cli"
        with mock.patch("benchmarks.models.subprocess.run", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(BackendUnavailableError, "not executable"):
                adapter.generate("prompt")

    def test_explicit_non_executable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "llama-cli"
            exe.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
            exe.chmod(0o644)
            with mock.patch("benchmarks.models.importlib.import_module", side_effect=ImportError):
                with self.assertRaisesRegex(BackendUnavailableError, "not executable"):
                    LlamaCppAdapter({"path": "tiny.gguf"}, executable=str(exe)).load()

    def test_does_not_treat_main_on_path_as_llama_cli(self) -> None:
        def fake_which(name: str) -> str | None:
            if name == "main":
                return "/usr/bin/main"
            return None

        with mock.patch("benchmarks.models.importlib.import_module", side_effect=ImportError):
            with mock.patch("benchmarks.models.shutil.which", side_effect=fake_which):
                with self.assertRaisesRegex(BackendUnavailableError, "llama-cli"):
                    LlamaCppAdapter({"path": "tiny.gguf"}).load()

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


class PackageExportTests(unittest.TestCase):
    def test_load_manifest_is_exported(self) -> None:
        self.assertIs(exported_load_manifest, load_manifest)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('{"runtime_format": "mock"}', encoding="utf-8")
            self.assertEqual(load_manifest(path)["runtime_format"], "mock")


if __name__ == "__main__":
    unittest.main()
