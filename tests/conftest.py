"""Shared pytest fixtures and configuration for elitelm tests."""

import importlib.util
from pathlib import Path
import pytest

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

MODULE_PATH = Path(__file__).resolve().parents[1] / "elitelm.py"


def load_elitelm_module():
    spec = importlib.util.spec_from_file_location("elitelm", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================================
# Pytest Markers and Configuration
# ============================================================================


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "requires_npu: mark test as requiring Qualcomm NPU hardware "
        "(deselect with '-m \"not requires_npu\"')",
    )


# ============================================================================
# QNN Availability Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def qnn_available():
    """Check if QNN is available on this system (compiled into onnxruntime-genai).

    Returns:
        bool: True if og.is_qnn_available() returns True, False otherwise.
    """
    if og is None:
        return False
    try:
        return og.is_qnn_available()
    except (AttributeError, RuntimeError):
        return False


@pytest.fixture(scope="session")
def skip_if_no_qnn(qnn_available):
    """Skip test if QNN is not available.

    Use this fixture when a test requires QNN to be available.
    If QNN is not available, the test is skipped with an appropriate message.
    """
    if not qnn_available:
        pytest.skip(
            "QNN not available in onnxruntime-genai on this system. "
            "Install onnxruntime-qnn or build onnxruntime-genai with QNN support."
        )


# ============================================================================
# QNN SDK and Hardware Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def qnn_sdk_root(skip_if_no_qnn):
    """Resolve the QNN SDK root directory.

    Returns:
        Path: Path to the QNN SDK root directory.

    Raises:
        pytest.skip: If QNN SDK cannot be found.
    """
    module = load_elitelm_module()
    try:
        return module._resolve_qnn_sdk_root(None).resolve()
    except FileNotFoundError as e:
        pytest.skip(f"QNN SDK not found: {e}")


@pytest.fixture(scope="session")
def qnn_backend(qnn_sdk_root):
    """Resolve the QNN backend DLL path.

    Returns:
        Path: Path to the QnnHtp backend DLL.

    Raises:
        pytest.skip: If backend not found.
    """
    module = load_elitelm_module()
    try:
        backend_path, _ = module._default_backend_path(qnn_sdk_root)
        return backend_path
    except FileNotFoundError as e:
        pytest.skip(f"QNN backend not found: {e}")


@pytest.fixture(scope="session")
def test_model_path():
    """Path to a quantized test model for NPU verification.

    Returns:
        Path: Path to the test model directory.

    Raises:
        pytest.skip: If test model not found.
    """
    # Try several common locations
    base_candidates = [
        Path("cpu_and_mobile") / "phi-2-mini",
        Path("cpu_and_mobile") / "tinyllama",
        Path("cpu_and_mobile") / "llama3-8b",
        Path("models") / "test_model",
    ]

    # Also check any subdirectory in cpu_and_mobile/
    cpu_mobile_dir = Path("cpu_and_mobile")
    if cpu_mobile_dir.exists():
        for subdir in cpu_mobile_dir.iterdir():
            if subdir.is_dir():
                base_candidates.append(subdir)

    # Model file names to look for (in order of preference)
    model_files = ["model.onnx", "ort_model.onnx"]

    for candidate in base_candidates:
        if candidate.exists():
            for model_file in model_files:
                if (candidate / model_file).exists():
                    return candidate.resolve()

    pytest.skip(
        "Test model not found. Looked for ONNX model (model.onnx or ort_model.onnx) in: "
        f"{', '.join(str(c) for c in base_candidates)}"
    )
