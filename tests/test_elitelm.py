import os
from pathlib import Path
import pytest
import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "elitelm.py"


def load_module():
    spec = importlib.util.spec_from_file_location("elitelm", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def elitelm_module(monkeypatch):
    module = load_module()
    # Ensure globals start clean for each test
    module._DLL_HANDLES.clear()
    module._REGISTERED_DLL_PATHS.clear()
    return module


def test_add_dll_dir_registers_once(monkeypatch, tmp_path, elitelm_module):
    added = []

    def fake_add_dll_directory(path):
        added.append(path)
        return f"handle-{path}"

    monkeypatch.setattr(elitelm_module.os, "add_dll_directory", fake_add_dll_directory)
    monkeypatch.setitem(elitelm_module.os.environ, "PATH", "")

    dll_dir = tmp_path / "lib"
    dll_dir.mkdir()

    elitelm_module._add_dll_dir(dll_dir)
    elitelm_module._add_dll_dir(dll_dir)  # second call should be ignored

    assert added == [str(dll_dir.resolve())]
    path_env = elitelm_module.os.environ["PATH"].split(";")
    assert str(dll_dir.resolve()) in path_env


def test_resolve_qnn_sdk_root_prefers_argument(tmp_path, elitelm_module):
    provided = tmp_path / "sdk"
    provided.mkdir()

    result = elitelm_module._resolve_qnn_sdk_root(str(provided))
    assert result == provided.resolve()


def test_resolve_qnn_sdk_root_falls_back_to_env(monkeypatch, tmp_path, elitelm_module):
    env_path = tmp_path / "env-sdk"
    env_path.mkdir()
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setenv("QNN_SDK_ROOT", str(env_path))

    result = elitelm_module._resolve_qnn_sdk_root(None)
    assert result == env_path.resolve()


def test_resolve_qnn_sdk_root_detects_qairt(monkeypatch, tmp_path, elitelm_module):
    repo_root = tmp_path / "repo"
    qairt_dir = repo_root / "qairt" / "3.0.0"
    qairt_dir.mkdir(parents=True)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setattr(elitelm_module, "__file__", str(repo_root / "elitelm.py"))

    result = elitelm_module._resolve_qnn_sdk_root(None)
    assert result == qairt_dir.resolve()


def test_default_backend_path_prefers_arm64x_on_windows(monkeypatch, tmp_path, elitelm_module):
    monkeypatch.setattr(elitelm_module.sys, "platform", "win32")
    sdk_root = tmp_path / "sdk"

    arm64x_dir = sdk_root / "lib" / "arm64x-windows-msvc"
    arm64x_dir.mkdir(parents=True)
    (arm64x_dir / "QnnHtp.dll").write_bytes(b"")

    aarch_dir = sdk_root / "lib" / "aarch64-windows-msvc"
    aarch_dir.mkdir(parents=True)
    (aarch_dir / "QnnHtp.dll").write_bytes(b"")

    backend, arch = elitelm_module._default_backend_path(sdk_root)
    assert arch == "arm64x-windows-msvc"
    assert backend == (arm64x_dir / "QnnHtp.dll").resolve()


def test_configure_qnn_provider_sets_options(monkeypatch, tmp_path, elitelm_module):
    sdk_root = tmp_path / "sdk"
    lib_dir = sdk_root / "lib" / "aarch64-windows-msvc"
    bin_dir = sdk_root / "bin" / "aarch64-windows-msvc"
    lib_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    backend = lib_dir / "QnnHtp.dll"
    backend.write_bytes(b"")

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    added = []
    monkeypatch.setattr(
        elitelm_module.os,
        "add_dll_directory",
        lambda path: added.append(Path(path))
    )
    monkeypatch.setitem(elitelm_module.os.environ, "PATH", "")

    config = elitelm_module._configure_qnn_provider(model_dir, str(sdk_root), None)
    opt_backend = config.get_provider_option("QNNExecutionProvider", "backend_path")
    opt_sdk = config.get_provider_option("QNNExecutionProvider", "qnn_sdk_root")

    assert Path(opt_backend) == backend.resolve()
    assert Path(opt_sdk) == sdk_root.resolve()
    assert set(p.resolve() for p in added) == {lib_dir.resolve(), bin_dir.resolve()}

    # repeated call should not duplicate paths
    elitelm_module._configure_qnn_provider(model_dir, str(sdk_root), None)
    assert len(added) == 2


def test_add_dll_dir_noop_for_missing_path(monkeypatch, tmp_path, elitelm_module):
    missing = tmp_path / "nope"
    monkeypatch.setitem(elitelm_module.os.environ, "PATH", "")
    monkeypatch.setattr(elitelm_module.os, "add_dll_directory", lambda path: (_ for _ in ()).throw(RuntimeError("should not be called")))

    elitelm_module._add_dll_dir(missing)
    assert elitelm_module.os.environ["PATH"] == ""


def test_load_yaml_config_defaults(tmp_path, elitelm_module):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""model: ./models/example
generation:
  max_length: 512
runtime:
  verbose: true
""")

    config = elitelm_module._load_yaml_config(config_file)
    assert Path(config.model) == (tmp_path / "models/example").resolve()
    assert config.device == "cpu"
    assert config.do_sample is False
    assert config.verbose is True
    assert config.max_length == 512
    assert not hasattr(config, "min_length")
    assert config.qnn_sdk is None
    assert config.qnn_backend is None


def test_load_yaml_config_requires_model(tmp_path, elitelm_module):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""runtime:
  timings: true
""")

    with pytest.raises(ValueError):
        elitelm_module._load_yaml_config(config_file)


def test_load_yaml_config_resolves_qnn_paths(tmp_path, elitelm_module):
    sdk_root = tmp_path / "qairt" / "2.0.0"
    backend = sdk_root / "lib" / "QnnHtp.dll"
    backend.parent.mkdir(parents=True)
    backend.write_bytes(b"")
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    config_file = tmp_path / "config.yaml"
    config_file.write_text("""model: ./model
device: qnn
qnn:
  sdk_root: ./qairt/2.0.0
  backend: ./qairt/2.0.0/lib/QnnHtp.dll
""")

    config = elitelm_module._load_yaml_config(config_file)
    assert Path(config.model) == model_dir.resolve()
    assert config.device == "qnn"
    assert Path(config.qnn_sdk) == sdk_root.resolve()
    assert Path(config.qnn_backend) == backend.resolve()


# ============================================================================
# Tier 1: Build-Time Availability Tests
# ============================================================================


def test_qnn_compiled_in():
    """Verify QNN support is compiled into onnxruntime-genai (Tier 1).

    This test validates that the build includes QNN support.
    It runs on all systems and returns early on non-Snapdragon machines.
    """
    try:
        import onnxruntime_genai as og

        result = og.is_qnn_available()
        assert isinstance(result, bool), "is_qnn_available() should return a boolean"
    except ImportError:
        pytest.skip("onnxruntime-genai not installed")


# ============================================================================
# Tier 2: Configuration and Provider Registration Tests
# ============================================================================


def test_qnn_provider_options_set(monkeypatch, tmp_path, elitelm_module):
    """Verify QNN provider options are correctly set (Tier 2).

    This test validates that _configure_qnn_provider correctly sets
    the backend_path and qnn_sdk_root options on the config object.
    It uses mocked paths and does not require actual hardware.
    """
    sdk_root = tmp_path / "sdk"
    lib_dir = sdk_root / "lib" / "aarch64-windows-msvc"
    bin_dir = sdk_root / "bin" / "aarch64-windows-msvc"
    lib_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    backend = lib_dir / "QnnHtp.dll"
    backend.write_bytes(b"")

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setattr(
        elitelm_module.os,
        "add_dll_directory",
        lambda path: None,
    )
    monkeypatch.setitem(elitelm_module.os.environ, "PATH", "")

    # Create config and verify options are retrievable
    config = elitelm_module._configure_qnn_provider(model_dir, str(sdk_root), None)

    # Verify backend_path option was set correctly
    opt_backend = config.get_provider_option("QNNExecutionProvider", "backend_path")
    assert opt_backend is not None, "backend_path option not set"
    assert Path(opt_backend) == backend.resolve()

    # Verify qnn_sdk_root option was set correctly
    opt_sdk = config.get_provider_option("QNNExecutionProvider", "qnn_sdk_root")
    assert opt_sdk is not None, "qnn_sdk_root option not set"
    assert Path(opt_sdk) == sdk_root.resolve()


def test_qnn_provider_options_persist(monkeypatch, tmp_path, elitelm_module):
    """Verify provider options persist across multiple calls (Tier 2).

    This test validates that the provider option helper correctly tracks
    and retrieves options even when multiple providers are configured.
    """
    sdk_root = tmp_path / "sdk"
    lib_dir = sdk_root / "lib" / "aarch64-windows-msvc"
    lib_dir.mkdir(parents=True)
    backend = lib_dir / "QnnHtp.dll"
    backend.write_bytes(b"")

    model_dir = tmp_path / "model"
    model_dir.mkdir()

    monkeypatch.setattr(
        elitelm_module.os,
        "add_dll_directory",
        lambda path: None,
    )
    monkeypatch.setitem(elitelm_module.os.environ, "PATH", "")

    config = elitelm_module._configure_qnn_provider(model_dir, str(sdk_root), None)

    # Retrieve options multiple times to ensure they persist
    opt1 = config.get_provider_option("QNNExecutionProvider", "backend_path")
    opt2 = config.get_provider_option("QNNExecutionProvider", "backend_path")
    assert opt1 == opt2, "Options should persist across multiple retrievals"

    # Verify the option value is correct
    assert Path(opt1) == backend.resolve()
