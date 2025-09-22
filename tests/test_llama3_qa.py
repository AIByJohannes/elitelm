import os
from pathlib import Path
import pytest
import importlib.util

MODULE_PATH = Path(__file__).resolve().parents[1] / "llama3-qa.py"


def load_module():
    spec = importlib.util.spec_from_file_location("llama3_qa", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def llama3_module(monkeypatch):
    module = load_module()
    # Ensure globals start clean for each test
    module._DLL_HANDLES.clear()
    module._REGISTERED_DLL_PATHS.clear()
    return module


def test_add_dll_dir_registers_once(monkeypatch, tmp_path, llama3_module):
    added = []

    def fake_add_dll_directory(path):
        added.append(path)
        return f"handle-{path}"

    monkeypatch.setattr(llama3_module.os, "add_dll_directory", fake_add_dll_directory)
    monkeypatch.setitem(llama3_module.os.environ, "PATH", "")

    dll_dir = tmp_path / "lib"
    dll_dir.mkdir()

    llama3_module._add_dll_dir(dll_dir)
    llama3_module._add_dll_dir(dll_dir)  # second call should be ignored

    assert added == [str(dll_dir.resolve())]
    path_env = llama3_module.os.environ["PATH"].split(";")
    assert str(dll_dir.resolve()) in path_env


def test_resolve_qnn_sdk_root_prefers_argument(tmp_path, llama3_module):
    provided = tmp_path / "sdk"
    provided.mkdir()

    result = llama3_module._resolve_qnn_sdk_root(str(provided))
    assert result == provided.resolve()


def test_resolve_qnn_sdk_root_falls_back_to_env(monkeypatch, tmp_path, llama3_module):
    env_path = tmp_path / "env-sdk"
    env_path.mkdir()
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setenv("QNN_SDK_ROOT", str(env_path))

    result = llama3_module._resolve_qnn_sdk_root(None)
    assert result == env_path.resolve()


def test_resolve_qnn_sdk_root_detects_qairt(monkeypatch, tmp_path, llama3_module):
    repo_root = tmp_path / "repo"
    qairt_dir = repo_root / "qairt" / "3.0.0"
    qairt_dir.mkdir(parents=True)
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    monkeypatch.setattr(llama3_module, "__file__", str(repo_root / "llama3-qa.py"))

    result = llama3_module._resolve_qnn_sdk_root(None)
    assert result == qairt_dir.resolve()


def test_default_backend_path_prefers_aarch64(tmp_path, llama3_module):
    sdk_root = tmp_path / "sdk"
    aarch_dir = sdk_root / "lib" / "aarch64-windows-msvc"
    aarch_dir.mkdir(parents=True)
    (aarch_dir / "QnnHtp.dll").write_bytes(b"")

    other_dir = sdk_root / "lib" / "arm64x-windows-msvc"
    other_dir.mkdir(parents=True)
    (other_dir / "QnnHtp.dll").write_bytes(b"")

    backend, arch = llama3_module._default_backend_path(sdk_root)
    assert arch == "aarch64-windows-msvc"
    assert backend == (aarch_dir / "QnnHtp.dll").resolve()


def test_configure_qnn_provider_sets_options(monkeypatch, tmp_path, llama3_module):
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
        llama3_module.os,
        "add_dll_directory",
        lambda path: added.append(Path(path))
    )
    monkeypatch.setitem(llama3_module.os.environ, "PATH", "")

    config = llama3_module._configure_qnn_provider(model_dir, str(sdk_root), None)
    opt_backend = config.get_provider_option("QNNExecutionProvider", "backend_path")
    opt_sdk = config.get_provider_option("QNNExecutionProvider", "qnn_sdk_root")

    assert opt_backend == backend.name
    assert Path(opt_sdk) == sdk_root.resolve()
    assert set(p.resolve() for p in added) == {lib_dir.resolve(), bin_dir.resolve()}

    # repeated call should not duplicate paths
    llama3_module._configure_qnn_provider(model_dir, str(sdk_root), None)
    assert len(added) == 2


def test_add_dll_dir_noop_for_missing_path(monkeypatch, tmp_path, llama3_module):
    missing = tmp_path / "nope"
    monkeypatch.setitem(llama3_module.os.environ, "PATH", "")
    monkeypatch.setattr(llama3_module.os, "add_dll_directory", lambda path: (_ for _ in ()).throw(RuntimeError("should not be called")))

    llama3_module._add_dll_dir(missing)
    assert llama3_module.os.environ["PATH"] == ""



def test_load_yaml_config_defaults(tmp_path, llama3_module):
    config_file = tmp_path / 'config.yaml'
    config_file.write_text("""model: ./models/example
generation:
  max_length: 512
runtime:
  verbose: true
""")

    config = llama3_module._load_yaml_config(config_file)
    assert config.model == './models/example'
    assert config.device == 'cpu'
    assert config.do_sample is False
    assert config.verbose is True
    assert config.max_length == 512
    assert not hasattr(config, 'min_length')



def test_load_yaml_config_requires_model(tmp_path, llama3_module):
    config_file = tmp_path / 'config.yaml'
    config_file.write_text("""runtime:
  timings: true
""")

    with pytest.raises(ValueError):
        llama3_module._load_yaml_config(config_file)
