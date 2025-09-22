import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "elitelm.py"


def load_elitelm_module():
    spec = importlib.util.spec_from_file_location("elitelm", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
