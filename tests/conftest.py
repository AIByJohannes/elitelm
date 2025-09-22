import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "llama3-qa.py"


def load_llama3_module():
    spec = importlib.util.spec_from_file_location("llama3_qa", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
