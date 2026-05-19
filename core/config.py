import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.json not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def project_path(*parts):
    return ROOT.joinpath(*parts)
