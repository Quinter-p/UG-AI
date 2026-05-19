import json
from datetime import datetime
from core.config import project_path

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_json(rel_path, default):
    path = project_path(rel_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = dict(default)
        data["created_at"] = now()
        data["updated_at"] = now()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return dict(default)

def write_json(rel_path, data):
    path = project_path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
