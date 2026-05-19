from datetime import datetime
from core.config import project_path

def log(tag, message):
    log_dir = project_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = log_dir / f"{day}_agent.txt"
    text = f"[{now}] {tag}:\n{message}\n\n"
    path.open("a", encoding="utf-8").write(text)
    print(text, end="")
