import subprocess
import shutil
import sys
import fnmatch
import os
from pathlib import Path


# =========================
# 你只需要改这里
# =========================

SOURCE_DIR = Path(r"D:\ai-quinter")
CLEAN_DIR = Path(r"D:\ai-quinter_clean_upload")

REMOTE_URL = "https://github.com/Quinter-p/UG-AI"
COMMIT_MESSAGE = "Clean project upload"


# =========================
# 不复制到 GitHub 的内容
# =========================

EXCLUDE_PATTERNS = [
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".idea",
    ".vscode",

    "*.pyc",
    "*.pyo",
    "*.pyd",

    "*.log",
    "logs",

    "record.wav",
    "*.wav",
    "*.mp3",

    "memory.json",
    "chat_memory.json",

    "*.db",

    "models",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.safetensors",
]


GITIGNORE_CONTENT = """
# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# Virtual environments
.venv/
venv/
env/

# IDE
.idea/
.vscode/

# Secrets
.env
config.json
secrets.json

# Logs
logs/
*.log

# Audio files
record.wav
*.wav
*.mp3

# Memory / private data
memory.json
chat_memory.json
*.db

# Large model files
models/
*.bin
*.pt
*.pth
*.onnx
*.safetensors

# OS
.DS_Store
Thumbs.db
""".strip()


def run(cmd, cwd=None, check=True):
    print(f"\n>>> {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        encoding="utf-8",
        errors="ignore",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False
    )

    if result.stdout:
        print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if check and result.returncode != 0:
        print(f"\n[ERROR] 命令失败：{' '.join(cmd)}")
        sys.exit(result.returncode)

    return result


def should_exclude(path: Path):
    name = path.name

    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(name, pattern):
            return True

        relative = str(path).replace("\\", "/")
        if fnmatch.fnmatch(relative, pattern):
            return True

    return False


def remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, 0o777)
        func(path)
    except Exception as e:
        print(f"[WARN] 无法删除：{path}")
        print(e)


def prepare_clean_folder():
    if CLEAN_DIR.exists():
        print(f"\n[INFO] 正在删除旧的干净副本：{CLEAN_DIR}")
        shutil.rmtree(CLEAN_DIR, onerror=remove_readonly)

    print(f"\n[INFO] 正在创建干净副本：{CLEAN_DIR}")
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def copy_project_files():
    print("\n[INFO] 正在复制项目文件，自动排除 .git、.venv、日志、录音、记忆文件...")

    for item in SOURCE_DIR.iterdir():
        if should_exclude(item):
            print(f"[SKIP] {item.name}")
            continue

        target = CLEAN_DIR / item.name

        if item.is_dir():
            shutil.copytree(
                item,
                target,
                ignore=lambda folder, names: [
                    name for name in names
                    if should_exclude(Path(folder) / name)
                ]
            )
        else:
            shutil.copy2(item, target)

    print("\n[OK] 项目文件复制完成")


def write_gitignore():
    gitignore_path = CLEAN_DIR / ".gitignore"
    gitignore_path.write_text(GITIGNORE_CONTENT, encoding="utf-8")
    print("\n[OK] 已写入 .gitignore")


def check_git():
    result = run(["git", "--version"], check=False)
    if result.returncode != 0:
        print("\n[ERROR] 没检测到 Git，请先安装 Git。")
        sys.exit(1)


def init_and_commit():
    run(["git", "init"], cwd=CLEAN_DIR)
    run(["git", "branch", "-M", "main"], cwd=CLEAN_DIR)
    run(["git", "remote", "add", "origin", REMOTE_URL], cwd=CLEAN_DIR)

    run(["git", "add", "."], cwd=CLEAN_DIR)

    status = run(["git", "status", "--porcelain"], cwd=CLEAN_DIR, check=False)

    if not status.stdout.strip():
        print("\n[ERROR] 干净副本里没有可提交文件。")
        sys.exit(1)

    run(["git", "commit", "-m", COMMIT_MESSAGE], cwd=CLEAN_DIR)


def push_to_github(yes):
    print("\n[WARN] 接下来会用干净副本覆盖 GitHub 仓库 main 分支。")
    print("[WARN] 这样可以清掉之前错误上传的 .venv / python.exe 等内容。")

    confirm = yes

    if confirm != yes:
        print("\n[CANCEL] 已取消上传。")
        sys.exit(0)

    run(["git", "fetch", "origin", "main"], cwd=CLEAN_DIR, check=False)
    run(["git", "push", "-u", "origin", "main", "--force-with-lease"], cwd=CLEAN_DIR)

    print("\n[SUCCESS] 已经从干净副本上传到 GitHub。")
    print(f"[INFO] 干净副本位置：{CLEAN_DIR}")


def main():
    print("=" * 60)
    print("GitHub 干净副本上传脚本")
    print(f"原项目目录：{SOURCE_DIR}")
    print(f"干净副本目录：{CLEAN_DIR}")
    print(f"远程仓库：{REMOTE_URL}")
    print("=" * 60)

    a = 1
    check_git()
    prepare_clean_folder()
    copy_project_files()
    write_gitignore()
    init_and_commit()
    push_to_github(a)


if __name__ == "__main__":
    main()