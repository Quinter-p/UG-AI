# fix_v7_patch_files_string_v2.py
# 用法：
# 1. 放到 D:\ugai_agent_langgraph_v0
# 2. python fix_v7_patch_files_string_v2.py
# 3. python apply_v7_event_reflection_patch.py

from pathlib import Path
from datetime import datetime
import shutil

TARGET = Path("apply_v7_event_reflection_patch.py")

def main():
    if not TARGET.exists():
        raise FileNotFoundError("找不到 apply_v7_event_reflection_patch.py。请把本脚本放到项目根目录运行。")

    text = TARGET.read_text(encoding="utf-8")

    if "if isinstance(FILES, str):\n    FILES = json.loads(FILES)" in text:
        print("[OK] 已经修复过，无需重复修改。")
        return

    marker = "FILES=json.loads("
    pos = text.find(marker)
    if pos < 0:
        marker = "FILES = json.loads("
        pos = text.find(marker)

    if pos < 0:
        raise RuntimeError("找不到 FILES=json.loads(...) 锚点，无法自动修复。")

    # 找到 FILES=... 这一整行末尾
    line_end = text.find("\n", pos)
    if line_end < 0:
        raise RuntimeError("FILES 行格式异常，无法自动修复。")

    patch = "\nif isinstance(FILES, str):\n    FILES = json.loads(FILES)\n"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"apply_v7_event_reflection_patch_before_fix_v2_{stamp}.py.bak")
    shutil.copy2(TARGET, backup)
    print("[OK] 已备份：", backup)

    text = text[:line_end] + patch + text[line_end:]
    TARGET.write_text(text, encoding="utf-8")

    print("[OK] 已修复 apply_v7_event_reflection_patch.py")
    print("")
    print("现在运行：")
    print("python apply_v7_event_reflection_patch.py")

if __name__ == "__main__":
    main()
