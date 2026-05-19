# fix_v7_patch_files_dict.py
# 用法：
# 1. 把本文件放到 D:\ugai_agent_langgraph_v0
# 2. python fix_v7_patch_files_dict.py
# 3. 再运行 python apply_v7_event_reflection_patch.py

from pathlib import Path
from datetime import datetime
import shutil

TARGET = Path("apply_v7_event_reflection_patch.py")

def main():
    if not TARGET.exists():
        raise FileNotFoundError("找不到 apply_v7_event_reflection_patch.py。请把本脚本放到项目根目录运行。")

    text = TARGET.read_text(encoding="utf-8")

    marker = "def write_clean"
    inject = '''
# v7 patch hotfix:
# 某些版本生成时 FILES 会变成字符串，这里转回 dict。
import ast
if isinstance(FILES, str):
    FILES = ast.literal_eval(FILES)

'''

    if "if isinstance(FILES, str):" in text:
        print("[OK] 已经修复过，无需重复修改。")
        return

    if marker not in text:
        raise RuntimeError("找不到 def write_clean 锚点，无法自动修复。")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = TARGET.with_name(f"apply_v7_event_reflection_patch_before_fix_{stamp}.py.bak")
    shutil.copy2(TARGET, backup)
    print("[OK] 已备份：", backup)

    text = text.replace(marker, inject + "\\n" + marker, 1)
    TARGET.write_text(text, encoding="utf-8")
    print("[OK] 已修复 apply_v7_event_reflection_patch.py")
    print("")
    print("现在重新运行：")
    print("python apply_v7_event_reflection_patch.py")

if __name__ == "__main__":
    main()
