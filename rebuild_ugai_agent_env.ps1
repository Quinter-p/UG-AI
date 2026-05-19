# rebuild_ugai_agent_env.ps1
# 用法：
# 1) 把本脚本放到 D:\ugai_agent_langgraph_v0
# 2) PowerShell 进入该目录
# 3) 运行：
#    powershell -ExecutionPolicy Bypass -File .\rebuild_ugai_agent_env.ps1

$ErrorActionPreference = "Stop"

Write-Host "=== UGAI Agent environment rebuild ===" -ForegroundColor Cyan

if (!(Test-Path ".\config.json")) {
    Write-Host "[ERROR] 当前目录没有 config.json。请把脚本放到 D:\ugai_agent_langgraph_v0 根目录运行。" -ForegroundColor Red
    exit 1
}

# 1. 找正常 Python，不用 Lumerical embedded Python
$pythonCmd = $null

try {
    $ver = & py -3 --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pythonCmd = "py -3"
        Write-Host "[OK] 使用 Python Launcher: $ver"
    }
} catch {}

if ($null -eq $pythonCmd) {
    try {
        $ver = & python --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonPath = (Get-Command python).Source
            if ($pythonPath -like "*Lumerical*") {
                Write-Host "[ERROR] 当前 python 指向 Lumerical embedded Python，不建议使用：" -ForegroundColor Red
                Write-Host $pythonPath -ForegroundColor Yellow
                Write-Host "请安装正常 Python 3.10/3.11，并勾选 Add python.exe to PATH。" -ForegroundColor Yellow
                exit 1
            }
            $pythonCmd = "python"
            Write-Host "[OK] 使用 python: $ver"
        }
    } catch {}
}

if ($null -eq $pythonCmd) {
    Write-Host "[ERROR] 找不到正常 Python。请安装 Python 3.10 或 3.11。" -ForegroundColor Red
    exit 1
}

# 2. 删除旧 .venv
if (Test-Path ".\.venv") {
    Write-Host "[INFO] 删除旧 .venv ..."
    Remove-Item ".\.venv" -Recurse -Force
}

# 3. 创建新 .venv
Write-Host "[INFO] 创建新虚拟环境 .venv ..."
Invoke-Expression "$pythonCmd -m venv .venv"

# 4. 使用虚拟环境 Python
$venvPython = ".\.venv\Scripts\python.exe"

if (!(Test-Path $venvPython)) {
    Write-Host "[ERROR] 虚拟环境 Python 创建失败。" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] 升级 pip ..."
& $venvPython -m pip install --upgrade pip

Write-Host "[INFO] 安装 requirements.txt ..."
& $venvPython -m pip install -r requirements.txt

# 5. 修复 v1 缩进/编码，如果脚本存在
if (Test-Path ".\apply_v1_ascii_indent_fix.py") {
    Write-Host "[INFO] 运行 v1 ASCII/indent fix ..."
    & $venvPython .\apply_v1_ascii_indent_fix.py
} elseif (Test-Path ".\apply_v1_fix_indentation.py") {
    Write-Host "[INFO] 运行 v1 indentation fix ..."
    & $venvPython .\apply_v1_fix_indentation.py
}

# 6. 语法检查
Write-Host "[INFO] 语法检查 ..."
& $venvPython -m py_compile .\main_qq.py
Get-ChildItem -Path .\core, .\adapters, .\storage -Filter *.py -Recurse | ForEach-Object {
    & $venvPython -m py_compile $_.FullName
}

Write-Host ""
Write-Host "[DONE] 环境重建完成。" -ForegroundColor Green
Write-Host "启动方式：" -ForegroundColor Cyan
Write-Host ".\.venv\Scripts\activate"
Write-Host "python main_qq.py"
Write-Host ""
Write-Host "或者直接运行：" -ForegroundColor Cyan
Write-Host ".\.venv\Scripts\python.exe .\main_qq.py"
