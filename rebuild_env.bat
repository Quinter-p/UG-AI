@echo off
cd /d %~dp0
powershell -ExecutionPolicy Bypass -File .\rebuild_ugai_agent_env.ps1
pause
