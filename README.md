# UGAI Agent LangGraph v0

这是从零重建的 **Agent Core** 版，不是旧项目继续打补丁。

## 当前目标

先做一个最小但结构正确的智能体：

```text
QQ/NapCat 入口
→ LangGraph Agent Core
→ 自动反应 / 命令 / 情绪 / 模型表达
→ QQ 单一出口回复
```

## 当前包含

```text
✅ QQ/NapCat/OneBot 接入
✅ LangGraph 主流程
✅ Ollama 本地模型
✅ 身份识别
✅ 基础情绪状态
✅ 群聊唤醒
✅ 群聊自动反应：沐/木/母/mu -> 哈！！
✅ /help
✅ /status
✅ 单一回复出口
```

## 当前不包含

```text
❌ 长期记忆
❌ RAG
❌ 设定审核
❌ 主动提醒
❌ 工具调用
❌ 语音
```

## 安装

```powershell
cd /d D:\ugai_agent_langgraph_v0
pip install -r requirements.txt
```

## NapCat 设置

HTTP Server:

```text
Host: 127.0.0.1
Port: 3000
Token: 空
```

HTTP Client:

```text
启用：开
URL: http://127.0.0.1:8765/onebot
messagePostFormat: string
Token: 空
reportSelfMessage: false
```

## 启动

```powershell
python main_qq.py
```

## 测试

私聊：

```text
你好
/help
/status
```

群聊唤醒：

```text
/ugai 你好
昆特 你好
器灵 你好
```

群聊无需 @ 的自动反应：

```text
沐
木
母
黑沐来了
```

应回复：

```text
哈！！
```

## LangGraph 流程

```text
parse_event
  ↓
identity
  ↓
reflex
  ├─ auto_reply → END
  ↓
command
  ├─ command_reply → END
  ↓
wakeup
  ├─ ignore → END
  ↓
emotion
  ↓
prompt
  ↓
expression
  ↓
output_filter
  ↓
END
```

Graph 只生成 `final_reply`，真正发送 QQ 消息只在 OneBot adapter 里做一次。
