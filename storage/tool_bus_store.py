import json
from datetime import datetime
from core.config import project_path


DEFAULT_REGISTRY = {
    "schema_version": 3,
    "created_at": "",
    "updated_at": "",
    "approval_policy": {
        "enabled": True,
        "auto_approve_low_risk": False,
        "block_high_risk_when_disabled": True,
        "allowed_risks": ["low", "medium", "high"],
        "owner_only_approval": True
    },
    "tools": [
        {
            "name": "task_registry",
            "description": "查看、登记、完成或取消任务。v8.3 支持审批后执行。",
            "enabled": True,
            "requires_approval": True,
            "risk": "low",
            "executable": True,
            "args_schema": {
                "action": "list/add/done/cancel",
                "task_id": "optional",
                "title": "optional",
                "description": "optional"
            }
        },
        {
            "name": "memory_write",
            "description": "写入事实记忆或长期记忆。v8.3 支持审批后执行。",
            "enabled": True,
            "requires_approval": True,
            "risk": "medium",
            "executable": True,
            "args_schema": {
                "memory_type": "fact/long",
                "content": "string",
                "evidence_event_id": "optional"
            }
        },
        {
            "name": "local_shell",
            "description": "本地命令行工具，占位。v8.3 不执行。",
            "enabled": False,
            "requires_approval": True,
            "risk": "high",
            "executable": False,
            "args_schema": {
                "command": "string"
            }
        },
        {
            "name": "browser",
            "description": "浏览器/网页工具，占位。v8.3 不执行。",
            "enabled": False,
            "requires_approval": True,
            "risk": "medium",
            "executable": False,
            "args_schema": {
                "url_or_query": "string",
                "action": "open/search/read"
            }
        }
    ]
}


DEFAULT_RUNTIME = {
    "schema_version": 3,
    "created_at": "",
    "updated_at": "",
    "next_call_id": 1,
    "calls": []
}


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def deep_copy(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False))


def read_json(path, default):
    full = project_path(path)

    if not full.exists():
        full.parent.mkdir(parents=True, exist_ok=True)
        data = deep_copy(default)
        data["created_at"] = now_text()
        data["updated_at"] = now_text()
        full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return data

    try:
        data = json.loads(full.read_text(encoding="utf-8"))
    except Exception:
        data = deep_copy(default)

    for k, v in default.items():
        data.setdefault(k, v)

    if "approval_policy" not in data and "approval_policy" in default:
        data["approval_policy"] = deep_copy(default["approval_policy"])

    if "tools" in data:
        existing = {t.get("name"): t for t in data.get("tools", []) if t.get("name")}
        for t in default.get("tools", []):
            name = t.get("name")
            if name and name not in existing:
                data.setdefault("tools", []).append(t)
            elif name in existing:
                # 温和迁移：补字段，不覆盖用户 enabled/risk 设置。
                for key, val in t.items():
                    existing[name].setdefault(key, val)

    return data


def write_json(path, data):
    full = project_path(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    data["updated_at"] = now_text()
    full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def next_call_id(data):
    value = int(data.get("next_call_id", 1))
    data["next_call_id"] = value + 1
    return value


class ToolBusStore:
    def __init__(
        self,
        registry_file="memory_runtime/tool_registry.json",
        runtime_file="memory_runtime/tool_calls.json",
    ):
        self.registry_file = registry_file
        self.runtime_file = runtime_file

    def registry(self):
        return read_json(self.registry_file, DEFAULT_REGISTRY)

    def runtime(self):
        return read_json(self.runtime_file, DEFAULT_RUNTIME)

    def save_registry(self, data):
        write_json(self.registry_file, data)

    def save_runtime(self, data):
        write_json(self.runtime_file, data)

    def policy(self):
        data = self.registry()
        policy = data.get("approval_policy") or {}
        default = DEFAULT_REGISTRY["approval_policy"]
        for k, v in default.items():
            policy.setdefault(k, v)
        return policy

    def format_policy(self):
        p = self.policy()
        return (
            "【工具审批策略】\n"
            f"enabled：{p.get('enabled')}\n"
            f"auto_approve_low_risk：{p.get('auto_approve_low_risk')}\n"
            f"block_high_risk_when_disabled：{p.get('block_high_risk_when_disabled')}\n"
            f"allowed_risks：{','.join(p.get('allowed_risks', []))}\n"
            f"owner_only_approval：{p.get('owner_only_approval')}\n"
            "\n当前阶段：只有 task_registry / memory_write 支持审批后执行。"
        )

    def list_tools(self):
        data = self.registry()
        tools = data.get("tools", [])

        if not tools:
            return "暂无工具注册。"

        lines = ["【工具注册表】"]
        for t in tools:
            enabled = "ON" if t.get("enabled") else "OFF"
            approval = "approval" if t.get("requires_approval", True) else "auto"
            executable = "exec" if t.get("executable") else "no-exec"
            lines.append(
                f"- {t.get('name')} [{enabled}/{approval}/{executable}/risk={t.get('risk', 'unknown')}]\n"
                f"  {t.get('description', '')}"
            )

        return "\n".join(lines)

    def find_tool(self, name):
        name = str(name or "").strip()
        data = self.registry()
        for t in data.get("tools", []):
            if t.get("name") == name:
                return t
        return None

    def set_tool_enabled(self, name, enabled):
        data = self.registry()
        name = str(name or "").strip()
        for t in data.get("tools", []):
            if t.get("name") == name:
                t["enabled"] = bool(enabled)
                self.save_registry(data)
                return t
        return None

    def set_tool_risk(self, name, risk):
        risk = str(risk or "").strip().lower()
        if risk not in ["low", "medium", "high"]:
            raise ValueError("risk must be low/medium/high")

        data = self.registry()
        name = str(name or "").strip()
        for t in data.get("tools", []):
            if t.get("name") == name:
                t["risk"] = risk
                self.save_registry(data)
                return t
        return None

    def add_call(
        self,
        tool_name,
        args=None,
        reason="",
        proposed_by="agent",
        source_text="",
        trace_id="",
        requires_approval=True,
    ):
        tool_name = str(tool_name or "").strip()
        tool = self.find_tool(tool_name)
        policy = self.policy()

        data = self.runtime()

        if not tool:
            status = "rejected"
            error = "unknown_tool"
            risk = "unknown"
            requires_approval = True
            executable = False
        else:
            enabled = bool(tool.get("enabled", False))
            risk = str(tool.get("risk", "unknown")).lower()
            executable = bool(tool.get("executable", False))
            tool_requires_approval = bool(tool.get("requires_approval", True))
            requires_approval = bool(requires_approval) or tool_requires_approval or bool(policy.get("enabled", True))
            status = "pending_approval" if requires_approval else "pending"
            error = ""

            if risk not in policy.get("allowed_risks", ["low", "medium", "high"]):
                status = "blocked"
                error = "risk_not_allowed"

            if not enabled:
                status = "blocked"
                error = "tool_disabled"

            if enabled and not executable:
                status = "blocked"
                error = "tool_not_executable_in_v8_3"

            if (
                enabled
                and executable
                and policy.get("enabled", True)
                and risk == "low"
                and policy.get("auto_approve_low_risk", False)
                and not tool_requires_approval
            ):
                status = "approved"
                requires_approval = False

        item = {
            "id": next_call_id(data),
            "tool_name": tool_name,
            "args": args or {},
            "reason": str(reason or "").strip(),
            "status": status,
            "error": error,
            "risk": risk,
            "executable": bool(executable),
            "requires_approval": bool(requires_approval),
            "proposed_by": str(proposed_by or "agent"),
            "source_text": str(source_text or "").strip(),
            "trace_id": str(trace_id or "").strip(),
            "created_at": now_text(),
            "updated_at": now_text(),
            "approved_at": "",
            "approved_by": "",
            "completed_at": "",
            "cancelled_at": "",
            "result": "",
        }

        data.setdefault("calls", []).append(item)
        self.save_runtime(data)
        return item

    def get_call(self, call_id):
        data = self.runtime()
        target = str(call_id).strip()

        for item in data.get("calls", []):
            if str(item.get("id")) == target:
                return item

        return None

    def update_call(self, call_id, status, result="", actor=""):
        status = str(status or "").strip()
        allowed = ["pending_approval", "pending", "approved", "done", "cancelled", "failed", "blocked", "rejected"]
        if status not in allowed:
            raise ValueError("invalid status")

        data = self.runtime()
        target = str(call_id).strip()

        for item in data.get("calls", []):
            if str(item.get("id")) == target:
                item["status"] = status
                item["updated_at"] = now_text()

                if status == "approved":
                    item["approved_at"] = now_text()
                    item["approved_by"] = str(actor or "")
                if status == "done":
                    item["completed_at"] = now_text()
                if status in ["cancelled", "rejected"]:
                    item["cancelled_at"] = now_text()
                if result:
                    item["result"] = str(result)

                self.save_runtime(data)
                return item

        return None

    def approve_call(self, call_id, actor=""):
        item = self.get_call(call_id)
        if not item:
            return None, "not_found"

        if item.get("status") == "blocked":
            return item, "blocked_call_cannot_be_approved"

        if item.get("status") in ["done", "cancelled", "rejected"]:
            return item, "call_already_closed"

        tool = self.find_tool(item.get("tool_name"))
        if not tool:
            return item, "unknown_tool"

        if not tool.get("enabled", False):
            return item, "tool_disabled"

        if not tool.get("executable", False):
            return item, "tool_not_executable_in_v8_3"

        return self.update_call(call_id, "approved", actor=actor), ""

    def reject_call(self, call_id, actor="", reason=""):
        item = self.get_call(call_id)
        if not item:
            return None, "not_found"

        if item.get("status") in ["done", "cancelled", "rejected"]:
            return item, "call_already_closed"

        return self.update_call(call_id, "rejected", result=reason, actor=actor), ""

    def list_calls(self, status="open", limit=20):
        data = self.runtime()
        calls = data.get("calls", [])

        status = str(status or "open").strip().lower()

        if status == "open":
            calls = [x for x in calls if x.get("status") in ["pending_approval", "pending", "approved", "blocked", "failed"]]
        elif status != "all":
            calls = [x for x in calls if str(x.get("status", "")).lower() == status]

        calls = calls[-int(limit):]

        if not calls:
            return f"暂无 {status} 工具调用。"

        lines = ["【工具调用记录】"]
        for c in calls:
            args_text = json.dumps(c.get("args", {}), ensure_ascii=False)
            if len(args_text) > 100:
                args_text = args_text[:100] + "..."
            reason = c.get("reason", "")
            if len(reason) > 100:
                reason = reason[:100] + "..."
            err = f" error={c.get('error')}" if c.get("error") else ""
            lines.append(
                f"#{c.get('id')} {c.get('tool_name')} "
                f"[{c.get('status')}/risk={c.get('risk')}/exec={c.get('executable')}{err}]\n"
                f"  reason: {reason}\n"
                f"  args: {args_text}"
            )

        return "\n".join(lines)

    def retrieve_open_calls(self, limit=5):
        data = self.runtime()
        calls = [
            x for x in data.get("calls", [])
            if x.get("status") in ["pending_approval", "pending", "approved", "blocked", "failed"]
        ]
        return calls[-int(limit):]

    def format_for_prompt(self, calls):
        if not calls:
            return "暂无待处理工具请求。"

        lines = []
        for c in calls:
            lines.append(
                f"- [工具请求#{c.get('id')}] {c.get('tool_name')} "
                f"状态={c.get('status')} 风险={c.get('risk')} 可执行={c.get('executable')} 原因={c.get('reason', '')}"
            )
            if c.get("error"):
                lines.append(f"  阻塞/失败原因：{c.get('error') or c.get('result')}")
            elif c.get("status") == "failed":
                lines.append(f"  失败结果：{c.get('result')}")
            args_text = json.dumps(c.get("args", {}), ensure_ascii=False)
            if len(args_text) > 200:
                args_text = args_text[:200] + "..."
            lines.append(f"  参数：{args_text}")

        return "\n".join(lines)
