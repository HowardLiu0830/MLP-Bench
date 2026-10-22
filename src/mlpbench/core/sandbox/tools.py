from __future__ import annotations

__all__ = [
    "AGENTIC_BLOCKED",
    "claude_agentic_args",
]

AGENTIC_BLOCKED: tuple[str, ...] = (
    "Task", "Agent", "Workflow", "ScheduleWakeup",
    "CronCreate", "CronDelete", "CronList",
    "WebFetch", "WebSearch", "ToolSearch",
    "SendMessage", "RemoteTrigger", "PushNotification", "DesignSync",
    "EnterWorktree", "ExitWorktree", "Monitor",
    "ListAgents", "ReportFindings", "TaskStop",
)


def claude_agentic_args() -> list[str]:
    return [
        "--mcp-config", '{"mcpServers":{}}',
        "--strict-mcp-config",
        "--disallowedTools", *AGENTIC_BLOCKED,
    ]
