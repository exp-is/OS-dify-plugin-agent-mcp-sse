import fnmatch
import re
from typing import Any, Generator

from dify_plugin.entities.model import ModelFeature
from dify_plugin.entities.model.llm import LLMUsage
from dify_plugin.entities.model.message import PromptMessageContentType, PromptMessage
from dify_plugin.interfaces.agent import AgentModelConfig


class FilterHistoryMessageByModelFeaturesMixin:

    @staticmethod
    def _iter_cleanup_history_prompt_messages(model: AgentModelConfig) -> Generator[PromptMessage, None, None]:
        """
        remove history_prompt_message if model not support
        :param model
        :return:
        """
        for msg in model.history_prompt_messages:
            if isinstance(msg.content, list):
                filtered_content = [
                    item
                    for item in msg.content
                    if (
                            item.type == PromptMessageContentType.TEXT
                            or (item.type in {
                        PromptMessageContentType.IMAGE, PromptMessageContentType.VIDEO,
                        PromptMessageContentType.DOCUMENT,
                    } and ModelFeature.VISION in model.entity.features)
                            or (item.type == PromptMessageContentType.AUDIO and ModelFeature.AUDIO in model.entity.features)
                            or (item.type == PromptMessageContentType.VIDEO and ModelFeature.VIDEO in model.entity.features)
                            or (item.type == PromptMessageContentType.DOCUMENT and ModelFeature.DOCUMENT in model.entity.features)
                    )
                ]
                new_msg = msg.__class__(
                    role=msg.role,
                    content=filtered_content,
                    name=msg.name,
                )
                yield new_msg
            else:
                yield msg


def build_execution_metadata(usage: LLMUsage | None) -> dict:
    """
    Build the agent node's execution_metadata from the accumulated LLM usage.
    Dify rebuilds LLMUsage via LLMUsage.from_metadata(), which defaults every
    missing field to 0, so all fields must be sent (not only totals).
    """
    usage = usage or LLMUsage.empty_usage()
    return {
        "prompt_tokens": usage.prompt_tokens,
        "prompt_unit_price": float(usage.prompt_unit_price),
        "prompt_price_unit": float(usage.prompt_price_unit),
        "prompt_price": float(usage.prompt_price),
        "completion_tokens": usage.completion_tokens,
        "completion_unit_price": float(usage.completion_unit_price),
        "completion_price_unit": float(usage.completion_price_unit),
        "completion_price": float(usage.completion_price),
        "total_tokens": usage.total_tokens,
        "total_price": float(usage.total_price),
        "currency": usage.currency,
        "latency": usage.latency,
    }


# Tools whose name starts with one of these only read data, so they are not recorded as actions
READ_ONLY_TOOL_PREFIXES = (
    "list_", "get_", "search_", "read_", "find_", "fetch_", "query_", "describe_", "count_", "check_",
    "resource__", "prompt__",
)
# Arguments that identify what an action was about, in order of preference
ACTION_LABEL_KEYS = ("name", "title", "displayName", "label", "slug", "id")
TOOL_ERROR_PREFIXES = ("tool invoke error", "there is not a tool")


def is_tool_error(result: Any) -> bool:
    return isinstance(result, str) and result.startswith(TOOL_ERROR_PREFIXES)


def format_action(tool_name: str, tool_args: Any, failed: bool) -> str | None:
    """
    One line describing a tool call that changed something, or None for read-only tools.
    Only a short identifying argument is kept, never the full arguments.
    """
    if tool_name.startswith(READ_ONLY_TOOL_PREFIXES):
        return None
    label = ""
    if isinstance(tool_args, dict):
        for key in ACTION_LABEL_KEYS:
            value = tool_args.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                value = str(value).strip().replace("\n", " ")
                label = f' "{value[:40]}{"…" if len(value) > 40 else ""}"'
                break
    return f"{tool_name}{label} {'✗ failed' if failed else '✓'}"


def build_actions_summary(actions: list[str]) -> str:
    """
    Block appended to the final answer. Dify's conversation memory keeps only the answer text,
    not tool calls, so without it the next turn cannot tell which actions already happened.
    """
    if not actions:
        return ""
    lines = "\n".join(f"- {action}" for action in actions)
    return f"\n\n<details>\n<summary>Actions done</summary>\n\n{lines}\n\n</details>"


def _patterns(value: str | None) -> list[str]:
    return [p.strip() for p in re.split(r"[,\n]", value or "") if p.strip()]


def filter_mcp_tools(tools: list[dict], include: str | None, exclude: str | None) -> list[dict]:
    """
    Keep only the MCP tools whose name matches an include pattern (all when empty) and no exclude
    pattern. Patterns are comma or newline separated shell-style globs, e.g. "list_*, create_plan".
    Every LLM round sends all tool definitions, so fewer tools means smaller, faster requests.
    """
    include_patterns, exclude_patterns = _patterns(include), _patterns(exclude)
    return [
        tool for tool in tools
        if (not include_patterns or any(fnmatch.fnmatchcase(tool["name"], p) for p in include_patterns))
        and not any(fnmatch.fnmatchcase(tool["name"], p) for p in exclude_patterns)
    ]


def truncate_tool_result(result: Any, max_chars: Any) -> Any:
    """
    Cap a tool result sent back to the model. The result stays in the context for every later
    round of the turn, so one large result slows down and costs every following LLM call.
    """
    try:
        max_chars = int(max_chars or 0)  # an empty number field can arrive as "" or a float
    except (TypeError, ValueError):
        max_chars = 0
    if max_chars <= 0 or not isinstance(result, str) or len(result) <= max_chars:
        return result
    return (
        f"{result[:max_chars]}\n\n[truncated: showing {max_chars} of {len(result)} characters. "
        f"Ask for a smaller page or a more specific query if you need the rest.]"
    )
