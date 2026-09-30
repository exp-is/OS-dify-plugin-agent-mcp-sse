from typing import Generator

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
