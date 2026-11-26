import unittest
from unittest.mock import Mock, patch

from dify_plugin.entities.agent import AgentRuntime
from dify_plugin.entities.model.llm import LLMResult, LLMUsage
from dify_plugin.entities.model.message import AssistantPromptMessage

from strategies.function_calling import FunctionCallingAgentStrategy


class McpToolAllowlistTest(unittest.TestCase):
    def setUp(self):
        self.discovered = [
            {"name": "list_documents"},
            {"name": "create_document"},
            {"name": "other__list_documents"},
            {"name": "get_document"},
        ]
        self.filter_tools = FunctionCallingAgentStrategy._filter_mcp_tools

    def test_blank_settings_keep_the_existing_tool_list(self):
        for value in (None, "", " "):
            with self.subTest(value=value):
                self.assertEqual(self.filter_tools(self.discovered, value), self.discovered)

    def test_keeps_exact_names_in_discovery_order(self):
        self.assertEqual(
            self.filter_tools(self.discovered, "get_document, other__list_documents, get_document"),
            [{"name": "other__list_documents"}, {"name": "get_document"}],
        )

    def test_typo_cannot_silently_expose_the_full_list(self):
        with self.assertRaisesRegex(ValueError, "get_documents"):
            self.filter_tools(self.discovered, "list_documents, get_documents")

    def test_rejects_empty_entries_in_a_list(self):
        with self.assertRaisesRegex(ValueError, "empty tool name"):
            self.filter_tools(self.discovered, "list_documents,")

    @patch("strategies.function_calling.McpClients")
    def test_only_allowlisted_tools_reach_the_model_and_dispatch(self, client_class):
        client = client_class.return_value
        client.fetch_tools.return_value = [
            {**tool, "inputSchema": {"type": "object"}} for tool in self.discovered
        ]
        client.execute_tool.return_value = [{"type": "text", "text": "Documents"}]
        session = Mock()
        session.model.llm.invoke.return_value = LLMResult(
            model="test-model",
            usage=LLMUsage.empty_usage(),
            message=AssistantPromptMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call-{name}",
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                    for name in ("list_documents", "create_document")
                ],
            ),
        )
        strategy = FunctionCallingAgentStrategy(AgentRuntime(user_id=None), session)

        list(
            strategy._invoke(
                {
                    "query": "List my documents",
                    "instruction": None,
                    "model": {
                        "model": "test-model",
                        "model_type": "llm",
                        "provider": "test",
                        "mode": "chat",
                    },
                    "tools": None,
                    "mcp_servers_config": "{}",
                    "mcp_allowed_tools": "list_documents",
                    "maximum_iterations": 1,
                }
            )
        )

        sent_tools = session.model.llm.invoke.call_args.kwargs["tools"]
        self.assertEqual([tool.name for tool in sent_tools], ["list_documents"])
        client.execute_tool.assert_called_once_with(tool_name="list_documents", tool_args={})
        client.close.assert_called_once()
