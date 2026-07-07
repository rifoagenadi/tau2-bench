import pytest

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import (
    generate,
    set_preserve_thinking,
    to_litellm_messages,
)


@pytest.fixture(autouse=True)
def reset_preserve_thinking():
    set_preserve_thinking(None)
    yield
    set_preserve_thinking(None)


@pytest.fixture
def model() -> str:
    return "gpt-4o-mini"


@pytest.fixture
def messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(role="user", content="What is the capital of the moon?"),
    ]
    return messages


@pytest.fixture
def tool() -> Tool:
    def calculate_square(x: int) -> int:
        """Calculate the square of a number.
            Args:
            x (int): The number to calculate the square of.
        Returns:
            int: The square of the number.
        """
        return x * x

    return as_tool(calculate_square)


@pytest.fixture
def tool_call_messages() -> list[Message]:
    messages = [
        SystemMessage(role="system", content="You are a helpful assistant."),
        UserMessage(
            role="user",
            content="What is the square of 5? Just give me the number, no explanation.",
        ),
    ]
    return messages


def test_to_litellm_messages_omits_reasoning_content_by_default(monkeypatch):
    monkeypatch.delenv("TAU2_PRESERVE_THINKING", raising=False)
    monkeypatch.delenv("PRESERVE_THINKING", raising=False)
    message = AssistantMessage(
        role="assistant",
        content="Done",
        reasoning_content="Internal reasoning",
    )

    [litellm_message] = to_litellm_messages([message])

    assert "reasoning_content" not in litellm_message


def test_to_litellm_messages_includes_reasoning_content_when_enabled():
    set_preserve_thinking(True)
    message = AssistantMessage(
        role="assistant",
        content="Done",
        reasoning_content="Internal reasoning",
    )

    [litellm_message] = to_litellm_messages([message])

    assert litellm_message["reasoning_content"] == "Internal reasoning"


def test_to_litellm_messages_preserve_thinking_env(monkeypatch):
    monkeypatch.setenv("TAU2_PRESERVE_THINKING", "true")
    message = AssistantMessage(
        role="assistant",
        content="Done",
        reasoning_content="Internal reasoning",
    )

    [litellm_message] = to_litellm_messages([message])

    assert litellm_message["reasoning_content"] == "Internal reasoning"


def test_generate_no_tool_call(model: str, messages: list[Message]):
    response = generate(model, messages)
    assert isinstance(response, AssistantMessage)
    assert response.content is not None


def test_generate_tool_call(model: str, tool_call_messages: list[Message], tool: Tool):
    response = generate(model, tool_call_messages, tools=[tool])
    assert isinstance(response, AssistantMessage)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculate_square"
    assert response.tool_calls[0].arguments == {"x": 5}
    follow_up_messages = [
        response,
        ToolMessage(role="tool", id=response.tool_calls[0].id, content="25"),
    ]
    response = generate(
        model,
        tool_call_messages + follow_up_messages,
        tools=[tool],
    )
    assert isinstance(response, AssistantMessage)
    assert response.tool_calls is None
    assert response.content == "25"
