from typing import Any

from pydantic import BaseModel


class ChatCompletionToolsParam(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class ChatCompletionNamedToolChoiceParam(BaseModel):
    type: str = "function"
    function: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    model: str = "gemma4"
    messages: list[Any] = []
    tools: list[ChatCompletionToolsParam] | None = None
    tool_choice: Any = "auto"
    skip_special_tokens: bool = True
