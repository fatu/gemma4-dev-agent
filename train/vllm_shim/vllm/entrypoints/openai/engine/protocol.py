# the six models the gemma4 parsers construct, copied from vllm 0.19.1 (OpenAIBaseModel → BaseModel)
from typing import Literal

from pydantic import BaseModel, Field

from vllm.entrypoints.chat_utils import make_tool_call_id


class FunctionCall(BaseModel):
    id: str | None = Field(default=None, exclude=True)
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=make_tool_call_id)
    type: Literal["function"] = "function"
    function: FunctionCall


class DeltaFunctionCall(BaseModel):
    name: str | None = None
    arguments: str | None = None


class DeltaToolCall(BaseModel):
    id: str | None = None
    type: Literal["function"] | None = None
    index: int
    function: DeltaFunctionCall | None = None


class ExtractedToolCallInformation(BaseModel):
    tools_called: bool
    tool_calls: list[ToolCall]
    content: str | None = None


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None
    tool_calls: list[DeltaToolCall] = Field(default_factory=list)
