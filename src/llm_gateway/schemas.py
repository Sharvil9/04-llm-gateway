from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=16_000)]


class ChatRequest(StrictModel):
    model: Annotated[str, Field(min_length=1, max_length=128)]
    messages: Annotated[list[Message], Field(min_length=1, max_length=32)]
    temperature: Annotated[float, Field(ge=0, le=2)] = 0.0
    top_p: Annotated[float, Field(gt=0, le=1)] = 1.0
    max_tokens: Annotated[int, Field(ge=1, le=2048)] = 128
    seed: Annotated[int, Field(ge=0, le=2**31 - 1)] | None = None
    # An after validator enforces false: Literal[False] accepts 0 in Pydantic.
    stream: bool = False

    @model_validator(mode="after")
    def supported_conversation(self) -> "ChatRequest":
        if self.stream:
            raise ValueError("streaming is not supported")
        if self.messages[-1].role != "user":
            raise ValueError("the final message must be a user message")
        if sum(len(message.content) for message in self.messages) > 32_000:
            raise ValueError("total message content exceeds 32000 characters")
        return self


class AssistantMessage(BaseModel):
    model_config = ConfigDict(strict=True)
    role: Literal["assistant"]
    content: str


class Choice(BaseModel):
    model_config = ConfigDict(strict=True)
    index: Annotated[int, Field(ge=0, le=0)]
    message: AssistantMessage
    finish_reason: Literal["stop", "length"]


class Usage(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt_tokens: Annotated[int, Field(ge=0)]
    completion_tokens: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def consistent_total(self) -> "Usage":
        if self.prompt_tokens + self.completion_tokens != self.total_tokens:
            raise ValueError("inconsistent token totals")
        return self


class ChatResponse(BaseModel):
    # Ignore upstream extras; only these fields cross the gateway boundary.
    model_config = ConfigDict(strict=True)
    id: Annotated[str, Field(min_length=1, max_length=256)]
    object: Literal["chat.completion"]
    created: Annotated[int, Field(ge=0)]
    model: Annotated[str, Field(min_length=1, max_length=128)]
    choices: Annotated[list[Choice], Field(min_length=1, max_length=1)]
    usage: Usage | None = None
