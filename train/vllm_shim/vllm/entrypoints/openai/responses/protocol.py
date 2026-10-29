from pydantic import BaseModel


class ResponsesRequest(BaseModel):
    skip_special_tokens: bool = True
