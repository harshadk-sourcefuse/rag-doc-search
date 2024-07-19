"""Callback handlers used in the app."""

from typing import Any

from langchain.callbacks.base import AsyncCallbackHandler

from rag_doc_search.src.models.chat_response import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """
        Asynchronously handles the event triggered when a new token is encountered in the language model and sends the token in websocket event.

        Parameters:
        - `token`: A string representing the new token encountered.
        - `**kwargs`: Additional keyword arguments that may be provided (optional).

        Returns:
        None
        """
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())
