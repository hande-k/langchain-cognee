"""Test chat model integration."""

from typing import Type

from langchain_cognee.chat_models import ChatCognee
from langchain_tests.unit_tests import ChatModelUnitTests


class TestChatCogneeUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[ChatCognee]:
        return ChatCognee

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
            "parrot_buffer_length": 50,
        }
