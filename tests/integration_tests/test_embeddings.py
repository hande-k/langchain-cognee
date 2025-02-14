"""Test Cognee embeddings."""

from typing import Type

from langchain_cognee.embeddings import CogneeEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[CogneeEmbeddings]:
        return CogneeEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
