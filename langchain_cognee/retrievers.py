"""Cognee retrievers.
 
This module provides a CogneeRetriever class that integrates cognee's knowledge graph
and vector store functionalities with LangChain's Retriever interface.

"""

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import os
import asyncio
from pydantic import ConfigDict, model_validator

def _import_cognee():
    """Lazy import Cognee to avoid ImportError if it's not installed."""
    try:
        import cognee
        from cognee.api.v1.search import SearchType
    except ImportError as e:
        raise ImportError(
            "Cognee is not installed. Please install it with:\n"
            "  pip install cognee\n"
            "or install langchain-cognee if you want to use CogneeRetriever."
        ) from e
    return cognee, SearchType


class CogneeRetriever(BaseRetriever):
    """A LangChain retriever that integrates with cognee, allowing you to:
        1. Add documents to a cognee dataset (via ``add_documents``).
        2. Process (cognify) the dataset into a knowledge graph (via ``process_data``).
        3. Retrieve relevant documents (via the standard Retriever interface).
   
    Setup:
        Install ``langchain-cognee`` and set environment variable ``LLM_API_KEY`` or pass in the key via the ``llm_api_key`` parameter.

        .. code-block:: bash

            pip install -U langchain-cognee
            export LLM_API_KEY="openai-api-key"

    Key init args:
        - llm_api_key (str): 
            Your LLM API key. If not provided, we attempt to read it from the environment variable ``LLM_API_KEY``.
        - llm_provider (str):
            The name of the LLM provider, please set it up if not "openai"
        - llm_model (str):
            The model to use, e.g., "gpt-4o-mini".
        - dataset_name (str):
            The name of the cognee dataset to which documents are added and from which they are retrieved.
        - k (int):
            Default number of documents to retrieve if not overridden during a query.

    Instantiate:
        .. code-block:: python

            from langchain-cognee import CogneeRetriever
            from langchain_core.documents import Document

            retriever = CogneeRetriever(
                llm_api_key="your-api-key",
                dataset_name="cognee_dataset",
                k=3
            )

    Usage:
        .. code-block:: python
            # Add documents
            docs = [
                Document(page_content="Elon Musk is the CEO of SpaceX."),
                Document(page_content="Tesla focuses on electric vehicles."),
            ]
            retriever.add_documents(docs)

            # Build the knowledge graph
            retriever.process_data()

            # Retrieve documents using the standard retriever interface
            results = retriever.invoke("Tell me about Elon Musk")
            for doc in results:
                print(doc.page_content)

    Use within a chain:
        .. code-block:: python
            
            from langchain_cognee.retrievers import CogneeRetriever
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnablePassthrough
            from langchain_openai import ChatOpenAI
            
            retriever = CogneeRetriever(llm_api_key="your-api-key", dataset_name="my_dataset")

            prompt = ChatPromptTemplate.from_template(
                \"\"\"Answer the question based only on the context provided.

            Context: {context}

            Question: {question}\"\"\"
            )

            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

            def format_docs(docs):
                return "\\n\\n".join(doc.page_content for doc in docs)

            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke("What do we know about Elon Musk?")
            print("Answer:", answer)

        .. code-block:: none

    """

    model_config = ConfigDict(extra='allow')

    llm_api_key: str = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    dataset_name: str = "default_dataset"
    k: int = 1

    @model_validator(mode="after")
    def configure_cognee(self):
        """Run after the object is constructed to set cognee config."""
        # If no key is provided, try environment variable fallback:
        if not self.llm_api_key:
            env_key = os.environ.get("LLM_API_KEY")
            if not env_key:
                raise ValueError("No LLM API key found. Provide via `llm_api_key` or env var `LLM_API_KEY`.")
            self.llm_api_key = env_key

        # Import Cognee
        cognee, _ = _import_cognee()

        # Configure cognee
        cognee.config.set_llm_config(
            {
                "llm_api_key": self.llm_api_key,
                "llm_provider": self.llm_provider,
                "llm_model": self.llm_model,
            }
        )
        return self
    
    def prune(self) -> None:
        """
        Prune (remove) data from the cognee dataset.
        Call this before adding documents if you want to ensure a clean state.
        Otherwise, you can skip this step to retain existing data in cognee.
        """
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._prune_async())

    async def _prune_async(self) -> None:
        """Async helper to prune a dataset in cognee."""
        cognee, _ = _import_cognee()
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

    def add_documents(self, docs: List[Document]) -> None:
        """Add LangChain Documents to the cognee dataset.

        Typically, you'd do this once before calling `process_data()` to build the knowledge graph.
        """
        # Convert each Document to text
        texts = [doc.page_content for doc in docs]
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._add_documents_async(texts))

    async def _add_documents_async(self, texts: List[str]) -> None:
        """Async helper to call cognee add(...)"""
        cognee, _ = _import_cognee()
        await cognee.add(texts, self.dataset_name)

    def process_data(self) -> None:
        """Process ingested data into a knowledge graph (aka 'cognify')."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._process_data_async())

    async def _process_data_async(self) -> None:
        """Async helper to 'cognify' a dataset in cognee."""
        cognee, _ = _import_cognee()
        user = await cognee.modules.users.methods.get_default_user()
        datasets = await cognee.modules.data.methods.get_datasets_by_name(
            self.dataset_name,
            user.id
        )
        await cognee.cognify(datasets, user)


    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        k = kwargs.get("k", self.k)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self._search_cognee(query))
    
        # Convert cognee results to LangChain Documents
        docs = [Document(page_content=str(r)) for r in results]
        return docs[:k]

    # optional: add custom async implementations here
    # async def _aget_relevant_documents(
    #     self,
    #     query: str,
    #     *,
    #     run_manager: AsyncCallbackManagerForRetrieverRun,
    #     **kwargs: Any,
    # ) -> List[Document]: ...

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any
    ) -> List[Document]:
        k = kwargs.get("k", self.k)
        results = await self._search_cognee(query)
        
        return [Document(page_content=r) for r in results][:k]


    async def _search_cognee(self, query: str) -> List[str]:
        """Async helper to call cognee search or rag_search."""
        cognee, SearchType = _import_cognee()
        user = await cognee.modules.users.methods.get_default_user()
        results = await cognee.search(
            query_type=SearchType.INSIGHTS,
            query_text=query,
            user=user,
        )
        return [str(r) for r in results]