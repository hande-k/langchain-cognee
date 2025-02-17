# langchain-cognee

This package contains the LangChain integration with [cognee](https://github.com/topoteretes/cognee).

This package enables you to:
- Ingest documents into cognee
- Build or update a knowledge graph
- Retrieve and query your data using LangChain's standard interfaces

For more information, check out [cognee documentation](https://docs.cognee.ai/).

## Installation

```bash
pip install -U langchain-cognee
```

## Configuration
Set your environment variables required by cognee:

```bash
export LLM_API_KEY="your-openai-api-key"
```

Cognee's default settings:
- LLM Provider: OpenAI 
- Databases: SQLite, LanceDB, networkx

In case you want to customize your settings, please refer [here](https://github.com/topoteretes/cognee/blob/dev/.env.template) and configure your env variables accordingly. 

Supported databases
- Relational databases: SQLite, PostgreSQL
- Vector databases: LanceDB, PGVector, QDrant, Weviate
- Graph databases: Neo4j, NetworkX

## Basic Usage
Below is a minimal example of how to use this integration:

```python

    from langchain_cognee.retrievers import CogneeRetriever
    from langchain_core.documents import Document

    # 1) Instantiate the retriever
    retriever = CogneeRetriever(
        llm_api_key="YOUR_KEY", 
        dataset_name="test_dataset", 
        k=3
    )

    # 2) (Optional) Reset dataset if you want a clean slate
    retriever.reset_dataset()

    # 3) Add documents
    docs = [
        Document(page_content="Elon Musk is the CEO of SpaceX."),
        Document(page_content="SpaceX focuses on rockets."),
    ]
    retriever.add_documents(docs)

    # 4) Build knowledge graph
    retriever.process_data()

    # 5) Retrieve documents
    results = retriever.invoke("Tell me about Elon Musk")
    for doc in results:
        print(doc.page_content)
```

You can also incorporate CogneeRetriever in any LangChain chain, such as a RetrievalQA chain or a custom pipeline.





