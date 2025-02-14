# langchain-cognee

This package contains the LangChain integration with Cognee

## Installation

```bash
pip install -U langchain-cognee
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatCognee` class exposes chat models from Cognee.

```python
from langchain_cognee import ChatCognee

llm = ChatCognee()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`CogneeEmbeddings` class exposes embeddings from Cognee.

```python
from langchain_cognee import CogneeEmbeddings

embeddings = CogneeEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`CogneeLLM` class exposes LLMs from Cognee.

```python
from langchain_cognee import CogneeLLM

llm = CogneeLLM()
llm.invoke("The meaning of life is")
```
