from importlib import metadata

from langchain_cognee.chat_models import ChatCognee
from langchain_cognee.document_loaders import CogneeLoader
from langchain_cognee.embeddings import CogneeEmbeddings
from langchain_cognee.retrievers import CogneeRetriever
from langchain_cognee.toolkits import CogneeToolkit
from langchain_cognee.tools import CogneeTool
from langchain_cognee.vectorstores import CogneeVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCognee",
    "CogneeVectorStore",
    "CogneeEmbeddings",
    "CogneeLoader",
    "CogneeRetriever",
    "CogneeToolkit",
    "CogneeTool",
    "__version__",
]
