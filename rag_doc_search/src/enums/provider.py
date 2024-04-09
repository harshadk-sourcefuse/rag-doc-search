from enum import Enum


class VectorStoreProvider(Enum):
    """
    Enum representing the providers available for a vector stores.

    Attributes:
    - `PGVector`: Refers to the PGVector provider.
    - `FAISS`: Refers to the FAISS provider.
    """
    PGVector = "PGVector"
    FAISS = "FAISS"


class AIProvider(Enum):
    """
    Enum representing providers available for the AI model API.

    Attributes:
    - `OPENAI`: Refers to the OpenAI provider.
    - `AZURE_OPENAI`: Refers to the Azure OpenAI provider.
    - `BEDROCK`: Refers to the Bedrock provider.
    """
    OPENAI = "OPENAI"
    AZURE_OPENAI = "AZURE_OPENAI"
    BEDROCK = "BEDROCK"
