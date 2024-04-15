# RAG doc search

## Overview
This package offers a lightweight and straightforward solution for implementing Retrieval-augmented generation (RAG) functionality with large language models (LLMs). RAG enhances prediction quality by incorporating external data storage during inference, enabling the construction of more contextually rich prompts. Leveraging combinations of context, history, and up-to-date knowledge, RAG LLMs empower users to generate more accurate and relevant responses. This package streamlines the integration of RAG capabilities into applications, facilitating the creation of more intelligent and context-aware conversational agents, search engines, and text generation systems.

## Config Json
```
{
    "ai_provider": "OPENAI" | "BEDROCK" | "AZURE_OPENAI",
    "embeddings_model": "<embedding model of openai or bedrock or azure openai as per ai_provider>",
    "llm": "<llm model of openai or bedrock as per ai_provider>",
    "llm_temperature": "<llm model of temperature>",
    "llm_max_output_tokens": "<llm model of mac output token>",
    "vector_store_provider": "FAISS" | "PGVector",
    "faiss_vector_embeddings_location": "./data", provide only if vector_store_provider is FAISS
    "faiss_index_name": "fiass-db", provide only if vector_store_provider is FAISS
    "name": "<name of your project>",
    "retriever": {
         use this link [https://python.langchain.com/docs/modules/data_connection/retrievers/vectorstore] for understanding retriver searchtype and search_args
        "search_type": "similarity" | "mmr" | "similarity_score_threshold" ,
        "search_args": {
            "k": 10,
            "fetch_k": 500, 
            "lambda_mult": 0.1 
            "score_threshold" : 0.1
        }
    }
}
```

### Note
Ensure that the necessary environment variables are set before initializing the configuration.

- If AI Provider is `OPENAI` then ENV `OPENAI_API_KEY` is required.
- If AI Provider is `BEDROCK` then ENV `AWS_ACCESS_KEY`,`AWS_SECRET_ACCESS_KEY` are required and AWS_REGION is optional, default is 'us-east-1'.
- If Vector Store Provider is `PGVector` then `PGVECTOR_HOST`, `PGVECTOR_PORT`,  `PGVECTOR_DATABASE`,  `PGVECTOR_USER` and `PGVECTOR_PASSWORD` are required

## How to Use
```python
from rag_doc_search import config_init, get_bot_instance
from rag_doc_search.utils.config import Config
from rag_doc_search.src.enums.provider import AIProvider

# provide your config JSON in init config_init method below also set the required ENV as mentioned in Note
config = config_init({})

# Method a bot instance as per AI provider it will give you the bot model for BedRock or for OpenAI
bot_instance = get_bot_instance(config.ai_provider)

qa_instance = bot_model.create_qa_instance()

result = qa_instance.invoke(input="Provide your prompt here")

print(result)

```

## Examples

To understand how to use the package, refer to the following examples:

1. To create an API that answers your questions, use the [example provided here](https://github.com/harshadk-sourcefuse/rag-doc-search/blob/main/examples/qna_api_example.py).

2. To create a WebSocket API for streaming answers to your questions, use the [example provided here](https://github.com/harshadk-sourcefuse/rag-doc-search/blob/main/examples/qna_stream_websocket_example.py).
