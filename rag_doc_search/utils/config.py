import os
from langchain.schema.embeddings import Embeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.faiss import FAISS
from langchain.schema.vectorstore import VectorStore

from rag_doc_search.src.enums.provider import AIProvider, VectorStoreProvider
from rag_doc_search.src.enums.search_type import RetrieverSearchType
from rag_doc_search.utils.miscellaneous import get_logger


class Config:
    """
    A class representing configuration settings for the System which can be used to initialize the llm model, vector stores and its retrivers argument

    ```
    Note: Ensure that the necessary environment variables are set before initializing the configuration.
    If AI Provider is OPENAI then ENV OPENAI_API_KEY is required.
    If AI Provider is AZURE OPENAI then ENV AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT are required and AZURE_OPENAI_API_VERSION is optional, default is '2023-05-15'.
    If AI Provider is BEDROCK then ENV AWS_ACCESS_KEY,AWS_SECRET_ACCESS_KEY are required and AWS_REGION is optional, default is 'us-east-1'.
    If Vector Store Provider is PGVector then PGVECTOR_HOST, PGVECTOR_PORT,  PGVECTOR_DATABASE,  PGVECTOR_USER and PGVECTOR_PASSWORD are required
    """

    _instance = None

    def __new__(cls, config: dict = None):
        if cls._instance is None:
            if config is not None:
                cls._instance = super(Config, cls).__new__(cls)
                cls._instance._validate_and_initialize(config)
            else:
                raise ValueError(
                    "Config is not provided while creating singleton instance."
                )
        return cls._instance

    def _validate_and_initialize_ai_provider(self, config_json: dict):
        """
        Validates and initializes the AI provider based on the configuration JSON.
        Also validates the required environment variables for the language model (LLM) as per provider
        and also initializes AI LLM config properties.

        Parameters:
        - `config_json`: A dictionary containing configuration settings.

        Raises:
        - `ValueError`: If the configuration is invalid or missing required properties.
        """
        ai_provider_str = config_json.get("ai_provider", "")
        try:
            self.ai_provider = AIProvider(ai_provider_str)
        except ValueError:
            raise ValueError(
                f"Invalid value for 'ai_provider'. Expected values: {', '.join(member.value for member in AIProvider)}"
            )

        match self.ai_provider:
            case AIProvider.OPENAI:
                if not os.environ.get("OPENAI_API_KEY"):
                    raise ValueError(
                        "OPENAI_API_KEY environment variable is required for OpenAI"
                    )
            
            case AIProvider.AZURE_OPENAI:
                if not os.environ.get("AZURE_OPENAI_API_KEY"):
                    raise ValueError(
                        "AZURE_OPENAI_API_KEY environment variable is required for Azure OpenAI"
                    )
                if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
                    raise ValueError(
                        "AZURE_OPENAI_ENDPOINT environment variable is required for Azure OpenAI"
                    )
                if not os.environ.get("AZURE_OPENAI_API_VERSION"):
                    os.environ["AZURE_OPENAI_API_VERSION"] = "2023-05-15"
                    print( os.environ.get("AZURE_OPENAI_API_VERSION") )
                    self.logger.warning(
                        'AZURE_OPENAI_API_VERSION key not found in enviroment variables, so "2023-05-15" will be used as default value.'
                    )
                    print(
                        'AZURE_OPENAI_API_VERSION key not found in enviroment variables, so "2023-05-15" will be used as default value.'
                    )

            case AIProvider.BEDROCK:
                if not os.environ.get("AWS_ACCESS_KEY") or not os.environ.get(
                    "AWS_SECRET_ACCESS_KEY"
                ):
                    raise ValueError(
                        "AWS_ACCESS_KEY and AWS_SECRET_ACCESS_KEY environment variables are required for Bedrock"
                    )

                if not os.environ.get("AWS_REGION"):
                    os.environ["AWS_REGION"] = "us-east-1"
                    print( os.environ.get("AWS_REGION") )
                    self.logger.warning(
                        'AWS_REGION key not found in enviroment variables, so "us-east-1" will be used as default value.'
                    )
                    print(
                        'AWS_REGION key not found in enviroment variables, so "us-east-1" will be used as default value.'
                    )
            case _:
                self.logger.warning("Default case AIProvider")

    def _validate_and_initialize_vector_store_provider(self, config_json: dict):
        """
        Validates and initializes the vector store provider based on the configuration JSON.
        Also validates and initializes vector store config properties and checks required environment variables as per provider.

        Parameters:
        - `config_json`: A dictionary containing configuration settings.

        Raises:
        - `ValueError`: If the configuration is invalid or missing required properties.
        """
        vector_store_provider_str = config_json.get("vector_store_provider", "")
        try:
            self.vector_store_provider = VectorStoreProvider(vector_store_provider_str)
        except ValueError:
            raise ValueError(
                f"Invalid value for 'vector_store_provider'. Expected values: {', '.join(member.value for member in VectorStoreProvider)}"
            )
        match self.vector_store_provider:
            case VectorStoreProvider.FAISS:
                self.faiss_vector_embeddings_location = config_json.get(
                    "faiss_vector_embeddings_location", ""
                )
                self.faiss_index_name = config_json.get("faiss_index_name", "index")
                if not self.faiss_vector_embeddings_location:
                    raise ValueError(
                        "faiss_vector_embeddings_location are required for FAISS"
                    )
                # if self.faiss_vector_embeddings_location.startswith("./"):
                #     self.faiss_vector_embeddings_location = (
                #         self.faiss_vector_embeddings_location.replace(
                #             "./", os.path.dirname(__file__) + "/"
                #         )
                #     )
                self.logger.info(f"{self.faiss_vector_embeddings_location}")
                if not (os.path.exists(self.faiss_vector_embeddings_location)):
                    raise ValueError(
                        "Invalid Directory path for FAISS vector embeddings"
                    )
            case VectorStoreProvider.PGVector:
                self.collection_name = config_json.get("collection_name", "")
                if not self.collection_name:
                    raise ValueError("collection_name is required for PGVECTOR")

                if (
                    not os.environ.get("PGVECTOR_HOST")
                    or not os.environ.get("PGVECTOR_PORT")
                    or not os.environ.get("PGVECTOR_DATABASE")
                    or not os.environ.get("PGVECTOR_USER")
                    or not os.environ.get("PGVECTOR_PASSWORD")
                ):
                    raise ValueError(
                        "PGVECTOR_HOST, PGVECTOR_PORT, PGVECTOR_DATABASE, PGVECTOR_USER and PGVECTOR_PASSWORD environment variables are required for PgVector"
                    )
            case _:
                self.logger.warning("Default case VectorStoreProvider")

    def _validate_and_initialize_retriever_arguments(self, config_json: dict):
        """
        Validates and initializes the retriever arguments based on the configuration JSON.
        Also validates and initializes retriever config properties.

        Parameters:
        - `config_json`: A dictionary containing configuration settings.

        Raises:
        - `ValueError`: If the configuration is invalid or missing required properties.
        """

        retriever_args = config_json.get("retriever", {})
        search_type_str = retriever_args.get("search_type", "")
        try:
            self.retriever_search_type = RetrieverSearchType(search_type_str)
        except ValueError:
            raise ValueError(
                f"Invalid value for 'search_type'. Expected values: {', '.join(member.value for member in RetrieverSearchType)}"
            )

        search_args = retriever_args.get("search_args", {})

        self.retriever_search_args_k = int(search_args.get("k", 4))
        self.retriever_search_args_fetch_k = int(search_args.get("fetch_k", 20))

        self.retriever_search_args_lambda_mult = float(
            config_json.get("lambda_mult", 0.5)
        )
        if not (0 <= self.retriever_search_args_lambda_mult <= 1):
            raise ValueError(
                "Invalid lambda_mult value in config -> retriever -> search args"
            )

        self.retriever_search_args_score_threshold = float(
            config_json.get("score_threshold", 0)
        )

    def _validate_and_initialize(self, config_json: dict):
        """
        Validates and initializes various configuration properties based on the provided JSON.

        Parameters:
        - `config_json`: A dictionary containing configuration settings.

        Raises:
        - `ValueError`: If the configuration is invalid or missing required properties.
        """
        self.logger = get_logger()
        self.logger.info("Vaidating the config")

        self._validate_and_initialize_ai_provider(config_json)
        self._validate_and_initialize_vector_store_provider(config_json)
        self._validate_and_initialize_retriever_arguments(config_json)

        # Validate Embeddings model, llm, llm_temperature, and llm_max_output_tokens
        self.embeddings_model = config_json.get("embeddings_model", "")
        if not self.embeddings_model:
            raise ValueError("No value provided for the key 'embeddings_model'")

        self.llm = config_json.get("llm", "")
        if not self.llm:
            raise ValueError("No value provided for the key 'llm'")

        self.llm_temperature = float(config_json.get("llm_temperature", 0.1))
        if not (0 <= self.llm_temperature <= 2):
            raise ValueError("Invalid llm_temperature value in config")

        self.llm_max_output_tokens = int(config_json.get("llm_max_output_tokens", 500))
        if self.llm_max_output_tokens <= 0:
            raise ValueError("Invalid llm_max_output_tokens value in config")

        # Set the name property
        self.name = config_json.get("name", "")

        self.vector_store = None

    def get_vector_store(self, embeddings: Embeddings) -> VectorStore:
        """
        Uses the vector store provider to create a requested provider using the required information for that vector store such as FAISS and PgVector.

        Parameters:
        - `embeddings`: An instance of the Embeddings class.

        Returns:
        A VectorStore instance based on the specified provider.

        Raises:
        - `ValueError`: If the specified vector store provider is not supported.
        """
        if self.vector_store is None:
            match self.vector_store_provider:
                case VectorStoreProvider.FAISS:
                    self.vector_store = FAISS.load_local(
                        folder_path=self.faiss_vector_embeddings_location,
                        embeddings=embeddings,
                        index_name=self.faiss_index_name,
                        allow_dangerous_deserialization=True,
                    )

                case VectorStoreProvider.PGVector:
                    CONNECTION_STRING = PGVector.connection_string_from_db_params(
                        driver="psycopg2",
                        host=os.environ.get("PGVECTOR_HOST", "localhost"),
                        port=int(os.environ.get("PGVECTOR_PORT", "5432")),
                        database=os.environ.get("PGVECTOR_DATABASE", "postgres"),
                        user=os.environ.get("PGVECTOR_USER", "postgres"),
                        password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
                    )
                    self.vector_store = PGVector(
                        collection_name="city_bot",
                        connection_string=CONNECTION_STRING,
                        embedding_function=self.embeddings,
                    )
                    self.vector_store.as_retriever
                case _:
                    self.logger.warning("Default case VectorStoreProvider")
        return self.vector_store

    def get_retriever_args(self) -> dict:
        """
        Retrieves retriever information from the configuration.

        Returns:
        A dictionary containing retriever configuration properties which can be used in vector store retriever.

        Raises:
        - `ValueError`: If the retriever configuration is invalid or missing required properties.
        """
        retriever_args = {
            "k": self.retriever_search_args_k,
            "fetch_k": self.retriever_search_args_fetch_k,
        }
        match self.vector_store_provider:
            case RetrieverSearchType.mmr:
                if self.retriever_search_args_lambda_mult <= 0:
                    retriever_args["lambda_mult"] = self.retriever_search_args_fetch_k
            case RetrieverSearchType.similarity_score_threshold:
                retriever_args[
                    "score_threshold"
                ] = self.retriever_search_args_score_threshold

        return {
            "search_type": self.retriever_search_type.value,
            "search_args": retriever_args,
        }
