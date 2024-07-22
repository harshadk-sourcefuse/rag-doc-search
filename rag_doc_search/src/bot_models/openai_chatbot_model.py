from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_doc_search import config
from rag_doc_search.src.bot_models.chatbot_model import ChatBotModel
from rag_doc_search.utils.callback import StreamingLLMCallbackHandler


class OpenAIChatBot(ChatBotModel):
    """
    A class representing the Bedrock ChatBot.

    This class serves as an implementation of a chatbot using the OpenAI.
    """

    def __init__(self):
        self.config = config
        self.embeddings = OpenAIEmbeddings(model=self.config.embeddings_model)
        super().__init__(
            embeddings=self.embeddings,
        )

    def create_qa_instance(
        self,
        index_or_collection_name: str = None,
        prompt_template: PromptTemplate = None,
        retriever_args: dict = None,
    ) -> RetrievalQA:
        """
        Creates and returns an instance of RetrivalQA using OpenAI Language Model.

        Args:
        - index_or_collection_name (str, optional): Collection or index name for which vector store
        needs to be initialized.
        - prompt_template (PromptTemplate, optional): Custom prompt template.
        - retriever_args (dict, optional): A dictionary containing configuration settings for retrievers.

        Example:
            retriever_args = {
                "search_type": "similarity" | "mmr" | "similarity_score_threshold"
                "search_args": {
                    "k": 10,
                    "fetch_k": 500,
                    "lambda_mult": 0.1,
                    "score_threshold": 0.1
                }
            }

        Returns:
        An instance of RetrievalQA.
        """
        cl_llm: BaseLanguageModel = ChatOpenAI(
            model_name=self.config.llm,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_output_tokens,
        )
        vector_store = self.config.get_vector_store(
            embeddings=self.embeddings,
            index_or_collection_name=index_or_collection_name,
        )
        retriever_args: dict = (
            self.config.validate_and_get_retriever_arguments(retriever_args)
            if retriever_args
            else self.config.retriever_args
        )

        qa = self.create_qa_chain(cl_llm, vector_store, prompt_template, retriever_args)
        return qa

    def create_conversational_qa_instance(
        self,
        stream_handler: StreamingLLMCallbackHandler,
        index_or_collection_name: str = None,
        prompt_template: PromptTemplate = None,
        retriever_args: dict = None,
        tracing: bool = False,
    ) -> ConversationalRetrievalChain:
        """
        Creates and returns an instance of ConversationalRetrievalChain for conversational question-answering using OpenAI Language Model.

        Parameters:
        - `stream_handler`: An instance of StreamingLLMCallbackHandler used for handling streaming callbacks.
        - index_or_collection_name (str, optional): Collection or index name for which vector store
        needs to be initialized.
        - prompt_template (PromptTemplate, optional): Custom prompt template.
        - retriever_args (dict, optional): A dictionary containing configuration settings for retrievers.
        - `tracing`: A boolean indicating whether tracing is enabled. Default is False.

        Example:
            retriever_args = {
                "search_type": "similarity" | "mmr" | "similarity_score_threshold"
                "search_args": {
                    "k": 10,
                    "fetch_k": 500,
                    "lambda_mult": 0.1,
                    "score_threshold": 0.1
                }
            }

        Returns:
        An instance of ConversationalRetrievalChain for conversational question-answering.
        """
        stream_manager = self.create_stream_manager(stream_handler, tracing)
        cl_llm: BaseLanguageModel = ChatOpenAI(
            model_name=self.config.llm,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_output_tokens,
            streaming=True,
            callback_manager=stream_manager,
        )
        vector_store = self.config.get_vector_store(
            embeddings=self.embeddings,
            index_or_collection_name=index_or_collection_name,
        )
        retriever_args: dict = (
            self.config.validate_and_get_retriever_arguments(retriever_args)
            if retriever_args
            else self.config.retriever_args
        )

        qa = self.create_conversational_qa_chain(
            cl_llm, vector_store, prompt_template, retriever_args
        )
        return qa
