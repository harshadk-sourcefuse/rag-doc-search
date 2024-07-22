import os

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

from rag_doc_search import config
from rag_doc_search.src.bot_models.chatbot_model import ChatBotModel


class AzureChatBot(ChatBotModel):
    """
    A class representing the Azure OpenAI ChatBot.

    This class serves as an implementation of a chatbot using the Azure OpenAI.
    """

    def __init__(self):
        self.config = config
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self.config.embeddings_model,
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        )
        super().__init__(embeddings=self.embeddings)

    def create_qa_instance(
        self,
        index_or_collection_name: str = None,
        prompt_template: PromptTemplate = None,
        retriever_args: dict = None,
    ) -> RetrievalQA:
        """
        Creates and returns an instance of RetrivalQA using Azure OpenAI Language Model.

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
        cl_llm: BaseLanguageModel = AzureChatOpenAI(
            azure_deployment=self.config.llm,
            model_name=self.config.llm,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_output_tokens,
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
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
        stream_handler,
        index_or_collection_name: str = None,
        prompt_template: PromptTemplate = None,
        retriever_args: dict = None,
        tracing: bool = False,
    ) -> ConversationalRetrievalChain:
        """
        Creates and returns an instance of ConversationalRetrievalChain for conversational question-answering using Azure OpenAI Language Model.

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
        cl_llm: BaseLanguageModel = AzureChatOpenAI(
            azure_deployment=self.config.llm,
            model_name=self.config.llm,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_output_tokens,
            streaming=True,
            callback_manager=stream_manager,
            openai_api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
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
