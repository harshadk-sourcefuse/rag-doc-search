import boto3
import os

from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.schema.language_model import BaseLanguageModel
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from rag_doc_search.src.bot_models.chatbot_model import ChatBotModel
from rag_doc_search import config
from rag_doc_search.utils.callback import StreamingLLMCallbackHandler


class BedrockChatBot(ChatBotModel):
    """
    A class representing the Bedrock ChatBot.

    This class serves as an implementation of a chatbot using the Bedrock.
    """

    def __init__(self):
        self.config = config
        session = boto3.Session(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        boto3_bedrock = session.client(
            service_name="bedrock-runtime", region_name=os.environ.get("AWS_REGION")
        )
        self.embeddings = BedrockEmbeddings(
            model_id=self.config.embeddings_model, client=boto3_bedrock
        )
        self.boto3_bedrock = boto3_bedrock
        super().__init__(
            embeddings=self.embeddings,
            retriever_args=self.config.get_retriever_args(),
        )

    def create_qa_instance(
        self,
        index_or_collection_name: str = None,
        prompt_template: PromptTemplate = None,
    ) -> RetrievalQA:
        """
        Creates and returns an instance of RetrivalQA using Bedrock Language Model.

        Args:
        - index_or_collection_name (str, optional): Collection or index name for which vector store 
        needs to be initialized. 
        - prompt_template (PromptTemplate, optional): Custom prompt template.

        Returns:
        An instance of RetrivalQA.
        """
        cl_llm: BaseLanguageModel = Bedrock(
            model_id=self.config.llm,
            client=self.boto3_bedrock,
            model_kwargs={
                "max_tokens_to_sample": self.config.llm_max_output_tokens,
                "temperature": self.config.llm_temperature,
            },
        )
        vector_store = self.config.get_vector_store(
            embeddings=self.embeddings,
            index_or_collection_name=index_or_collection_name,
        )
        qa = self.create_qa_chain(cl_llm, vector_store, prompt_template)
        return qa

    def create_conversational_qa_instance(
        self,
        stream_handler: StreamingLLMCallbackHandler,
        index_or_collection_name: str = None,
        prompt_template: PromptTemplate = None,
        tracing: bool = False,
    ) -> ConversationalRetrievalChain:
        """
        Creates and returns an instance of ConversationalRetrievalChain for conversational question-answering using Bedrock Language Model.

        Parameters:
        - `stream_handler`: An instance of StreamingLLMCallbackHandler used for handling streaming callbacks.
        - index_or_collection_name (str, optional): Collection or index name for which vector store 
        needs to be initialized. 
        - prompt_template (PromptTemplate, optional): Custom prompt template.
        - `tracing`: A boolean indicating whether tracing is enabled. Default is False.

        Returns:
        An instance of ConversationalRetrievalChain for conversational question-answering.
        """
        stream_manager = self.create_stream_manager(stream_handler, tracing)
        cl_llm: BaseLanguageModel = Bedrock(
            model_id=self.config.llm,
            client=self.boto3_bedrock,
            model_kwargs={
                "max_tokens_to_sample": self.config.llm_max_output_tokens,
                "temperature": self.config.llm_temperature,
            },
            streaming=True,
            callback_manager=stream_manager,
        )
        vector_store = self.config.get_vector_store(
            embeddings=self.embeddings,
            index_or_collection_name=index_or_collection_name,
        )
        qa = self.create_conversational_qa_chain(cl_llm, vector_store, prompt_template)
        return qa
