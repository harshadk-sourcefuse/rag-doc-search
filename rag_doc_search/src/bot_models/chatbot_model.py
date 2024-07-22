from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.vectorstore import VectorStore

from rag_doc_search.src.prompt_templates.default_prompt_templates import (
    DEFAULT_CHAT_HISTORY_PROMPT,
    DEFAULT_PROMPT_TEMPLATE,
)
from rag_doc_search.utils.miscellaneous import get_chat_history, get_logger


class ChatBotModel:
    """
    A class representing the Base ChatBot.

    This class serves as an implementation of a base chatbot for diff LLM's.
    """

    def __init__(self, embeddings: Embeddings):
        self.prompt_template = DEFAULT_PROMPT_TEMPLATE
        self.embeddings = embeddings
        self.logger = get_logger()

    def create_stream_manager(self, stream_handler, tracing) -> AsyncCallbackManager:
        """
        Creates and returns an instance of AsyncCallbackManager for handling asynchronous callbacks.

        Parameters:
        - `stream_handler`: The handler for managing the stream.
        - `tracing`: A boolean indicating whether tracing is enabled.

        Returns:
        An instance of AsyncCallbackManager.
        """
        manager = AsyncCallbackManager([])
        stream_manager = AsyncCallbackManager([stream_handler])
        if tracing:
            tracer = LangChainTracer()
            tracer.load_default_session()
            manager.add_handler(tracer)
            stream_manager.add_handler(tracer)

        return stream_manager

    def create_qa_chain(
        self,
        cl_llm: BaseLanguageModel,
        vector_store: VectorStore,
        prompt_template: PromptTemplate = None,
        retriever_args: dict = None,
    ) -> RetrievalQA:
        """
        Creates and returns an instance of RetrievalQA for question-answering using a provided language model.

        Parameters:
        - `cl_llm`: An instance of LanguageModel such as OpenAI or Bedrock Model.
        - vector_store (VectorStore): The vector store used for retrieving documents.
        - prompt_template (PromptTemplate, optional): Custom prompt template.
            If not provided, it will use DEFAULT_PROMPT_TEMPLATE from rag_doc_search.src.prompt_templates.default_prompt_templates.

        Returns:
        An instance of RetrievalQA.
        """
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate(
                template=self.prompt_template, input_variables=["context", "question"]
            )
        )

        self.logger.info(
            f"search_type: {retriever_args.get('search_type')} \n search_args: {retriever_args.get('search_args')}"
        )

        qa = RetrievalQA.from_chain_type(
            llm=cl_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type=retriever_args.get("search_type"),
                search_kwargs=retriever_args.get("search_args"),
            ),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template,
            },
        )
        return qa

    def create_conversational_qa_chain(
        self,
        cl_llm: BaseLanguageModel,
        vector_store: VectorStore,
        prompt_template: PromptTemplate = None,
        retriever_args: dict = None,
    ) -> ConversationalRetrievalChain:
        """
        Creates and returns an instance of ConversationalRetrievalChain for handling conversational question-answering
        using a provided language model.

        Parameters:
        - `cl_llm`: An instance of LanguageModel such as OpenAI or Bedrock Model.
        - vector_store (VectorStore): The vector store used for retrieving documents.
        - prompt_template (PromptTemplate, optional): Custom prompt template.
            If not provided, it will use DEFAULT_PROMPT_TEMPLATE from rag_doc_search.src.prompt_templates.default_prompt_templates.

        Returns:
        An instance of ConversationalRetrievalChain.
        """
        memory_chain = ConversationBufferWindowMemory(
            memory_key="chat_history",
            ai_prefix="Assistant",
            return_messages=True,
            k=0,
            output_key="answer",
        )

        # the condense prompt for Claude
        condense_prompt = PromptTemplate.from_template(DEFAULT_CHAT_HISTORY_PROMPT)
        prompt_template = (
            prompt_template
            if prompt_template
            else PromptTemplate.from_template(self.prompt_template)
        )

        self.logger.info(
            f"search_type: {retriever_args.get('search_type')} \n search_args: {retriever_args.get('search_args')}"
        )
        qa = ConversationalRetrievalChain.from_llm(
            llm=cl_llm,
            retriever=vector_store.as_retriever(
                search_type=retriever_args.get("search_type"),
                search_kwargs=retriever_args.get("search_args"),
            ),
            # return_source_documents=True,
            memory=memory_chain,
            get_chat_history=get_chat_history,
            condense_question_prompt=condense_prompt,
            chain_type="stuff",  # 'refine',
        )

        # the LLMChain prompt to get the answer. the ConversationalRetrievalChange does not expose this parameter in the constructor
        qa.combine_docs_chain.llm_chain.prompt = prompt_template
        return qa
