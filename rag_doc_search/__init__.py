
from .src.enums.provider import AIProvider
from .src.bot_models.chatbot_model import ChatBotModel
from .utils.config import Config

def config_init(config_json:dict)-> Config:
    """
    Initializes the configuration based on the provided JSON config.

    Parameters:
    - config_json (dict): A dictionary containing configuration parameters.

    Returns: (Config) -> Initialized Config instance

    None
    """
    global config
    if(config_json is None):
        raise ValueError("Config Json can not be None or Null")
    config = Config(config_json)
    
    return config

def get_bot_instance(ai_provider : AIProvider) -> ChatBotModel: 
    """
    Returns an instance of the ChatBotModel based on the specified AI provider.

    Parameters:
    - ai_provider (AIProvider): An enum specifying the AI provider for the ChatBotModel.

    Returns:
    ChatBotModel: An instance of the ChatBotModel corresponding to the specified AI provider.

    Raises:
    ValueError: If the specified AI provider is not supported.
    """
    match ai_provider:
        case AIProvider.BEDROCK:
            from rag_doc_search.src.bot_models.bedrock_chatbot_model import BedrockChatBot
            return BedrockChatBot()
        case AIProvider.OPENAI:
            from rag_doc_search.src.bot_models.openai_chatbot_model import OpenAIChatBot
            return OpenAIChatBot()
        case AIProvider.AZURE_OPENAI:
            from rag_doc_search.src.bot_models.azure_chatbot_model import AzureChatBot
            return AzureChatBot()
        case _:
            raise ValueError("cannot initiate llm base model for given ai provider")
        
# Export the init function
__all__ = ["init"]
