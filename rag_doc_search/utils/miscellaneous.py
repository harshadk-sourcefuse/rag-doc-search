import json
import logging

from langchain.schema import BaseMessage


def get_logger() -> logging.Logger:
    """
        provides logger instance

    Returns:
        logging.Logger: logger for application
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.propagate = 0
        logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter(
            "%(levelname)s.%(module)s.%(funcName)s Line:%(lineno)d: %(asctime)s %(message)s"
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)
    return logger


def get_chat_history(chat_history):
    """
    Converts the given chat history into a formatted string representation.

    The chat history may contain messages from both humans and the AI. The function supports
    different formats for representing each dialogue turn.

    Parameters:
    - `chat_history`: A list representing the chat history. Each element can be a `BaseMessage`
      instance or a tuple containing a human message and an AI response.

    Returns:
    A formatted string representation of the chat history.
    """

    _ROLE_MAP = {"human": "\n\nHuman: ", "ai": "\n\nAssistant: "}
    buffer = ""
    for dialogue_turn in chat_history:
        if isinstance(dialogue_turn, BaseMessage):
            role_prefix = _ROLE_MAP.get(dialogue_turn.type, f"{dialogue_turn.type}: ")
            buffer += f"\n{role_prefix}{dialogue_turn.content}"
        elif isinstance(dialogue_turn, tuple):
            human = "\n\nHuman: " + dialogue_turn[0]
            ai = "\n\nAssistant: " + dialogue_turn[1]
            buffer += "\n" + "\n".join([human, ai])
        else:
            raise ValueError(
                f"Unsupported chat history format: {type(dialogue_turn)}."
                f" Full chat history: {chat_history} "
            )
    return buffer


def json_to_dict(path: str) -> dict:
    """
    Convert JSON data from the specified path to a Python dictionary.

    Args:
        path (str): The file path to the JSON file.

    Returns:
        dict: A dictionary containing the data parsed from the JSON file.
    """
    with open(path, "r") as json_file:
        parsed_json = json.load(json_file)
    return parsed_json
