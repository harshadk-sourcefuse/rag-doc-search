import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from websockets.exceptions import ConnectionClosedOK

from rag_doc_search import config_init, get_bot_instance
from rag_doc_search.src.models.chat_response import ChatResponse
from rag_doc_search.utils.callback import StreamingLLMCallbackHandler
from rag_doc_search.utils.miscellaneous import get_logger

root_path = "path to your env file"
dotenv_path = os.path.join(root_path, ".env")
load_dotenv(dotenv_path)


app = FastAPI()
logger = get_logger()

# add your config here in config_init
config = config_init({})

bot_model = get_bot_instance(config.ai_provider)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    stream_handler = StreamingLLMCallbackHandler(websocket)
    qa_chain = bot_model.create_conversational_qa_instance(stream_handler)

    while True:
        try:
            # Receive and send back the client message
            user_msg = await websocket.receive_text()
            logger.info(f"User Message Received : {user_msg}")

            if len(user_msg.strip()) <= 0:
                start_resp = ChatResponse(
                    sender="bot",
                    message="A string that should not be empty or contain only whitespace ",
                    type="error",
                )
                await websocket.send_json(start_resp.dict())
            else:
                if not user_msg.endswith("?"):
                    user_msg = f"{user_msg}?"

                resp = ChatResponse(sender="human", message=user_msg, type="end")
                await websocket.send_json(resp.dict())
                logger.info("User stream send")

                # Construct a response
                start_resp = ChatResponse(sender="bot", message="", type="start")
                await websocket.send_json(start_resp.dict())

                logger.info("Bot Start send")
                output = await qa_chain.ainvoke(input=user_msg)
                logger.info(output)
                # # Send the end-response back to the client
                # end_resp = ChatResponse(sender="bot", message=output, type="stream")
                # await websocket.send_json(end_resp.dict())

                # Send the end-response back to the client
                end_resp = ChatResponse(sender="bot", message="", type="end")
                await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logger.info("WebSocketDisconnect")
            # TODO try to reconnect with back-off
            break
        except ConnectionClosedOK:
            logger.info("ConnectionClosedOK")
            # TODO handle this?
            break
        except Exception as e:
            logger.error(e, stack_info=True)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


uvicorn.run(app, host="0.0.0.0", port=8000)
