import logging

from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from waitress import serve

logging.basicConfig(format='%(levelname)s : %(asctime)s : %(message)s ', level=logging.INFO,
                    handlers=[logging.StreamHandler(), logging.FileHandler("logs.txt", 'w+')])

app = Flask(__name__)

OPENAI_API_VERSION = "2023-05-15"
AZURE_ENDPOINT = "https://biopenai4.openai.azure.com/"
OPENAI_API_KEY = "694eb87fb5364882a9a1ba04a7a17562"
DEPLOYMENT_NAME = "gpt4v32k"


@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question', '')
    response = ask_chat_gpt(question)
    return jsonify({"response": response})


def ask_chat_gpt(text):
    """
    Sends a text to an Azure-hosted ChatGPT model and retrieves the response.

    Args:
    - text (str): The text message to send to ChatGPT.

    Returns:
    - dict: The response from ChatGPT, parsed as a JSON object.
    """
    llm = AzureChatOpenAI(deployment_name=DEPLOYMENT_NAME,
                          openai_api_version=OPENAI_API_VERSION,
                          openai_api_key=OPENAI_API_KEY,
                          azure_endpoint=AZURE_ENDPOINT)
    msg = [HumanMessage(content=text)]
    response = llm.invoke(msg)
    answer = response.content
    return answer


if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5005)
