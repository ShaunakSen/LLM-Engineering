import streamlit as st
import numpy as np
from loguru import logger


# models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# prompts
from langchain_core.prompts import load_prompt, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# env
from dotenv import load_dotenv

load_dotenv()


def setup_model():
    # model = ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     temperature=0,
    #     max_tokens=None,
    #     timeout=None,
    # )
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )
    return model

class ModelResponse():
    def __init__(self, current_message: str, history: list, character: str):
        self.current_message = current_message
        self.history = history
        self.character = character

    def get_response(self) -> str:
        chat_template = ChatPromptTemplate(messages=[
            ("system", f"You are an assistant of character: {self.character}. You can answer questions and help with tasks based on the chat history."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}"),
        ])
        prompt = chat_template.invoke({"query": self.current_message, "chat_history": self.history, "character": self.character})
        logger.info(f"Prompt: {prompt}")
        return model.invoke(prompt).content
        

model = setup_model()

if "history" not in st.session_state:
    st.session_state.history = []

clear_history_button = st.button("Clear History")
if clear_history_button:
    st.warning("Clearing history")
    st.session_state.history = []

character = st.selectbox("Select Character", ["Serious", "Hip hop artist", "Friend", "Comedian"])

# Display chat messages from history on app rerun
for message in st.session_state.history:
    with st.chat_message(message.type):
        st.markdown(message.content)


chat_input = st.chat_input("Enter a message")
if chat_input:
    
    response = ModelResponse(chat_input,st.session_state.history, character).get_response()
    st.session_state.history.append(HumanMessage(content=chat_input))
    st.session_state.history.append(AIMessage(content=response))
    logger.info(f"History after response: {st.session_state.history}")
    with st.chat_message("assistant"):
        st.write(response)

