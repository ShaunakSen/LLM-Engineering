import streamlit as st
import numpy as np

# models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# prompts
from langchain_core.prompts import load_prompt

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


def get_assistant_response(user_message):
    return model.invoke(f"{user_message}").content


model = setup_model()

st.title("Langchain Prompt UI")

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)

template = load_prompt("./prompt_templates/template.json")

clear_history_button = st.button("Clear History")

if clear_history_button:
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_message := st.chat_input("Enter a message"):
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.write(user_message)

    # compute the assistant response
    response = get_assistant_response(user_message)
    with st.chat_message("assistant"):
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})