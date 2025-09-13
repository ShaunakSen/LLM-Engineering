import streamlit as st

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

button = st.button("Generate Research")

if button:
    st.write("Generating research...")
    prompt = template.format(
        paper_input=paper_input, style_input=style_input, length_input=length_input
    )
    # display the prompt in a code block
    st.code(prompt)
    st.write("Generating research...")
    result = model.invoke(prompt)
    st.write(result.content)
