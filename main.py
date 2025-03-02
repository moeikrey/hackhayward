import os
from dotenv import dotenv_values
import streamlit as st
from groq import Groq


st.set_page_config(
    page_title="The Tech Buddy ",
    page_icon="",
    layout="centered",
)


try:
    secrets = dotenv_values(".env")  # for dev env
    GROQ_API_KEY = secrets["GROQ_API_KEY"]
except Exception:
    secrets = st.secrets  # for streamlit deployment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INITIAL_RESPONSE = secrets["INITIAL_RESPONSE"]
INITIAL_MSG = secrets["INITIAL_MSG"]
CHAT_CONTEXT = secrets["CHAT_CONTEXT"]


# Initialize the chat history if present as Streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": INITIAL_RESPONSE
         },
    ]

client = Groq()
