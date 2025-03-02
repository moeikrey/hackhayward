import os
from dotenv import dotenv_values
import streamlit as st
from groq import Groq
import model as model

st.set_page_config(
    page_title="The Tech Buddy ",
    page_icon="",
    layout="centered",
)

try:
    secrets = dotenv_values(".env")
    GROQ_API_KEY = secrets["GROQ_API_KEY"]
except Exception:
    secrets = st.secrets  # streamlit deployment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

# Save the API key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INITIAL_RESPONSE = secrets["INITIAL_RESPONSE"]
INITIAL_MSG = secrets["INITIAL_MSG"]
CHAT_CONTEXT = secrets["CHAT_CONTEXT"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": INITIAL_RESPONSE
         },
    ]

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

client = Groq()

st.title("Welcome to ASL Buddy! 🤘")
st.caption("I'm here to assist with American Sign Language interpretation only. Let's start!")

# Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"], avatar='assets/asl-avatar.jpg'):
        st.markdown(message["content"])

user_prompt = st.chat_input("Let's chat!")


def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


if user_prompt:
    with st.chat_message("user", avatar="assets/asl-avatar.jpg"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt})

    messages = [
        {"role": "system", "content": CHAT_CONTEXT
         },
        {"role": "assistant", "content": INITIAL_MSG},
        *st.session_state.chat_history
    ]

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        stream=True
    )
    response = st.write_stream(parse_groq_stream(stream))
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response})

col1, col2 = st.columns(2)

with col1:
    if st.button("Start ASL Sign Detection"):
        st.session_state.camera_running = True

with col2:
    if st.button("Stop ASL Sign Detection"):
        st.session_state.camera_running = False

if st.session_state.camera_running:
    st_frame = st.empty()  # Placeholder for video feed
    try:
        for frame in model.start_camera():
            st_frame.image(frame, channels="RGB")  # Display frame in Streamlit
            if not st.session_state.camera_running:
                break
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        st.session_state.camera_running = False
        st_frame.empty()
