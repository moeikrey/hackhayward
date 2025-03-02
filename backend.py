# import requests
# import os
# from dotenv import load_dotenv 
# from groq import Groq
# import logging
# import time
# from urllib.request import ProxyHandler

# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# PROXY_URL = os.getenv("PROXY_URL")
# BASE_URL = "https://api.groq.ai/v1/completions"

# logging.basicConfig(level=logging.ERROR)

# def get_groq_response(prompt):
#     if not prompt:
#         logging.error("Prompt is empty")
#         return None

#     try:
#         proxy_handler = ProxyHandler({"http": PROXY_URL, "https": PROXY_URL})
#         session = requests.Session()
#         session.proxies = {"http": PROXY_URL, "https": PROXY_URL}

#         headers = {
#             'Authorization': f'Bearer {GROQ_API_KEY}',
#             'Content-Type': 'application/json'
#         }
#         data = {
#             'prompt': prompt,
#             'max_tokens': 100
#         }
#         response = session.post(f'{BASE_URL}', headers=headers, json=data)
#         if response.status_code == 200:
#             return response.json()
#         else:
#             logging.error(f"Failed to get response. Status code: {response.status_code}")
#             return None
#     except Exception as e:
#         logging.error(f"An error occurred: {str(e)}")
#         return None

# prompt = "Hello, Groq AI!"
# attempts = 0
# while attempts < 5:
#     response = get_groq_response(prompt)
#     if response:
#         logging.info(f"Response: {response}")
#         break
#     attempts += 1
#     time.sleep(1)  # Wait for 1 second before trying again
import os
from dotenv import dotenv_values
import streamlit as st
from groq import Groq


def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


# streamlit page configuration
st.set_page_config(
    page_title="The Tech Buddy 🧑‍💻",
    page_icon="🤖",
    layout="centered",
)


try:
    secrets = dotenv_values(".env")  # for dev env
    GROQ_API_KEY = secrets["GROQ_API_KEY"]
except:
    secrets = st.secrets  # for streamlit deployment
    GROQ_API_KEY = secrets["GROQ_API_KEY"]

# save the api_key to environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

INITIAL_RESPONSE = secrets["INITIAL_RESPONSE"]
INITIAL_MSG = secrets["INITIAL_MSG"]
CHAT_CONTEXT = secrets["CHAT_CONTEXT"]


client = Groq()

# initialize the chat history if present as streamlit session
if "chat_history" not in st.session_state:
    # print("message not in chat session")
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": INITIAL_RESPONSE
         },
    ]

# page title
st.title("Welcome Hacker!")
st.caption("Helping You Level Up Your Coding Game.")
# the messages in chat_history will be stored as {"role":"user/assistant", "content":"msg}
# display chat history
for message in st.session_state.chat_history:
    # print("message in chat session")
    with st.chat_message("role", avatar='🤖'):
        st.markdown(message["content"])


# user input field
user_prompt = st.chat_input("Ask me")

if user_prompt:
    # st.chat_message("user").markdown
    with st.chat_message("user", avatar="🗨️"):
        st.markdown(user_prompt)
    st.session_state.chat_history.append(
        {"role": "user", "content": user_prompt})

    # get a response from the LLM
    messages = [
        {"role": "system", "content": CHAT_CONTEXT
         },
        {"role": "assistant", "content": INITIAL_MSG},
        *st.session_state.chat_history
    ]

    # Display assistant response in chat message container
    with st.chat_message("assistant", avatar='🤖'):
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            stream=True  # for streaming the message
        )
        response = st.write_stream(parse_groq_stream(stream))
    st.session_state.chat_history.append(
        {"role": "assistant", "content": response})
