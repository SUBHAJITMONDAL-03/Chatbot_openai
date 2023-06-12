import streamlit as st
import pandas as pd
from pandasai import PandasAI
from dotenv import load_dotenv
from pandasai.llm.openai import OpenAI
import matplotlib
import sys
import os
matplotlib.use("TkAgg")
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


class StreamlitConsole:
    def __init__(self, st_output):
        self.st_output = st_output
        self.console_output = ""

    def write(self, message):
        self.console_output += message + "\n"

    def flush(self):
        pass


def chat_with_csv(dataframes, prompt):
    llm = OpenAI(api_token=openai_api_key)
    pandas_ai = PandasAI(llm, verbose=True)

    # Redirect console output to Streamlit
    sys.stdout = StreamlitConsole(st)

    result = ""
    for df in dataframes:
        result += pandas_ai.run(df, prompt=prompt) + "\n\n"

    console_output = sys.stdout.console_output

    # Reset console output
    sys.stdout = sys.__stdout__

    return result, console_output


class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


st.set_page_config(layout='wide')
st.title("PwCbot🤖")

input_csv = st.file_uploader(
    "Upload your CSV files", type=['csv'], accept_multiple_files=True)

dataframes = []
if input_csv is not None:
    
    for file in input_csv:
        data = pd.read_csv(file)
        dataframes.append(data)

if input_csv is not None:
   
    col1, col2 = st.columns([1, 1])

    with col1:

        #st.info("CSV Uploaded Successfully")
        for i, data in enumerate(dataframes):
            st.subheader(f"CSV File {i+1}")
            st.dataframe(data, use_container_width=True)
            st.info("CSV Uploaded Successfully")

    with col2:

        st.info("Chat Below")

        # Retrieve the session state
        if 'state' not in st.session_state:
            st.session_state.state = SessionState(prompt="")

        input_text = st.text_area(
            "Enter your query", key="input_area", value=st.session_state.state.prompt)

        if input_text is not None:
            st.session_state.state.prompt = input_text

            if st.button("Chat with CSV"):
                st.info("Your Query: " + input_text)
                result, console_output = chat_with_csv(dataframes, input_text)
                st.success(result)

                st.subheader("Console Output:")
                st.code(console_output)

        if st.button("Reset"):
            st.session_state.state.prompt = ""
            st.experimental_rerun()
