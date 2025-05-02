import streamlit as st
from langchain import OpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.memory import ConversationBufferMemory
import pandas as pd

from dotenv import load_dotenv

load_dotenv()
import warnings


warnings.filterwarnings("ignore", category=UserWarning, module="langchain")

st.title("CSV Agent with LangChain")
st.write("This app allows you to upload a CSV file and interact with it using a language model agent.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv","xlsx"])

if uploaded_file is not None:

    csv_filepath = uploaded_file.name

    with open(csv_filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())


    df = pd.read_csv(csv_filepath)
    st.write("DataFrame Preview:")
    st.dataframe(df.head())


    agent = create_csv_agent(
        OpenAI(temperature=0), 
        csv_filepath, 
        verbose=True, 
        allow_dangerous_code=True
    )    

    query = st.text_input("Ask a question about the CSV file:")

    if query:
        resonse = agent.run(query)
        st.write("Response:")
        st.write(resonse)
    
