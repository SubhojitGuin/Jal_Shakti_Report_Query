import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_core.output_parsers import StrOutputParser
from typing import AsyncGenerator
import time
import asyncio

load_dotenv()
# genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

st.set_page_config("Chat with multiple PDFs")
st.header("Chat with multiple PDFs using GPT💁")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000 , chunk_overlap =1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    embeddings = OpenAIEmbeddings()
    #model = "gpt-3.5-turbo-0125"
    vector_store = FAISS.from_texts(text_chunks, embedding = embeddings)
    return vector_store

def get_conversational_chain():
    prompt_template = """
        Answer the following question in a detailed manner based only on the provided context. 
        Think step by step before providing an answer,make sure to provide all the details
        I will tip you $1000 if the user finds the answer helpful.
        if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n 
        <context>
        {context}
        </context>
        Question: {question}"""
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0 , streaming=True, stream_usage=True)
    #  model = ChatGoogleGenerativeAI(model="gemini-1.5-flash" , temeperature = 0.3)
    prompt = PromptTemplate(template = prompt_template , input_variables={"context","question"})
    #  chain = load_qa_chain(model , chain_type="stuff" , prompt=prompt)
    chain = prompt | llm | StrOutputParser()
    return chain

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def user_input(user_question):
    start_time = time.time()
    # embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    # embeddings = OpenAIEmbeddings()
    
    new_db = st.session_state.get("vector_store")
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    for r in chain.stream(
      {"context": docs, "question": user_question},
      ):
        yield r

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            start_time = time.time()
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)
            st.session_state.vector_store = vector_store
            st.success(f"Done \n Time Taken: {time.time() - start_time}" )

if prompt := st.chat_input("Enter your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # response1 = user_input(prompt)
        # response1 = next(response1)
        response = st.write_stream(user_input(prompt))
    st.session_state.messages.append({"role": "assistant", "content": response})

