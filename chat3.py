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
import logging

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up Streamlit
st.set_page_config("Chat with multiple PDFs")
st.header("Chat with multiple PDFs using GPT💁")

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:  # Ensure we handle pages with no extractable text
                    text += extracted_text
                else:
                    logging.warning(f"No text extracted from a page in {pdf.name}")
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {str(e)}")
            logging.error(f"Error reading {pdf.name}: {str(e)}")
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text into chunks: {str(e)}")
        logging.error(f"Error splitting text into chunks: {str(e)}")
        return []

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        logging.error(f"Error creating vector store: {str(e)}")
        return None

# Function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
        Answer the following question in a detailed manner based only on the provided context.
        Think step by step before providing an answer, and make sure to provide all the details.
        I will tip you $1000 if the user finds the answer helpful.
        If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
        <context>
        {context}
        </context>
        Question: {question}
    """
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True, stream_usage=True)
        prompt = PromptTemplate(template=prompt_template, input_variables={"context", "question"})
        chain = prompt | llm | StrOutputParser()
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        logging.error(f"Error creating conversational chain: {str(e)}")
        return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to handle user input and generate a response
def user_input(user_question):
    try:
        new_db = st.session_state.get("vector_store")
        if not new_db:
            raise ValueError("No vector store available. Please upload and process PDFs first.")
        
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        
        if chain is None:
            raise ValueError("Conversational chain could not be initialized.")
        
        for r in chain.stream({"context": docs, "question": user_question}):
            yield r
    except Exception as e:
        st.error(f"Error processing your input: {str(e)}")
        logging.error(f"Error processing user input: {str(e)}")

# Sidebar for PDF upload and processing
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    
    if st.button("Submit & Process"):
        if pdf_docs:
            with st.spinner("Processing..."):
                start_time = time.time()
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text:
                        raise ValueError("No text extracted from the uploaded PDFs.")
                    
                    text_chunks = get_text_chunks(raw_text)
                    if not text_chunks:
                        raise ValueError("Failed to split the text into chunks.")
                    
                    vector_store = get_vector_store(text_chunks)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.success(f"Processing completed. Time Taken: {time.time() - start_time:.2f} seconds")
                    else:
                        raise ValueError("Vector store creation failed.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    logging.error(f"Error in processing: {str(e)}")
        else:
            st.warning("Please upload at least one PDF file before submitting.")

# Chat input and response display
if prompt := st.chat_input("Enter your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            response = st.write_stream(user_input(prompt))
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error generating assistant response: {str(e)}")
            logging.error(f"Error generating assistant response: {str(e)}")
