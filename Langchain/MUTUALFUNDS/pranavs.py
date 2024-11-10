import os
import streamlit as st
import pickle
import time
from typing import List, Optional
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from urllib.parse import urlparse
import tempfile
import json

# Configuration
AZURE_API_VERSION = "2023-05-15"
AZURE_DEPLOYMENT = "gpt-4o"
AZURE_ENDPOINT = ""
MAX_URLS = 3
CHUNK_SIZE = 10000
FILE_PATH = "faiss_store_openai2.pkl"
METADATA_PATH = "url_metadata.json"

def init_page() -> None:
    """Initialize Streamlit page with title and configuration."""
    st.set_page_config(page_title="RockyBot: News Research Tool", page_icon="📈")
    st.title("RockyBot: News Research Tool 📈")
    st.sidebar.title("News Article URLs")

def validate_url(url: str) -> bool:
    """Validate if the provided URL is properly formatted."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def init_azure_clients(api_key: str):
    """Initialize Azure OpenAI clients with proper configuration."""
    llm = AzureChatOpenAI(
        openai_api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key,
        streaming=True,
        temperature=0.1,
        max_tokens=500
    )

    embedding = AzureOpenAIEmbeddings(
        openai_api_version=AZURE_API_VERSION,
        azure_deployment="text-embedding-3-small",
        azure_endpoint=AZURE_ENDPOINT,
        api_key=api_key
    )

    return llm, embedding

def save_url_metadata(urls: List[str]) -> None:
    """Save URL metadata separately from the FAISS index."""
    metadata = {
        "urls": urls,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

def load_url_metadata() -> dict:
    """Load URL metadata if it exists."""
    try:
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"urls": [], "timestamp": None}

def process_urls(urls: List[str], embedding_model, status_placeholder) -> Optional[FAISS]:
    """Process URLs and create FAISS index."""
    try:
        # Filter out empty URLs and validate
        valid_urls = [url.strip() for url in urls if url.strip() and validate_url(url)]
        
        if not valid_urls:
            st.error("Please provide at least one valid URL.")
            return None

        # Load and process data
        loader = UnstructuredURLLoader(urls=valid_urls)
        status_placeholder.text("Loading data from URLs... ⏳")
        data = loader.load()

        if not data:
            st.error("No content could be loaded from the provided URLs.")
            return None

        # Split text
        status_placeholder.text("Processing text... ⏳")
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=CHUNK_SIZE
        )
        docs = text_splitter.split_documents(data)

        # Create embeddings
        status_placeholder.text("Creating embeddings... ⏳")
        vectorstore = FAISS.from_documents(docs, embedding_model)

        # Save metadata separately
        save_url_metadata(valid_urls)

        # Save FAISS index using a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "temp_faiss")
            
            # Remove existing FAISS store if it exists
            if os.path.exists(FILE_PATH):
                try:
                    os.remove(FILE_PATH)
                except Exception as e:
                    st.error(f"Failed to remove existing index file: {str(e)}")
                    return None
                    
            vectorstore.save_local(temp_path)
            
            # Load and save again to ensure clean serialization
            new_vectorstore = FAISS.load_local(temp_path, embedding_model, allow_dangerous_deserialization=True)
            
            try:
                new_vectorstore.save_local(FILE_PATH)
            except Exception as e:
                st.error(f"Failed to save new index file: {str(e)}")
                return None

        status_placeholder.text("Processing complete! ✅")
        return new_vectorstore

    except Exception as e:
        st.error(f"An error occurred while processing URLs: {str(e)}")
        return None

def main():
    init_page()

    # Secure API key input
    api_key = st.sidebar.text_input("Enter Azure OpenAI API Key", type="password")
    
    if not api_key:
        st.warning("Please enter your Azure OpenAI API key to continue.")
        return

    # URL inputs
    urls = []
    for i in range(MAX_URLS):
        url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
        urls.append(url)

    # Initialize Azure clients
    llm, embedding_model = init_azure_clients(api_key)

    # Process URLs button
    process_url_clicked = st.sidebar.button("Process URLs")
    
    # Main content area
    main_placeholder = st.empty()
    
    # Display currently processed URLs
    metadata = load_url_metadata()
    if metadata["timestamp"]:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Currently processed URLs")
        for url in metadata["urls"]:
            st.sidebar.markdown(f"- {url}")
        st.sidebar.markdown(f"Last processed: {metadata['timestamp']}")
    
    if process_url_clicked:
        vectorstore = process_urls(urls, embedding_model, main_placeholder)
        if vectorstore:
            st.success("URLs processed successfully!")

    # Query input and processing
    query = st.text_input("Question: ")
    
    if query:
        try:
            if not os.path.exists(FILE_PATH):
                st.error("Please process URLs before asking questions.")
                return

            # Load FAISS index
            vectorstore = FAISS.load_local(FILE_PATH, embedding_model,allow_dangerous_deserialization=True)

            chain = RetrievalQAWithSourcesChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever()
            )

            with st.spinner("Generating answer..."):
                result = chain({"question": query}, return_only_outputs=True)

            # Display results
            if "answer" in result:
                st.header("Answer")
                st.write(result["answer"])

                if "sources" in result and result["sources"]:
                    st.subheader("Sources")
                    for source in result["sources"].split("\n"):
                        if source.strip():
                            st.write(source)
            else:
                st.warning("No answer was generated. Please try rephrasing your question.")

        except Exception as e:
            st.error(f"An error occurred while processing your question: {str(e)}")

if __name__ == "__main__":
    main()