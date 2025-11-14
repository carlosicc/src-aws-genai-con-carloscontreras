# Credits: github.com/somilg050/rag-aws-bedrock/tree/master
import os
import boto3
import streamlit as st
import shutil
import logging
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_aws import ChatBedrock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load the Titan Embeddings using Bedrock client.
logger.info("Initializing Bedrock client and Titan embeddings...")
bedrock = boto3.client(service_name='bedrock-runtime')
titan_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",
                                     client=bedrock)
logger.info("Bedrock client and embeddings initialized successfully")

# Load the PDFs from the directory
def data_ingestion():
    logger.info("Starting data ingestion from PDF directory...")
    loader = PyPDFDirectoryLoader("data_pdf_electronics")
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from PDFs")
    
    # Split the text into chunks
    logger.info("Splitting documents into chunks (chunk_size=1000, overlap=250)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                   chunk_overlap=250)
    docs = text_splitter.split_documents(documents)
    logger.info(f"Created {len(docs)} text chunks from documents")
    return docs


# Vector Store for Vector Embeddings
def setup_vector_store(documents):
    logger.info(f"Creating FAISS vector store from {len(documents)} documents...")
    # Create a vector store using FAISS from the documents and the embeddings
    vector_store = FAISS.from_documents(
        documents,
        titan_embeddings,
    )
    logger.info("Vector store created successfully")
    
    # Save the vector store locally
    # - See: python.langchain.com/docs/integrations/vectorstores/faiss
    logger.info("Saving vector store to 'faiss_index_electronics'...")
    vector_store.save_local("faiss_index_electronics")
    logger.info("Vector store saved successfully")


# Load the LLM from the Bedrock
def load_llm():
    logger.info("Loading ChatBedrock LLM (Claude Sonnet 4.5)...")
    llm = ChatBedrock(model_id="global.anthropic.claude-sonnet-4-5-20250929-v1:0", client=bedrock, model_kwargs={"max_tokens": 512})
    logger.info("LLM loaded successfully")
    return llm


# System prompt for the agent
system_prompt = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If the answer is not within the context knowledge, kindly state that you do not know, rather than attempting to fabricate a response.
2. If you find the answer, please craft a detailed and concise response to the question at the end. Aim for a summary of max 250 words, ensuring that your explanation is thorough.

You have access to a retrieval tool that can search the knowledge base. Use it to find relevant information before answering."""


# Create a retrieval tool for the agent
def create_retrieval_tool(vector_store):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information from the QLED TV knowledge base."""
        logger.info(f"Retrieval tool called with query: '{query}'")
        retrieved_docs = vector_store.similarity_search(query, k=3)
        logger.info(f"Retrieved {len(retrieved_docs)} documents from vector store")
        
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        logger.debug(f"Serialized context length: {len(serialized)} characters")
        return serialized, retrieved_docs
    
    return retrieve_context


# Create an agent and get response
def get_response(llm, vector_store, query):
    logger.info(f"Processing query: '{query}'")
    
    # Create the retrieval tool
    logger.info("Creating retrieval tool for agent...")
    retrieval_tool = create_retrieval_tool(vector_store)
    
    # Create agent with the retrieval tool
    logger.info("Initializing LangChain agent with retrieval tool...")
    agent = create_agent(
        model=llm,
        tools=[retrieval_tool],
        system_prompt=system_prompt
    )
    logger.info("Agent created successfully")
    
    # Invoke the agent
    logger.info("Invoking agent to process query...")
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    logger.info("Agent completed processing")
    
    # Extract the final answer from the agent's response
    final_answer = response['messages'][-1].content
    logger.info(f"Generated response length: {len(final_answer)} characters")
    return final_answer


def streamlit_ui():
    st.set_page_config("RAG on Bedrock with LangChain")
    st.header("RAG on Bedrock, powered by LangChain")

    user_question = st.text_input("Ask me about your QLED TV; e.g. What can I do to prevent it from falling?")

    with st.sidebar:
        st.title("Carga PDF Docs")

        if st.button("Actualiza Vector Store"):
            with st.spinner("Procesando..."):
                logger.info("=== Starting Vector Store Update ===")
                docs = data_ingestion()
                setup_vector_store(docs)
                logger.info("=== Vector Store Update Complete ===")
                st.success("Listo!")
        
        if st.button("Borra Vector Store"):
            if os.path.exists("faiss_index_electronics"):
                logger.info("Deleting vector store directory...")
                shutil.rmtree("faiss_index_electronics")
                logger.info("Vector store deleted successfully")
                st.success("Vector store cleared successfully!")
            else:
                logger.warning("No vector store found to clear")
                st.info("No vector store found to clear.")

    if st.button("Generate Response") or user_question:
        if not os.path.exists("faiss_index_electronics"):
            st.error("Please create the vector store first from the sidebar.")
            return
        
        if not user_question:
            st.error("Please enter a question.")
            return
        
        with st.spinner("Processing..."):
            logger.info("=== Starting Query Processing ===")
            logger.info("Loading FAISS vector store from disk...")
            faiss_index = FAISS.load_local("faiss_index_electronics", embeddings=titan_embeddings,
                                           allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
            
            llm = load_llm()
            answer = get_response(llm, faiss_index, user_question)
            st.write(answer)
            logger.info("=== Query Processing Complete ===")
            st.success("Done")


if __name__ == "__main__":
    streamlit_ui()
