from langchain_core import embeddings
import streamlit as st
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader
import tempfile
import os
import asyncio
from PyPDF2 import PdfReader




# Initialize Ollama models and embeddings
@st.cache_resource
def init_models():
    model_name = "phi4"
    chat_model = ChatOllama(model=model_name, temperature=0.1)
    embeddings = OllamaEmbeddings(model=model_name, temperature=0.1)
    return chat_model, embeddings

# Initialize vector store for RAG
@st.cache_resource
def init_vectorstore():
    return Chroma(embedding_function=embeddings, persist_directory="./chroma_db")

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = {
            "generic": [],
            "rag": [],
            "summary": []
        }
    if "memories" not in st.session_state:
        st.session_state.memories = {
            "generic": ConversationBufferMemory(),
            "rag": ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            "summary": ConversationBufferMemory()
        }
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = init_vectorstore()

async def process_uploaded_files(files, vectorstore):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for file in files:
        file_extension = os.path.splitext(file.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_file.flush()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(tmp_file.name)
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(tmp_file.name)
            else:  # .txt files
                loader = TextLoader(tmp_file.name)
                
            documents = loader.load()
            chunks = text_splitter.split_documents(documents)
            vectorstore.add_documents(chunks)
        os.unlink(tmp_file.name)
    return vectorstore

async def summarize_text(file, chat_model):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    prompt = f"Please provide a long summary of the following text:\n\n{text}"
    return chat_model.predict(prompt)

async def chat(user_input, chat_model):
    return chat_model.predict(user_input)
    

def main():
    st.set_page_config(page_title="Multi-Purpose Chat App", layout="wide")
    st.title("Multi-Purpose Chat Application")
    
    # Initialize models and session state
    chat_model, embeddings = init_models()
    init_session_state()
    
    # Create tabs for different areas
    tab1, tab2, tab3, tab4 = st.tabs(["Generic Chat", "RAG Chat", "Summarization", "Instructions"])
    
    # Tab 1: Generic Chat
    with tab1:
        st.header("Generic Chat")
        if st.button("Clear chat", key="clear_generic"):
            st.session_state.messages["generic"] = []
            st.session_state.memories["generic"].clear()
        for message in st.session_state.messages["generic"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if generic_input := st.chat_input("Type your message here (Generic Chat)...", key="generic"):
            st.session_state.messages["generic"].append({"role": "user", "content": generic_input})
            with st.chat_message("user"):
                st.write(generic_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Getting response..."):
                    # Add message to memory
                    st.session_state.memories["generic"].save_context({"input": generic_input}, {"output": ""})
                    # Get chat history
                    chat_history = st.session_state.memories["generic"].load_memory_variables({})
                    # Generate response with context
                    response = asyncio.run(chat(f"Previous conversation: {chat_history}\n\nUser: {generic_input}", chat_model))
                    # Save response to memory
                    st.session_state.memories["generic"].save_context({"input": ""}, {"output": response})
                st.write(response)
                st.session_state.messages["generic"].append({"role": "assistant", "content": response})
    
    # Tab 2: RAG Chat
    with tab2:
        st.header("RAG Chat")
        if st.button("Clear chat", key="clear_rag"):
            st.session_state.messages["rag"] = []
            st.session_state.memories["rag"].clear()
        uploaded_files = st.file_uploader(
            "Upload files for RAG (PDF, Word, or Text files)", 
            accept_multiple_files=True,
            type=["pdf", "docx", "doc", "txt"],
            key="rag_files"
        )
        
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    st.session_state.vectorstore = asyncio.run(process_uploaded_files(
                        uploaded_files,
                        st.session_state.vectorstore
                    ))
                st.success("Files processed successfully!")
        
        for message in st.session_state.messages["rag"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if rag_input := st.chat_input("Type your message here (RAG Chat)...", key="rag"):
            st.session_state.messages["rag"].append({"role": "user", "content": rag_input})
            with st.chat_message("user"):
                st.write(rag_input)
            
            retriever = st.session_state.vectorstore.as_retriever()
            rag_chain = ConversationalRetrievalChain.from_llm(
                chat_model,
                retriever=retriever,
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )
            
            with st.chat_message("assistant"):
                response = rag_chain.invoke({"question": rag_input})["answer"]
                st.write(response)
                st.session_state.messages["rag"].append({"role": "assistant", "content": response})
    
    # Tab 3: Summarization
    with tab3:
        st.header("Text Summarization")
        if st.button("Clear chat", key="clear_summary"):
            st.session_state.messages["summary"] = []
            st.session_state.memories["summary"].clear()
        uploaded_file = st.file_uploader(
            "Upload a file to summarize (PDF, Word, or Text file)",
            type=["pdf", "docx", "doc", "txt"],
            key="summary_file"
        )
        
        if uploaded_file and st.button("Summarize"):
            with st.spinner("Generating summary..."):
                summary = asyncio.run(summarize_text(uploaded_file, chat_model))
                st.session_state.messages["summary"].append({"role": "assistant", "content": summary})
            
        for message in st.session_state.messages["summary"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Tab 4: Instructions
    with tab4:
        st.header("Instructions")
        st.markdown("""
        ### How to Use This App
        
        1. **Generic Chat**
           - Simply type your message and get responses from the AI
           - Use this for general questions and conversations
        
        2. **RAG Chat**
           - Upload one or more text files to create a knowledge base
           - Click 'Process Files' to add them to the vector database
           - Ask questions about the uploaded documents
           - The AI will use the relevant information to answer your questions
        
        3. **Summarization**
           - Upload a single text file
           - Click 'Summarize' to get a concise summary of the content
           - View the summary in the chat area below
        
        """)

if __name__ == "__main__":
    main()