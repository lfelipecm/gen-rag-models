import streamlit as st
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
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
    model_name = "gemma3:12b"
    embedding_model = "llama3"
    chat_model = ChatOllama(model=model_name, temperature=0.1)
    embeddings = OllamaEmbeddings(model=embedding_model, temperature=0.1)
    return chat_model, embeddings

# Initialize vector store for RAG
@st.cache_resource
def init_vectorstore(_embeddings):
    return Chroma(embedding_function=_embeddings, persist_directory="./chroma_db")

# Initialize session state
def init_session_state(_embeddings, chat_model):
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
        st.session_state.vectorstore = init_vectorstore(_embeddings)
    if "llm" not in st.session_state:
        st.session_state.llm = chat_model

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

async def summarize_text(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    prompt = f"“Can you provide a comprehensive summary of the given text:\n\n{text}? \
    The summary should cover all the key points and main ideas presented in \
    the original text, while also condensing the information into a concise and \
        easy-to-understand format. Please ensure that the summary includes relevant \
            details and examples that support the main ideas, while avoiding any \
                unnecessary information or repetition. The length of the summary \
                    should be appropriate for the length and complexity of the original \
                        text, providing a clear and accurate overview without omitting any \
                        important information.”"
                        
    return st.session_state.llm.predict(prompt)

async def chat(user_input):
    return st.session_state.llm.predict(user_input)
    

def main():
    st.set_page_config(page_title="Multi-Purpose Chat App", layout="wide", page_icon="🤖")
    st.title("🤖 Multi-Purpose Chat Application")
    
    # Initialize models and session state
    chat_model, embeddings = init_models()
    init_session_state(embeddings, chat_model)
    
    # Create tabs for different areas
    tab1, tab2, tab3 = st.tabs(["Generic Chat", "RAG Chat", "Summarization"])
    
    # Tab 1: Generic Chat
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Generic Chat")
            st.markdown("""
                - Simply type your message and get responses from the AI
                - Use this for general questions and conversations
                """)
            if st.button("Clear chat", key="clear_generic"):
                st.session_state.messages["generic"] = []
                st.session_state.memories["generic"].clear()
            if generic_input := st.chat_input("Type your message here (Generic Chat)...", key="generic"):
                st.session_state.messages["generic"].append({"role": "user", "content": generic_input})
                with st.spinner("Getting response..."):
                    # Add message to memory
                    st.session_state.memories["generic"].save_context({"input": generic_input}, {"output": ""})
                    # Get chat history
                    chat_history = st.session_state.memories["generic"].load_memory_variables({})
                    # Generate response with context
                    response = asyncio.run(chat(f"Previous conversation: {chat_history}\n\nUser: {generic_input}"))
                    # Save response to memory
                    st.session_state.memories["generic"].save_context({"input": ""}, {"output": response})
                    st.session_state.messages["generic"].append({"role": "assistant", "content": response})
        with col2:
            st.header("Conversation")
            for message in st.session_state.messages["generic"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    
    # Tab 2: RAG Chat
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.header("RAG Chat")
            st.markdown("""
                - Upload one or more text files to create a knowledge base
                - Click 'Process Files' to add them to the vector database
                - Ask questions about the uploaded documents
                - The AI will use the relevant information to answer your questions
                """)
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
            
            if rag_input := st.chat_input("Type your message here (RAG Chat)...", key="rag"):
                st.session_state.messages["rag"].append({"role": "user", "content": rag_input})
                retriever = st.session_state.vectorstore.as_retriever()
                rag_chain = ConversationalRetrievalChain.from_llm(
                    st.session_state.llm,
                    retriever=retriever,
                    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                )
                
                with st.spinner("Processing files..."):
                    response = rag_chain.invoke({"question": rag_input})["answer"]
                    st.session_state.messages["rag"].append({"role": "assistant", "content": response})
        
        with col2:
            st.header("Conversation")
            for message in st.session_state.messages["rag"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
    
    # Tab 3: Summarization
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Text Summarization")
            st.markdown("""
                - Upload a single text file
                - Click 'Summarize' to get a concise summary of the content
                - View the summary in the chat area below
                """)
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
                    summary = asyncio.run(summarize_text(uploaded_file))
                    st.session_state.messages["summary"].append({"role": "assistant", "content": summary})
        
        with col2:
            st.header("Conversation")
            for message in st.session_state.messages["summary"]:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

if __name__ == "__main__":
    main()