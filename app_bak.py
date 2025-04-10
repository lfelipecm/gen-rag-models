import streamlit as st
import asyncio
from core.llm_chat import LLMChat
from core.rag_chat import RAGChat
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load CSS
def load_css():
    with open("styles/main.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize chat instances
@st.cache_resource
def init_chats():
    return LLMChat(), RAGChat(index_name=os.getenv("PINECONE_INDEX_NAME", "default-index"))

def main():
    st.set_page_config(page_title="Comparison Chat", layout="wide")
    load_css()
    
    st.title("LLM vs RAG Chat Comparison")
    
    # Initialize chat instances
    llm_chat, rag_chat = init_chats()
    
    # Initialize session state for chat histories
    if "messages" not in st.session_state:
        st.session_state.messages = {
            "llm": [],
            "rag": []
        }
    
    # Create two columns for the chats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chat-title">Generic LLM Chat</div>', unsafe_allow_html=True)
        for message in st.session_state.messages["llm"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    with col2:
        st.markdown('<div class="chat-title">RAG-Enhanced Chat</div>', unsafe_allow_html=True)
        for message in st.session_state.messages["rag"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to both chats
        st.session_state.messages["llm"].append({"role": "user", "content": user_input})
        st.session_state.messages["rag"].append({"role": "user", "content": user_input})
        
        # Get responses from both models
        async def get_responses():
            llm_response = await llm_chat.get_response(user_input)
            rag_response = await rag_chat.get_response(user_input)
            return llm_response, rag_response
        
        with st.spinner("Getting responses..."):
            llm_response, rag_response = asyncio.run(get_responses())
        
        # Add assistant responses to both chats
        st.session_state.messages["llm"].append({"role": "assistant", "content": llm_response})
        st.session_state.messages["rag"].append({"role": "assistant", "content": rag_response})
        
        # Rerun to update the UI
        st.rerun()

    # Add a clear button
    if st.button("Clear Chat"):
        st.session_state.messages = {"llm": [], "rag": []}
        llm_chat.clear_history()
        rag_chat.clear_history()
        st.rerun()

if __name__ == "__main__":
    main()