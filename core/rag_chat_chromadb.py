from langchain.schema import HumanMessage, SystemMessage
from core.documents_services import documents_services
from .prompts import PromptManager
from typing import List, Dict
from langchain_ollama import ChatOllama

class RAGChromaDB:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5",
            temperature=0.1
        )
        self.document_services = documents_services()
        self.prompt_manager = PromptManager()
        self.conversation_history: List[Dict] = []
    
    async def get_response(self, user_input: str) -> str:
        """Get response using RAG approach"""
        response = self.document_services.load_retriever_chain(user_input)
        
        return response['answer']
    
    async def _rephrase_query(self, query: str) -> str:
        """Rephrase the query for better retrieval"""
        messages = [
            SystemMessage(content=self.prompt_manager.rephrase_prompt["system"]),
            HumanMessage(content=self.prompt_manager.rephrase_prompt["human"].format(question=query))
        ]
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from ChromaDB"""
        results = self.index.query(
            vector=[0] * 1536,  # Replace with actual query embedding
            top_k=top_k,
            include_metadata=True
        )
        
        contexts = [match.metadata.get("text", "") for match in results.matches]
        return "\n".join(contexts)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
