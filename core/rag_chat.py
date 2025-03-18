from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from pinecone import Pinecone
from typing import List, Dict
import os
from .prompts import PromptManager

class RAGChat:
    def __init__(self, index_name: str):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        self.prompt_manager = PromptManager()
        self.conversation_history: List[Dict] = []
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(index_name)
    
    async def get_response(self, user_input: str) -> str:
        """Get response using RAG approach"""
        # First, rephrase the question for better retrieval
        rephrased_query = await self._rephrase_query(user_input)
        
        # Retrieve relevant context from Pinecone
        context = self._retrieve_context(rephrased_query)
        
        # Format prompt with context
        prompt = self.prompt_manager.retrieval_prompt["human"].format(
            context=context,
            question=user_input
        )
        
        messages = [
            SystemMessage(content=self.prompt_manager.retrieval_prompt["system"]),
            *[HumanMessage(content=msg["content"]) for msg in self.conversation_history if msg["role"] == "user"],
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        self.conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response.content}
        ])
        
        return response.content
    
    async def _rephrase_query(self, query: str) -> str:
        """Rephrase the query for better retrieval"""
        messages = [
            SystemMessage(content=self.prompt_manager.rephrase_prompt["system"]),
            HumanMessage(content=self.prompt_manager.rephrase_prompt["human"].format(question=query))
        ]
        response = await self.llm.ainvoke(messages)
        return response.content
    
    def _retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve relevant context from Pinecone"""
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
