from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from typing import List, Dict
from .prompts import PromptManager

class LLMChat:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7
        )
        self.prompt_manager = PromptManager()
        self.conversation_history: List[Dict] = []
    
    async def get_response(self, user_input: str) -> str:
        """Get response from the LLM"""
        messages = [
            SystemMessage(content=self.prompt_manager.retrieval_prompt["system"]),
            *[HumanMessage(content=msg["content"]) for msg in self.conversation_history if msg["role"] == "user"],
            HumanMessage(content=user_input)
        ]
        
        response = await self.llm.ainvoke(messages)
        
        self.conversation_history.extend([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response.content}
        ])
        
        return response.content
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
