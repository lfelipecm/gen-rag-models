from huggingface_hub import hf_hub_download
from typing import Dict, Any
import json
import os

class PromptManager:
    def __init__(self):
        self.retrieval_prompt = self._load_prompt("retrieval")
        self.rephrase_prompt = self._load_prompt("rephrase")
    
    def _load_prompt(self, prompt_type: str) -> Dict[str, Any]:
        """Load prompt from HuggingFace Hub"""
        try:
            # Note: Replace with actual HF repo and file names
            repo_id = "your-hf-repo/prompts"
            filename = f"{prompt_type}_prompt.json"
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=os.getenv("HUGGINGFACE_TOKEN")
            )
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            # Fallback prompts if HF loading fails
            fallback_prompts = {
                "retrieval": {
                    "system": "You are a helpful assistant that provides accurate information based on the given context.",
                    "human": "Using the following context, please answer the question:\nContext: {context}\nQuestion: {question}"
                },
                "rephrase": {
                    "system": "You are a helpful assistant that rephrases questions to improve search results.",
                    "human": "Rephrase the following question to make it more suitable for search: {question}"
                }
            }
            return fallback_prompts.get(prompt_type, {})
