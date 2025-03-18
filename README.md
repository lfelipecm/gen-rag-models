# Comparison Chat Application

This application provides a side-by-side comparison between a generic LLM chat and a RAG-enhanced chat interface. Both chats process the same user input, allowing users to compare the responses and evaluate the benefits of RAG-based approaches.

## Features

- Dual chat interface
- Generic LLM chat using OpenAI's GPT models
- RAG-enhanced chat using Pinecone for vector search
- Dynamic prompt management with HuggingFace integration
- Modern, responsive UI with Streamlit

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with the following variables:
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
HUGGINGFACE_TOKEN=your_hf_token
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application
- `core/`
  - `llm_chat.py`: Generic LLM chat implementation
  - `rag_chat.py`: RAG-enhanced chat implementation
  - `prompts.py`: Prompt management and HuggingFace integration
- `styles/`
  - `main.css`: Application styling

## Note

Make sure to have the appropriate HuggingFace prompts repository set up and Pinecone index configured before running the application.
