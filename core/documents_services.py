from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

class documents_services:
    def __init__(self):
        """Initialize the load_documents class"""
        self.embeddings = OllamaEmbeddings(
            model="mxbai-embed-large:latest",
            temperature=0.7
        )
        self.llm = ChatOllama(
            model="qwen2.5",
            temperature=0.1
        )
        self.vector_store = None
        self.retriever = None
    

    def upload_documents(self, directory_path: str):
        """Load documents from a directory and create a vector store"""
        loader = DirectoryLoader(directory_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        
        # directory for DB persistence
        persist_directory = "./db"

        # Create embeddings and vector store
        
        vector_store = Chroma.from_documents(documents=docs, embedding=self.embeddings, persist_directory=persist_directory)
        # Persist the vector store
        vector_store.persist()
        vector_store = None

    def load_retriever_chain(self, query: str):
        if self.vector_store is None:
            self.vector_store = Chroma(persist_directory="./db", embedding_function=self.embeddings)

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        system_prompt = (
            "Use the given context to answer the question. "
            "Do not include your own opinion or analysis. "
            "Be concise and clear. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Use the context to answer the question {context} "
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        # Use the LLM to generate a responseRetrievalQA
        qa_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=qa_chain,

        )
        return chain.invoke({"input": query})
   
    def summarize_document(self, path: str):

        """Customize the prompt for summarization"""
        prompt_template = """Write a summary of the following document.
            Keep the summary within 2000 words.
            Only include information that is part of the document. 
            Do not include your own opinion or analysis.
            Be concise and clear.

            Document:
            "{context}"
            Summary:"""
        prompt = PromptTemplate.from_template(prompt_template)
        """Summarize a document using the LLM"""
        
        # Load the document
        loader = PyPDFLoader(path)
        documents = loader.load()
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(documents)
        
        # Create a summary chain
        summary_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        # Generate the summary
        summary = summary_chain.invoke({"context": docs})
        
        return summary
    
if __name__ == "__main__":
    #directory_path = "/Users/felipe/Workspace/gen-rag-models/documents"
    loader = documents_services()
    #vector_store = loader.upload_documents(directory_path)
    #print("Documents loaded and vector store created.")

    # full example
    #query = "What is AWS Well Architected Framework?"
    #llm_response = loader.load_retriever_chain(query)
    #print("LLM Response:", llm_response['answer'])

    doc_path = "/Users/felipe/Workspace/gen-rag-models/documents/wellarchitected-machine-learning-lens.pdf"
    summary = loader.summarize_document(doc_path)
    print("LLM Response:", summary)


