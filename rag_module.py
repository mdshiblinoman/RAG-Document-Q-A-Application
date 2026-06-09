"""
RAG (Retrieval Augmented Generation) Module using Groq API
"""

import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import requests
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document


class RAGApplication:
    """RAG Application class for document Q&A using Groq API"""
    
    def __init__(self, grok_api_key: str):
        """Initialize RAG Application with Groq API key."""
        # Store Groq API key
        self.groq_api_key = grok_api_key
        # Initialize embeddings model (runs locally, no HuggingFace token required)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        # Load environment variables (if any additional)
        load_dotenv()
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.vectorstore = None
        self.chat_history: List = []
    
    def load_pdf(self, pdf_path: str) -> List:
        """Load PDF document"""
        loader = PyPDFLoader(pdf_path)
        return loader.load()
    
    def load_text(self, text_path: str) -> List:
        """Load text document"""
        loader = TextLoader(text_path, encoding='utf-8')
        return loader.load()
    
    def process_documents(self, documents: List) -> List:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def create_vectorstore(self, chunks: List) -> FAISS:
        """Create FAISS vectorstore from document chunks"""
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        return self.vectorstore
    
    def _format_docs(self, docs: List) -> str:
        """Combine docs but limit total length for LLM context."""
        max_len = 1500  # maximum characters sent to the model
        combined = "\n\n".join(doc.page_content for doc in docs)
        return combined[:max_len]
    
    def ask_question(self, question: str) -> dict:
        """Ask a question and get response based on documents"""
        if not self.vectorstore:
            raise ValueError("Vectorstore not initialized. Please upload documents first.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(question)
        context = self._format_docs(retrieved_docs)
        
        # Use Groq API for LLM inference
        api_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        
        # Build system prompt with context
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "If you cannot find the answer in the context, say so clearly.\n\n"
            f"Context:\n{context}"
        )
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        # Append chat history if any
        for msg in self.chat_history:
            role = "assistant" if isinstance(msg, AIMessage) else "user"
            messages.append({"role": role, "content": msg.content})
        # Add current user question
        messages.append({"role": "user", "content": question})
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512,
        }
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Grok API error: {response.status_code} - {response.text}")
        data = response.json()
        answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Update chat history
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        return {
            "answer": answer,
            "source_documents": retrieved_docs
        }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history.clear()
    
    def process_uploaded_file(self, file_path: str, file_type: str = "pdf"):
        """Process uploaded file and create vectorstore"""
        if file_type == "pdf":
            documents = self.load_pdf(file_path)
        elif file_type == "txt":
            documents = self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        chunks = self.process_documents(documents)
        self.create_vectorstore(chunks)
        
        return len(chunks)
    
    def process_text(self, text_content: str):
        """Process raw text content"""
        documents = [Document(page_content=text_content, metadata={"source": "user_input"})]
        chunks = self.process_documents(documents)
        self.create_vectorstore(chunks)
        return len(chunks)
    
    def add_documents(self, file_path: str, file_type: str = "pdf"):
        """Add more documents to existing vectorstore"""
        if file_type == "pdf":
            documents = self.load_pdf(file_path)
        elif file_type == "txt":
            documents = self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        chunks = self.process_documents(documents)
        
        if self.vectorstore:
            self.vectorstore.add_documents(chunks)
        else:
            self.create_vectorstore(chunks)
        
        return len(chunks)
