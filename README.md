# 📚 RAG Document Q&A Application

A RAG (Retrieval Augmented Generation) based document Q&A application built with Google Gemini API.

## ✨ Features

- 📄 PDF and TXT file support
- 💬 Conversational Q&A with memory
- 🔍 Semantic search with FAISS vector store
- 🌐 Beautiful Streamlit web interface
- 🔄 Multi-document support

## 🚀 Installation

### 1. Clone the repository or download the files

```bash
cd /home/noman/MyFiles/RAG
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click the "Create API Key" button
3. Copy the API key

## 🎮 Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

## 📖 User Guide

1. **Set API Key:** Enter your Gemini API Key in the sidebar
2. **Upload Documents:** Upload PDF or TXT files
3. **Process:** Click the "Process Documents" button
4. **Ask Questions:** Type your question in the chat input

### Alternative: Direct Text Input

You can also enter text directly without uploading files.

## 🛠️ Project Structure

```
RAG/
├── app.py              # Streamlit web application
├── rag_module.py       # RAG core functionality
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .env.example        # Environment variables example
```

## 🔧 Configuration

### RAG Parameters (can be modified in rag_module.py)

- `chunk_size`: 1000 (document chunk size)
- `chunk_overlap`: 200 (overlap between chunks)
- `temperature`: 0.3 (Gemini response randomness)
- `k`: 3 (number of retrieved documents)

## 📝 Code Example (Using from Python)

```python
from rag_module import RAGApplication

# Initialize
rag = RAGApplication(api_key="your-gemini-api-key")

# Load PDF
rag.process_uploaded_file("document.pdf", "pdf")

# Ask question
response = rag.ask_question("What is this document about?")
print(response["answer"])
```

## ⚠️ Troubleshooting

### "API Key invalid" error
- Check if your API key is correct
- Generate a new key from Google AI Studio

### "Module not found" error
```bash
pip install -r requirements.txt
```

### PDF loading error
- Make sure the PDF file is not corrupted
- Check if the PDF has selectable text

## 📄 License

MIT License - Free to use for any purpose.

## 🤝 Contributing

Pull requests are welcome! Open an issue for any improvements.

---

**Made with ❤️ using Google Gemini & Streamlit**
