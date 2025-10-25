# 🤖 AI RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content using advanced AI technology.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- 📄 **PDF Document Upload**: Drag-and-drop or click to upload PDF files (up to 50MB)
- 💬 **Intelligent Chat Interface**: Ask questions about uploaded documents and receive context-aware answers
- 🔍 **Semantic Search**: Uses vector embeddings for accurate document retrieval
- 💾 **Persistent Chat History**: Conversations are stored in Firestore and persist across sessions
- 🔄 **Session Management**: Support for multiple independent chat sessions
- 🎨 **Modern UI**: Beautiful, responsive interface with smooth animations
- ⚡ **Real-time Responses**: Fast processing with loading indicators

## 🏗️ Architecture

The application follows a client-server architecture with the following components:

```
┌─────────────────┐
│   Web Browser   │
│   (Frontend)    │
└────────┬────────┘
         │ HTTP/AJAX
         ▼
┌─────────────────┐
│  Flask Server   │
│   (Backend)     │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────────┐
    ▼         ▼          ▼              ▼
┌────────┐ ┌──────┐  ┌──────┐    ┌──────────┐
│ FAISS  │ │ LLM  │  │ PDF  │    │Firestore │
│Vector  │ │Model │  │Loader│    │ (Chat    │
│ Store  │ │(API) │  │      │    │ History) │
└────────┘ └──────┘  └──────┘    └──────────┘
```

### Architecture Flow

1. **Document Upload**: User uploads PDF → Backend processes with PyPDFLoader → Document chunked → Embeddings generated → Stored in FAISS vector database
2. **Query Processing**: User asks question → Semantic search retrieves relevant chunks → Context + query sent to LLM → Response generated and returned
3. **Chat History**: All conversations stored in Google Firestore for persistence and retrieval

## 🛠️ Technologies Used

### Backend

- **Flask**: Web framework for Python
- **LangChain**: Framework for building LLM applications
  - `langchain-huggingface`: HuggingFace embeddings integration
  - `langchain-community`: Vector stores and document loaders
  - `langchain-openai`: OpenAI-compatible LLM integration
  - `langchain-google-firestore`: Firestore chat history management
- **FAISS**: Vector similarity search library
- **PyPDF**: PDF document parsing
- **Google Cloud Firestore**: NoSQL database for chat history

### Frontend

- **HTML5/CSS3**: Modern, responsive UI
- **Vanilla JavaScript**: Client-side functionality
- **Gradient Design**: Beautiful purple gradient theme

### AI/ML

- **Sentence Transformers**: `all-MiniLM-L6-v2` for document embeddings
- **OpenAI-compatible API**: For language model inference (supports OpenAI, Azure OpenAI, or compatible endpoints)

## 📦 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Google Cloud Project with Firestore enabled
- OpenAI API key or compatible LLM API endpoint
- Service account credentials for Firestore

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-rag-assistant.git
cd ai-rag-assistant
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Google Cloud Setup

1. Create a Google Cloud Project
2. Enable Firestore API
3. Create a service account with Firestore permissions
4. Download the service account JSON key file

### 2. Environment Variables

Create a `.env` file in the project root:

```env
# Flask Configuration
FLASK_SECRET_KEY=your-secret-key-here

# LLM Configuration
MODEL_NAME=gpt-3.5-turbo
API_KEY=your-openai-api-key
BASE_URL=https://api.openai.com/v1

# Firestore Configuration
SERVICE_ACCOUNT_PATH=path/to/your/service-account-key.json
FIRESTORE_PROJECT_ID=your-firestore-project-id
```

### Configuration Options

- **MODEL_NAME**: The LLM model to use (e.g., `gpt-3.5-turbo`, `gpt-4`)
- **API_KEY**: Your OpenAI or compatible API key
- **BASE_URL**: API endpoint (change for Azure OpenAI or other providers)
- **SERVICE_ACCOUNT_PATH**: Path to your Google Cloud service account JSON file
- **FIRESTORE_PROJECT_ID**: Your Google Cloud project ID

## 🎯 Running the Application

### 1. Start the Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 2. Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## 📖 Usage

### Uploading a Document

1. Click the upload area or drag and drop a PDF file
2. Wait for the upload and processing to complete
3. The chat interface will appear once processing is done

### Asking Questions

1. Type your question in the input field
2. Press Enter or click the "Send" button
3. The AI will respond based on the document content

### Managing Sessions

- **New Chat**: Click "New Chat" to start a fresh conversation (keeps the same document)
- **Upload PDF**: Click "Upload PDF" to load a new document

## 📁 Project Structure

```
ai-rag-assistant/
│
├── app.py                  # Main Flask application
├── templates/
│   └── index.html         # Frontend interface
├── uploads/               # Uploaded PDFs and FAISS indices (created automatically)
├── .env                   # Environment variables (not in repo)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🔄 How It Works

### Document Processing Pipeline

1. **Upload**: PDF file is uploaded and saved securely
2. **Loading**: PyPDFLoader extracts text from the PDF
3. **Chunking**: RecursiveCharacterTextSplitter breaks text into manageable chunks (1000 chars, 200 overlap)
4. **Embedding**: Each chunk is converted to a vector using `all-MiniLM-L6-v2`
5. **Indexing**: Vectors are stored in FAISS for fast similarity search

### Query Processing Pipeline

1. **Retrieval**: User query is embedded and compared to document chunks
2. **Context Selection**: Top 3 most relevant chunks are retrieved
3. **Prompt Construction**: Context + query + system message are combined
4. **LLM Processing**: The prompt is sent to the language model
5. **Response**: AI-generated answer is returned and displayed

### Chat History Management

- Each session has a unique ID stored in browser cookies
- All messages (user and AI) are stored in Firestore
- System message provides instructions to the AI about its behavior
- History is loaded on session continuation

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'X'`
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue**: `Error processing PDF`
- **Solution**: Ensure the PDF is not corrupted and is under 50MB

**Issue**: `Firestore connection error`
- **Solution**: Verify service account credentials and project ID in `.env`

**Issue**: `OpenAI API error`
- **Solution**: Check API key validity and account balance

**Issue**: Warning about `urllib3` or `tokenizers`
- **Solution**: These warnings are suppressed in the code but don't affect functionality

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the excellent RAG framework
- HuggingFace for sentence transformers
- FAISS for efficient vector search
- OpenAI for LLM capabilities

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ using Flask, LangChain, and AI**
