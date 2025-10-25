from flask import Flask, render_template, request, jsonify, session
import os
import uuid
import warnings
import shutil

# Fix 1: Suppress urllib3 OpenSSL warning
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

# Fix 2: Set tokenizers parallelism BEFORE importing any tokenizers/transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore
from google.oauth2 import service_account
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=os.path.join(basedir, 'templates'))
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")

# Upload configuration
UPLOAD_FOLDER = os.path.join(basedir, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize models
model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Firestore setup
service_account_path = os.getenv("SERVICE_ACCOUNT_PATH")
firestore_project_id = os.getenv("FIRESTORE_PROJECT_ID")

credentials = service_account.Credentials.from_service_account_file(service_account_path)
client = firestore.Client(project=firestore_project_id, credentials=credentials)

model = ChatOpenAI(
    model_name=model_name,
    openai_api_key=api_key,
    base_url=base_url
)

# Store vector stores and chat histories by session
vector_stores = {}
chat_histories = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_index_path(session_id):
    return os.path.join(UPLOAD_FOLDER, f"faiss_index_{session_id}")

def create_vector_store(pdf_path, session_id):
    """Create a vector store from a PDF file"""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    # Save the vector store
    index_path = get_session_index_path(session_id)
    vector_store.save_local(index_path)
    
    return vector_store

def get_vector_store(session_id):
    """Get or create vector store for session"""
    if session_id not in vector_stores:
        index_path = get_session_index_path(session_id)
        if os.path.exists(index_path):
            vector_stores[session_id] = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
    return vector_stores.get(session_id)

def get_chat_history(session_id):
    if session_id not in chat_histories:
        chat_history = FirestoreChatMessageHistory(
            collection="chat_history",
            client=client,
            session_id=f"session_{session_id}"
        )
        # Add system message
        system_message = SystemMessage(
            content="""You are an AI assistant. Use the following context to answer the question.
            If you don't know the answer, just say you don't know. Do not try to make up an answer.
            If the user input is not a question, engage in friendly small talk.
            If the user's question is not related to the context, politely inform them that you are only able to answer questions related to the context."""
        )
        if not chat_history.messages:
            chat_history.add_message(system_message)
        chat_histories[session_id] = chat_history
    return chat_histories[session_id]

def answer_query(query_text: str, session_id: str) -> dict:
    """Answer a query using the vector store"""
    vector_store = get_vector_store(session_id)
    
    if not vector_store:
        return {
            'response': 'Please upload a PDF document first before asking questions.',
            'error': True
        }
    
    chat_history = get_chat_history(session_id)
    
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query_text)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    user_input = ChatPromptTemplate.from_template(
        "Context : {context}.\n Question : {question}"
    )
    full_input = user_input.invoke({"context": context, "question": query_text})
    chat_history.add_user_message(full_input.to_string())
    
    response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(response.content)
    
    return {
        'response': response.content,
        'error': False
    }

@app.route('/')
def index():
    # Initialize session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = uuid.uuid4().hex
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = session.get('session_id')
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{filename}")
        file.save(filepath)
        
        # Create vector store
        vector_store = create_vector_store(filepath, session_id)
        vector_stores[session_id] = vector_store
        
        # Reset chat history for new document
        if session_id in chat_histories:
            del chat_histories[session_id]
        
        return jsonify({
            'success': True,
            'filename': filename,
            'message': 'PDF uploaded and processed successfully!'
        })
    except Exception as e:
        return jsonify({'error': f'Error processing PDF: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    session_id = session.get('session_id')
    
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        result = answer_query(user_message, session_id)
        
        if result['error']:
            return jsonify({'error': result['response']}), 400
        
        return jsonify({
            'response': result['response'],
            'session_id': session_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/new-session', methods=['POST'])
def new_session():
    new_session_id = uuid.uuid4().hex
    session['session_id'] = new_session_id
    return jsonify({'session_id': new_session_id})

@app.route('/check-document', methods=['GET'])
def check_document():
    session_id = session.get('session_id')
    has_document = get_vector_store(session_id) is not None
    return jsonify({'has_document': has_document})

if __name__ == '__main__':
    app.run(debug=True, port=5000)