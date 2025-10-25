import os
import uuid
import warnings

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
from langchain_core.messages import HumanMessage , SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

from google.oauth2 import service_account


# Load environment variables
load_dotenv()
model_name = os.getenv("MODEL_NAME")
api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
if os.path.exists("faiss_index"):
    print("Loading existing FAISS index...")
    # Load documents
    vector_store = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
else:
    print("Creating new FAISS index...")
    pdf_path = "/Users/mohamedazizbalti/sites/rag_application/db/Comprehensive Guide to Artificial Intelligence.pdf"
    loader = PyPDFLoader(pdf_path)
    pages = []
    for page in loader.load_and_split():
        pages.append(page)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("faiss_index")

retriever  = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3,"similarity_threshold":0.5})


service_account_path = os.getenv("SERVICE_ACCOUNT_PATH")
firestore_project_id = os.getenv("FIRESTORE_PROJECT_ID")

credentials = service_account.Credentials.from_service_account_file(
    service_account_path
)
client = firestore.Client(
    project=firestore_project_id,
    credentials=credentials
)

session_id = uuid.uuid4().hex
print("session id : ", session_id)


chat_history = FirestoreChatMessageHistory(
    collection="chat_history",
    client=client,
    session_id="session_" + session_id
)

# Create system message with context
system_message = SystemMessage(
"""You are an AI assistant. Use the following context to answer the question.
    If you don't know the answer, just say you don't know. Do not try to make up an answer.
    If the user input is not a question, engage in friendly small talk.
    If the user's question is not related to the context, politely inform them that you are only able to answer questions related to the context."""
    )
chat_history.add_message(system_message)


model = ChatOpenAI(
    model_name=model_name,
    openai_api_key=api_key,
    base_url=base_url
)

def answer_query(query_text: str) -> str:
    # Retrieve relevant documents
    relevant_docs = retriever.invoke(query_text)
    context = "\n".join([doc.page_content for doc in relevant_docs])
    
    user_input = ChatPromptTemplate.from_template(
        "Context : {context}.\n Question : {question}"
    )
    full_input = user_input.invoke({"context" : context , "question" : query_text})
    #print(full_input.to_string())
    chat_history.add_user_message(full_input.to_string())
    response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(response.content)
    return response.content

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            print("Chat History : ",chat_history.messages)
            break
        answer = answer_query(user_query)
        print("Answer:", answer)