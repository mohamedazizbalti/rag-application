from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter , CharacterTextSplitter

# Load documents
loader = TextLoader("/Users/mohamedazizbalti/sites/rag_application/db/text.txt")
documents = loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
texts = text_splitter.split_documents(documents)

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
# Create vector store
vectorstore = FAISS.from_documents(texts, embeddings)

query = "What is NLP?"

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3,"similarity_threshold":0.75}
)
docs = retriever.invoke(query)
print(f"Found {len(docs)} documents:")
for i, doc in enumerate(docs, 1):
    print(f"\n--- Document {i} ---")
    print(doc.page_content[:200])  # First 200 chars
    print(f"Metadata: {doc.metadata}")