from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import  HuggingFaceEmbedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from  dotenv import load_dotenv

load_dotenv()

pdffile = PyPDFLoader("")