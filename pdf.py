from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

pdffile = PyPDFLoader("books/csc808/AdvancedComputerArchitectureNote.pdf")

pages = pdffile.load_and_split()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(pages, embeddings)

query = input("Enter the query: ")

docs = db.similarity_search(query)

relevant_search = "\n".join([x.page_content for x in docs])
gemini_prompts = (
    "use the following pieces of context to answer the question."
    "if you dont know the answer just say you dont know dont make it up"
)

input_prompt = gemini_prompts + "\nContext" + relevant_search + "\nUser Question" + query

# Replace "mixtral-8x7b-32768" with the new model ID from Groq's documentation.
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile") #example model, verify the correct model.

messages = [HumanMessage(content=input_prompt)]

result = llm.invoke(messages)

print(result.content)