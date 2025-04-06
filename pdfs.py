from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

def load_and_summarize_documents(file_paths):
    """Loads and summarizes documents from given file paths."""

    all_pages = []
    for file_path in file_paths:
        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(".docx"):
            loader = UnstructuredWordDocumentLoader(file_path)
        elif file_path.lower().endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue

        all_pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) #Adjust chunk size.
    texts = text_splitter.split_documents(all_pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    # Summarize the documents in chunks.
    summary_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    chunk_summaries = []
    for chunk in texts:
        summary_prompt = f"""Summarize the following text: {chunk.page_content}"""
        summary_messages = [HumanMessage(content=summary_prompt)]
        summary_result = summary_llm.invoke(summary_messages)
        chunk_summaries.append(summary_result.content)

    # Summarize the chunk summaries.
    final_summary_prompt = f"""Summarize the following summaries: {'\n'.join(chunk_summaries)}"""
    final_summary_messages = [HumanMessage(content=final_summary_prompt)]
    final_summary_result = summary_llm.invoke(final_summary_messages)
    summary = final_summary_result.content

    return db, summary

file_paths = ["books/AdvancedComputerArchitectureNote.pdf", ]
db, summary = load_and_summarize_documents(file_paths)

print(f"Document Summary:\n{summary}\n")

query = input("Enter the query: ")

docs = db.similarity_search(query)

relevant_search = "\n".join([x.page_content for x in docs])
qa_prompt = f"""Use the following context to answer the user's question. If you don't know the answer, say "I don't know."
Context: {relevant_search}
Question: {query}
Answer:"""

messages = [HumanMessage(content=qa_prompt)]

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
result = llm.invoke(messages)

print(f"Answer:\n{result.content}")