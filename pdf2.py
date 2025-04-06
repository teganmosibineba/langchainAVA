import streamlit as st
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pytesseract  # OCR library
from PIL import Image  # image processing

load_dotenv()

def extract_text_from_image(image_path):
    """Extracts text from an image using OCR."""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        st.error(f"Error processing image {image_path}: {e}")
        return None

def load_and_summarize_documents(file_paths):
    """Loads and summarizes documents from given file paths."""

    all_pages = []
    for file_path in file_paths:
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                all_pages.extend(loader.load())
            elif file_path.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
                all_pages.extend(loader.load())
            elif file_path.lower().endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(file_path)
                all_pages.extend(loader.load())
            elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                image_text = extract_text_from_image(file_path)
                if image_text:
                    all_pages.append(Document(page_content=image_text, metadata={"source": file_path}))
            else:
                st.warning(f"Unsupported file type: {file_path}. Skipping.")
                continue

        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None, None

    if not all_pages:
        st.error("No pages loaded from any documents.")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(all_pages)

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None, None

    summary_llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    chunk_summaries = []
    for chunk in texts:
        summary_prompt = f"""Summarize the following text: {chunk.page_content}"""
        summary_messages = [HumanMessage(content=summary_prompt)]
        try:
            summary_result = summary_llm.invoke(summary_messages)
            chunk_summaries.append(summary_result.content)
        except Exception as e:
            st.error(f"Error during chunk summarization: {e}")
            return None, None

    final_summary_prompt = f"""Summarize the following summaries: {'\n'.join(chunk_summaries)}"""
    final_summary_messages = [HumanMessage(content=final_summary_prompt)]
    try:
        final_summary_result = summary_llm.invoke(final_summary_messages)
        summary = final_summary_result.content
    except Exception as e:
        st.error(f"Error during final summarization: {e}")
        return None, None

    return db, summary

def main():
    st.title("Nile MscCS Summarization and Q&A")

    # Recursively find all documents in the 'books' directory and its subdirectories.
    books_dir = "books"
    file_paths = []
    try:
        for root, dirs, files in os.walk(books_dir):
            for file in files:
                file_paths.append(os.path.join(root, file))
    except FileNotFoundError:
        st.error(f"Directory 'books' not found.")
        return

    db, summary = load_and_summarize_documents(file_paths)

    if db and summary:
        st.subheader("Document Summary:")
        st.write(summary)

        query = st.text_input("Enter your question:")
        if query:
            docs = db.similarity_search(query)
            relevant_search = "\n".join([x.page_content for x in docs])
            qa_prompt = f"""Use the following context to answer the user's question. If you don't know the answer, say "I don't know."
            Context: {relevant_search}
            Question: {query}
            Answer:"""

            messages = [HumanMessage(content=qa_prompt)]
            llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
            try:
                result = llm.invoke(messages)
                st.subheader("Answer:")
                st.write(result.content)
            except Exception as e:
                st.error(f"Error during question answering: {e}")

    else:
        st.warning("Error processing documents.")

if __name__ == "__main__":
    main()