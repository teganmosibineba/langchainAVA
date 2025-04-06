import streamlit as st
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
            st.error(f"Unsupported file type: {file_path}")
            return None, None

        all_pages.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(all_pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    summary_llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
    chunk_summaries = []
    for chunk in texts:
        summary_prompt = f"""Summarize the following text: {chunk.page_content}"""
        summary_messages = [HumanMessage(content=summary_prompt)]
        summary_result = summary_llm.invoke(summary_messages)
        chunk_summaries.append(summary_result.content)

    final_summary_prompt = f"""Summarize the following summaries: {'\n'.join(chunk_summaries)}"""
    final_summary_messages = [HumanMessage(content=final_summary_prompt)]
    final_summary_result = summary_llm.invoke(final_summary_messages)
    summary = final_summary_result.content

    return db, summary

def main():
    st.title("Document Summarization and Q&A")

    uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, PPTX)", type=["pdf", "docx", "pptx"], accept_multiple_files=True)

    if uploaded_files:
        file_paths = []
        for uploaded_file in uploaded_files:
            temp_file_path = f"temp_{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.read())
            file_paths.append(temp_file_path)

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
                llm = ChatGroq(temperature=0, model_name="llama3-70b-8192") #Corrected model id.
                result = llm.invoke(messages)

                st.subheader("Answer:")
                st.write(result.content)
        else:
            st.warning("Please upload valid documents.")

if __name__ == "__main__":
    main()