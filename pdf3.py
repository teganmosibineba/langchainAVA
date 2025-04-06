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
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import asyncio
from langchain.tools import DuckDuckGoSearchResults
from duckduckgo_search.exceptions import DuckDuckGoSearchException

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def invoke_with_retry(llm, messages, max_retries=3):
    """Handles Groq API calls with retry logic for rate limits."""
    retries = 0
    while retries < max_retries:
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                try:
                    retry_after = float(str(e).split("Please try again in ")[1].split("s")[0])
                except:
                    retry_after = 5  # Default wait time.
                time.sleep(retry_after)
                retries += 1
            else:
                raise e  # Re-raise other errors.
    return None  # Return None if max retries are exceeded.


async def load_and_process_documents(file_paths, include_summary=False, directory_name=None):
    """Loads and processes documents, optionally creating a summary."""

    all_pages = []
    for file_path in file_paths:
        try:
            if file_path.lower().endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            elif file_path.lower().endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(file_path)
            elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                st.warning(f"Image file detected: {file_path}. Images not supported.")
                continue  # Skip image files.
            else:
                st.warning(f"Unsupported file type: {file_path}. Skipping.")
                continue

            all_pages.extend(loader.load())

        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return None, None

    if not all_pages:
        st.error("No content loaded from documents.")
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_pages)

    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(texts, embeddings)
    except Exception as e:
        st.error(f"Error creating vector database: {e}")
        return None, None

    if include_summary:
        summary_llm = ChatGroq(temperature=0, model_name="deepseek-r1-distill-llama-70b")
        combined_chunks = [" ".join([chunk.page_content for chunk in texts[i:i + 3]]) for i in range(0, len(texts), 3)]
        chunk_summaries = []
        for combined_chunk in combined_chunks:
            summary_prompt = f"Summarize text related to {directory_name or 'the documents'}: {combined_chunk}"
            summary_messages = [HumanMessage(content=summary_prompt)]
            summary_result = invoke_with_retry(summary_llm, summary_messages)
            if summary_result is None:
                st.error("Groq API limit exceeded during summarization.")
                return None, None
            chunk_summaries.append(summary_result.content)

        final_summary_prompt = f"Summarize the summaries related to {directory_name or 'the documents'}: {'\\n'.join(chunk_summaries)}"
        final_summary_messages = [HumanMessage(content=final_summary_prompt)]
        final_summary_result = invoke_with_retry(summary_llm, final_summary_messages)
        if final_summary_result is None:
            st.error("Groq API limit exceeded during final summary.")
            return None, None
        return db, final_summary_result.content
    else:
        return db, None


def main():
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] > .main { padding-top: 2rem; }
        [data-testid="stSidebar"] { background-color: #f0f2f6; padding-top: 2rem; }
        [data-testid="stHeader"] { background-color: rgba(0,0,0,0); }
        .css-1avcm0n { padding-top: 0px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div style="position: fixed; top: 15px; left: 15px; font-size: 20px;">TeganAI</div>', unsafe_allow_html=True)
    st.title("Nile MscCS Q&A and Summaries")

    books_dir = "books"
    directories = [d for d in os.listdir(books_dir) if os.path.isdir(os.path.join(books_dir, d))]

    if not directories:
        st.error(f"No directories found in '{books_dir}'.")
        return

    selected_directory = st.selectbox("Select a Directory", directories)
    file_paths = [os.path.join(books_dir, selected_directory, f) for f in os.listdir(os.path.join(books_dir, selected_directory))]
    file_paths = [f for f in file_paths if os.path.isfile(f)]

    if not file_paths:
        st.warning(f"No files found in '{selected_directory}'.")
        return

    query = st.text_input("Enter your question or summary request:")

    if query:
        if "summary" in query.lower():
            db, summary = asyncio.run(load_and_process_documents(file_paths, include_summary=True, directory_name=selected_directory))
            if summary:
                st.subheader(f"Summary of: {selected_directory}")
                st.write(summary)
            else:
                st.warning("Error generating summary.")
        else:
            db, _ = asyncio.run(load_and_process_documents(file_paths, include_summary=False))
            if db:
                docs = db.similarity_search(query)
                relevant_search = "\n".join([x.page_content for x in docs])
                qa_prompt = f"Answer using the context. If unsure, say 'I don't know.'\nContext: {relevant_search}\nQuestion: {query}\nAnswer:"
                messages = [HumanMessage(content=qa_prompt)]
                llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
                result = invoke_with_retry(llm, messages)
                if result is None:
                    st.error("Groq API limit exceeded during question answering.")
                    return

                st.subheader("Answer:")
                st.write(result.content)

                try:
                    search = DuckDuckGoSearchResults(max_results=5)
                    search_results = search.run(query)

                    if search_results:
                        st.subheader("Related Links:")
                        if isinstance(search_results, list):
                            for item in search_results:
                                if isinstance(item, dict):
                                    try:
                                        st.write(f"[{item['title']}]({item['link']})")
                                    except KeyError:
                                        st.write("Link data incomplete.")
                                else:
                                    st.write(item)
                        else:
                            st.write(search_results)  # display the string, if the results are not a list.

                    else:
                        st.write("No related links found.")

                except DuckDuckGoSearchException as e:
                    st.error(f"Search timed out. Please try again later. Error: {e}")

                st.download_button(
                    label="Download Context",
                    data=relevant_search.encode('utf-8'),
                    file_name="context.txt",
                    mime="text/plain",
                )
            else:
                st.warning("Error processing documents.")
    else:
        st.write("Please enter a question or a summary request.")

if __name__ == "__main__":
    main()