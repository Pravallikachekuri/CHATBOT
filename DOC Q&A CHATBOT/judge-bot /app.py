import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import numpy as np
import faiss

os.environ["STREAMLIT_WATCHDOG_MODE"] = "none"

# Models
ANSWER_MODEL = "llama3.2:1b"
JUDGE_MODEL = "llama3.2:3b"

embedder = OllamaEmbeddings(model=ANSWER_MODEL)
qa_model = OllamaLLM(model=ANSWER_MODEL)
judge_model = OllamaLLM(model=JUDGE_MODEL)


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return " ".join(page.extract_text() or "" for page in reader.pages)


def split_text(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)


def build_faiss_index(chunks):
    embeddings = np.array(embedder.embed_documents(chunks), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def faiss_search(query, index, chunks, k=3):
    query_emb = embedder.embed_documents([query])[0]
    query_emb = np.array(query_emb, dtype=np.float32)
    query_emb /= np.linalg.norm(query_emb)
    D, I = index.search(np.array([query_emb]), k)
    return [(chunks[i], D[0][idx]) for idx, i in enumerate(I[0])]


def bm25_search(query, chunks, k=3):
    tokenized = [chunk.split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.split())
    top_indices = np.argsort(scores)[::-1][:k]
    return [(chunks[i], scores[i]) for i in top_indices]


def hybrid_retrieval(query, index, chunks, top_k=3):
    faiss_results = faiss_search(query, index, chunks, top_k)
    bm25_results = bm25_search(query, chunks, top_k)
    combined = {}
    for chunk, score in faiss_results + bm25_results:
        combined[chunk] = max(score, combined.get(chunk, 0))
    sorted_chunks = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks[:top_k]


def generate_answer(question, relevant_chunks):
    context = "\n\n".join(chunk for chunk, _ in relevant_chunks)
    prompt = (
        f"Answer the question based only on the following context:\n\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
    response = qa_model.invoke(prompt)
    return response.text if hasattr(response, "text") else str(response)


def evaluate_answer(question, context, answer):
    prompt = (
        f"You are a strict evaluator. Assess the accuracy of the answer based only on the given context.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer}\n\n"
        f"Reply with:\nAccuracy: <0-100>\nReason: <brief justification>\nImproved Answer: <optional improved answer>"
    )
    response = judge_model.invoke(prompt)
    return response.text if hasattr(response, "text") else str(response)


def main():
    st.title("📘 Smart PDF Q&A with Judge Evaluation")

    with st.sidebar:
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        question = st.text_input("Ask a Question")

        if pdf_file and st.button("Process PDF"):
            text = extract_text_from_pdf(pdf_file)
            chunks = split_text(text)
            index = build_faiss_index(chunks)
            st.session_state.update({
                "chunks": chunks,
                "index": index,
                "full_text": text
            })
            st.success("PDF processed successfully!")

    if "index" in st.session_state and question:
        with st.spinner("Retrieving answer..."):
            relevant_chunks = hybrid_retrieval(question, st.session_state["index"], st.session_state["chunks"])
            context = "\n\n".join(chunk for chunk, _ in relevant_chunks)
            answer = generate_answer(question, relevant_chunks)
            evaluation = evaluate_answer(question, context, answer)

        st.markdown("### ✅ Answer")
        st.write(answer)

        st.markdown("### 🧠 Judge Evaluation")
        st.text_area("Judge Feedback", evaluation, height=200)

        st.markdown("### 🔍 Top Relevant Chunks")
        for i, (chunk, _) in enumerate(relevant_chunks, 1):
            st.markdown(f"**Chunk {i}:**")
            st.write(chunk)


if __name__ == "__main__":
    main()

