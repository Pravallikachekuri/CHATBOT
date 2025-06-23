import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import re

os.environ["STREAMLIT_WATCHDOG_MODE"] = "none"

# Load the MinLM model once
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # Using MinLM model

def extract_text_from_pdf(pdf):
    reader = PdfReader(pdf)
    return " ".join([page.extract_text() or "" for page in reader.pages])

def split_text_by_sentence(text, max_chunk_words=150):
    """
    Splits text into chunks by sentences, ensuring each chunk does not exceed max_chunk_words.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, chunk, word_count = [], [], 0

    for sent in sentences:
        words_in_sent = len(sent.split())
        if word_count + words_in_sent > max_chunk_words and chunk:
            chunks.append(" ".join(chunk))
            chunk, word_count = [], 0
        chunk.append(sent)
        word_count += words_in_sent
    
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

def get_embeddings(chunks):
    return embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Using inner product for cosine similarity
    index.add(embeddings)
    return index

def search_index(query, faiss_index, chunks, top_k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(query_embedding, top_k)
    return [(chunks[i], float(D[0][idx])) for idx, i in enumerate(I[0])]

def extract_answer(query, chunk):
    sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
    if len(sentences) == 1:
        return sentences[0]
    
    sent_embeds = embedder.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    query_embed = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = np.dot(sent_embeds, query_embed)
    
    best_idx = scores.argmax()
    return sentences[best_idx] if scores[best_idx] > 0.2 else sentences[0]

def generate_answer(query, chunks_scores):
    answers = []
    for chunk, _ in chunks_scores:
        answers.append(extract_answer(query, chunk))
    return "\n\n".join(answers)

def main():
    st.title("Chat with PDF :books:")

    with st.sidebar:
        pdf = st.file_uploader("Upload PDF", type="pdf")
        question = st.text_input("Ask a question")

        if pdf and st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                text = extract_text_from_pdf(pdf)
                chunks = split_text_by_sentence(text)
                embeddings = get_embeddings(chunks)
                index = build_faiss_index(embeddings)
                st.session_state['index'] = index
                st.session_state['chunks'] = chunks
                st.session_state['text'] = text
                st.success("PDF processed successfully!")
                st.subheader("Extracted Text")
                st.text_area("Full Text", text, height=300)
                st.subheader("Text Chunks")
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(chunk)

    if question and 'index' in st.session_state:
        with st.spinner("Searching for answer..."):
            relevant = search_index(question, st.session_state['index'], st.session_state['chunks'])
            st.markdown("### Answer:")
            answer = generate_answer(question, relevant)
            st.write(answer)
            st.markdown("### Related Chunks:")
            for i, (chunk, score) in enumerate(relevant, 1):
                st.markdown(f"**Chunk {i} (score: {score:.3f}):**")
                st.write(chunk.strip())

if __name__ == "__main__":
    main()

