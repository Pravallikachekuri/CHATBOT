# 🧠 Smart PDF Chatbots

Interact with your PDF documents using two intelligent chatbot applications powered by different LLM and embedding architectures. These bots allow you to upload any PDF, ask natural-language questions, and receive context-aware answers—one with lightweight sentence-transformers, and one with dual LLaMA models that even **score the quality of the generated answers**.

---

## 📦 Project Overview

This repository features **two separate chatbot implementations**, each showcasing a unique architecture for document-based Q&A:

---

### 1. 🟢 **LiteBot** – Lightweight Q&A with Sentence-Transformers

An efficient, minimalistic chatbot designed for fast and simple document question-answering. Built using the `all-MiniLM-L6-v2` model from [SentenceTransformers](https://www.sbert.net/), it provides accurate results with minimal dependencies.

#### 🔹 Technology Stack:
- `sentence-transformers` (MiniLM model)
- FAISS for vector similarity search
- Streamlit UI
- PyPDF2 for text extraction

#### 🔹 Key Features:
- Fast PDF parsing and sentence-level chunking
- Semantic search using normalized embeddings
- Returns top 3 relevant chunks for any question
- Highlights the most relevant sentence as the answer

📂 Code Location: [`lite-bot`](./DOC%20Q%26A%20CHATBOT/lite-bot)

---

### 2. 🟣 **JudgeBot** – Advanced Q&A with Dual LLaMA Models (Ollama)

A more powerful Q&A system that uses two LLaMA models via [Ollama](https://ollama.com/): one to answer your questions, and another to **evaluate** the response for relevance and accuracy.

#### 🔹 Technology Stack:
- Ollama for LLaMA model inference
- FAISS + BM25 hybrid retrieval
- LangChain integration for modular components
- Recursive text chunking via LangChain
- Rank-BM25 for lexical scoring

#### 🔹 Key Features:
- Embedding model for semantic retrieval
- Answer model generates responses using top-k context
- Judge model evaluates the answer:
  - ✅ **Accuracy Score**
  - ✅ **Reasoning Explanation**
  - ✅ **Improved Answer Suggestion**
- Displays top 3 relevant chunks

📂 Code Location: [`judge-bot`](./DOC%20Q%26A%20CHATBOT/judge-bot)

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Pravallikachekuri/pdf-chatbots.git
cd pdf-chatbots
````

---

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Alternatively, refer to `env activate command.txt` inside the `DOC Q&A CHATBOT/` folder.

---

### 3. Install Dependencies

#### For LiteBot:

```bash
cd "DOC Q&A CHATBOT/lite-bot"
pip install -r requirements.txt
streamlit run app.py
```

#### For JudgeBot:

```bash
cd "DOC Q&A CHATBOT/judge-bot"
pip install -r requirements.txt
streamlit run app.py
```

> ⚠️ **JudgeBot requires Ollama** to be installed and running locally with models `llama3.2:1b` and `llama3.2:3b`.

---

## 📂 Folder Structure

```
pdf-chatbots/
├── DOC Q&A CHATBOT/
│   ├── .streamlit/
│   │   └── config.toml
│   ├── lite-bot/
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── judge-bot/
│   │   ├── app.py
│   │   └── requirements.txt
│   ├── env activate command.txt
│   └── requirements1.txt
├── LICENSE
└── README.md
```

---

## 📝 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🤝 Contributing

Contributions are welcome!
To contribute:

1. Fork this repo
2. Create a new branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a Pull Request

---

## 📬 Contact

**Maintainer**: *Your Name*
📧 Email: *[pravallika.chekuri711@gmail.com](mailto:pravallika.chekuri711@gmail.com)*
🔗 GitHub: [Pravallikachekuri](https://github.com/Pravallikachekuri)


