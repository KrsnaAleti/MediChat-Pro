

# 🩺 MediChat Pro

**MediChat Pro** is an intelligent **medical document assistant** built with [Streamlit](https://streamlit.io/).  
It allows users to upload their **medical documents (PDF)** and interact with them through a chat interface.  

The application uses a **Retrieval-Augmented Generation (RAG)** pipeline to retrieve relevant information from uploaded documents and generate accurate, context-aware answers.

---

## ✨ Core Features

- 📂 **Document Upload**  
  Upload one or more PDF files through a simple web interface.  

- 📝 **Text Extraction & Processing**  
  Extract text from PDFs, split into manageable chunks, and convert into numerical **vector embeddings**.  

- 📦 **Vector Storage**  
  Store embeddings in a **FAISS vector database** for efficient similarity searching.  

- 💬 **Chat Interface**  
  Ask questions in a user-friendly chat interface.  

- 🤖 **Retrieval & Generation**  
  Relevant chunks are retrieved and passed to a **Large Language Model (LLM)** to generate context-aware answers.  

---

## 🚀 Tech Stack

- [Streamlit](https://streamlit.io/) – Frontend & App Framework  
- [FAISS](https://github.com/facebookresearch/faiss) – Vector Database  
- [LangChain](https://www.langchain.com/) – RAG Pipeline  
- [OpenAI API](https://platform.openai.com/) – Large Language Model  
- [PyPDF2 / pdfplumber] – PDF text extraction  

---

## 📌 Usage

1. Upload one or more PDF medical documents.  
2. Ask your medical-related questions in the chat.  
3. Get **accurate, contextual answers** powered by RAG + LLM.  

---

## 🔒 Security Note

- API keys and sensitive configs are managed using a `.env` file (not pushed to GitHub).  
- Use **Streamlit Cloud secrets** or **Hugging Face Spaces secrets** when deploying.  

---

## 🛠️ Future Improvements

- ✅ Support for multiple document formats (DOCX, TXT).  
- ✅ Enhanced UI with history tracking.  
- ✅ Deployment on Hugging Face Spaces / Streamlit Cloud.  

---
## Streamlit-App
 - Application is up and running at: [MediChat-Pro](https://medichat-pro-01.streamlit.app/)
 - Demo: [DemoMediChat-Pro in Action](https://github.com/user-attachments/assets/fa38957a-8236-4eb6-b0fc-919cd2232ca7)

---

## 👨‍💻 Author
Developed by **Krishna Aleti**
