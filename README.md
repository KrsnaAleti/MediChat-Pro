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
## 📷 Demo
Application is up and running at: [MediChat-Pro]([https://medichat-pro-01.streamlit.ap](https://medichat-pro-01.streamlit.app/)

### Landing Page:
<img width="1512" height="947" alt="image" src="https://github.com/user-attachments/assets/38059b6a-e081-439a-b3a5-3f79735d8c9d" />

### Upload Files:
<img width="1512" height="946" alt="image" src="https://github.com/user-attachments/assets/ef271537-b4c6-4039-8602-155090afd979" />

### Intelligent Assitant ready:
<img width="1509" height="945" alt="image" src="https://github.com/user-attachments/assets/e5224b19-f5b5-4275-89d1-ef5646e03c23" />

### Assistant in Action:
<img width="1512" height="939" alt="image" src="https://github.com/user-attachments/assets/91cadbfa-6802-4aac-a9b2-5760f16b7ab1" />




---

## 👨‍💻 Author

Developed by **Krishna Aleti** 🚀  
