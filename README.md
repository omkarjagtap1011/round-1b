# Adobe "Connecting the Dots" Hackathon Solutions

This repository contains solutions for Adobe's "Connecting the Dots" hackathon challenges focusing on intelligent document processing and analysis.

## 🏆 Challenge Overview

Adobe's hackathon presents real-world document intelligence problems that require innovative solutions combining PDF processing, natural language understanding, and user-centric design.

## 📁 Project Structure

```
adobe/
├── Challenge_1a/          # PDF Outline Extractor
│   ├── main.py           # Core extraction algorithm
│   ├── requirements.txt  # Dependencies
│   ├── Dockerfile        # Container configuration
│   └── README.md         # Challenge-specific documentation
│
├── Challenge_1b/          # Persona-Driven Document Intelligence
│   ├── main.py           # Core intelligence algorithm
│   ├── requirements.txt  # Dependencies
│   ├── Dockerfile        # Container configuration
│   └── README.md         # Challenge-specific documentation
│
├── .gitignore            # Git ignore rules
└── README.md             # This file
```
## 🚀 Solutions Summary

### Challenge 1B: Persona-Driven Document Intelligence
- **Purpose**: Intelligent document analysis based on user personas and tasks
- **Technology**: Python 3.9, PyMuPDF, Advanced NLP techniques
- **Features**: Relevance scoring, section prioritization, multi-document processing
- **Performance**: Scalable algorithm with contextual understanding

---

## 🧠 Approach

Our system identifies and returns the most relevant sections of a PDF tailored to a persona's intent by combining:

•  Text-based PDF parsing
•  Lightweight offline embedding
•  Vector similarity search
•  Persona-aware reranking

---

## 🔍 Pipeline Overview

1. *PDF Section Extraction*  
   → Parse PDFs with *PyMuPDF* into structured sections saved as JSON.

2. *Text Chunking*  
   → Split long sections into *smaller coherent chunks* for better semantic representation.

3. *Embedding & Storage*  
   → Embed each chunk using a *lightweight offline model. Store in **ChromaDB* for fast vector search.

4. *Semantic Querying*  
   → Embed the combined *persona + task query* and retrieve *top 10 relevant chunks* from ChromaDB.

5. *Reranking with Nomic*  
   → Refine relevance using the *nomic-embed-text* model for *intent-aware ranking*.

6. *Output Formatting*  
   → Return *top 5 matched sections* in a structured  output.json .


## 🛠️ Technical Stack

| Component         | Tool / Library                | Purpose                                               |
|------------------|-------------------------------|-------------------------------------------------------|
| Language          | Python 3.10                    | Core programming language                             |
| PDF Parsing       | PyMuPDF (`fitz`)               | Extracts section-wise text from PDFs                 |
| Structuring Logic | `extract_struct.py`            | Identifies section headers and chunks content         |
| Embeddings        | `nomic-embed-text` (via Ollama)| Generates semantic vectors and reranking embeddings   |
| Vector Store      | ChromaDB                       | Stores and queries vectors using cosine similarity    |
| Reranking Model   | CrossEncoder (MiniLM)          | Reranks top sections based on persona/task relevance  |
| LLM Access (opt.) | TinyLLaMA via `llama-cpp-python` | Local model loading via HuggingFace (optional)       |
| Embedding Server  | Ollama                         | Serves embedding models via local API                |
| Data Format       | JSON                           | For structured input and output                      |
| Containerization  | Docker (AMD64)                 | Optional for reproducible, isolated execution         |

## 🎯 Key Features

### Intelligent Document Retrieval  
- Persona-based section filtering  
- Task-aware document prioritization  
- Semantic chunk indexing and reranking  
- Structured, persona-aligned JSON output

### Offline & Efficient  
- Entirely local execution (no internet needed)  
- Lightweight models for embedding and reranking  
- Fast semantic search with ChromaDB  
- Docker-ready deployment with AMD64 support

---

## 📊 Performance Highlights  
- **Execution Time**: < 60 seconds for multi-document inputs  
- **Model Size**: encoder : 784 mb,  yolo :5.6 mb
                           yolo :5.6 mb 
nomic-embed-text:latest: 274 mb- **Accuracy**: Persona-aware semantic reranking using `nomic-embed-text`  
- **Scalability**: Supports multi-document, multi-persona batches

---

## 🔧 Development Approach  
- **Modular Reuse**: Leveraged the YOLO-based PDF layout extraction pipeline from Round 1A to ensure consistent and high-accuracy section detection across rounds.

Our Round 1B solution emphasizes:

- **Accuracy**: Persona + JTBD mapped to semantic query, refined with reranking  
- **Performance**: ChromaDB + fast embedding/reranking with minimal overhead  
- **Scalability**: Easily handles multiple PDFs and inputs  
- **Usability**: JSON-driven I/O, logs, and auto-output generator  
- **Automation**: End-to-end script processes PDFs → JSON without manual intervention  

---

## 📋 Submission Requirements Met  

✅ **Challenge 1B Requirements**  
- Intelligent section extraction for persona and job-to-be-done  
- Relevance ranking and filtering  
- Structured JSON format with metadata  
- Full offline execution (no external API)  
- Docker containerization (AMD64 compatible)  

---

## 🏗️ Architecture Decisions  

### PDF Processing  
- **PyMuPDF**: High-performance, per-page text extraction  
- **Section Chunking**: Large sections split into manageable text blocks  
- **Structured JSON Output**: Easy downstream use and validation  

### Vector Search & Reranking  
- **ChromaDB**: Fast, persistent vector storage and retrieval  
- **Ollama + `nomic-embed-text`**: Used for embedding and semantic reranking  
- **CrossEncoder**: Further refines the relevance of retrieved chunks  

### Optimization  
- **Offline-First**: Embeddings, search, and reranking run locally  
- **Low Footprint**: Lightweight models meet all system constraints  
- **Batch Mode**: Multi-document, persona-based processing enabled  

---

## 🎉 Innovation Highlights  

- **Persona-Aware Relevance**: Semantic understanding of user's role and intent  
- **Hybrid Ranking**: Combines vector search with cross-encoder refinement  
- **Efficient & Offline**: No internet or cloud calls; fully CPU-compliant  
- **Plug-and-Play Architecture**: Easily extendable to new personas or tasks  
- **Auto Structuring**: Output JSON aligned exactly to Adobe’s schema  

---

## 📝 Documentation

Each challenge folder includes a README with:

- Implementation details  
- Pipeline explanation  
- Model setup  
- Docker usage  
- Output samples  
- Compliance checklist  

   












