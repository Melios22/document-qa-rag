# Vietnamese RAG System - Complete Project Guide

## 🎯 High-Level Overview

This is a **complete Vietnamese Retrieval-Augmented Generation (RAG) system** that processes PDF documents, builds a hybrid vector database, and provides intelligent question-answering capabilities using advanced language models.

### 🔧 **System Architecture**

```
📄 PDF Documents → 🔄 Preprocessing → 🗃️ Vector Database → 🤖 RAG Query System
```

**Key Components:**
- **Document Processing**: PDF extraction and text cleaning
- **Hybrid Search**: BGE-M3 dense+sparse embeddings with Milvus vector database  
- **Unified RAG Class**: Single class with internal modularity for complete RAG flow
- **Multi-LLM Support**: Google Gemini and IBM Watsonx integration
- **Vietnamese Language**: Optimized for Vietnamese text processing

---

## 📂 **Project Structure**

```
vietnamese-rag-system/
│
├── 🚀 **MAIN APPLICATIONS**
│   ├── main_preprocess.py      # Step 1: Process PDF documents
│   ├── main_build_rag.py       # Step 2: Build vector database  
│   ├── main_search_rag.py      # Step 3: Interactive RAG queries
│   ├── streamlit_app.py        # Web UI interface
│   └── simple_api.py           # REST API interface
│
├── 🧠 **CORE RAG SYSTEM** (src/)
│   ├── preprocess/             # Document preprocessing
│   │   ├── process_pdf.py      # PDF extraction
│   │   ├── cleaner.py          # Text cleaning
│   │   ├── metadata_gen.py     # Metadata generation
│   │   └── storage.py          # Document storage
│   │
│   ├── rag_builder/            # Vector database building
│   │   ├── builder.py          # Main RAG builder
│   │   ├── connection_manager.py # Database connections
│   │   ├── milvus.py           # Milvus operations
│   │   ├── model_loader.py     # Model loading
│   │   └── vector_bge_m3.py    # BGE-M3 embeddings
│   │
│   ├── rag_retriever/          # **UNIFIED RAG SYSTEM** ⭐
│   │   ├── vietnamese_rag.py   # **MAIN UNIFIED CLASS**
│   │   ├── hybrid_search.py    # Hybrid search logic
│   │   ├── llm_caller.py       # LLM integration
│   │   ├── model_loader.py     # Model utilities
│   │   └── models/             # LLM implementations
│   │       ├── base_llm.py     # Base LLM interface
│   │       ├── factory.py      # Model factory
│   │       ├── gemini_llm.py   # Google Gemini
│   │       └── watsonx_llm.py  # IBM Watsonx
│   │
│   ├── utils/
│   │   └── logging.py          # Centralized logging
│   │
│   ├── config.json             # System configuration
│   └── constant.py             # System constants
│
├── 📊 **DATA & STORAGE**
│   ├── data/
│   │   ├── pdf/                # Input PDF documents
│   │   ├── processed/          # Processed documents
│   │   ├── milvus.db           # Vector database
│   │   └── vietnamese-stopwords.txt
│   └── logs/                   # System logs
│
├── 📓 **EXAMPLES & DOCS**
│   ├── notebooks/
│   │   └── example.ipynb       # Usage examples
│   ├── ESSENTIAL_STRUCTURE.md  # This guide
│   └── requirements.txt        # Dependencies
│
└── 🔧 **CONFIG & DEPLOYMENT**
    ├── .env                    # Environment variables
    ├── docker-compose.yml      # Container deployment
    └── .gitignore             # Git configuration
```

---

## 🚀 **How to Use the System**

### **Prerequisites**
```bash
# 1. Install Python 3.8+
# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables (create .env file)
GOOGLE_API_KEY=your_gemini_api_key
WATSONX_URL=your_watsonx_url
WATSONX_API_KEY=your_watsonx_key
WATSONX_PROJECT_ID=your_project_id
```

### **Step-by-Step Usage**

#### **Step 1: Process Documents** 📄➡️🔄
```bash
python main_preprocess.py
```
**What it does:**
- Reads PDF files from `data/pdf/`
- Extracts and cleans text content
- Generates metadata for each document
- Saves processed documents to `data/processed/`

#### **Step 2: Build Vector Database** 🔄➡️🗃️
```bash
python main_build_rag.py
```
**What it does:**
- Loads processed documents
- Creates BGE-M3 embeddings (dense + sparse)
- Builds Milvus vector database
- Optimizes for hybrid search

#### **Step 3: Query the System** 🗃️➡️🤖
```bash
python main_search_rag.py
```
**What it does:**
- Loads the complete RAG system
- Provides interactive command-line interface
- Performs hybrid search + LLM answer generation
- Logs all Q&A interactions

#### **Alternative Interfaces**

**Web UI (Streamlit):**
```bash
streamlit run streamlit_app.py
```

**REST API:**
```bash
python simple_api.py
```

---

## 🧠 **Core RAG System - Unified Architecture**

### **Main Class: `VietnameseRAG`** ⭐

```python
from src.rag_retriever import VietnameseRAG

# Initialize the unified system
rag = VietnameseRAG(
    model_type="gemini",  # or "watsonx"
    k=10,                 # initial retrieval count
    rerank_top_k=5,       # final reranked results
)

# Complete RAG flow - One method does everything!
result = rag.answer("What is machine learning?")
print(result['answer'])

# Or just search documents
docs = rag.search("machine learning")

# Switch models easily
rag.switch_model("watsonx")

# Check system status
print(rag.status)
```

### **Internal Architecture (Modular Subclasses)**

The `VietnameseRAG` class uses **internal subclasses** for clean modularity:

```python
class VietnameseRAG:
    class DocumentRetriever:    # Internal: handles search & reranking
    class AnswerGenerator:      # Internal: handles LLM generation
    
    # Main API methods:
    def answer(query):         # Complete RAG flow
    def search(query):         # Document search only  
    def switch_model(type):    # Change LLM model
    def status():              # System information
```

**Benefits:**
- ✅ **Single file** with complete functionality
- ✅ **Internal modularity** via subclasses
- ✅ **Clean API** - main `answer()` method
- ✅ **Easy model switching**
- ✅ **Backward compatibility**

---

## 🔧 **Technical Deep Dive**

### **Hybrid Search System**
- **BGE-M3 Model**: Dense + sparse embeddings in one model
- **Milvus Database**: High-performance vector storage
- **RRF Fusion**: Combines dense and sparse search results
- **Reranking**: Final relevance scoring for top results

### **LLM Integration**
- **Factory Pattern**: Easy addition of new models
- **Multiple Models**: Google Gemini, IBM Watsonx
- **Unified Interface**: Same API for all models
- **Error Handling**: Graceful fallbacks and logging

### **Vietnamese Language Support**
- **Stopwords**: Vietnamese-specific text filtering
- **BGE-M3**: Multilingual model with Vietnamese support
- **Text Cleaning**: Vietnamese text preprocessing
- **Encoding**: Proper UTF-8 handling throughout

### **Logging System**
- **File-based**: Separate logs for each component
- **Milestone Tracking**: Important events highlighted
- **Q&A History**: Complete interaction logging
- **Error Management**: Comprehensive error tracking

---

## 📋 **Configuration**

### **Main Configuration (`src/config.json`)**
```json
{
  "models": {
    "embedding": "BAAI/bge-m3",
    "reranker": "BAAI/bge-reranker-v2-m3"
  },
  "retrieval": {
    "k": 10,
    "rerank_top_k": 5,
    "similarity_threshold": 0.3
  },
  "llm": {
    "default_model": "gemini",
    "temperature": 0.1
  }
}
```

### **Environment Variables (`.env`)**
```bash
# Google Gemini
GOOGLE_API_KEY=your_gemini_api_key

# IBM Watsonx
WATSONX_URL=your_watsonx_url
WATSONX_API_KEY=your_watsonx_key
WATSONX_PROJECT_ID=your_project_id

# Optional: Database settings
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

---

## 🧪 **Testing & Development**

### **Quick Test**
```bash
python test_essentials.py
```

### **Development Workflow**
1. **Add documents**: Place PDFs in `data/pdf/`
2. **Preprocess**: Run `main_preprocess.py`
3. **Build**: Run `main_build_rag.py`  
4. **Test**: Run `main_search_rag.py`
5. **Deploy**: Use `streamlit_app.py` or `simple_api.py`

### **Adding New LLM Models**
1. Create new model class in `src/rag_retriever/models/`
2. Inherit from `BaseLLM`
3. Add to factory in `factory.py`
4. Use with `rag.switch_model("your_model")`

---

## 🐳 **Deployment**

### **Docker Deployment**
```bash
docker-compose up -d
```

### **Production Considerations**
- **GPU Support**: For faster embedding generation
- **API Rate Limits**: Monitor LLM API usage
- **Vector Database**: Consider Milvus cluster for scale
- **Caching**: Implement query result caching
- **Security**: Secure API endpoints and environment variables

---

## 📈 **Performance & Scaling**

### **Current Capabilities**
- **Document Capacity**: 1000+ documents tested
- **Query Speed**: ~2-5 seconds per query
- **Accuracy**: High relevance with BGE-M3 + reranking
- **Languages**: Vietnamese + multilingual support

### **Optimization Tips**
- **Batch Processing**: Process documents in batches
- **GPU Acceleration**: Use CUDA for model inference
- **Index Tuning**: Optimize Milvus index parameters
- **Model Caching**: Cache loaded models in memory

---

## 🆘 **Troubleshooting**

### **Common Issues**

**"No module named 'langchain'"**
```bash
pip install -r requirements.txt
```

**"No documents found"**
- Check `data/pdf/` folder has PDF files
- Run `main_preprocess.py` first

**"Vector database not found"**
- Run `main_build_rag.py` after preprocessing

**"LLM API errors"**
- Check your API keys in `.env` file
- Verify internet connection
- Check API quota/rate limits

### **Log Files**
- `logs/preprocess.log` - Document processing
- `logs/builder.log` - Vector database building
- `logs/retriever.log` - RAG query operations
- `logs/errors.log` - All errors
- `logs/qa_history.log` - Question-answer history

---

## 🎉 **Summary**

This Vietnamese RAG system provides:

✅ **Complete End-to-End Pipeline** from PDF to intelligent answers  
✅ **Unified Architecture** with single main class and internal modularity  
✅ **Multiple LLM Support** (Gemini, Watsonx) with easy switching  
✅ **Hybrid Search** using state-of-the-art BGE-M3 embeddings  
✅ **Vietnamese Language Optimization** throughout the pipeline  
✅ **Production Ready** with logging, error handling, and deployment options  
✅ **Developer Friendly** with clean APIs and comprehensive documentation  

**Perfect for:** Vietnamese document Q&A systems, knowledge bases, research assistants, and educational applications.

---

*Need help? Check the logs, review the configuration, or examine the example notebook for detailed usage patterns.*
