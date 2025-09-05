# Simple LLM Caller Usage Guide

## What Changed

I've simplified the LLM caller to exactly what you asked for:

1. **GeminiLLM class** - Direct copy of your Gemini model code
2. **RAGLLMCaller class** - All-in-one: prepare prompt, init Gemini, call model, return structured output
3. **No complex abstractions** - Just the basics you need

## Quick Test

```bash
python test_simple_llm.py
```

## Basic Usage

### 1. With RAG System (Recommended)

```python
from src import VietnameseRAG

# Initialize RAG (includes LLM caller automatically)
rag = VietnameseRAG()

# Ask question - everything handled automatically
result = rag.generate_answer("Python là gì?")

print(f"Answer: {result['answer']}")
print(f"Success: {result['success']}")
print(f"Sources: {len(result['sources'])}")
```

### 2. Direct LLM Caller Usage

```python
from src.rag_retriever.llm_caller import RAGLLMCaller
from langchain.schema import Document

# Get LLM caller
llm_caller = RAGLLMCaller()

# Your documents
docs = [
    Document(page_content="Python is great", metadata={"source": "doc1"})
]

# Generate answer
result = llm_caller.generate_answer(
    query="What is Python?",
    documents=docs,
    max_tokens=512,      # Optional Gemini params
    temperature=0.1
)

print(result['answer'])
```

### 3. Just Gemini (No RAG)

```python
from src.rag_retriever.llm_caller import GeminiLLM

# Direct Gemini usage
gemini = GeminiLLM(max_tokens=256, temperature=0.0)
response = gemini.generate("Tell me about Python")
print(response)
```

## What You Get

### RAGLLMCaller Methods

- `generate_answer(query, documents, **gemini_kwargs)` - Complete pipeline
- `_prepare_context(documents)` - Format documents  
- `_prepare_prompt(query, context)` - Create prompt
- `_init_gemini(**kwargs)` - Initialize Gemini

### Response Structure

```python
{
    "answer": "Generated answer text",
    "context_length": 1234,
    "success": True,
    "error": None  # or error message if failed
}
```

## Setup Requirements

1. **Environment**: Add to `.env`
   ```
   GEMINI_API_KEY=your_key_here
   ```

2. **Dependencies**:
   ```bash
   pip install google-genai python-dotenv
   ```

3. **File Structure**:
   ```
   RAG/
   ├── .env
   ├── src/rag_retriever/llm_caller.py  # Simple version
   └── test_simple_llm.py               # Test script
   ```

## Key Features

✅ **Simple**: Only essential code, no over-engineering  
✅ **Direct**: Uses your exact Gemini class code  
✅ **Integrated**: Works seamlessly with existing RAG system  
✅ **Flexible**: Pass custom Gemini parameters as needed  
✅ **Error Handling**: Returns structured responses with error info  

## Migration

Your existing code still works! The new system provides:
- `RAGLLMCaller` (new, simple)
- `LLMCaller` (alias for compatibility)
- `get_rag_llm_caller()` (new function)
- `get_llm_caller()` (alias for compatibility)

## Testing

Run the test to make sure everything works:

```bash
python test_simple_llm.py
```

Expected output:
```
🧪 Testing Direct Gemini...
✅ Direct Gemini works!
📖 Response: Python là một ngôn ngữ lập trình...

🧪 Testing Simple LLM Caller...
✅ Success!
📖 Answer: Có, Python rất phù hợp cho Machine Learning...
```

That's it! Simple, clean, and exactly what you requested.
