# RAG (Retrieval-Augmented Generation) Guide

## 🎯 What is RAG?

RAG allows your chatbot to answer questions based on **your own documents**. Instead of relying only on training data, the bot retrieves relevant information from uploaded documents and uses it to generate accurate, context-aware responses.

## 🚀 Quick Start

### 1. Train Your Model (if not done yet)

```bash
python train.py
```

This now trains on **OpenAssistant** dataset (better for conversations).

### 2. Start the RAG Chatbot

```bash
python chatbot_rag.py
```

Or with a document pre-loaded:

```bash
python chatbot_rag.py --document "path/to/your/document.txt"
```

### 3. Use the Chatbot

```
You: add research_paper.pdf
✓ Document added: research_paper.pdf

You: What is the main finding of this paper?
Bot: Based on the document, the main finding is...
```

## 📚 Supported Document Formats

| Format | Extension | Requirements |
|--------|-----------|--------------|
| **Text** | `.txt` | None (built-in) |
| **PDF** | `.pdf` | `pip install PyPDF2` |
| **Word** | `.docx` | `pip install python-docx` |

## 🎮 Commands

### Document Management
- `add <filepath>` - Add a document to knowledge base
- `docs` - List loaded documents
- `save kb` - Save knowledge base to disk
- `load kb` - Load saved knowledge base

### Chat Commands
- `quit` or `exit` - Exit chatbot
- `clear` - Clear conversation history
- `history` - View conversation history

## 💡 How RAG Works

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                           │
└─────────────────────────────────────────────────────────────┘

1. DOCUMENT LOADING
   User uploads document.txt
   ├── Split into chunks (500 chars each)
   └── Create embeddings for each chunk

2. QUESTION ANSWERING
   User: "What is machine learning?"
   ├── Create embedding for question
   ├── Find 3 most similar chunks
   ├── Build context with relevant chunks
   └── Generate answer using context + question

3. RESPONSE
   Bot: "Based on the document, machine learning is..."
```

## 🔧 Configuration

### Chunk Settings (in `rag_pipeline.py`)

```python
doc_store = DocumentStore(
    embedder=embedder,
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50     # Overlap between chunks
)
```

**Recommendations:**
- **Small documents** (< 10 pages): `chunk_size=300`
- **Medium documents** (10-50 pages): `chunk_size=500` (default)
- **Large documents** (> 50 pages): `chunk_size=700`

### Retrieval Settings

```python
relevant_chunks = doc_store.search(
    query=user_question,
    top_k=3    # Number of chunks to retrieve
)
```

**Recommendations:**
- **Short answers**: `top_k=2`
- **Detailed answers**: `top_k=4-5`

## 🎯 Two Embedding Options

### Option 1: SimpleEmbedder (Default, No Installation)
- ✅ No extra dependencies
- ✅ Fast
- ✅ Works offline
- ⚠️ Less accurate

### Option 2: SentenceTransformer (Better Quality)

Install:
```bash
pip install sentence-transformers
```

The chatbot will automatically use it if available.

**Quality Comparison:**
- SimpleEmbedder: ~60-70% accuracy
- SentenceTransformer: ~85-95% accuracy

## 📊 Example Use Cases

### 1. Research Paper Q&A
```
You: add research_paper.pdf
You: What methodology was used?
Bot: According to the paper, they used...
```

### 2. Technical Documentation
```
You: add api_docs.txt
You: How do I authenticate?
Bot: Based on the documentation, authentication requires...
```

### 3. Course Materials
```
You: add lecture_notes.pdf
You: Explain neural networks
Bot: From the lecture notes, neural networks are...
```

### 4. Company Policies
```
You: add employee_handbook.docx
You: What is the vacation policy?
Bot: According to the handbook, the vacation policy is...
```

## 🔬 Advanced Features

### Multiple Documents

```bash
You: add document1.pdf
You: add document2.txt
You: add document3.docx
You: docs

📚 Loaded documents:
  1. document1.pdf
  2. document2.txt
  3. document3.docx
```

The bot will search across **all** documents.

### Persistent Knowledge Base

Save your knowledge base to avoid re-processing documents:

```bash
You: save kb
✓ Knowledge base saved

# Later...
You: load kb
✓ Knowledge base loaded
```

### Relevance Threshold

In `rag_pipeline.py`, line ~370:

```python
if score > 0.1:  # Only use chunks with >10% relevance
    context_parts.append(chunk)
```

Adjust this threshold:
- **Higher** (0.3-0.5): Only very relevant chunks → shorter, focused answers
- **Lower** (0.05-0.1): More chunks → longer, comprehensive answers

## 🐛 Troubleshooting

### Issue: "sentence-transformers not available"

**Solution:**
```bash
pip install sentence-transformers
```

Or use the built-in SimpleEmbedder (automatic fallback).

### Issue: "PyPDF2 not installed"

**Solution:**
```bash
pip install PyPDF2
```

### Issue: "Poor answers even with document"

**Solutions:**
1. **Use SentenceTransformer** instead of SimpleEmbedder
2. **Reduce chunk_size** for better granularity
3. **Increase top_k** to retrieve more context
4. **Check document quality** - ensure it's readable

### Issue: "Out of memory when loading large document"

**Solutions:**
1. **Reduce chunk_size**: `chunk_size=300`
2. **Process in batches**: Split large PDFs
3. **Use CPU**: `python chatbot_rag.py --device cpu`

### Issue: "Bot ignores the document"

**Check:**
1. Document was added successfully (see "✓ Document added")
2. Question is related to document content
3. Try more specific questions

## 📈 Performance

### Memory Usage

| Component | Memory |
|-----------|--------|
| Model (inference) | ~500 MB |
| Document embeddings | ~10 MB per 100 pages |
| Total | ~600 MB for small docs |

### Speed

| Operation | Time (GTX 1650) |
|-----------|-----------------|
| Add document (10 pages) | ~2-5 seconds |
| Search chunks | < 0.1 second |
| Generate answer | ~3-5 seconds |

## 🎓 Best Practices

### 1. Document Preparation
- ✅ Use clear, well-formatted documents
- ✅ Remove unnecessary headers/footers
- ✅ Ensure text is machine-readable (not scanned images)

### 2. Question Formulation
- ✅ Be specific: "What is the methodology?" (good)
- ❌ Too vague: "Tell me everything" (bad)

### 3. Multiple Documents
- ✅ Group related documents together
- ✅ Use clear filenames
- ✅ Save knowledge base for reuse

### 4. Quality Checks
- ✅ Test with known questions first
- ✅ Compare answers with document manually
- ✅ Adjust settings if needed

## 🔄 Comparison: Regular vs RAG Mode

| Feature | Regular Chatbot | RAG Chatbot |
|---------|----------------|-------------|
| Knowledge source | Training data only | Training data + Documents |
| Answer accuracy | General | Specific to documents |
| Up-to-date info | Only from training | From uploaded docs |
| Speed | Faster | Slightly slower |
| Memory usage | Lower | Higher |
| Best for | General chat | Document Q&A |

## 🚀 Next Steps

1. **Train on OpenAssistant** (already configured):
   ```bash
   python train.py
   ```

2. **Test RAG chatbot**:
   ```bash
   python chatbot_rag.py
   ```

3. **Add your documents**:
   ```bash
   You: add my_document.pdf
   ```

4. **Ask questions**:
   ```bash
   You: What does the document say about...?
   ```

5. **Improve quality** (optional):
   ```bash
   pip install sentence-transformers
   ```

## 📚 Additional Resources

- **Embeddings**: [sentence-transformers docs](https://www.sbert.net/)
- **RAG concepts**: [LangChain RAG guide](https://python.langchain.com/docs/use_cases/question_answering/)
- **Document processing**: [PyPDF2 docs](https://pypdf2.readthedocs.io/)

---

**Enjoy your RAG-powered chatbot!** 🎉

For questions or issues, check the troubleshooting section above.
