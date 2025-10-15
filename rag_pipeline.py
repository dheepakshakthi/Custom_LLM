"""
RAG (Retrieval-Augmented Generation) Pipeline
Allows the chatbot to answer questions based on uploaded documents.
"""
import torch
import numpy as np
from typing import List, Tuple
import os
from pathlib import Path
import json

class SimpleEmbedder:
    """Simple TF-IDF style embeddings for document chunks."""
    
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.vocab_size = 0
    
    def fit(self, texts: List[str]):
        """Build vocabulary and IDF from texts."""
        # Build vocabulary
        word_counts = {}
        doc_counts = {}
        
        for text in texts:
            words = set(text.lower().split())
            for word in words:
                doc_counts[word] = doc_counts.get(word, 0) + 1
        
        # Assign IDs and calculate IDF
        for word, doc_count in doc_counts.items():
            self.vocab[word] = len(self.vocab)
            self.idf[word] = np.log(len(texts) / (1 + doc_count))
        
        self.vocab_size = len(self.vocab)
    
    def embed(self, text: str) -> np.ndarray:
        """Create embedding for text."""
        words = text.lower().split()
        embedding = np.zeros(max(100, self.vocab_size))  # Fixed size
        
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        for word, count in word_counts.items():
            if word in self.vocab:
                idx = self.vocab[word] % len(embedding)
                tf = count / len(words)
                embedding[idx] = tf * self.idf.get(word, 1.0)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding


class SentenceTransformerEmbedder:
    """Use sentence-transformers for better embeddings (requires installation)."""
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError:
            print("sentence-transformers not available. Install with: pip install sentence-transformers")
            self.available = False
    
    def embed(self, text: str) -> np.ndarray:
        """Create embedding for text."""
        if not self.available:
            raise RuntimeError("sentence-transformers not installed")
        return self.model.encode(text, convert_to_numpy=True)


class DocumentStore:
    """Store and retrieve document chunks."""
    
    def __init__(self, embedder=None, chunk_size=500, chunk_overlap=50):
        """
        Args:
            embedder: Embedding model (SimpleEmbedder or SentenceTransformerEmbedder)
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks
        """
        self.embedder = embedder or SimpleEmbedder()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.chunks = []
        self.embeddings = []
        self.metadata = []
    
    def load_document(self, file_path: str):
        """Load a document and add it to the store."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Read document based on file type
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_path.suffix == '.pdf':
            text = self._read_pdf(file_path)
        elif file_path.suffix == '.docx':
            text = self._read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        # Chunk the document
        chunks = self._chunk_text(text)
        
        # Create embeddings
        print(f"Processing {len(chunks)} chunks from {file_path.name}...")
        
        # Fit embedder if using SimpleEmbedder
        if isinstance(self.embedder, SimpleEmbedder) and not self.embedder.vocab:
            self.embedder.fit(chunks)
        
        for i, chunk in enumerate(chunks):
            embedding = self.embedder.embed(chunk)
            
            self.chunks.append(chunk)
            self.embeddings.append(embedding)
            self.metadata.append({
                'file': file_path.name,
                'chunk_id': i,
                'total_chunks': len(chunks)
            })
        
        print(f"Added {len(chunks)} chunks from {file_path.name}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > self.chunk_size * 0.5:  # At least 50% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - self.chunk_overlap
        
        return [c for c in chunks if len(c.strip()) > 20]
    
    def _read_pdf(self, file_path: Path) -> str:
        """Read PDF file."""
        try:
            import PyPDF2
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except ImportError:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
    
    def _read_docx(self, file_path: Path) -> str:
        """Read DOCX file."""
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])
        except ImportError:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float, dict]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of (chunk_text, similarity_score, metadata) tuples
        """
        if not self.chunks:
            return []
        
        # Embed query
        query_embedding = self.embedder.embed(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        results = []
        for idx, score in similarities[:top_k]:
            results.append((self.chunks[idx], float(score), self.metadata[idx]))
        
        return results
    
    def save(self, save_path: str):
        """Save document store to disk."""
        data = {
            'chunks': self.chunks,
            'embeddings': [emb.tolist() for emb in self.embeddings],
            'metadata': self.metadata,
            'embedder_vocab': self.embedder.vocab if isinstance(self.embedder, SimpleEmbedder) else None,
            'embedder_idf': self.embedder.idf if isinstance(self.embedder, SimpleEmbedder) else None,
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        print(f"Document store saved to {save_path}")
    
    def load(self, load_path: str):
        """Load document store from disk."""
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.embeddings = [np.array(emb) for emb in data['embeddings']]
        self.metadata = data['metadata']
        
        if isinstance(self.embedder, SimpleEmbedder) and data['embedder_vocab']:
            self.embedder.vocab = data['embedder_vocab']
            self.embedder.idf = data['embedder_idf']
            self.embedder.vocab_size = len(self.embedder.vocab)
        
        print(f"Document store loaded from {load_path}")


class RAGChatbot:
    """Chatbot with RAG capabilities."""
    
    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: Trained LLM model
            tokenizer: Tokenizer instance
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize document store
        try:
            # Try to use sentence-transformers for better embeddings
            embedder = SentenceTransformerEmbedder()
            if not embedder.available:
                embedder = SimpleEmbedder()
        except:
            embedder = SimpleEmbedder()
        
        self.doc_store = DocumentStore(embedder=embedder)
        
        print("RAG Chatbot initialized")
        print(f"Embedder: {type(self.doc_store.embedder).__name__}")
    
    def add_document(self, file_path: str):
        """Add a document to the knowledge base."""
        self.doc_store.load_document(file_path)
    
    def generate_with_context(self, query: str, max_new_tokens=150, temperature=0.7, top_k=50, top_p=0.9):
        """
        Generate response using RAG.
        
        Args:
            query: User query
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            Generated response
        """
        # Retrieve relevant context
        if self.doc_store.chunks:
            relevant_chunks = self.doc_store.search(query, top_k=3)
            
            # Build context
            context_parts = []
            for chunk, score, metadata in relevant_chunks:
                if score > 0.1:  # Threshold for relevance
                    context_parts.append(f"[From {metadata['file']}]: {chunk[:300]}...")
            
            if context_parts:
                context = "\n\n".join(context_parts)
                prompt = f"Context from documents:\n{context}\n\nUser: {query}\nAssistant:"
            else:
                prompt = f"User: {query}\nAssistant:"
        else:
            prompt = f"User: {query}\nAssistant:"
        
        # Generate response
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.shape[1] > self.model.params.max_seq_len:
                    input_chunk = generated[:, -self.model.params.max_seq_len:]
                else:
                    input_chunk = generated
                
                outputs = self.model(input_chunk)
                logits = outputs[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS
                if hasattr(self.tokenizer, 'tokenizer') and next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                    break
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0].tolist())
        
        # Extract response
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:")[-1].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        # Clean up
        response = response.split('\n')[0].strip()
        
        return response if response else "I'm not sure how to respond to that."
    
    def save_knowledge_base(self, save_path='knowledge_base.json'):
        """Save the document store."""
        self.doc_store.save(save_path)
    
    def load_knowledge_base(self, load_path='knowledge_base.json'):
        """Load the document store."""
        self.doc_store.load(load_path)
