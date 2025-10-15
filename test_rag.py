"""
Quick test script for RAG functionality.
Tests document loading and question answering.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_pipeline import DocumentStore, SimpleEmbedder

def test_rag():
    """Test RAG pipeline with sample document."""
    print("="*70)
    print("Testing RAG Pipeline")
    print("="*70)
    
    # Initialize document store
    print("\n1. Initializing document store...")
    embedder = SimpleEmbedder()
    doc_store = DocumentStore(embedder=embedder, chunk_size=500, chunk_overlap=50)
    
    # Load sample document
    print("\n2. Loading sample document...")
    try:
        doc_store.load_document('sample_document.txt')
        print(f"✓ Successfully loaded document with {len(doc_store.chunks)} chunks")
    except FileNotFoundError:
        print("✗ sample_document.txt not found!")
        print("Creating sample document...")
        # The file should already be created
        return False
    
    # Test queries
    print("\n3. Testing queries...")
    test_queries = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "Explain neural networks",
        "What is overfitting?",
        "What libraries are used for machine learning?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {query}")
        
        results = doc_store.search(query, top_k=2)
        
        if results:
            print(f"Found {len(results)} relevant chunks:")
            for j, (chunk, score, metadata) in enumerate(results, 1):
                print(f"\nChunk {j} (score: {score:.3f}):")
                print(f"  {chunk[:200]}...")
        else:
            print("No results found")
    
    # Test saving and loading
    print("\n4. Testing save/load...")
    doc_store.save('test_knowledge_base.json')
    print("✓ Saved knowledge base")
    
    # Load into new store
    new_store = DocumentStore(embedder=SimpleEmbedder())
    new_store.load('test_knowledge_base.json')
    print(f"✓ Loaded knowledge base with {len(new_store.chunks)} chunks")
    
    # Verify it works
    results = new_store.search("What is machine learning?", top_k=1)
    if results:
        print("✓ Loaded knowledge base works correctly")
    
    print("\n" + "="*70)
    print("RAG Pipeline Test Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Train your model: python train.py")
    print("2. Start RAG chatbot: python chatbot_rag.py")
    print("3. Add document: 'add sample_document.txt'")
    print("4. Ask questions about the document!")
    
    return True


if __name__ == '__main__':
    test_rag()
