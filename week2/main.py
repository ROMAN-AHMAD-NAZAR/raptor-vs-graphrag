"""Week 2 Main: Embedding Pipeline - Enhanced Version"""
import os
import sys
import pickle

# Add directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 60)
    print("WEEK 2: Creating Intelligent Embeddings")
    print("=" * 60)
    
    # Step 1: Load Week 1 chunks
    print("\n📂 Step 1: Loading Week 1 chunks...")
    chunks_path = os.path.join(PROJECT_ROOT, 'outputs', 'week1_chunks.pkl')
    
    try:
        with open(chunks_path, 'rb') as f:
            data = pickle.load(f)
        
        # Handle both old format (list) and new format (dict)
        if isinstance(data, dict):
            chunks = data['chunks']
            chunk_metadata = data.get('metadata', [])
        else:
            chunks = data
            chunk_metadata = [{'chunk_id': i} for i in range(len(chunks))]
        
        print(f"✅ Loaded {len(chunks)} chunks from Week 1")
        
        # Show sample
        print(f"\nSample chunks from Week 1:")
        for i in range(min(3, len(chunks))):
            preview = chunks[i][:80] + "..." if len(chunks[i]) > 80 else chunks[i]
            print(f"  [{i}] {preview}")
    except FileNotFoundError:
        print("❌ ERROR: Week 1 output not found!")
        print(f"   Expected at: {chunks_path}")
        print("   Run Week 1 first: python week1/main.py")
        return
    
    # Step 2: Initialize Embeddings Manager
    print("\n" + "=" * 40)
    print("Step 2: Initializing Embedding Model")
    print("=" * 40)
    
    from embeddings_manager import EmbeddingsManager
    
    # You can change the model here:
    # - 'all-MiniLM-L6-v2' (default, fast)
    # - 'all-mpnet-base-v2' (slower, better)
    # - 'BAAI/bge-small-en-v1.5' (optimized for retrieval)
    embedder = EmbeddingsManager(model_name='all-MiniLM-L6-v2')
    
    # Test semantic understanding
    embedder.test_semantic_understanding()
    
    # Step 3: Create Embeddings
    print("\n" + "=" * 40)
    print("Step 3: Creating Embeddings")
    print("=" * 40)
    
    embeddings = embedder.create_embeddings(chunks, batch_size=16)
    
    # Validate embeddings
    embedder.validate_embeddings(embeddings)
    
    # Step 4: Optimize Chunks
    print("\n" + "=" * 40)
    print("Step 4: Optimizing Chunks")
    print("=" * 40)
    
    from chunk_optimizer import ChunkOptimizer
    
    optimizer = ChunkOptimizer(similarity_threshold=0.85)
    
    # Analyze similarities first
    optimizer.analyze_chunk_similarities(chunks, embeddings, top_n=5)
    
    # Find outliers
    optimizer.find_outliers(chunks, embeddings, threshold=0.3)
    
    # Merge similar chunks
    optimized_chunks, optimized_embeddings = optimizer.merge_similar_chunks(
        chunks, embeddings
    )
    
    # Step 5: Save Everything
    print("\n" + "=" * 40)
    print("Step 5: Saving Results")
    print("=" * 40)
    
    from embedding_storage import EmbeddingStorage
    
    embeddings_dir = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
    storage = EmbeddingStorage(save_dir=embeddings_dir)
    
    # Create metadata for chunks
    metadata = []
    for i, chunk in enumerate(optimized_chunks):
        metadata.append({
            'id': i,
            'word_count': len(chunk.split()),
            'char_count': len(chunk),
            'is_optimized': True,
            'original_chunk_count': 1  # Simplified - in real case track merges
        })
    
    # Save
    storage.save_embeddings(
        optimized_chunks, 
        optimized_embeddings, 
        metadata,
        filename="week2_embeddings"
    )
    
    # Also save FAISS index for fast search
    print("\n📦 Building FAISS index...")
    from embedder import FAISSIndex
    import numpy as np
    
    index = FAISSIndex(embedding_dim=optimized_embeddings.shape[1])
    index.add(optimized_embeddings.copy())
    index_path = os.path.join(PROJECT_ROOT, "outputs", "week2_faiss.index")
    index.save(index_path)
    
    # Step 6: Test Search
    print("\n" + "=" * 40)
    print("Step 6: Testing Semantic Search")
    print("=" * 40)
    
    test_queries = [
        "What is RAPTOR?",
        "How does clustering work in the paper?",
        "What are the experimental results?"
    ]
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        query_embedding = embedder.embed_query(query)
        
        # Compute similarities
        similarities = cosine_similarity([query_embedding], optimized_embeddings)[0]
        
        # Get top 3
        top_indices = similarities.argsort()[-3:][::-1]
        
        print("   Top 3 results:")
        for rank, idx in enumerate(top_indices):
            preview = optimized_chunks[idx][:60].replace('\n', ' ')
            print(f"   {rank+1}. [score: {similarities[idx]:.3f}] {preview}...")
    
    # Step 7: Summary
    print("\n" + "=" * 60)
    print("WEEK 2 SUMMARY")
    print("=" * 60)
    
    stats = storage.get_stats("week2_embeddings")
    if stats:
        print(f"\n📊 Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print(f"\n🎯 Key Achievements:")
    print(f"  1. Converted text to {optimized_embeddings.shape[1]}-dimensional vectors")
    reduction = ((len(chunks) - len(optimized_chunks)) / len(chunks) * 100)
    print(f"  2. Reduced {len(chunks)} → {len(optimized_chunks)} chunks ({reduction:.1f}% reduction)")
    print(f"  3. All vectors normalized (magnitude ~1.0)")
    print(f"  4. Semantic understanding verified")
    print(f"  5. Search functionality tested")
    
    print(f"\n📁 Files created:")
    for f in storage.list_saved_files():
        if "week2" in f['name']:
            print(f"  - {f['name']} ({f['size_kb']:.1f} KB)")
    print(f"  - week2_faiss.index")
    
    print(f"\n✅ WEEK 2 COMPLETE!")
    print(f"   Next: Week 3 - Clustering & Building the Tree Structure")


if __name__ == "__main__":
    main()
