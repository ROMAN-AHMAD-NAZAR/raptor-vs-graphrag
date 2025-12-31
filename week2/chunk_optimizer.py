# week2/chunk_optimizer.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ChunkOptimizer:
    def __init__(self, similarity_threshold=0.85):
        """
        WHY merge similar chunks?
        
        Problem without merging:
        - "The cat sat" → Chunk 1
        - "on the mat." → Chunk 2  
        - These are separate but should be together
        
        Solution: Merge chunks with similarity > threshold
        """
        self.threshold = similarity_threshold
    
    def merge_similar_chunks(self, chunks, embeddings):
        """
        Merge chunks that are semantically similar
        """
        if len(chunks) <= 1:
            return chunks, embeddings
        
        print(f"\n🔄 Optimizing chunks (merging similar ones)...")
        print(f"   Similarity threshold: {self.threshold}")
        print(f"   Starting with {len(chunks)} chunks")
        
        merged_chunks = []
        merged_embeddings = []
        skip_indices = set()
        
        for i in range(len(chunks)):
            if i in skip_indices:
                continue
            
            # Start with current chunk
            current_chunk = chunks[i]
            current_embedding = embeddings[i].copy()
            merged_count = 1
            
            # Look ahead for similar chunks (only check nearby chunks)
            for j in range(i + 1, min(i + 10, len(chunks))):  # Only check next 10
                if j in skip_indices:
                    continue
                
                similarity = cosine_similarity(
                    [current_embedding], 
                    [embeddings[j]]
                )[0][0]
                
                if similarity > self.threshold:
                    # Merge them
                    current_chunk += " " + chunks[j]
                    
                    # Weighted average of embeddings
                    weight_i = len(chunks[i].split())
                    weight_j = len(chunks[j].split())
                    current_embedding = (
                        current_embedding * weight_i + 
                        embeddings[j] * weight_j
                    ) / (weight_i + weight_j)
                    
                    # Normalize
                    current_embedding = current_embedding / np.linalg.norm(current_embedding)
                    
                    skip_indices.add(j)
                    merged_count += 1
            
            merged_chunks.append(current_chunk)
            merged_embeddings.append(current_embedding)
            
            if merged_count > 1:
                print(f"   Merged {merged_count} chunks at index {i}")
        
        merged_embeddings = np.array(merged_embeddings)
        
        print(f"✅ Optimization complete!")
        print(f"   Final chunks: {len(merged_chunks)}")
        reduction = ((len(chunks) - len(merged_chunks)) / len(chunks)) * 100
        print(f"   Reduction: {reduction:.1f}%")
        
        return merged_chunks, merged_embeddings
    
    def analyze_chunk_similarities(self, chunks, embeddings, top_n=10):
        """
        Show the most similar chunk pairs
        Useful for debugging
        """
        print(f"\n📈 Analyzing chunk similarities...")
        
        # Compute similarity matrix
        similarities = cosine_similarity(embeddings)
        
        # Find top similar pairs (excluding self-similarity)
        similar_pairs = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):
                similar_pairs.append((i, j, similarities[i][j]))
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"Top {min(top_n, len(similar_pairs))} most similar pairs:")
        for idx, (i, j, sim) in enumerate(similar_pairs[:top_n]):
            chunk_i_preview = chunks[i][:40] + "..." if len(chunks[i]) > 40 else chunks[i]
            chunk_j_preview = chunks[j][:40] + "..." if len(chunks[j]) > 40 else chunks[j]
            print(f"\n  Pair {idx+1} (similarity: {sim:.3f}):")
            print(f"    [{i}] '{chunk_i_preview}'")
            print(f"    [{j}] '{chunk_j_preview}'")
        
        # Stats
        avg_sim = np.mean([s[2] for s in similar_pairs])
        max_sim = similar_pairs[0][2] if similar_pairs else 0
        min_sim = similar_pairs[-1][2] if similar_pairs else 0
        
        print(f"\n  📊 Similarity Stats:")
        print(f"     Average: {avg_sim:.3f}")
        print(f"     Maximum: {max_sim:.3f}")
        print(f"     Minimum: {min_sim:.3f}")
        
        return similar_pairs
    
    def find_outliers(self, chunks, embeddings, threshold=0.3):
        """Find chunks that are very different from others"""
        print(f"\n🔍 Finding outlier chunks...")
        
        similarities = cosine_similarity(embeddings)
        
        outliers = []
        for i in range(len(chunks)):
            # Average similarity to other chunks (excluding self)
            avg_sim = (np.sum(similarities[i]) - 1) / (len(chunks) - 1)
            if avg_sim < threshold:
                outliers.append((i, avg_sim, chunks[i][:50]))
        
        if outliers:
            print(f"  Found {len(outliers)} outlier chunks:")
            for idx, avg_sim, preview in outliers:
                print(f"    [{idx}] avg_sim={avg_sim:.3f}: '{preview}...'")
        else:
            print(f"  No outliers found (threshold: {threshold})")
        
        return outliers
