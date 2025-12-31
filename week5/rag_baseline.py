# week5/rag_baseline.py
"""
Traditional RAG Baseline for Comparison

This is what most systems use - flat retrieval without hierarchy
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict


class RAGBaseline:
    """
    Traditional RAG implementation for comparison
    
    Features:
    - Flat retrieval (no hierarchy)
    - Direct chunk similarity search
    - No context from summaries
    """
    
    def __init__(self, qdrant_manager, embedder_model='all-MiniLM-L6-v2'):
        """
        Initialize RAG baseline
        
        Args:
            qdrant_manager: QdrantManager instance
            embedder_model: Sentence transformer model name
        """
        self.qdrant = qdrant_manager
        self.embedder = SentenceTransformer(embedder_model)
        
        print(f"📊 Initializing normal RAG baseline...")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Standard RAG search - just find similar chunks
        
        No hierarchy, no context, just pure similarity
        """
        # Embed query
        query_vector = self.embedder.encode(query).tolist()
        
        # Search in normal_rag collection
        results = self.qdrant.search_similar(
            query_vector=query_vector,
            limit=top_k,
            collection="normal_rag"
        )
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                'id': result['id'],
                'score': result['score'],
                'text': result['text'],
                'source': 'normal_rag',
                'type': 'chunk',
                'depth': 0,
                'is_summary': False
            })
        
        return formatted
    
    def explain_retrieval(self, query: str, results: List[Dict]) -> str:
        """Explain normal RAG retrieval"""
        explanation = []
        explanation.append(f"Normal RAG Query: '{query}'")
        explanation.append(f"Retrieved {len(results)} chunks:")
        
        for i, result in enumerate(results[:5]):
            explanation.append(f"\n{i+1}. Score: {result['score']:.3f}")
            text_preview = result.get('text', '')[:100]
            explanation.append(f"   Text: {text_preview}...")
        
        return "\n".join(explanation)
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict]]:
        """
        Search for multiple queries
        
        Returns list of results for each query
        """
        all_results = []
        
        for query in queries:
            results = self.search(query, top_k)
            all_results.append(results)
        
        return all_results
    
    def get_stats(self) -> Dict:
        """Get statistics about the RAG collection"""
        stats = self.qdrant.get_collection_stats("normal_rag")
        return {
            'total_chunks': stats.get('vectors_count', 0),
            'collection': 'normal_rag'
        }
