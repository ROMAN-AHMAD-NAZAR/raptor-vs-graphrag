# week5/raptor_retriever.py
"""
RAPTOR Hierarchical Retriever

Implements multi-level search that leverages the tree structure
"""

from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


class RaptorRetriever:
    """
    The RAPTOR hierarchical retriever
    
    Key innovation: Searches at multiple levels of the tree
    to provide both context (summaries) and details (chunks)
    """
    
    def __init__(self, qdrant_manager, embedder_model='all-MiniLM-L6-v2'):
        """
        Initialize RAPTOR retriever
        
        Args:
            qdrant_manager: QdrantManager instance
            embedder_model: Sentence transformer model name
        """
        self.qdrant = qdrant_manager
        self.embedder = SentenceTransformer(embedder_model)
        
        print(f"🚀 Initializing RAPTOR retriever...")
        print(f"   Using model: {embedder_model}")
    
    def hierarchical_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform hierarchical search (RAPTOR's core algorithm)
        
        Strategy:
        1. Start at root/summary level
        2. Find relevant high-level topics
        3. Drill down into those topics
        4. Combine results from multiple levels
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of results with scores and metadata
        """
        print(f"\n🌳 Performing hierarchical search for: '{query[:50]}...'")
        
        # Step 1: Embed query
        query_vector = self.embedder.encode(query).tolist()
        
        # Step 2: Multi-level search
        all_results = []
        
        # Search at summary levels first (depth 0 and 1)
        for depth in [0, 1]:
            try:
                filters = Filter(
                    must=[
                        FieldCondition(key="depth", match=MatchValue(value=depth)),
                        FieldCondition(key="is_summary", match=MatchValue(value=True))
                    ]
                )
                
                results = self.qdrant.search_similar(
                    query_vector=query_vector,
                    limit=3,  # Few top summaries at each level
                    filters=filters
                )
                
                print(f"   Depth {depth} (summaries): Found {len(results)} relevant nodes")
                
                for result in results:
                    result['search_level'] = 'summary'
                    result['summary_depth'] = depth
                    all_results.append(result)
                    
            except Exception as e:
                print(f"   Depth {depth}: No summaries found ({e})")
        
        # Step 3: Search at leaf level (actual chunks)
        try:
            leaf_filters = Filter(
                must=[
                    FieldCondition(key="is_summary", match=MatchValue(value=False))
                ]
            )
            
            leaf_results = self.qdrant.search_similar(
                query_vector=query_vector,
                limit=top_k * 2,  # Get more leaves initially
                filters=leaf_filters
            )
        except:
            # If filter fails, search all
            leaf_results = self.qdrant.search_similar(
                query_vector=query_vector,
                limit=top_k * 2
            )
        
        print(f"   Leaf level: Found {len(leaf_results)} relevant chunks")
        
        # Step 4: Combine and rerank
        combined_results = self._combine_and_rerank(all_results, leaf_results, query_vector)
        
        # Step 5: Return top-k
        final_results = combined_results[:top_k]
        
        print(f"✅ Found {len(final_results)} final results")
        print(f"   Sources: {len([r for r in final_results if r.get('is_summary', False)])} summaries, "
              f"{len([r for r in final_results if not r.get('is_summary', False)])} chunks")
        
        return final_results
    
    def _combine_and_rerank(self, summary_results: List[Dict], 
                           leaf_results: List[Dict],
                           query_vector: List[float]) -> List[Dict]:
        """
        Combine results from different levels and rerank
        
        Strategy:
        - Boost scores from relevant summaries
        - Include their children (if any)
        - Remove duplicates
        """
        combined = []
        
        # Add summary results (with score boost for being summaries)
        for result in summary_results:
            # Summaries get a small boost because they provide context
            boosted_result = result.copy()
            boosted_result['score'] = result['score'] * 1.1  # 10% boost
            boosted_result['is_context_summary'] = True
            combined.append(boosted_result)
        
        # Add leaf results
        for result in leaf_results:
            result['is_context_summary'] = False
            combined.append(result)
        
        # Sort by score
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates (same text)
        seen_texts = set()
        unique_results = []
        
        for result in combined:
            text = result.get('text', '')[:100]  # Use first 100 chars as key
            if text not in seen_texts:
                seen_texts.add(text)
                unique_results.append(result)
        
        return unique_results
    
    def simple_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Simple search without hierarchy (for comparison)
        """
        query_vector = self.embedder.encode(query).tolist()
        
        results = self.qdrant.search_similar(
            query_vector=query_vector,
            limit=top_k
        )
        
        return results
    
    def explain_retrieval(self, query: str, results: List[Dict]) -> str:
        """
        Generate an explanation of why these results were retrieved
        Great for debugging and understanding RAPTOR!
        """
        explanation = []
        explanation.append(f"Query: '{query}'")
        explanation.append(f"Retrieved {len(results)} results:")
        
        for i, result in enumerate(results[:5]):  # Top 5 only
            source = "📝 Summary" if result.get('is_summary', False) else "📄 Chunk"
            depth = f"Depth {result.get('depth', 0)}"
            score = f"Score: {result['score']:.3f}"
            
            explanation.append(f"\n{i+1}. {source} ({depth}) - {score}")
            text_preview = result.get('text', '')[:100]
            explanation.append(f"   Text: {text_preview}...")
        
        # Statistics
        num_summaries = len([r for r in results if r.get('is_summary', False)])
        num_chunks = len([r for r in results if not r.get('is_summary', False)])
        
        explanation.append(f"\n📊 Statistics:")
        explanation.append(f"   Total: {len(results)} results")
        explanation.append(f"   Summaries: {num_summaries}")
        explanation.append(f"   Chunks: {num_chunks}")
        
        if results:
            avg_depth = sum(r.get('depth', 0) for r in results) / len(results)
            explanation.append(f"   Avg depth: {avg_depth:.1f}")
        
        return "\n".join(explanation)
    
    def retrieve_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve with context from parent summaries
        
        Returns chunks with their parent summary as context
        This is RAPTOR's superpower!
        """
        # First get hierarchical results
        results = self.hierarchical_search(query, top_k * 2)
        
        # Enhance chunks with parent summaries
        enhanced_results = []
        
        for result in results:
            if not result.get('is_summary', False):
                # Try to find parent summary
                result['context'] = self._find_context_for_chunk(result, results)
            
            enhanced_results.append(result)
        
        return enhanced_results[:top_k]
    
    def _find_context_for_chunk(self, chunk_result: Dict, all_results: List[Dict]) -> str:
        """Find relevant context/summary for a chunk"""
        # Look for summaries at shallower depth
        possible_contexts = []
        
        for result in all_results:
            if (result.get('is_summary', False) and 
                result.get('depth', 0) < chunk_result.get('depth', 0) and
                result['score'] > 0.5):  # Only good summaries
                possible_contexts.append(result)
        
        if possible_contexts:
            # Return the best summary
            best_context = max(possible_contexts, key=lambda x: x['score'])
            return f"Context (Depth {best_context.get('depth', 0)}): {best_context.get('text', '')[:150]}..."
        
        return "No context available"
    
    def get_tree_overview(self) -> Dict:
        """Get overview of the stored tree"""
        stats = self.qdrant.count_by_type()
        return {
            'total_nodes': stats.get('total', 0),
            'summaries': stats.get('summary', 0),
            'chunks': stats.get('chunk', 0),
            'by_depth': stats.get('by_depth', {})
        }
