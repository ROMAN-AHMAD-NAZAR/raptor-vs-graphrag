# week4_graph_retrieval/hybrid_retriever.py
"""
Hybrid Retriever for GraphRAG
Combines semantic similarity and graph traversal approaches
with multiple fusion strategies and optional reranking
"""

from typing import List, Dict, Tuple, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from .graph_retriever import RetrievalResult


class RetrievalMode(Enum):
    """Available retrieval modes"""
    GRAPH_ONLY = "graph_only"
    SEMANTIC_ONLY = "semantic_only"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


@dataclass
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval"""
    semantic_weight: float = 0.6
    graph_weight: float = 0.4
    max_semantic_results: int = 20
    max_graph_results: int = 20
    fusion_method: str = "weighted_sum"  # weighted_sum, reciprocal_rank, comb_mnz
    use_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    min_score_threshold: float = 0.1


class HybridRetriever:
    """
    Advanced hybrid retriever combining:
    1. Semantic similarity (vector search)
    2. Graph traversal (knowledge graph)
    3. Optional: Cross-encoder reranking
    """
    
    def __init__(self, graph_retriever, config: HybridRetrievalConfig = None):
        self.graph_retriever = graph_retriever
        self.config = config or HybridRetrievalConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize reranker if configured
        self.reranker = None
        if self.config.use_reranking:
            self._initialize_reranker()
        
        self.logger.info("✅ HybridRetriever initialized")
        self.logger.info(f"   Semantic weight: {self.config.semantic_weight}")
        self.logger.info(f"   Graph weight: {self.config.graph_weight}")
        self.logger.info(f"   Fusion method: {self.config.fusion_method}")
    
    def _initialize_reranker(self):
        """Initialize cross-encoder reranker"""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.config.reranker_model)
            self.logger.info(f"✅ Cross-encoder reranker loaded: {self.config.reranker_model}")
        except ImportError:
            self.logger.warning("CrossEncoder not available, skipping reranking")
            self.logger.info("Install with: pip install sentence-transformers")
            self.config.use_reranking = False
        except Exception as e:
            self.logger.warning(f"Failed to load reranker: {e}")
            self.config.use_reranking = False
    
    def retrieve(self, query: str, top_k: int = 10, 
                mode: RetrievalMode = RetrievalMode.HYBRID) -> List[Dict]:
        """
        Hybrid retrieval with multiple fusion strategies
        
        Args:
            query: Search query
            top_k: Number of results to return
            mode: Retrieval mode (semantic, graph, hybrid, ensemble)
        
        Returns:
            List of result dictionaries
        """
        self.logger.info(f"🎯 Hybrid retrieval for: '{query[:50]}...'")
        self.logger.info(f"   Mode: {mode.value}, Top-K: {top_k}")
        
        # Get results from different retrieval methods
        semantic_results = []
        graph_results = []
        
        if mode in [RetrievalMode.SEMANTIC_ONLY, RetrievalMode.HYBRID, RetrievalMode.ENSEMBLE]:
            semantic_results = self.graph_retriever._semantic_retrieval(
                query, self.config.max_semantic_results
            )
            self.logger.info(f"   Semantic retrieval: {len(semantic_results)} results")
        
        if mode in [RetrievalMode.GRAPH_ONLY, RetrievalMode.HYBRID, RetrievalMode.ENSEMBLE]:
            graph_results = self.graph_retriever._graph_traversal_retrieval(
                query, self.config.max_graph_results
            )
            self.logger.info(f"   Graph retrieval: {len(graph_results)} results")
        
        # Fuse results based on configuration
        if mode == RetrievalMode.SEMANTIC_ONLY:
            final_results = semantic_results
        elif mode == RetrievalMode.GRAPH_ONLY:
            final_results = graph_results
        else:
            final_results = self._fuse_results(
                semantic_results, graph_results, query
            )
        
        # Apply reranking if configured
        if self.config.use_reranking and self.reranker and final_results:
            final_results = self._rerank_results(query, final_results)
        
        # Filter by minimum score threshold
        final_results = [r for r in final_results 
                        if r.score >= self.config.min_score_threshold]
        
        # Convert to dict format
        return [r.to_dict() for r in final_results[:top_k]]
    
    def _fuse_results(self, semantic_results: List[RetrievalResult], 
                     graph_results: List[RetrievalResult], 
                     query: str) -> List[RetrievalResult]:
        """
        Fuse results from multiple retrieval methods
        """
        if not semantic_results and not graph_results:
            return []
        
        if not semantic_results:
            return graph_results
        if not graph_results:
            return semantic_results
        
        # Combine results
        all_results = {}
        
        # Add semantic results with rank
        for i, result in enumerate(semantic_results):
            all_results[result.node_id] = {
                'result': result,
                'semantic_score': result.score,
                'graph_score': 0.0,
                'semantic_rank': i + 1
            }
        
        # Add graph results
        for i, result in enumerate(graph_results):
            if result.node_id in all_results:
                # Update existing result
                all_results[result.node_id]['graph_score'] = result.score
                all_results[result.node_id]['graph_rank'] = i + 1
            else:
                all_results[result.node_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'graph_score': result.score,
                    'graph_rank': i + 1
                }
        
        # Apply fusion method
        fused_results = []
        for data in all_results.values():
            result = self._create_fused_result(
                data, 
                len(semantic_results), 
                len(graph_results)
            )
            fused_results.append(result)
        
        # Sort by fused score
        fused_results.sort(key=lambda x: x.score, reverse=True)
        
        return fused_results
    
    def _create_fused_result(self, data: Dict, 
                            num_semantic: int, 
                            num_graph: int) -> RetrievalResult:
        """Create a fused result from combined data"""
        result = data['result']
        
        if self.config.fusion_method == "weighted_sum":
            fused_score = (
                data['semantic_score'] * self.config.semantic_weight +
                data['graph_score'] * self.config.graph_weight
            )
        elif self.config.fusion_method == "reciprocal_rank":
            # Combine reciprocal ranks
            rr_semantic = 1.0 / data.get('semantic_rank', num_semantic + 1)
            rr_graph = 1.0 / data.get('graph_rank', num_graph + 1)
            fused_score = rr_semantic + rr_graph
        elif self.config.fusion_method == "comb_mnz":
            # CombMNZ: (sum of scores) * (number of systems that retrieved it)
            num_systems = sum([
                1 if data['semantic_score'] > 0 else 0,
                1 if data['graph_score'] > 0 else 0
            ])
            fused_score = (data['semantic_score'] + data['graph_score']) * num_systems
        else:
            # Default to weighted sum
            fused_score = data['semantic_score'] * 0.5 + data['graph_score'] * 0.5
        
        result.score = float(fused_score)
        result.retrieval_method = "hybrid_fusion"
        
        # Add fusion evidence
        evidence = []
        if data['semantic_score'] > 0:
            evidence.append(f"Semantic: {data['semantic_score']:.3f}")
        if data['graph_score'] > 0:
            evidence.append(f"Graph: {data['graph_score']:.3f}")
        
        if evidence:
            result.evidence = evidence
        
        return result
    
    def _rerank_results(self, query: str, 
                       results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Re-rank results using cross-encoder for better precision
        """
        if not results or len(results) <= 1:
            return results
        
        self.logger.info(f"   Reranking {len(results)} results...")
        
        try:
            # Prepare query-document pairs
            pairs = []
            for result in results:
                # Create document text from result context
                doc_text = f"{result.node_name} ({result.node_type}). {result.context}"
                pairs.append((query, doc_text))
            
            # Get cross-encoder scores
            scores = self.reranker.predict(pairs)
            
            # Update result scores
            for i, result in enumerate(results):
                if i < len(scores):
                    # Combine original score with reranker score
                    rerank_score = float(scores[i])
                    result.score = result.score * 0.3 + rerank_score * 0.7
                    result.evidence.append(f"Reranker score: {rerank_score:.3f}")
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            self.logger.info(f"   Reranking complete")
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
        
        return results
    
    def compare_retrieval_modes(self, query: str, top_k: int = 10) -> Dict:
        """
        Compare different retrieval modes for analysis
        """
        comparison = {
            'query': query,
            'top_k': top_k,
            'results': {}
        }
        
        for mode in RetrievalMode:
            try:
                results = self.retrieve(query, top_k, mode)
                comparison['results'][mode.value] = {
                    'count': len(results),
                    'top_scores': [r['score'] for r in results[:3]],
                    'top_nodes': [
                        {
                            'name': r['node_name'],
                            'type': r['node_type'],
                            'score': r['score']
                        }
                        for r in results[:3]
                    ]
                }
                self.logger.info(f"   {mode.value}: {len(results)} results")
            except Exception as e:
                self.logger.error(f"Failed to retrieve with {mode}: {e}")
                comparison['results'][mode.value] = {'error': str(e)}
        
        return comparison
    
    def generate_detailed_report(self, query: str) -> str:
        """Generate detailed retrieval report for analysis"""
        report = []
        report.append("=" * 70)
        report.append("HYBRID RETRIEVAL DETAILED REPORT")
        report.append("=" * 70)
        report.append(f"\nQuery: {query}")
        
        # Get comparison of all modes
        comparison = self.compare_retrieval_modes(query, top_k=5)
        
        report.append("\n📊 RETRIEVAL MODE COMPARISON:")
        report.append("-" * 40)
        
        for mode, data in comparison['results'].items():
            if 'error' in data:
                report.append(f"\n{mode.upper()}: ERROR - {data['error']}")
            else:
                report.append(f"\n{mode.upper()}:")
                report.append(f"  Results: {data['count']}")
                report.append(f"  Top scores: {[f'{s:.3f}' for s in data['top_scores']]}")
                
                for i, node in enumerate(data['top_nodes'][:3], 1):
                    report.append(f"  {i}. {node['name']} [{node['type']}] - {node['score']:.3f}")
        
        # Get hybrid results with explanation
        hybrid_results = self.retrieve(query, top_k=10, mode=RetrievalMode.HYBRID)
        
        report.append("\n🔍 HYBRID RETRIEVAL RESULTS (Top 10):")
        report.append("-" * 40)
        
        for i, result in enumerate(hybrid_results[:10], 1):
            report.append(f"\n{i}. {result['node_name']} [{result['node_type']}]")
            report.append(f"   Score: {result['score']:.4f}")
            report.append(f"   Context: {result['context'][:100]}..." if result['context'] else "   Context: N/A")
            
            if result.get('evidence'):
                report.append(f"   Evidence:")
                for evidence in result['evidence'][:2]:
                    report.append(f"     • {evidence}")
        
        # Statistics
        report.append("\n📈 STATISTICS:")
        report.append("-" * 40)
        
        if hybrid_results:
            scores = [r['score'] for r in hybrid_results]
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            
            report.append(f"Average score: {avg_score:.4f}")
            report.append(f"Max score: {max_score:.4f}")
            report.append(f"Min score: {min_score:.4f}")
            report.append(f"Score range: {max_score - min_score:.4f}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
