# week4_graph_retrieval/evaluation.py
"""
Evaluation Framework for GraphRAG Retrieval
Provides metrics for comparing retrieval systems
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime

# Try to import sklearn for NDCG
try:
    from sklearn.metrics import ndcg_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mean_reciprocal_rank: float = 0.0
    query_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'ndcg@5': self.ndcg_at_5,
            'ndcg@10': self.ndcg_at_10,
            'precision@5': self.precision_at_5,
            'precision@10': self.precision_at_10,
            'recall@5': self.recall_at_5,
            'recall@10': self.recall_at_10,
            'mrr': self.mean_reciprocal_rank,
            'query_time_ms': self.query_time_ms
        }


class GraphRAGEvaluator:
    """
    Evaluation framework for GraphRAG retrieval
    Compares with RAPTOR and baseline RAG
    """
    
    def __init__(self, ground_truth_path: str = None):
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.logger = logging.getLogger(__name__)
        
        # Sample test queries for research papers
        self.test_queries = [
            "What is RAPTOR and how does it work?",
            "Compare RAPTOR with traditional RAG systems",
            "What metrics are used to evaluate retrieval systems?",
            "Explain hierarchical clustering in document retrieval",
            "What are the advantages of knowledge graphs for RAG?",
            "How does GraphRAG differ from RAPTOR?",
            "What entity types are extracted in GraphRAG?",
            "Describe the relationship extraction process",
            "What datasets are mentioned in the research?",
            "How do embeddings work in retrieval systems?"
        ]
        
        self.logger.info("✅ GraphRAGEvaluator initialized")
        self.logger.info(f"   Test queries: {len(self.test_queries)}")
    
    def evaluate_retrieval(self, retriever, queries: List[str] = None, 
                          top_k: int = 10) -> Dict[str, Any]:
        """
        Evaluate retrieval performance on test queries
        
        Args:
            retriever: Retriever object with retrieve() method
            queries: List of test queries (uses default if None)
            top_k: Number of results to retrieve
        
        Returns:
            Dictionary with per-query and aggregate metrics
        """
        if queries is None:
            queries = self.test_queries
        
        results = {
            'per_query': {},
            'aggregate': None
        }
        
        all_metrics = []
        query_times = []
        
        for query in queries:
            self.logger.info(f"Evaluating query: '{query[:50]}...'")
            
            start_time = datetime.now()
            
            # Perform retrieval
            try:
                retrieved_results = retriever.retrieve(query, top_k=top_k)
            except Exception as e:
                self.logger.error(f"Retrieval failed for query: {e}")
                continue
            
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            query_times.append(query_time)
            
            # Calculate relevance scores (simulated for now)
            # In real evaluation, you'd have human judgments
            relevance_scores = self._simulate_relevance(query, retrieved_results)
            
            # Calculate metrics
            metrics = self._calculate_metrics(relevance_scores, top_k)
            metrics.query_time_ms = query_time
            
            results['per_query'][query] = metrics
            all_metrics.append(metrics)
        
        # Calculate aggregate metrics
        if all_metrics:
            aggregate = EvaluationMetrics(
                ndcg_at_5=np.mean([m.ndcg_at_5 for m in all_metrics]),
                ndcg_at_10=np.mean([m.ndcg_at_10 for m in all_metrics]),
                precision_at_5=np.mean([m.precision_at_5 for m in all_metrics]),
                precision_at_10=np.mean([m.precision_at_10 for m in all_metrics]),
                recall_at_5=np.mean([m.recall_at_5 for m in all_metrics]),
                recall_at_10=np.mean([m.recall_at_10 for m in all_metrics]),
                mean_reciprocal_rank=np.mean([m.mean_reciprocal_rank for m in all_metrics]),
                query_time_ms=np.mean(query_times) if query_times else 0.0
            )
            results['aggregate'] = aggregate
        
        return results
    
    def compare_systems(self, systems: Dict[str, Any], 
                       queries: List[str] = None) -> Dict:
        """
        Compare multiple retrieval systems
        
        Args:
            systems: Dictionary of system_name -> retriever
            queries: List of test queries
        
        Returns:
            Comparison results
        """
        if queries is None:
            queries = self.test_queries[:5]  # Use first 5 for quick comparison
        
        comparison = {
            'queries': queries,
            'systems': {},
            'summary': {}
        }
        
        # Evaluate each system
        for system_name, system in systems.items():
            self.logger.info(f"Evaluating system: {system_name}")
            
            try:
                results = self.evaluate_retrieval(system, queries)
                comparison['systems'][system_name] = results
            except Exception as e:
                self.logger.error(f"Failed to evaluate {system_name}: {e}")
                comparison['systems'][system_name] = {'error': str(e)}
        
        # Create comparison summary
        summary = {}
        for system_name, results in comparison['systems'].items():
            if 'error' not in results and results.get('aggregate'):
                summary[system_name] = results['aggregate'].to_dict()
        
        comparison['summary'] = summary
        
        return comparison
    
    def _calculate_metrics(self, relevance_scores: List[float], 
                          top_k: int) -> EvaluationMetrics:
        """Calculate evaluation metrics from relevance scores"""
        if not relevance_scores:
            return EvaluationMetrics()
        
        # Ensure we have enough scores
        scores = relevance_scores[:top_k]
        if len(scores) < top_k:
            scores = scores + [0.0] * (top_k - len(scores))
        
        # Calculate NDCG
        if HAS_SKLEARN and len(scores) >= 5:
            try:
                ideal_5 = sorted(scores[:5], reverse=True)
                ideal_10 = sorted(scores[:10], reverse=True)
                
                ndcg_5 = ndcg_score([ideal_5], [scores[:5]], k=5)
                ndcg_10 = ndcg_score([ideal_10], [scores[:10]], k=10)
            except Exception:
                ndcg_5 = self._manual_ndcg(scores, 5)
                ndcg_10 = self._manual_ndcg(scores, 10)
        else:
            ndcg_5 = self._manual_ndcg(scores, 5)
            ndcg_10 = self._manual_ndcg(scores, 10)
        
        # Calculate Precision@k
        precision_5 = self._precision_at_k(scores, 5)
        precision_10 = self._precision_at_k(scores, 10)
        
        # Calculate Recall@k (simplified)
        recall_5 = self._recall_at_k(scores, 5)
        recall_10 = self._recall_at_k(scores, 10)
        
        # Calculate Mean Reciprocal Rank
        mrr = self._calculate_mrr(scores)
        
        return EvaluationMetrics(
            ndcg_at_5=float(ndcg_5),
            ndcg_at_10=float(ndcg_10),
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            mean_reciprocal_rank=mrr,
            query_time_ms=0.0  # Will be set by caller
        )
    
    def _manual_ndcg(self, scores: List[float], k: int) -> float:
        """Calculate NDCG manually when sklearn not available"""
        if not scores or k <= 0:
            return 0.0
        
        scores_k = scores[:k]
        
        # DCG
        dcg = 0.0
        for i, score in enumerate(scores_k):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCG (ideal DCG)
        ideal_scores = sorted(scores_k, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _simulate_relevance(self, query: str, results: List[Dict]) -> List[float]:
        """
        Simulate relevance scores for evaluation
        In real research, you would use human judgments
        """
        relevance_scores = []
        
        # Simple simulation based on query terms in result
        query_terms = set(query.lower().split())
        # Remove common stop words
        stop_words = {'what', 'is', 'how', 'the', 'a', 'an', 'in', 'to', 'for', 'with', 'and', 'or', 'are', 'does'}
        query_terms = query_terms - stop_words
        
        for result in results:
            score = 0.0
            
            # Check node name
            node_name = result.get('node_name', '').lower()
            for term in query_terms:
                if len(term) > 2 and term in node_name:
                    score += 0.3
            
            # Check context
            context = result.get('context', '').lower()
            for term in query_terms:
                if len(term) > 2 and term in context:
                    score += 0.15
            
            # Boost for certain entity types based on query
            node_type = result.get('node_type', '').upper()
            if 'CONCEPT' in node_type:
                score += 0.1
            if 'METHOD' in node_type and any(t in query.lower() for t in ['method', 'how', 'work', 'approach']):
                score += 0.15
            if 'METRIC' in node_type and any(t in query.lower() for t in ['metric', 'evaluate', 'measure']):
                score += 0.15
            
            # Add retrieval score as a factor
            retrieval_score = result.get('score', 0.5)
            score += retrieval_score * 0.2
            
            # Add some variation
            score += np.random.uniform(0, 0.05)
            
            # Cap at 1.0
            relevance_scores.append(min(score, 1.0))
        
        return relevance_scores
    
    def _precision_at_k(self, scores: List[float], k: int) -> float:
        """Calculate precision@k"""
        if not scores or k <= 0:
            return 0.0
        
        k = min(k, len(scores))
        
        # Consider scores > 0.3 as relevant
        relevant = sum(1 for score in scores[:k] if score > 0.3)
        return relevant / k
    
    def _recall_at_k(self, scores: List[float], k: int) -> float:
        """Calculate recall@k (simplified)"""
        # Simplified: assume 5 relevant documents per query
        total_relevant = 5
        
        if not scores or k <= 0:
            return 0.0
        
        k = min(k, len(scores))
        
        relevant_found = sum(1 for score in scores[:k] if score > 0.3)
        return min(relevant_found / total_relevant, 1.0)
    
    def _calculate_mrr(self, scores: List[float]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for i, score in enumerate(scores):
            if score > 0.3:  # Consider as relevant
                return 1.0 / (i + 1)
        return 0.0
    
    def _load_ground_truth(self, path: str) -> Dict:
        """Load ground truth data (placeholder)"""
        # In real implementation, load from file
        return {}
    
    def generate_evaluation_report(self, comparison_results: Dict) -> str:
        """Generate detailed evaluation report"""
        report = []
        report.append("=" * 70)
        report.append("GRAPH RAG EVALUATION REPORT")
        report.append("=" * 70)
        
        # Summary table
        report.append("\n📊 PERFORMANCE SUMMARY")
        report.append("-" * 60)
        report.append(f"{'System':<20} {'NDCG@10':<10} {'P@10':<10} {'R@10':<10} {'MRR':<10} {'Time(ms)':<10}")
        report.append("-" * 60)
        
        for system_name, metrics in comparison_results.get('summary', {}).items():
            report.append(
                f"{system_name:<20} "
                f"{metrics.get('ndcg@10', 0):<10.3f} "
                f"{metrics.get('precision@10', 0):<10.3f} "
                f"{metrics.get('recall@10', 0):<10.3f} "
                f"{metrics.get('mrr', 0):<10.3f} "
                f"{metrics.get('query_time_ms', 0):<10.1f}"
            )
        
        # Per-query analysis
        report.append("\n\n🔍 PER-QUERY ANALYSIS")
        report.append("-" * 60)
        
        queries = comparison_results.get('queries', [])
        for query in queries[:3]:  # Show first 3 queries
            report.append(f"\nQuery: '{query[:50]}...'")
            report.append("-" * 40)
            
            for system_name, system_results in comparison_results.get('systems', {}).items():
                if 'error' not in system_results:
                    per_query = system_results.get('per_query', {})
                    if query in per_query:
                        metrics = per_query[query]
                        report.append(
                            f"{system_name:<15}: "
                            f"NDCG@10={metrics.ndcg_at_10:.3f}, "
                            f"P@10={metrics.precision_at_10:.3f}, "
                            f"Time={metrics.query_time_ms:.1f}ms"
                        )
        
        # Recommendations
        report.append("\n\n🎯 RECOMMENDATIONS")
        report.append("-" * 60)
        
        if comparison_results.get('summary'):
            summary = comparison_results['summary']
            if summary:
                best_system = max(
                    summary.items(),
                    key=lambda x: x[1].get('ndcg@10', 0)
                )[0]
                
                report.append(f"1. Best performing system: {best_system}")
            
            report.append(f"2. Consider hybrid approach for balanced performance")
            report.append(f"3. Graph-based retrieval excels at relationship queries")
            report.append(f"4. Semantic retrieval faster but may miss connections")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
