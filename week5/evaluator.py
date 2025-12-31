# week5/evaluator.py
"""
Performance Evaluator for RAPTOR vs RAG Comparison

Measures and compares retrieval quality metrics
"""

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd


class PerformanceEvaluator:
    """
    Compare RAPTOR vs Normal RAG performance
    
    Metrics:
    - NDCG (Normalized Discounted Cumulative Gain)
    - Precision@K
    - Recall simulation
    - Context coverage
    """
    
    def __init__(self):
        print(f"📈 Initializing performance evaluator...")
    
    def create_test_queries(self, chunks: List[str]) -> List[Dict]:
        """
        Create test queries based on document content
        
        In real scenario, you'd have actual queries
        """
        print(f"\n🧪 Creating test queries...")
        
        # Extract potential queries from chunks
        test_queries = []
        
        # Use first sentence of some chunks as queries
        for i, chunk in enumerate(chunks[:10]):  # Use up to 10 chunks
            sentences = chunk.split('. ')
            if sentences:
                query = sentences[0]
                # Filter for reasonable query length
                words = query.split()
                if 5 <= len(words) <= 30:
                    test_queries.append({
                        'query_id': i,
                        'text': query,
                        'source_chunk': i,
                        'expected_topics': self._extract_topics(chunk)
                    })
        
        print(f"   Created {len(test_queries)} test queries")
        for q in test_queries[:3]:
            print(f"   - '{q['text'][:50]}...'")
        
        return test_queries
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract key topics from text"""
        # Simple keyword extraction
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'have', 'from',
                    'are', 'was', 'were', 'has', 'had', 'but', 'not', 'you', 'your'}
        
        words = text.lower().split()
        words = [w.strip('.,!?()[]') for w in words]
        words = [w for w in words if len(w) > 3 and w not in stopwords]
        
        # Get most common words
        from collections import Counter
        counter = Counter(words)
        return [word for word, count in counter.most_common(5)]
    
    def simulate_relevance_scores(self, results: List[Dict], query: Dict) -> List[float]:
        """
        Simulate relevance scores for evaluation
        
        In real use, you'd have human judgments
        Here we simulate based on text overlap
        """
        relevance_scores = []
        query_text = query['text'].lower()
        query_terms = set(query_text.split())
        
        for result in results:
            result_text = result.get('text', '').lower()
            
            # Simple relevance: term overlap
            result_terms = set(result_text.split())
            overlap = len(query_terms & result_terms) / max(1, len(query_terms))
            
            # Boost for summaries (they provide context)
            if result.get('is_summary', False):
                overlap = min(1.0, overlap * 1.3)
            
            # Boost if from same source chunk
            if result.get('id') == query.get('source_chunk'):
                overlap = min(1.0, overlap * 1.5)
            
            relevance_scores.append(overlap)
        
        return relevance_scores
    
    def calculate_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """
        Calculate NDCG (Normalized Discounted Cumulative Gain)
        
        Measures ranking quality - higher is better (max 1.0)
        """
        if not relevance_scores:
            return 0.0
        
        # Limit to top-k
        scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = scores[0] if scores else 0
        for i, score in enumerate(scores[1:], 2):
            dcg += score / np.log2(i + 1)
        
        # Calculate ideal DCG (perfect ranking)
        ideal_scores = sorted(scores, reverse=True)
        idcg = ideal_scores[0] if ideal_scores else 0
        for i, score in enumerate(ideal_scores[1:], 2):
            idcg += score / np.log2(i + 1)
        
        # Avoid division by zero
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_precision_at_k(self, relevance_scores: List[float], 
                                 k: int = 5, threshold: float = 0.3) -> float:
        """
        Calculate Precision@K
        
        Percentage of top-k results that are relevant
        """
        if not relevance_scores:
            return 0.0
        
        top_k = relevance_scores[:k]
        relevant = sum(1 for score in top_k if score >= threshold)
        
        return relevant / k
    
    def calculate_context_coverage(self, results: List[Dict]) -> float:
        """
        Calculate context coverage score
        
        Measures how much context (summaries) is included
        This is RAPTOR's advantage!
        """
        if not results:
            return 0.0
        
        # Count summaries at different depths
        summary_count = sum(1 for r in results if r.get('is_summary', False))
        
        # Calculate coverage
        coverage = summary_count / len(results)
        
        # Bonus for multiple depth levels
        depths = set(r.get('depth', 0) for r in results)
        depth_bonus = min(0.2, len(depths) * 0.05)
        
        return min(1.0, coverage + depth_bonus)
    
    def evaluate_systems(self, raptor_results: List[List[Dict]], 
                        rag_results: List[List[Dict]],
                        queries: List[Dict]) -> Dict:
        """
        Compare RAPTOR vs Normal RAG
        """
        print(f"\n⚖️  Evaluating RAPTOR vs Normal RAG...")
        
        metrics = {
            'raptor': {
                'ndcg_scores': [], 
                'precision_scores': [], 
                'context_scores': []
            },
            'rag': {
                'ndcg_scores': [], 
                'precision_scores': [],
                'context_scores': []
            }
        }
        
        for i, query in enumerate(queries):
            if i >= len(raptor_results) or i >= len(rag_results):
                continue
            
            # Get relevance scores
            raptor_relevance = self.simulate_relevance_scores(raptor_results[i], query)
            rag_relevance = self.simulate_relevance_scores(rag_results[i], query)
            
            # Calculate NDCG@5 and NDCG@10
            raptor_ndcg5 = self.calculate_ndcg(raptor_relevance, 5)
            raptor_ndcg10 = self.calculate_ndcg(raptor_relevance, 10)
            
            rag_ndcg5 = self.calculate_ndcg(rag_relevance, 5)
            rag_ndcg10 = self.calculate_ndcg(rag_relevance, 10)
            
            # Calculate precision@5
            raptor_p5 = self.calculate_precision_at_k(raptor_relevance, 5)
            rag_p5 = self.calculate_precision_at_k(rag_relevance, 5)
            
            # Calculate context coverage
            raptor_context = self.calculate_context_coverage(raptor_results[i])
            rag_context = self.calculate_context_coverage(rag_results[i])
            
            # Store metrics
            metrics['raptor']['ndcg_scores'].append(raptor_ndcg10)
            metrics['raptor']['precision_scores'].append(raptor_p5)
            metrics['raptor']['context_scores'].append(raptor_context)
            
            metrics['rag']['ndcg_scores'].append(rag_ndcg10)
            metrics['rag']['precision_scores'].append(rag_p5)
            metrics['rag']['context_scores'].append(rag_context)
            
            print(f"\n   Query {i+1}: '{query['text'][:40]}...'")
            print(f"     RAPTOR - NDCG@5: {raptor_ndcg5:.3f}, NDCG@10: {raptor_ndcg10:.3f}, P@5: {raptor_p5:.3f}")
            print(f"     Normal RAG - NDCG@5: {rag_ndcg5:.3f}, NDCG@10: {rag_ndcg10:.3f}, P@5: {rag_p5:.3f}")
            
            # Calculate improvement
            if rag_ndcg10 > 0:
                improvement = ((raptor_ndcg10 - rag_ndcg10) / rag_ndcg10) * 100
                print(f"     Improvement: {improvement:+.1f}%")
        
        # Calculate averages
        for system in ['raptor', 'rag']:
            if metrics[system]['ndcg_scores']:
                metrics[system]['avg_ndcg'] = np.mean(metrics[system]['ndcg_scores'])
                metrics[system]['std_ndcg'] = np.std(metrics[system]['ndcg_scores'])
                metrics[system]['avg_precision'] = np.mean(metrics[system]['precision_scores'])
                metrics[system]['avg_context'] = np.mean(metrics[system]['context_scores'])
        
        # Overall comparison
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   RAPTOR Average NDCG@10: {metrics['raptor'].get('avg_ndcg', 0):.3f}")
        print(f"   Normal RAG Average NDCG@10: {metrics['rag'].get('avg_ndcg', 0):.3f}")
        
        if metrics['rag'].get('avg_ndcg', 0) > 0:
            overall_improvement = ((metrics['raptor'].get('avg_ndcg', 0) - 
                                   metrics['rag'].get('avg_ndcg', 0)) / 
                                   metrics['rag'].get('avg_ndcg', 0)) * 100
            print(f"   Overall Improvement: {overall_improvement:+.1f}%")
            metrics['overall_improvement'] = overall_improvement
        
        print(f"\n   Context Coverage:")
        print(f"     RAPTOR: {metrics['raptor'].get('avg_context', 0):.3f}")
        print(f"     Normal RAG: {metrics['rag'].get('avg_context', 0):.3f}")
        
        return metrics
    
    def create_comparison_report(self, metrics: Dict, output_path: str = None) -> Tuple:
        """Create a detailed comparison report"""
        print(f"\n📋 Creating comparison report...")
        
        # Create DataFrame for analysis
        data = {
            'System': ['RAPTOR', 'Normal RAG'],
            'Avg_NDCG@10': [
                metrics['raptor'].get('avg_ndcg', 0),
                metrics['rag'].get('avg_ndcg', 0)
            ],
            'Std_NDCG': [
                metrics['raptor'].get('std_ndcg', 0),
                metrics['rag'].get('std_ndcg', 0)
            ],
            'Avg_Precision@5': [
                metrics['raptor'].get('avg_precision', 0),
                metrics['rag'].get('avg_precision', 0)
            ],
            'Context_Coverage': [
                metrics['raptor'].get('avg_context', 0),
                metrics['rag'].get('avg_context', 0)
            ]
        }
        
        df = pd.DataFrame(data)
        print(f"\n{df.to_string(index=False)}")
        
        fig = None
        
        # Try to create visualization
        try:
            import plotly.graph_objects as go
            
            fig = go.Figure(data=[
                go.Bar(
                    name='RAPTOR',
                    x=['NDCG@10', 'Precision@5', 'Context'],
                    y=[
                        metrics['raptor'].get('avg_ndcg', 0),
                        metrics['raptor'].get('avg_precision', 0),
                        metrics['raptor'].get('avg_context', 0)
                    ],
                    marker_color='rgb(55, 83, 109)'
                ),
                go.Bar(
                    name='Normal RAG',
                    x=['NDCG@10', 'Precision@5', 'Context'],
                    y=[
                        metrics['rag'].get('avg_ndcg', 0),
                        metrics['rag'].get('avg_precision', 0),
                        metrics['rag'].get('avg_context', 0)
                    ],
                    marker_color='rgb(26, 118, 255)'
                )
            ])
            
            fig.update_layout(
                title='RAPTOR vs Normal RAG Performance Comparison',
                yaxis_title='Score (higher is better)',
                barmode='group',
                yaxis=dict(range=[0, 1.1])
            )
            
            if output_path:
                # Save as HTML
                html_path = output_path.replace('.png', '.html')
                fig.write_html(html_path)
                print(f"✅ Report saved: {html_path}")
                
                # Try to save image
                try:
                    fig.write_image(output_path)
                    print(f"✅ Image saved: {output_path}")
                except Exception as e:
                    print(f"⚠️  Could not save image: {e}")
                    
        except ImportError:
            print("⚠️  Plotly not available for visualization")
        except Exception as e:
            print(f"⚠️  Could not create visualization: {e}")
        
        return df, fig
    
    def print_summary(self, metrics: Dict):
        """Print formatted summary"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        print(f"\n🌳 RAPTOR (Hierarchical Retrieval):")
        print(f"   NDCG@10: {metrics['raptor'].get('avg_ndcg', 0):.3f} ± {metrics['raptor'].get('std_ndcg', 0):.3f}")
        print(f"   Precision@5: {metrics['raptor'].get('avg_precision', 0):.3f}")
        print(f"   Context Coverage: {metrics['raptor'].get('avg_context', 0):.3f}")
        
        print(f"\n📊 Normal RAG (Flat Retrieval):")
        print(f"   NDCG@10: {metrics['rag'].get('avg_ndcg', 0):.3f} ± {metrics['rag'].get('std_ndcg', 0):.3f}")
        print(f"   Precision@5: {metrics['rag'].get('avg_precision', 0):.3f}")
        print(f"   Context Coverage: {metrics['rag'].get('avg_context', 0):.3f}")
        
        if 'overall_improvement' in metrics:
            print(f"\n🚀 Overall Improvement: {metrics['overall_improvement']:+.1f}%")
        
        print("=" * 60)
