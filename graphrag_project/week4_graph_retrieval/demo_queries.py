# week4_graph_retrieval/demo_queries.py
"""
Demonstration queries and examples for GraphRAG
Shows capabilities of the graph-based retrieval system
"""

from typing import List, Dict, Any
import logging

from .hybrid_retriever import RetrievalMode


class GraphRAGDemo:
    """
    Demonstration of GraphRAG capabilities with example queries
    """
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.logger = logging.getLogger(__name__)
        
        # Define demonstration queries
        self.demo_queries = [
            {
                "query": "What is RAPTOR and how does it compare to traditional RAG?",
                "description": "Complex query requiring multi-hop reasoning",
                "expected": ["RAPTOR", "traditional RAG", "comparison", "metrics"]
            },
            {
                "query": "What metrics are used to evaluate retrieval systems?",
                "description": "Entity-focused query looking for specific types",
                "expected": ["Recall", "Precision", "NDCG", "evaluation"]
            },
            {
                "query": "How does hierarchical clustering work in document retrieval?",
                "description": "Technical query about specific methodology",
                "expected": ["hierarchical clustering", "GMM", "clustering", "document structure"]
            },
            {
                "query": "What are the advantages of knowledge graphs for RAG?",
                "description": "Conceptual query about system benefits",
                "expected": ["knowledge graph", "GraphRAG", "relationships", "retrieval"]
            },
            {
                "query": "What datasets are used for evaluation?",
                "description": "Dataset entity query",
                "expected": ["Natural Questions", "dataset", "evaluation", "benchmark"]
            }
        ]
        
        self.logger.info(f"✅ GraphRAGDemo initialized with {len(self.demo_queries)} demo queries")
    
    def run_demo(self, query_index: int = None) -> Dict:
        """
        Run demonstration with specific or all queries
        
        Args:
            query_index: Index of specific query to run (None for all)
        
        Returns:
            Dictionary with demo results
        """
        if query_index is not None and 0 <= query_index < len(self.demo_queries):
            queries = [self.demo_queries[query_index]]
        else:
            queries = self.demo_queries
        
        results = {}
        
        for i, query_info in enumerate(queries):
            query = query_info["query"]
            description = query_info["description"]
            expected = query_info["expected"]
            
            print(f"\n🔍 DEMO QUERY {i+1}: {description}")
            print(f"   Query: '{query}'")
            print(f"   Expected concepts: {', '.join(expected)}")
            
            # Run retrieval with different strategies
            retrieval_results = self._run_comparison(query)
            results[query] = retrieval_results
            
            # Print summary
            for mode, analysis in retrieval_results.items():
                if 'error' not in analysis:
                    print(f"   {mode.upper():<10}: {analysis['count']} results, "
                          f"Top score: {analysis['top_score']:.3f}")
        
        return results
    
    def _run_comparison(self, query: str) -> Dict:
        """Compare different retrieval strategies for a query"""
        comparison = {}
        
        # Test different retrieval modes
        modes = [
            ("semantic", RetrievalMode.SEMANTIC_ONLY),
            ("graph", RetrievalMode.GRAPH_ONLY),
            ("hybrid", RetrievalMode.HYBRID)
        ]
        
        for mode_name, mode in modes:
            try:
                results = self.retriever.retrieve(query, top_k=5, mode=mode)
                
                # Analyze results
                analysis = self._analyze_results(results, mode_name)
                comparison[mode_name] = analysis
                
            except Exception as e:
                self.logger.error(f"   {mode_name.upper()} failed: {e}")
                comparison[mode_name] = {"error": str(e)}
        
        return comparison
    
    def _analyze_results(self, results: List[Dict], mode: str) -> Dict:
        """Analyze retrieval results"""
        if not results:
            return {
                "count": 0, 
                "top_score": 0.0, 
                "entity_types": {},
                "top_node": "",
                "top_type": "",
                "avg_score": 0.0,
                "results": []
            }
        
        # Count entity types
        entity_types = {}
        for result in results:
            node_type = result.get('node_type', 'unknown')
            entity_types[node_type] = entity_types.get(node_type, 0) + 1
        
        # Get top result
        top_result = results[0] if results else {}
        
        return {
            "count": len(results),
            "top_score": top_result.get('score', 0.0),
            "top_node": top_result.get('node_name', ''),
            "top_type": top_result.get('node_type', ''),
            "entity_types": entity_types,
            "avg_score": sum(r.get('score', 0) for r in results) / len(results) if results else 0,
            "results": results[:3]  # Top 3 results
        }
    
    def generate_demo_report(self, demo_results: Dict) -> str:
        """Generate demonstration report"""
        report = []
        report.append("=" * 70)
        report.append("GRAPH RAG DEMONSTRATION REPORT")
        report.append("=" * 70)
        
        for query, comparisons in demo_results.items():
            report.append(f"\n📝 QUERY: {query}")
            report.append("-" * 60)
            
            for mode, analysis in comparisons.items():
                if 'error' in analysis:
                    report.append(f"\n{mode.upper()}: ERROR - {analysis['error']}")
                else:
                    report.append(f"\n{mode.upper()}:")
                    report.append(f"  Results: {analysis['count']}")
                    report.append(f"  Top score: {analysis['top_score']:.3f}")
                    report.append(f"  Top result: {analysis['top_node']} [{analysis['top_type']}]")
                    
                    if analysis['entity_types']:
                        report.append(f"  Entity types: {analysis['entity_types']}")
                    
                    # Show top 3 results
                    if analysis.get('results'):
                        report.append(f"  Top 3 results:")
                        for i, result in enumerate(analysis['results'], 1):
                            report.append(f"    {i}. {result['node_name']} "
                                       f"[{result['node_type']}] - {result['score']:.3f}")
        
        # Summary
        report.append("\n" + "=" * 70)
        report.append("📊 DEMONSTRATION SUMMARY")
        report.append("-" * 60)
        
        # Calculate averages
        mode_stats = {}
        for query, comparisons in demo_results.items():
            for mode, analysis in comparisons.items():
                if 'error' not in analysis:
                    if mode not in mode_stats:
                        mode_stats[mode] = {'counts': [], 'scores': []}
                    
                    mode_stats[mode]['counts'].append(analysis['count'])
                    mode_stats[mode]['scores'].append(analysis['avg_score'])
        
        for mode, stats in mode_stats.items():
            if stats['counts']:
                avg_count = sum(stats['counts']) / len(stats['counts'])
                avg_score = sum(stats['scores']) / len(stats['scores'])
                
                report.append(f"{mode.upper():<10}: Avg results: {avg_count:.1f}, "
                            f"Avg score: {avg_score:.3f}")
        
        report.append("\n🎯 KEY OBSERVATIONS:")
        report.append("1. Hybrid retrieval combines strengths of both approaches")
        report.append("2. Graph traversal finds connected concepts semantic might miss")
        report.append("3. Semantic retrieval faster for simple fact queries")
        report.append("4. Graph retrieval better for relationship-heavy queries")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)
    
    def interactive_demo(self):
        """Run an interactive demonstration"""
        print("\n" + "=" * 70)
        print("INTERACTIVE GRAPHRAG DEMONSTRATION")
        print("=" * 70)
        print("\nAvailable demo queries:")
        
        for i, query_info in enumerate(self.demo_queries):
            print(f"  {i+1}. {query_info['description']}")
            print(f"     Query: '{query_info['query'][:60]}...'")
        
        print("\nEnter a query number (1-5), 'all' for all queries, or 'custom' for your own query:")
        
        try:
            user_input = input("> ").strip().lower()
            
            if user_input == 'all':
                return self.run_demo()
            elif user_input == 'custom':
                custom_query = input("Enter your query: ").strip()
                if custom_query:
                    results = self._run_comparison(custom_query)
                    return {custom_query: results}
            elif user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(self.demo_queries):
                    return self.run_demo(idx)
            
            print("Invalid input. Running all demo queries.")
            return self.run_demo()
            
        except EOFError:
            # Non-interactive mode, run all demos
            return self.run_demo()
