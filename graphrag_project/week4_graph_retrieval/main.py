# week4_graph_retrieval/main.py
"""
WEEK 4: Graph-Based Retrieval (The Core of GraphRAG)
Main execution script for graph retrieval system
"""

import sys
from pathlib import Path
import logging
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import GraphRAGConfig
from week3_graph_construction.neo4j_manager import Neo4jGraphManager
from week4_graph_retrieval.embedding_manager import GraphEmbeddingManager
from week4_graph_retrieval.graph_retriever import GraphRetriever
from week4_graph_retrieval.hybrid_retriever import HybridRetriever, RetrievalMode, HybridRetrievalConfig
from week4_graph_retrieval.evaluation import GraphRAGEvaluator
from week4_graph_retrieval.demo_queries import GraphRAGDemo


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_file = output_dir / 'graph_retrieval.log'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def main():
    """Main Week 4: Graph-Based Retrieval"""
    print("=" * 70)
    print("WEEK 4: GRAPH-BASED RETRIEVAL (GraphRAG Core)")
    print("=" * 70)
    
    # Load config
    config = GraphRAGConfig()
    config.ensure_directories()
    
    # Setup
    logger = setup_logging(config.OUTPUT_DIR)
    
    # Create retrieval output directory
    retrieval_dir = config.OUTPUT_DIR / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Initialize Neo4j connection
    print("\n1️⃣  Connecting to Neo4j...")
    print(f"   URI: {config.NEO4J_URI}")
    
    neo4j_manager = Neo4jGraphManager(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    if not neo4j_manager.driver:
        print("\n❌ Failed to connect to Neo4j. Please check:")
        print("   1. Is Neo4j Desktop running?")
        print("   2. Is the database started?")
        print("   3. Are credentials correct in config.py?")
        print("\n💡 Run Week 3 first to create the knowledge graph!")
        return
    
    print("   ✅ Connected to Neo4j!")
    
    # Check if graph has data
    stats = neo4j_manager.get_graph_statistics()
    entity_count = stats.get('entity_count', {}).get('count', 0) if stats else 0
    
    if entity_count == 0:
        print("\n⚠️  No entities found in graph!")
        print("   Please run Week 3 first to build the knowledge graph.")
        neo4j_manager.close()
        return
    
    print(f"   Found {entity_count} entities in graph")
    
    # Step 2: Initialize embedding manager
    print("\n2️⃣  Initializing embedding manager...")
    
    embedding_manager = GraphEmbeddingManager(
        model_name=config.LOCAL_EMBEDDING_MODEL
    )
    
    # Step 3: Initialize graph retriever
    print("\n3️⃣  Initializing graph retriever...")
    
    graph_retriever = GraphRetriever(neo4j_manager, embedding_manager)
    
    # Step 4: Initialize hybrid retriever
    print("\n4️⃣  Initializing hybrid retriever...")
    
    hybrid_config = HybridRetrievalConfig(
        semantic_weight=0.6,
        graph_weight=0.4,
        fusion_method="weighted_sum",
        use_reranking=False  # Set to True if you have sentence-transformers
    )
    
    hybrid_retriever = HybridRetriever(graph_retriever, hybrid_config)
    
    # Step 5: Run demonstration
    print("\n5️⃣  Running demonstration queries...")
    
    demo = GraphRAGDemo(hybrid_retriever)
    demo_results = demo.run_demo()
    
    # Save demo results
    demo_output = retrieval_dir / "demo_results.json"
    
    try:
        # Convert to serializable format
        serializable_results = {}
        for query, comparisons in demo_results.items():
            serializable_results[query] = {}
            for mode, analysis in comparisons.items():
                if 'error' in analysis:
                    serializable_results[query][mode] = analysis
                else:
                    serializable_results[query][mode] = {
                        'count': analysis.get('count', 0),
                        'top_score': float(analysis.get('top_score', 0)),
                        'top_node': analysis.get('top_node', ''),
                        'top_type': analysis.get('top_type', ''),
                        'avg_score': float(analysis.get('avg_score', 0)),
                        'entity_types': analysis.get('entity_types', {}),
                        'results': analysis.get('results', [])
                    }
        
        with open(demo_output, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"   ✅ Demo results saved: {demo_output}")
    except Exception as e:
        logger.error(f"Failed to save demo results: {e}")
    
    # Step 6: Generate demo report
    demo_report = demo.generate_demo_report(demo_results)
    
    demo_report_path = retrieval_dir / "demo_report.txt"
    with open(demo_report_path, 'w', encoding='utf-8') as f:
        f.write(demo_report)
    
    print(f"   ✅ Demo report saved: {demo_report_path}")
    
    # Step 7: Run evaluation
    print("\n6️⃣  Running evaluation...")
    
    evaluator = GraphRAGEvaluator()
    
    # Define test queries
    test_queries = [
        "What is RAPTOR and how does it work?",
        "Compare RAPTOR with traditional RAG systems",
        "What metrics are used to evaluate retrieval systems?",
        "Explain hierarchical clustering in document retrieval",
        "What are the advantages of knowledge graphs for RAG?"
    ]
    
    # Evaluate hybrid retriever
    evaluation_results = evaluator.evaluate_retrieval(
        hybrid_retriever, test_queries, top_k=10
    )
    
    # Save evaluation results
    eval_output = retrieval_dir / "evaluation_results.json"
    try:
        serializable_eval = {
            'queries': test_queries,
            'per_query': {},
            'aggregate': None
        }
        
        for query, metrics in evaluation_results.get('per_query', {}).items():
            serializable_eval['per_query'][query] = metrics.to_dict()
        
        if evaluation_results.get('aggregate'):
            serializable_eval['aggregate'] = evaluation_results['aggregate'].to_dict()
        
        with open(eval_output, 'w', encoding='utf-8') as f:
            json.dump(serializable_eval, f, indent=2)
        
        print(f"   ✅ Evaluation results saved: {eval_output}")
    except Exception as e:
        logger.error(f"Failed to save evaluation results: {e}")
    
    # Step 8: Generate comparison report
    print("\n7️⃣  Generating comparison report...")
    
    # Compare different modes
    comparison_results = {
        'queries': test_queries[:3],
        'systems': {},
        'summary': {}
    }
    
    # Evaluate each mode
    for mode_name, mode in [("Hybrid", RetrievalMode.HYBRID), 
                            ("Semantic", RetrievalMode.SEMANTIC_ONLY),
                            ("Graph", RetrievalMode.GRAPH_ONLY)]:
        try:
            # Create a wrapper that uses specific mode
            class ModeRetriever:
                def __init__(self, retriever, mode):
                    self.retriever = retriever
                    self.mode = mode
                
                def retrieve(self, query, top_k=10):
                    return self.retriever.retrieve(query, top_k, self.mode)
            
            mode_retriever = ModeRetriever(hybrid_retriever, mode)
            results = evaluator.evaluate_retrieval(mode_retriever, test_queries[:3], top_k=10)
            comparison_results['systems'][mode_name] = results
            
            if results.get('aggregate'):
                comparison_results['summary'][mode_name] = results['aggregate'].to_dict()
                
        except Exception as e:
            logger.error(f"Failed to evaluate {mode_name}: {e}")
    
    # Generate comparison report
    comparison_report = evaluator.generate_evaluation_report(comparison_results)
    
    comparison_path = retrieval_dir / "comparison_report.txt"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(comparison_report)
    
    print(f"   ✅ Comparison report saved: {comparison_path}")
    
    # Step 9: Close connection
    neo4j_manager.close()
    
    # Step 10: Summary
    print("\n" + "=" * 70)
    print("✅ WEEK 4 COMPLETE!")
    
    print(f"\n📁 OUTPUTS CREATED:")
    print(f"   Demo results: {demo_output}")
    print(f"   Demo report: {demo_report_path}")
    print(f"   Evaluation results: {eval_output}")
    print(f"   Comparison report: {comparison_path}")
    
    print(f"\n📊 KEY METRICS COLLECTED:")
    
    if evaluation_results.get('aggregate'):
        metrics = evaluation_results['aggregate']
        print(f"   NDCG@10: {metrics.ndcg_at_10:.3f}")
        print(f"   Precision@10: {metrics.precision_at_10:.3f}")
        print(f"   Recall@10: {metrics.recall_at_10:.3f}")
        print(f"   MRR: {metrics.mean_reciprocal_rank:.3f}")
        print(f"   Avg query time: {metrics.query_time_ms:.1f} ms")
    
    print(f"\n🔍 RETRIEVAL STRATEGIES IMPLEMENTED:")
    print(f"   ✓ Semantic similarity search")
    print(f"   ✓ Graph traversal retrieval")
    print(f"   ✓ Hybrid fusion approach")
    print(f"   ✓ Multiple fusion methods (weighted_sum, reciprocal_rank, comb_mnz)")
    
    print(f"\n📈 FOR PAPER COMPARISON:")
    print(f"   Compare GraphRAG metrics with RAPTOR's metrics from your Raptor project")
    print(f"   Key comparison dimensions:")
    print(f"   - NDCG@10 for ranking quality")
    print(f"   - MRR for finding relevant results quickly")
    print(f"   - Query time for efficiency")
    
    print(f"\n🚀 NEXT: Week 5 - Final Comparison & Paper Preparation")
    print("=" * 70)
    
    # Print demo report highlights
    print("\n" + "=" * 70)
    print("DEMONSTRATION HIGHLIGHTS")
    print("=" * 70)
    
    # Print first few lines of demo report
    lines = demo_report.split('\n')[:35]
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
