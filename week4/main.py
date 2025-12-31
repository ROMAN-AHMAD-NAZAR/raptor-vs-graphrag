# week4/main.py
"""
Week 4 Main Pipeline: Intelligent Summarization for RAPTOR

This adds abstractive summaries to the RAPTOR tree from Week 3
"""

import sys
import pickle
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from week4.summarization_engine import SummarizationEngine
from week4.summary_enhancer import SummaryEnhancer
from week4.tree_enricher import TreeEnricher, TreeNavigator


def main():
    print("=" * 60)
    print("🌲 RAPTOR Week 4: Intelligent Summarization")
    print("=" * 60)
    
    # Paths
    results_path = Path("outputs/week3_results.pkl")
    chunks_path = Path("outputs/week1_chunks.pkl")
    output_dir = Path("outputs/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. Initialize Summarization Engine =====
    print("\n" + "=" * 40)
    print("STEP 1: Initialize Summarization Engine")
    print("=" * 40)
    
    # Choose model based on available resources
    # Options:
    # - TinyLlama (2.2GB RAM) - Recommended for most systems
    # - Rule-based (no download) - Use use_lightweight=True
    
    summarizer = SummarizationEngine(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device="cpu",
        use_lightweight=False  # Set True for lightweight mode
    )
    
    # Test summarization
    summarizer.test_summarization()
    
    # ===== 2. Initialize Summary Enhancer =====
    print("\n" + "=" * 40)
    print("STEP 2: Initialize Summary Enhancer")
    print("=" * 40)
    
    enhancer = SummaryEnhancer()
    
    # ===== 3. Load Data =====
    print("\n" + "=" * 40)
    print("STEP 3: Load Week 3 Tree and Chunks")
    print("=" * 40)
    
    enricher = TreeEnricher(summarizer, enhancer)
    
    # Load week 3 results which contains tree and chunks
    with open(results_path, 'rb') as f:
        week3_results = pickle.load(f)
    
    # Convert tree_nodes dict to the expected format
    tree_nodes = week3_results.get('tree_nodes', {})
    nodes_list = list(tree_nodes.values())
    
    tree_data = {
        'nodes': nodes_list,
        'total_nodes': len(nodes_list)
    }
    print(f"✅ Loaded tree with {len(nodes_list)} nodes")
    
    # Get chunks from results
    chunks = week3_results.get('chunks', [])
    print(f"✅ Loaded {len(chunks)} chunks")
    
    # ===== 4. Enrich Tree with Summaries =====
    print("\n" + "=" * 40)
    print("STEP 4: Enrich Tree with Summaries")
    print("=" * 40)
    
    enriched_tree = enricher.enrich_tree(tree_data, chunks)
    
    # ===== 5. Save Results =====
    print("\n" + "=" * 40)
    print("STEP 5: Save Enriched Tree")
    print("=" * 40)
    
    enricher.save_enriched_tree(enriched_tree, str(output_dir / "enriched_tree.pkl"))
    
    # Create summary index
    index = enricher.create_summary_index(enriched_tree)
    with open(output_dir / "summary_index.pkl", 'wb') as f:
        pickle.dump(index, f)
    print(f"💾 Saved summary index")
    
    # ===== 6. Display Results =====
    print("\n" + "=" * 40)
    print("STEP 6: Results Summary")
    print("=" * 40)
    
    enricher.print_tree_summary(enriched_tree)
    
    # ===== 7. Demo Navigation =====
    print("\n" + "=" * 40)
    print("STEP 7: Tree Navigation Demo")
    print("=" * 40)
    
    navigator = TreeNavigator(enriched_tree)
    
    print("\n🔝 Root Summary:")
    print(f"   {navigator.get_root_summary()}")
    
    print("\n📊 Level 1 Summaries:")
    for i, summary in enumerate(navigator.get_level_summaries(1)[:3]):
        preview = summary[:100] + "..." if len(summary) > 100 else summary
        print(f"   [{i+1}] {preview}")
    
    # Search demo
    print("\n🔍 Search Demo (query: 'RAPTOR'):")
    results = navigator.search_summaries("RAPTOR")
    for r in results[:3]:
        print(f"   Node {r['node_id']} (depth {r['depth']}): {r['summary'][:60]}...")
    
    print("\n" + "=" * 60)
    print("✅ Week 4 Complete!")
    print("=" * 60)
    print(f"""
    Outputs saved to: {output_dir}
    
    Files created:
    - enriched_tree.pkl - Tree with summaries
    - enriched_tree.json - JSON version for inspection
    - summary_index.pkl - Searchable summary index
    
    Next: Week 5 - Multi-level Retrieval System
    """)


if __name__ == "__main__":
    main()
