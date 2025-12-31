# week4/main_lightweight.py
"""
Week 4 Lightweight Pipeline: Rule-Based Summarization

Use this if you:
- Have limited RAM (< 4GB)
- Don't want to download ML models
- Want faster processing
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
    print("🌲 RAPTOR Week 4: Lightweight Summarization")
    print("=" * 60)
    print("Using rule-based extraction (no ML model download)")
    
    # Paths
    results_path = Path("outputs/week3_results.pkl")
    chunks_path = Path("outputs/week1_chunks.pkl")
    output_dir = Path("outputs/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== 1. Initialize Lightweight Summarizer =====
    print("\n" + "=" * 40)
    print("STEP 1: Initialize Lightweight Summarizer")
    print("=" * 40)
    
    summarizer = SummarizationEngine(
        use_lightweight=True  # Skip model loading
    )
    
    # Test
    summarizer.test_summarization()
    
    # ===== 2. Initialize Enhancer =====
    print("\n" + "=" * 40)
    print("STEP 2: Initialize Summary Enhancer")
    print("=" * 40)
    
    enhancer = SummaryEnhancer()
    
    # ===== 3. Load Data =====
    print("\n" + "=" * 40)
    print("STEP 3: Load Week 3 Data")
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
    
    # ===== 4. Enrich Tree =====
    print("\n" + "=" * 40)
    print("STEP 4: Enrich Tree with Summaries")
    print("=" * 40)
    
    enriched_tree = enricher.enrich_tree(tree_data, chunks)
    
    # ===== 5. Save =====
    print("\n" + "=" * 40)
    print("STEP 5: Save Results")
    print("=" * 40)
    
    enricher.save_enriched_tree(enriched_tree, str(output_dir / "enriched_tree.pkl"))
    
    index = enricher.create_summary_index(enriched_tree)
    with open(output_dir / "summary_index.pkl", 'wb') as f:
        pickle.dump(index, f)
    
    # ===== 6. Display =====
    enricher.print_tree_summary(enriched_tree)
    
    # ===== 7. Quick Demo =====
    print("\n" + "=" * 40)
    print("Quick Navigation Demo")
    print("=" * 40)
    
    navigator = TreeNavigator(enriched_tree)
    
    print("\n🔝 Document Overview (Root Summary):")
    root = navigator.get_root_summary()
    print(f"   {root}")
    
    print("\n📊 Main Sections (Level 1):")
    for i, s in enumerate(navigator.get_level_summaries(1)[:5]):
        print(f"   {i+1}. {s[:80]}...")
    
    print("\n" + "=" * 60)
    print("✅ Week 4 Lightweight Complete!")
    print("=" * 60)
    print(f"\nOutputs: {output_dir}")


if __name__ == "__main__":
    main()
