# week3/main.py
import sys
import os
import pickle
import numpy as np

# Add directories to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
week2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "week2")
sys.path.insert(0, week2_path)

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 70)
    print("WEEK 3: Clustering & Building the RAPTOR Tree")
    print("=" * 70)
    
    # Step 1: Load Week 2 embeddings
    print("\n📂 Step 1: Loading Week 2 embeddings...")
    
    embeddings_dir = os.path.join(PROJECT_ROOT, "outputs", "embeddings")
    
    try:
        from embedding_storage import EmbeddingStorage
        
        storage = EmbeddingStorage(save_dir=embeddings_dir)
        embeddings, chunks, metadata = storage.load_embeddings("week2_embeddings")
        
        if embeddings is None:
            print("❌ Failed to load embeddings. Run Week 2 first!")
            return
        
        print(f"✅ Loaded {len(chunks)} chunks with {embeddings.shape[1]}-dim embeddings")
        
        # Show sample
        print(f"\nSample chunk (0): {chunks[0][:80]}...")
        print(f"Embedding shape: {embeddings[0].shape}")
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Dimensionality Reduction
    print("\n" + "=" * 50)
    print("Step 2: Dimensionality Reduction with UMAP")
    print("=" * 50)
    
    from dimensionality_reducer import DimensionalityReducer
    
    # Adjust components based on data size
    n_components = min(50, len(chunks) - 2, embeddings.shape[1] - 1)
    n_components = max(2, n_components)  # At least 2 for visualization
    
    reducer = DimensionalityReducer(n_components=n_components)
    reduced_embeddings = reducer.reduce_dimensions(embeddings)
    
    # Visualize reduction
    vis_dir = os.path.join(PROJECT_ROOT, "outputs", "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    reducer.visualize_reduction(
        reduced_embeddings, 
        chunks,
        save_path=os.path.join(vis_dir, "week3_umap_reduction.png")
    )
    
    # Step 3: Clustering with GMM
    print("\n" + "=" * 50)
    print("Step 3: Clustering with Gaussian Mixture Models")
    print("=" * 50)
    
    from hierarchical_clusterer import HierarchicalClusterer
    
    # Determine max clusters based on data size
    max_clusters = min(10, len(chunks) // 3)
    max_clusters = max(2, max_clusters)
    
    clusterer = HierarchicalClusterer(max_clusters=max_clusters, min_clusters=2)
    
    # Find optimal clusters and cluster
    labels, probabilities, model, metrics = clusterer.cluster(reduced_embeddings)
    
    # Analyze what's in each cluster
    clusters = clusterer.analyze_clusters(labels, chunks, top_words=5)
    
    # Step 4: Visualize Clusters
    print("\n" + "=" * 50)
    print("Step 4: Visualizing Clusters")
    print("=" * 50)
    
    from visualizer import TreeVisualizer
    
    visualizer = TreeVisualizer()
    
    # 2D cluster plot
    visualizer.plot_clusters_2d(
        reduced_embeddings[:, :2],  # First 2 dimensions for visualization
        labels,
        chunks,
        save_path=os.path.join(vis_dir, "week3_clusters.png")
    )
    
    # Cluster size distribution
    visualizer.plot_cluster_sizes(
        labels,
        save_path=os.path.join(vis_dir, "week3_cluster_sizes.png")
    )
    
    # Interactive HTML plot
    html_path = os.path.join(PROJECT_ROOT, "outputs", "week3_interactive_plot.html")
    visualizer.create_html_visualization(
        reduced_embeddings[:, :2],
        labels,
        chunks,
        save_path=html_path
    )
    
    # Step 5: Build the RAPTOR Tree
    print("\n" + "=" * 50)
    print("Step 5: Building the RAPTOR Tree Structure")
    print("=" * 50)
    
    from tree_builder import RaptorTreeBuilder
    
    tree_builder = RaptorTreeBuilder()
    
    # Build tree from clusters
    cluster_nodes = tree_builder.build_tree(chunks, embeddings, labels, depth=1)
    
    # Set root and connect everything
    tree_builder.set_root(cluster_nodes)
    
    print("\n📜 Tree Structure:")
    print("-" * 40)
    tree_builder.print_tree()
    
    # Visualize tree
    visualizer.plot_tree_structure(
        tree_builder,
        save_path=os.path.join(vis_dir, "week3_tree_structure.png")
    )
    
    # Step 6: Save Tree and Results
    print("\n" + "=" * 50)
    print("Step 6: Saving Results")
    print("=" * 50)
    
    # Save tree structure
    trees_dir = os.path.join(PROJECT_ROOT, "outputs", "trees")
    os.makedirs(trees_dir, exist_ok=True)
    tree_builder.save_tree(os.path.join(trees_dir, "week3_raptor_tree.json"))
    
    # Save clustering results
    results = {
        'chunks': chunks,
        'embeddings': embeddings,
        'reduced_embeddings': reduced_embeddings,
        'cluster_labels': labels,
        'cluster_probabilities': probabilities,
        'cluster_metrics': metrics,
        'tree_nodes': {node_id: node.to_dict() for node_id, node in tree_builder.nodes.items()}
    }
    
    results_path = os.path.join(PROJECT_ROOT, "outputs", "week3_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"💾 Results saved to {results_path}")
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("WEEK 3 SUMMARY")
    print("=" * 70)
    
    num_clusters = len(np.unique(labels))
    avg_cluster_size = len(chunks) / num_clusters
    tree_stats = tree_builder.get_tree_stats()
    
    print(f"\n🎯 Key Achievements:")
    print(f"  1. Reduced dimensions: {embeddings.shape[1]}D → {reduced_embeddings.shape[1]}D")
    print(f"  2. Found {num_clusters} natural clusters in the document")
    print(f"  3. Average cluster size: {avg_cluster_size:.1f} chunks")
    print(f"  4. Built tree with {tree_stats['total_nodes']} nodes")
    print(f"  5. Tree depth: {tree_stats['max_depth']}")
    print(f"  6. Summary nodes: {tree_stats['summary_nodes']}, Leaf nodes: {tree_stats['leaf_nodes']}")
    
    sil_score = metrics.get('silhouette', 0)
    if sil_score > 0.5:
        print(f"  7. ✅ Excellent cluster separation! (silhouette: {sil_score:.3f})")
    elif sil_score > 0.25:
        print(f"  7. ⚠️  Fair cluster separation (silhouette: {sil_score:.3f})")
    else:
        print(f"  7. ❌ Poor cluster separation (silhouette: {sil_score:.3f})")
    
    print(f"\n📁 Files created:")
    print(f"  visualizations/week3_umap_reduction.png")
    print(f"  visualizations/week3_clusters.png")
    print(f"  visualizations/week3_cluster_sizes.png")
    print(f"  visualizations/week3_tree_structure.png")
    print(f"  week3_interactive_plot.html")
    print(f"  trees/week3_raptor_tree.json")
    print(f"  week3_results.pkl")
    
    print(f"\n🔮 What's Next (Week 4):")
    print(f"  - Generate intelligent summaries for each cluster using LLM")
    print(f"  - Create embeddings for summaries")
    print(f"  - Build the complete retrieval-ready tree")
    
    print(f"\n✅ WEEK 3 COMPLETE!")
    print(f"   You've built the core RAPTOR structure!")


if __name__ == "__main__":
    main()
