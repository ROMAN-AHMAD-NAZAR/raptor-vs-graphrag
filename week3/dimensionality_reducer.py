# week3/dimensionality_reducer.py
import umap
import numpy as np
import matplotlib.pyplot as plt


class DimensionalityReducer:
    def __init__(self, n_components=50, random_state=42):
        """
        WHY UMAP instead of PCA?
        
        PCA: Linear reduction, good for simple data
        UMAP: Non-linear, preserves local AND global structure
        
        Example: Imagine a Swiss roll (3D curved surface)
        - PCA: Flattens it, destroys structure
        - UMAP: Unrolls it, preserves neighbors
        """
        print(f"📉 Initializing UMAP reducer...")
        print(f"   Target dimensions: {n_components} (from 384)")
        print(f"   Reduction: {((384 - n_components) / 384 * 100):.1f}%")
        
        self.reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,           # Balance local/global structure
            min_dist=0.1,             # How tight clusters are (0.0-1.0)
            metric='cosine',          # Same as our embeddings
            random_state=random_state,
            verbose=False
        )
        self.n_components = n_components
    
    def reduce_dimensions(self, embeddings):
        """
        Reduce high-dimensional embeddings to lower dimensions
        
        BEFORE: (n_chunks, 384) - Hard to cluster
        AFTER:  (n_chunks, n_components)  - Easier to cluster, preserves structure
        """
        print(f"\n🔽 Reducing dimensions for {embeddings.shape[0]} embeddings...")
        print(f"   Input shape: {embeddings.shape}")
        
        # Adjust n_components if needed
        actual_components = min(self.n_components, embeddings.shape[0] - 1, embeddings.shape[1])
        if actual_components != self.n_components:
            print(f"   Adjusting components to {actual_components} (limited by data size)")
            self.reducer = umap.UMAP(
                n_components=actual_components,
                n_neighbors=min(15, embeddings.shape[0] - 1),
                min_dist=0.1,
                metric='cosine',
                random_state=42,
                verbose=False
            )
        
        # UMAP works better if data is normalized (ours already is)
        reduced = self.reducer.fit_transform(embeddings)
        
        print(f"✅ Reduction complete!")
        print(f"   Output shape: {reduced.shape}")
        print(f"   Memory savings: {(1 - reduced.nbytes/embeddings.nbytes)*100:.1f}%")
        
        # Analyze the reduction
        self._analyze_reduction(embeddings, reduced)
        
        return reduced
    
    def _analyze_reduction(self, original, reduced):
        """Check if reduction preserved important information"""
        print("\n📊 Analyzing reduction quality...")
        
        # Check variance explained (approximation)
        from sklearn.decomposition import PCA
        n_pca_components = min(reduced.shape[1], original.shape[0] - 1, original.shape[1])
        pca = PCA(n_components=n_pca_components)
        pca.fit(original)
        variance_explained = pca.explained_variance_ratio_.sum()
        
        print(f"   PCA variance explained with {n_pca_components} components: {variance_explained:.1%}")
        print(f"   (UMAP preserves structure better than PCA, so this is a lower bound)")
        
        # Check if clusters are more separable
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = min(5, original.shape[0] - 1)
        nn_original = NearestNeighbors(n_neighbors=n_neighbors).fit(original)
        nn_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(reduced)
        
        # Compare neighbor preservation
        sample_idx = 0
        distances_orig, indices_orig = nn_original.kneighbors(original[sample_idx:sample_idx+1])
        distances_red, indices_red = nn_reduced.kneighbors(reduced[sample_idx:sample_idx+1])
        
        common_neighbors = len(set(indices_orig[0]) & set(indices_red[0]))
        print(f"   Neighbor preservation: {common_neighbors/n_neighbors:.0%} of nearest neighbors kept")
        
        return True
    
    def visualize_reduction(self, reduced_embeddings, chunks, save_path=None):
        """
        Visualize the reduced embeddings (first 2 dimensions)
        Great for debugging!
        """
        print("\n🎨 Creating reduction visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Plot each point
        scatter = plt.scatter(
            reduced_embeddings[:, 0],  # First dimension
            reduced_embeddings[:, 1],  # Second dimension
            alpha=0.7,
            s=100,
            c=range(len(reduced_embeddings)),
            cmap='viridis',
            edgecolors='w',
            linewidth=0.5
        )
        
        # Add chunk preview as annotations (first few only)
        for i, (x, y) in enumerate(reduced_embeddings[:min(10, len(reduced_embeddings)), :2]):
            plt.annotate(
                f"[{i}]", 
                (x, y),
                fontsize=8,
                alpha=0.7
            )
        
        plt.colorbar(scatter, label='Chunk Index')
        plt.title(f'UMAP Reduction: 384D → 2D (showing first 2 of {reduced_embeddings.shape[1]} dimensions)')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # Add explanation
        plt.figtext(0.5, 0.01, 
                   f'Points are chunks. Closer points = more similar content.\n'
                   f'Total chunks: {len(chunks)}, Reduced to {reduced_embeddings.shape[1]}D',
                   ha='center', fontsize=9, style='italic')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return True
