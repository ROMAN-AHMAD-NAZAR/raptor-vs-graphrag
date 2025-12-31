# week3/hierarchical_clusterer.py
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


class HierarchicalClusterer:
    def __init__(self, max_clusters=10, min_clusters=2):
        """
        WHY Gaussian Mixture Model (GMM) instead of K-means?
        
        K-means: Hard assignment, spherical clusters
        GMM: Soft assignment, elliptical clusters, probabilistic
        
        Real data clusters aren't perfect spheres!
        """
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
    
    def find_optimal_clusters(self, data):
        """
        Find the best number of clusters automatically
        
        We use Bayesian Information Criterion (BIC):
        - Lower BIC = better model
        - Balances fit and complexity
        """
        print(f"\n🔍 Finding optimal number of clusters...")
        
        # Adjust range based on data size
        max_possible = min(self.max_clusters, len(data) - 1)
        min_possible = min(self.min_clusters, max_possible)
        
        if max_possible < 2:
            print("   ⚠️ Not enough data points for clustering")
            return None, {'n_clusters': 1}
        
        print(f"   Testing from {min_possible} to {max_possible} clusters")
        
        bics = []
        silhouette_scores = []
        models = []
        
        for n in range(min_possible, max_possible + 1):
            try:
                # Create GMM
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',  # Most flexible
                    random_state=42,
                    n_init=3,  # Multiple initializations
                    max_iter=200
                )
                
                # Fit model
                gmm.fit(data)
                
                # Calculate metrics
                labels = gmm.predict(data)
                bic = gmm.bic(data)
                
                # Silhouette score (only if meaningful)
                if len(set(labels)) > 1:
                    sil = silhouette_score(data, labels)
                else:
                    sil = -1
                
                bics.append(bic)
                silhouette_scores.append(sil)
                models.append(gmm)
                
                print(f"   {n} clusters: BIC={bic:.1f}, Silhouette={sil:.3f}")
            except Exception as e:
                print(f"   {n} clusters: Failed ({e})")
                continue
        
        if not models:
            print("   ⚠️ No valid clustering found")
            return None, {'n_clusters': 1}
        
        # Find best by BIC (lower is better)
        best_idx = np.argmin(bics)
        best_n = best_idx + min_possible
        
        print(f"\n✅ Optimal clusters: {best_n}")
        print(f"   Best BIC: {bics[best_idx]:.1f}")
        print(f"   Silhouette score: {silhouette_scores[best_idx]:.3f}")
        print(f"   (Silhouette > 0.5 = good separation, > 0.7 = excellent)")
        
        return models[best_idx], {
            'n_clusters': best_n,
            'bic': bics[best_idx],
            'silhouette': silhouette_scores[best_idx],
            'all_bics': bics,
            'all_silhouettes': silhouette_scores
        }
    
    def cluster(self, data, n_clusters=None):
        """
        Cluster the reduced embeddings
        
        Returns:
        - labels: Which cluster each chunk belongs to
        - probabilities: How confident (soft clustering)
        - model: The trained GMM
        """
        if len(data) < 2:
            print("⚠️ Not enough data to cluster")
            return np.zeros(len(data), dtype=int), np.ones((len(data), 1)), None, {}
        
        if n_clusters is None:
            model, metrics = self.find_optimal_clusters(data)
            if model is None:
                return np.zeros(len(data), dtype=int), np.ones((len(data), 1)), None, metrics
        else:
            n_clusters = min(n_clusters, len(data) - 1)
            print(f"\n🏷️  Clustering into {n_clusters} clusters...")
            model = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=42
            )
            model.fit(data)
            metrics = {'n_clusters': n_clusters}
        
        # Get cluster assignments
        labels = model.predict(data)
        probabilities = model.predict_proba(data)
        
        print(f"\n📊 Cluster distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = (count / len(labels)) * 100
            print(f"   Cluster {cluster_id}: {count} chunks ({percentage:.1f}%)")
        
        # Calculate cluster quality
        if len(unique) > 1:
            sil_score = silhouette_score(data, labels)
            db_score = davies_bouldin_score(data, labels)
            print(f"\n📈 Cluster quality metrics:")
            print(f"   Silhouette score: {sil_score:.3f}")
            print(f"   Davies-Bouldin index: {db_score:.3f} (lower is better)")
            
            metrics['silhouette'] = sil_score
            metrics['davies_bouldin'] = db_score
            
            if sil_score > 0.5:
                print("   ✅ Good cluster separation!")
            elif sil_score > 0.25:
                print("   ⚠️  Fair cluster separation")
            else:
                print("   ❌ Poor cluster separation - chunks may not form natural groups")
        else:
            print("⚠️  Only one cluster found - all chunks are similar")
        
        return labels, probabilities, model, metrics
    
    def analyze_clusters(self, labels, chunks, top_words=5):
        """
        Understand what each cluster is about
        """
        print(f"\n🧠 Analyzing cluster themes...")
        
        from collections import Counter
        import re
        
        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(chunks[i])
        
        for cluster_id in sorted(clusters.keys()):
            cluster_chunks = clusters[cluster_id]
            print(f"\n--- Cluster {cluster_id} ({len(cluster_chunks)} chunks) ---")
            
            # Extract most common words (excluding stopwords)
            all_text = ' '.join(cluster_chunks).lower()
            words = re.findall(r'\b[a-z]{3,}\b', all_text)  # Words 3+ chars
            
            # Simple stopwords list
            stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'have', 'from', 
                        'are', 'was', 'were', 'has', 'had', 'but', 'not', 'you', 'your',
                        'can', 'will', 'each', 'which', 'their', 'more', 'than', 'been',
                        'into', 'also', 'such', 'when', 'over', 'these', 'other', 'all',
                        'published', 'conference', 'paper', 'iclr'}
            
            word_counts = Counter([w for w in words if w not in stopwords])
            
            top = word_counts.most_common(top_words)
            if top:
                print(f"   Top words: {', '.join([w for w, _ in top])}")
            
            # Show sample chunk
            sample = cluster_chunks[0]
            preview = sample[:100] + "..." if len(sample) > 100 else sample
            print(f"   Sample: '{preview}'")
        
        return clusters
