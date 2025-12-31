class HierarchicalClustering:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.clusters = None

    def fit(self, embeddings):
        from sklearn.cluster import AgglomerativeClustering
        clustering_model = AgglomerativeClustering(n_clusters=self.n_clusters)
        self.clusters = clustering_model.fit_predict(embeddings)
        return self.clusters

    def calculate_cluster_quality(self, embeddings):
        from sklearn.metrics import silhouette_score
        if self.clusters is None:
            raise ValueError("You must fit the model before calculating cluster quality.")
        score = silhouette_score(embeddings, self.clusters)
        return score

    def get_clusters(self):
        if self.clusters is None:
            raise ValueError("You must fit the model before accessing clusters.")
        return self.clusters