class EfficientSummarizer:
    def __init__(self, clustering_model):
        self.clustering_model = clustering_model

    def generate_hierarchical_summary(self, text_chunks):
        clusters = self.clustering_model.cluster(text_chunks)
        summaries = []
        for cluster in clusters:
            summary = self._summarize_cluster(cluster)
            summaries.append(summary)
        return self._combine_summaries(summaries)

    def _summarize_cluster(self, cluster):
        # Implement summarization logic for a single cluster
        # This could involve extracting key sentences or generating a summary using a language model
        return "Summary of cluster"

    def _combine_summaries(self, summaries):
        # Combine individual summaries into a final hierarchical summary
        return "\n".join(summaries)