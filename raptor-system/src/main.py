from document_processing.processor import EnhancedDocumentProcessor
from embeddings.embedder import OptimizedEmbeddings
from clustering.clusterer import HierarchicalClustering
from summarization.summarizer import EfficientSummarizer
from evaluation.evaluator import RaptorEvaluator

def main():
    # Step 1: Document Processing
    doc_processor = EnhancedDocumentProcessor()
    documents = doc_processor.load_documents('path/to/documents')
    cleaned_docs = doc_processor.clean_documents(documents)
    text_chunks = doc_processor.chunk_documents(cleaned_docs)

    # Step 2: Generate Embeddings
    embedder = OptimizedEmbeddings()
    embeddings = embedder.generate_embeddings(text_chunks)

    # Step 3: Clustering
    clusterer = HierarchicalClustering()
    clusters = clusterer.cluster_embeddings(embeddings)

    # Step 4: Summarization
    summarizer = EfficientSummarizer()
    summaries = summarizer.generate_summaries(clusters)

    # Step 5: Evaluation
    evaluator = RaptorEvaluator()
    evaluation_results = evaluator.evaluate(summaries, 'path/to/ground_truth')

    # Output evaluation results
    print(evaluation_results)

if __name__ == "__main__":
    main()