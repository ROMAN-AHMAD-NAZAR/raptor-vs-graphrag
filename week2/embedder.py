"""Week 2: Text Embedding Module"""
import numpy as np
from typing import List, Optional


class TextEmbedder:
    """Generate embeddings for text chunks using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder with a sentence-transformer model.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Default: "all-MiniLM-L6-v2" (fast, 384 dimensions)
                       Other options: 
                       - "all-mpnet-base-v2" (better quality, 768 dims)
                       - "paraphrase-MiniLM-L6-v2" (good for paraphrasing)
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
    
    def load_model(self):
        """Load the embedding model (lazy loading)"""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string"""
        self.load_model()
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, 
                    show_progress: bool = True) -> np.ndarray:
        """
        Embed multiple texts efficiently in batches.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once
            show_progress: Whether to show a progress bar
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        self.load_model()
        
        print(f"Embedding {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, 
                           embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        from numpy.linalg import norm
        return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
    
    def find_similar(self, query_embedding: np.ndarray, 
                     corpus_embeddings: np.ndarray, 
                     top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings in a corpus.
        
        Args:
            query_embedding: The query embedding vector
            corpus_embeddings: Matrix of corpus embeddings
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        from numpy.linalg import norm
        
        # Normalize embeddings
        query_norm = query_embedding / norm(query_embedding)
        corpus_norms = corpus_embeddings / norm(corpus_embeddings, axis=1, keepdims=True)
        
        # Compute similarities
        similarities = np.dot(corpus_norms, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        return results


class FAISSIndex:
    """FAISS-based vector index for efficient similarity search"""
    
    def __init__(self, embedding_dim: int):
        """
        Initialize FAISS index.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
        """
        import faiss
        
        self.embedding_dim = embedding_dim
        # Use L2 (Euclidean) distance - for cosine, normalize vectors first
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product (cosine with normalized vectors)
        self.num_vectors = 0
    
    def add(self, embeddings: np.ndarray):
        """Add embeddings to the index"""
        import faiss
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings.astype(np.float32))
        self.num_vectors += len(embeddings)
        print(f"Added {len(embeddings)} vectors. Total: {self.num_vectors}")
    
    def search(self, query_embeddings: np.ndarray, top_k: int = 5) -> tuple:
        """
        Search for similar vectors.
        
        Args:
            query_embeddings: Query embedding(s) - shape (num_queries, dim)
            top_k: Number of results per query
            
        Returns:
            (distances, indices) - both of shape (num_queries, top_k)
        """
        import faiss
        
        # Handle single query
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Normalize query for cosine similarity
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        
        distances, indices = self.index.search(query_embeddings, top_k)
        return distances, indices
    
    def save(self, path: str):
        """Save index to disk"""
        import faiss
        faiss.write_index(self.index, path)
        print(f"Index saved to {path}")
    
    def load(self, path: str):
        """Load index from disk"""
        import faiss
        self.index = faiss.read_index(path)
        self.num_vectors = self.index.ntotal
        print(f"Index loaded from {path}. Contains {self.num_vectors} vectors.")
