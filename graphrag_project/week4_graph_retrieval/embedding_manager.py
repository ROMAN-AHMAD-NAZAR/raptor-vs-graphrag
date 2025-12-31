# week4_graph_retrieval/embedding_manager.py
"""
Graph Embedding Manager for GraphRAG
Manages embeddings for graph nodes and enables semantic similarity search
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import pickle
from pathlib import Path

# Try to import sentence-transformers, fall back gracefully
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Try sklearn for similarity
try:
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class GraphEmbeddingManager:
    """
    Manages embeddings for graph nodes and queries
    Enables semantic similarity search on graph
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        self.logger = logging.getLogger(__name__)
        self.embedding_dim = 384  # Default for MiniLM
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            self.logger.warning("sentence-transformers not installed. Using fallback embeddings.")
            self.logger.info("Install with: pip install sentence-transformers")
            return
        
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"✅ GraphEmbeddingManager initialized with {self.model_name}")
            self.logger.info(f"   Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Embed single text"""
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
        
        if self.model is None:
            # Fallback: simple hash-based embedding
            embedding = self._fallback_embedding(text)
        else:
            embedding = self.model.encode(text, normalize_embeddings=True)
        
        self.embeddings_cache[text] = embedding
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently"""
        if not texts:
            return np.array([])
        
        # Check cache first
        to_embed = []
        indices_to_embed = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            if text in self.embeddings_cache:
                cached_embeddings[i] = self.embeddings_cache[text]
            else:
                to_embed.append(text)
                indices_to_embed.append(i)
        
        # Embed new texts
        if to_embed:
            if self.model is None:
                new_embeddings = np.array([self._fallback_embedding(t) for t in to_embed])
            else:
                new_embeddings = self.model.encode(
                    to_embed,
                    batch_size=32,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            
            # Cache new embeddings
            for text, emb in zip(to_embed, new_embeddings):
                self.embeddings_cache[text] = emb
        
        # Combine results in correct order
        all_embeddings = np.zeros((len(texts), self.embedding_dim))
        
        for i, text in enumerate(texts):
            all_embeddings[i] = self.embeddings_cache[text]
        
        return all_embeddings
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Create a simple fallback embedding when model not available"""
        # Simple hash-based embedding for fallback
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self.embedding_dim)
        return embedding / np.linalg.norm(embedding)
    
    def calculate_similarity(self, query_embedding: np.ndarray, 
                           target_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and targets"""
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if HAS_SKLEARN:
            similarities = cosine_similarity(query_embedding, target_embeddings)
            return similarities[0]
        else:
            # Manual cosine similarity
            similarities = []
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            for target in target_embeddings:
                target_norm = target / np.linalg.norm(target)
                sim = np.dot(query_norm.flatten(), target_norm)
                similarities.append(sim)
            return np.array(similarities)
    
    def find_similar_nodes(self, query: str, node_texts: List[str], 
                          top_k: int = 10) -> List[Tuple[int, float, str]]:
        """Find most similar nodes to query"""
        if not node_texts:
            return []
        
        query_embedding = self.embed_text(query)
        node_embeddings = self.embed_batch(node_texts)
        
        similarities = self.calculate_similarity(query_embedding, node_embeddings)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include positive similarity
                results.append((int(idx), float(similarities[idx]), node_texts[idx]))
        
        return results
    
    def embed_graph_nodes(self, nodes: List[Dict]) -> Dict[str, Dict]:
        """Create embeddings for all graph nodes"""
        node_embeddings = {}
        
        # Prepare node texts for embedding
        node_texts = []
        node_info = []
        
        for node in nodes:
            # Create rich text representation for embedding
            node_text = self._create_node_text(node)
            node_texts.append(node_text)
            node_info.append({
                'id': node.get('id'),
                'name': node.get('properties', {}).get('name', ''),
                'type': node.get('properties', {}).get('type', '')
            })
        
        # Embed all nodes
        embeddings = self.embed_batch(node_texts)
        
        # Store embeddings by node ID
        for i, info in enumerate(node_info):
            node_embeddings[info['id']] = {
                'embedding': embeddings[i],
                'text': node_texts[i],
                'name': info['name'],
                'type': info['type']
            }
        
        self.logger.info(f"✅ Embedded {len(node_embeddings)} graph nodes")
        return node_embeddings
    
    def _create_node_text(self, node: Dict) -> str:
        """Create rich text representation of a node for embedding"""
        props = node.get('properties', {})
        
        # Combine relevant properties
        texts = []
        
        # Name is most important
        if 'name' in props:
            texts.append(props['name'])
        
        # Type information
        if 'type' in props:
            texts.append(f"type: {props['type']}")
        
        # Mention context
        if 'mention' in props:
            texts.append(f"mentioned as: {props['mention']}")
        
        return " ".join(texts)
    
    def save_embeddings(self, embeddings: Dict, path: Path):
        """Save embeddings to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for serialization
        serializable = {}
        for key, value in embeddings.items():
            if isinstance(value, dict) and 'embedding' in value:
                serializable[key] = {
                    **value,
                    'embedding': value['embedding'].tolist()
                }
            else:
                serializable[key] = value
        
        with open(path, 'wb') as f:
            pickle.dump(serializable, f)
        self.logger.info(f"💾 Saved embeddings to {path}")
    
    def load_embeddings(self, path: Path) -> Optional[Dict]:
        """Load embeddings from disk"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Convert lists back to numpy arrays
            for key, value in data.items():
                if isinstance(value, dict) and 'embedding' in value:
                    if isinstance(value['embedding'], list):
                        value['embedding'] = np.array(value['embedding'])
            
            self.logger.info(f"📂 Loaded embeddings from {path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            return None
    
    def get_node_context(self, node_id: str, neo4j_manager, depth: int = 2) -> str:
        """
        Get contextual information around a node by traversing the graph
        This creates rich context for better embeddings
        """
        query = f"""
        MATCH path = (start:Entity {{id: $node_id}})-[*1..{depth}]-(neighbor)
        WHERE neighbor:Entity
        RETURN DISTINCT neighbor.id as id, 
               neighbor.name as name,
               neighbor.type as type,
               length(path) as distance
        ORDER BY distance
        LIMIT 20
        """
        
        try:
            results = neo4j_manager.execute_cypher(query, {'node_id': node_id})
            
            context_parts = []
            for result in results:
                if result['id'] != node_id:
                    context_text = f"{result.get('name', '')} ({result.get('type', 'unknown')})"
                    context_parts.append(context_text)
            
            return " | ".join(context_parts[:10])  # Limit context length
            
        except Exception as e:
            self.logger.error(f"Failed to get node context: {e}")
            return ""
    
    def clear_cache(self):
        """Clear the embeddings cache"""
        self.embeddings_cache = {}
        self.logger.info("🗑️ Embeddings cache cleared")
