# week5/qdrant_manager.py
"""
Qdrant Vector Database Manager for RAPTOR

Handles storage and retrieval of RAPTOR tree and normal RAG chunks
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue
)
import numpy as np
from typing import List, Dict, Optional
import time


class QdrantManager:
    """
    Manage Qdrant vector database for RAPTOR tree storage
    
    Features:
    - Store hierarchical tree structure
    - Store flat RAG chunks for comparison
    - Filtered search by depth, type, etc.
    """
    
    def __init__(self, location=":memory:"):
        """
        Initialize Qdrant client
        
        Options for location:
        - ":memory:" - In-memory (fast, for testing)
        - "./qdrant_data" - Local filesystem
        - "http://localhost:6333" - Docker/Qdrant server
        """
        print(f"🔌 Initializing Qdrant client ({location})...")
        
        self.client = QdrantClient(location=location)
        self.collection_name = "raptor_tree"
        
        # Test connection
        try:
            collections = self.client.get_collections()
            print(f"✅ Connected to Qdrant (in-memory mode)")
        except Exception as e:
            print(f"⚠️  Could not connect: {e}")
            print("   Using in-memory mode...")
            self.client = QdrantClient(":memory:")
    
    def create_collection(self, vector_size=384, recreate=True):
        """
        Create collection for RAPTOR tree
        
        Special features:
        - Stores vectors (embeddings)
        - Stores payload (metadata, text, tree structure)
        - Enables filtering by depth, is_summary, etc.
        """
        print(f"\n🗂️  Creating collection '{self.collection_name}'...")
        print(f"   Vector size: {vector_size}")
        
        # Delete if exists and recreate is requested
        if recreate:
            try:
                self.client.delete_collection(self.collection_name)
                print("   Old collection deleted")
            except:
                pass
        
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        # Get collection info
        info = self.client.get_collection(self.collection_name)
        print(f"✅ Collection created!")
        print(f"   Status: {info.status}")
        print(f"   Points count: {info.points_count}")
        
        return info
    
    def store_raptor_tree(self, retrieval_tree: Dict) -> int:
        """
        Store the entire RAPTOR tree in Qdrant
        
        Each tree node becomes a point with:
        - Vector: embedding
        - Payload: text, depth, parent_id, children_ids, is_summary, etc.
        """
        print(f"\n💾 Storing RAPTOR tree in Qdrant...")
        
        points = []
        embeddings = retrieval_tree.get('embeddings', [])
        texts = retrieval_tree.get('texts', [])
        metadata_list = retrieval_tree.get('metadata', [])
        
        if len(embeddings) == 0:
            print("❌ No embeddings found in retrieval tree")
            return 0
        
        print(f"   Preparing {len(embeddings)} nodes...")
        
        for i in range(len(embeddings)):
            embedding = embeddings[i]
            text = texts[i] if i < len(texts) else f"Node {i}"
            metadata = metadata_list[i] if i < len(metadata_list) else {}
            
            # Create payload with tree structure
            payload = {
                "text": text,
                "depth": metadata.get('depth', 0),
                "is_summary": metadata.get('is_summary', False),
                "node_id": metadata.get('id', f"node_{i}"),
                "num_children": metadata.get('num_children', 0),
                "type": "summary" if metadata.get('is_summary', False) else "chunk",
                "timestamp": time.time()
            }
            
            # Convert embedding to list if needed
            if hasattr(embedding, 'tolist'):
                vector = embedding.tolist()
            else:
                vector = list(embedding)
            
            # Create point
            point = PointStruct(
                id=i,
                vector=vector,
                payload=payload
            )
            
            points.append(point)
        
        # Upload to Qdrant
        print(f"   Uploading {len(points)} points...")
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Verify upload
        info = self.client.get_collection(self.collection_name)
        print(f"✅ Tree stored successfully!")
        print(f"   Total points: {info.points_count}")
        print(f"   Status: {operation_info.status}")
        
        return info.points_count
    
    def store_normal_rag(self, chunks: List[str], embeddings: np.ndarray) -> int:
        """
        Store chunks in flat structure (normal RAG baseline)
        
        This is what traditional RAG does - no hierarchy, just chunks
        """
        print(f"\n💾 Storing normal RAG (flat structure)...")
        
        rag_collection = "normal_rag"
        
        # Create collection for normal RAG
        try:
            self.client.delete_collection(rag_collection)
        except:
            pass
        
        vector_size = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings[0])
        
        self.client.create_collection(
            collection_name=rag_collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        
        # Create points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            payload = {
                "text": chunk,
                "chunk_id": i,
                "type": "chunk",
                "is_summary": False,
                "depth": 0
            }
            
            vector = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            
            point = PointStruct(
                id=i,
                vector=vector,
                payload=payload
            )
            points.append(point)
        
        # Upload
        self.client.upsert(
            collection_name=rag_collection,
            points=points
        )
        
        info = self.client.get_collection(rag_collection)
        print(f"✅ Normal RAG stored!")
        print(f"   Chunks: {info.points_count}")
        
        return info.points_count
    
    def search_similar(self, query_vector: List[float], 
                      limit: int = 10,
                      collection: str = None,
                      filters: Optional[Filter] = None) -> List[Dict]:
        """
        Search for similar vectors
        
        Can be used for both RAPTOR and normal RAG
        """
        if collection is None:
            collection = self.collection_name
        
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            query_filter=filters,
            limit=limit,
            with_payload=True
        ).points
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                'id': result.id,
                'score': result.score,
                'text': result.payload.get('text', ''),
                'depth': result.payload.get('depth', 0),
                'is_summary': result.payload.get('is_summary', False),
                'node_id': result.payload.get('node_id', ''),
                'payload': result.payload
            })
        
        return formatted
    
    def get_collection_stats(self, collection_name: str = None) -> Dict:
        """Get statistics about a collection"""
        if collection_name is None:
            collection_name = self.collection_name
        
        try:
            info = self.client.get_collection(collection_name)
            return {
                'name': collection_name,
                'vectors_count': info.points_count,
                'status': str(info.status),
                'points_count': info.points_count
            }
        except:
            return {}
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        collections = self.client.get_collections()
        return [col.name for col in collections.collections]
    
    def count_by_type(self, collection: str = None) -> Dict:
        """Count points by type (summary vs chunk)"""
        if collection is None:
            collection = self.collection_name
        
        try:
            # Get all points
            all_points = self.client.scroll(
                collection_name=collection,
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            
            counts = {'summary': 0, 'chunk': 0, 'total': 0}
            depth_counts = {}
            
            for point in all_points[0]:
                counts['total'] += 1
                if point.payload.get('is_summary', False):
                    counts['summary'] += 1
                else:
                    counts['chunk'] += 1
                
                depth = point.payload.get('depth', 0)
                depth_counts[depth] = depth_counts.get(depth, 0) + 1
            
            counts['by_depth'] = depth_counts
            return counts
            
        except Exception as e:
            print(f"⚠️  Error counting: {e}")
            return {}
