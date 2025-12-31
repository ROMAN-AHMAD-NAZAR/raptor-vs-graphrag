# week2/embedding_storage.py
import pickle
import numpy as np
import json
import os
from datetime import datetime


class EmbeddingStorage:
    def __init__(self, save_dir="./embeddings_data"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        # Create metadata
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.0',
            'description': 'RAPTOR Week 2 Embeddings'
        }
    
    def save_embeddings(self, chunks, embeddings, metadata=None, filename="embeddings"):
        """
        Save in multiple formats for different use cases:
        1. .npy - Fast loading for numpy
        2. .pkl - Preserves Python objects
        3. .json - Human readable
        4. .txt - Simple text format
        """
        print(f"\n💾 Saving embeddings to {self.save_dir}/...")
        
        # 1. Save as numpy array (most efficient)
        np.save(f"{self.save_dir}/{filename}_vectors.npy", embeddings)
        print(f"   ✓ {filename}_vectors.npy - {embeddings.shape}")
        
        # 2. Save chunks and metadata as pickle
        data_to_save = {
            'chunks': chunks,
            'metadata': metadata if metadata else {},
            'system_metadata': self.metadata,
            'embedding_shape': embeddings.shape
        }
        
        with open(f"{self.save_dir}/{filename}_data.pkl", 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"   ✓ {filename}_data.pkl - {len(chunks)} chunks")
        
        # 3. Save human-readable JSON
        json_data = {
            'info': {
                'num_chunks': len(chunks),
                'embedding_dim': int(embeddings.shape[1]),
                'total_vectors': int(embeddings.shape[0]),
                'created_at': self.metadata['created_at']
            },
            'sample_chunks': [
                {
                    'id': i,
                    'text': chunk[:100] + "..." if len(chunk) > 100 else chunk,
                    'word_count': len(chunk.split())
                }
                for i, chunk in enumerate(chunks[:5])  # First 5 only
            ]
        }
        
        with open(f"{self.save_dir}/{filename}_info.json", 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"   ✓ {filename}_info.json - Human readable info")
        
        # 4. Save chunks as text file
        with open(f"{self.save_dir}/{filename}_chunks.txt", 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"=== Chunk {i} ===\n")
                f.write(f"{chunk}\n\n")
        print(f"   ✓ {filename}_chunks.txt - All chunks as text")
        
        total_size = 0
        for file in os.listdir(self.save_dir):
            if filename in file:
                total_size += os.path.getsize(f"{self.save_dir}/{file}")
        
        print(f"\n✅ All files saved!")
        print(f"   Total size: {total_size/1024:.1f} KB")
        print(f"   Location: {os.path.abspath(self.save_dir)}")
    
    def load_embeddings(self, filename="embeddings"):
        """Load embeddings and chunks"""
        print(f"📂 Loading embeddings from {self.save_dir}/...")
        
        try:
            # Load vectors
            vectors_path = f"{self.save_dir}/{filename}_vectors.npy"
            embeddings = np.load(vectors_path)
            print(f"   ✓ Loaded vectors: {embeddings.shape}")
            
            # Load data
            data_path = f"{self.save_dir}/{filename}_data.pkl"
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            chunks = data['chunks']
            metadata = data.get('metadata', {})
            
            print(f"   ✓ Loaded chunks: {len(chunks)}")
            print(f"   ✓ Created: {data.get('system_metadata', {}).get('created_at', 'unknown')}")
            
            return embeddings, chunks, metadata
            
        except FileNotFoundError as e:
            print(f"❌ Error loading: {e}")
            print(f"   Try running Week 2 first to create embeddings")
            return None, None, None
    
    def get_stats(self, filename="embeddings"):
        """Get statistics about saved embeddings"""
        try:
            with open(f"{self.save_dir}/{filename}_info.json", 'r', encoding='utf-8') as f:
                info = json.load(f)
            return info['info']
        except:
            return None
    
    def list_saved_files(self):
        """List all saved embedding files"""
        files = []
        if os.path.exists(self.save_dir):
            for f in os.listdir(self.save_dir):
                size = os.path.getsize(os.path.join(self.save_dir, f)) / 1024
                files.append({'name': f, 'size_kb': size})
        return files
