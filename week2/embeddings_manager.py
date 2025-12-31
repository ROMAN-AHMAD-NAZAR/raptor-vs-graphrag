# week2/embeddings_manager.py
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


class EmbeddingsManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Model Choices (pick one):
        
        1. 'all-MiniLM-L6-v2' (RECOMMENDED for testing)
           - 384 dimensions, fast, decent quality
           - 80MB download
        
        2. 'all-mpnet-base-v2'
           - 768 dimensions, slower, better quality
           - 420MB download
        
        3. 'BAAI/bge-small-en-v1.5'
           - 384 dimensions, optimized for retrieval
           - 33MB download
        """
        print(f"🚀 Loading embedding model: {model_name}")
        print("This may take a minute on first run (downloading model)...")
        
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded successfully!")
            print(f"   Dimension: {self.dimension}")
            print(f"   Max sequence length: {self.model.max_seq_length}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Falling back to a simpler model...")
            self.model = SentenceTransformer('paraphrase-albert-small-v2')
            self.dimension = self.model.get_sentence_embedding_dimension()
    
    def create_embeddings(self, chunks, batch_size=16):
        """
        WHY batch_size matters:
        - Too small (1-4): Very slow
        - Too large (64+): Might run out of memory
        - Sweet spot (16-32): Good for most systems
        """
        print(f"\n📊 Creating embeddings for {len(chunks)} chunks...")
        print(f"   Batch size: {batch_size}")
        print(f"   Estimated time: {len(chunks)/100:.1f} seconds")
        
        # Convert to embeddings
        embeddings = self.model.encode(
            chunks,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=False,  # Use numpy for compatibility
            normalize_embeddings=True,  # CRITICAL for cosine similarity
            device='cpu'  # Use 'cuda' if you have GPU
        )
        
        print(f"✅ Embeddings created!")
        print(f"   Shape: {embeddings.shape}")
        print(f"   Type: {type(embeddings)}")
        
        return embeddings
    
    def validate_embeddings(self, embeddings):
        """Check if embeddings are valid"""
        print("\n🔍 Validating embeddings...")
        
        # Check for NaN values
        if np.any(np.isnan(embeddings)):
            print("❌ WARNING: Embeddings contain NaN values!")
        else:
            print("✅ No NaN values")
        
        # Check magnitude (should be ~1.0 after normalization)
        magnitudes = np.linalg.norm(embeddings, axis=1)
        avg_magnitude = np.mean(magnitudes)
        print(f"✅ Average vector magnitude: {avg_magnitude:.4f}")
        print(f"   (Should be close to 1.0 after normalization)")
        
        # Check similarity of identical text
        test_text = "This is a test sentence."
        test_emb = self.model.encode([test_text, test_text], normalize_embeddings=True)
        similarity = np.dot(test_emb[0], test_emb[1])
        print(f"✅ Identical text similarity: {similarity:.6f}")
        print(f"   (Should be 1.0 exactly)")
        
        return True
    
    def test_semantic_understanding(self):
        """Test if the model understands semantics"""
        print("\n🧠 Testing semantic understanding...")
        
        test_pairs = [
            ("The cat sat on the mat", "The feline rested on the rug"),
            ("I love programming", "I enjoy coding"),
            ("The weather is sunny", "It's raining heavily"),
            ("Apple the company", "Apple the fruit")
        ]
        
        results = []
        for text1, text2 in test_pairs:
            emb1, emb2 = self.model.encode([text1, text2], normalize_embeddings=True)
            similarity = np.dot(emb1, emb2)
            results.append((text1, text2, similarity))
            
            # Visual indicator
            if similarity > 0.7:
                indicator = "✅ HIGH"
            elif similarity < 0.3:
                indicator = "❌ LOW"
            else:
                indicator = "⚠️  MEDIUM"
            
            print(f"   '{text1[:20]}...' vs '{text2[:20]}...'")
            print(f"      Similarity: {similarity:.3f} {indicator}")
        
        return results
    
    def embed_query(self, query):
        """Embed a single query for search"""
        return self.model.encode(query, normalize_embeddings=True)
