class OptimizedEmbeddings:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def create_semantic_chunks(self, text, chunk_size=512):
        import nltk
        nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []

        for sentence in sentences:
            if len(' '.join(current_chunk + [sentence])) <= chunk_size:
                current_chunk.append(sentence)
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def generate_embeddings(self, text_chunks):
        return self.model.encode(text_chunks, convert_to_tensor=True)