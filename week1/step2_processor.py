"""Text processor for cleaning and chunking documents"""
import re


class TextProcessor:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def clean_text(self, text):
        """Clean text by removing extra spaces and normalizing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        return text.strip()
    
    def split_into_sentences(self, text):
        """Simple sentence splitting using NLTK"""
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    
    def create_chunks(self, text):
        """Create overlapping chunks from text"""
        cleaned = self.clean_text(text)
        sentences = self.split_into_sentences(cleaned)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and chunks:
                    # Take last few sentences for overlap
                    overlap_sentences = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_sentences
                    current_length = sum(len(s.split()) for s in overlap_sentences)
                else:
                    current_chunk = []
                    current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
