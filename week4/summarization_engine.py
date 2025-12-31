# week4/summarization_engine.py
"""
Summarization Engine for RAPTOR Tree Enhancement

Supports multiple backends:
1. Transformers (HuggingFace) - Best quality, requires more RAM
2. Rule-based - Lightweight fallback, no model download needed
"""

from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class SummarizationEngine:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                 device: str = "cpu", use_lightweight: bool = False):
        """
        Initialize the summarization model
        
        Available models (choose based on your RAM):
        - "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  (2.2GB RAM)  ← RECOMMENDED
        - "microsoft/phi-2"                     (5.5GB RAM)
        - "mistralai/Mistral-7B-Instruct-v0.2"  (14GB RAM)
        
        Args:
            model_name: HuggingFace model name
            device: "cpu", "cuda", or "auto"
            use_lightweight: If True, skip model loading and use rule-based
        """
        self.model_name = model_name
        self.device = device
        self.generator = None
        self.tokenizer = None
        self.use_lightweight = use_lightweight
        
        if use_lightweight:
            print("🤖 Using lightweight rule-based summarization (no model download)")
            return
        
        print(f"🤖 Initializing summarization engine...")
        print(f"   Model: {model_name}")
        print(f"   Device: {device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            print("   Loading model (this may take a few minutes on first run)...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="./models_cache",
                trust_remote_code=True
            )
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine dtype based on device
            if self.device == "cuda" and torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir="./models_cache",
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            
            # Create pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" and torch.cuda.is_available() else -1
            )
            
            print(f"✅ Model loaded successfully!")
            print(f"   Parameters: {self.model.num_parameters():,}")
            
        except Exception as e:
            print(f"⚠️  Could not load model: {e}")
            print("   Falling back to lightweight rule-based summarization")
            self.use_lightweight = True
            self.generator = None
    
    def _create_prompt(self, text: str, depth: int) -> str:
        """Create summarization prompt based on tree depth"""
        # Truncate text if too long
        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        if depth == 0:
            prompt = f"""<|system|>
You are a helpful assistant that creates clear, concise summaries.
<|user|>
Provide a comprehensive overview of this text in 2-3 sentences. Focus on main topics and key concepts.

Text: {text}
<|assistant|>
Overview:"""
        
        elif depth == 1:
            prompt = f"""<|system|>
You are a helpful assistant that creates clear, concise summaries.
<|user|>
Summarize this section in 1-2 sentences. Capture the main ideas.

Text: {text}
<|assistant|>
Summary:"""
        
        else:
            prompt = f"""<|system|>
You are a helpful assistant that extracts key information.
<|user|>
Extract the most important point from this text in one sentence.

Text: {text}
<|assistant|>
Key point:"""
        
        return prompt
    
    def generate_summary(self, text: str, depth: int = 1, max_length: int = 100) -> str:
        """
        Generate an abstractive summary
        
        Args:
            text: The text to summarize
            depth: Tree depth (0=root, higher=more detailed)
            max_length: Maximum summary length in tokens
        """
        if self.use_lightweight or self.generator is None:
            return self._rule_based_summary(text, depth)
        
        try:
            prompt = self._create_prompt(text, depth)
            
            result = self.generator(
                prompt,
                max_new_tokens=max_length,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            generated = result[0]['generated_text']
            summary = generated[len(prompt):].strip()
            summary = self._clean_summary(summary)
            
            return summary
            
        except Exception as e:
            print(f"⚠️  Generation error: {e}")
            return self._rule_based_summary(text, depth)
    
    def _rule_based_summary(self, text: str, depth: int) -> str:
        """
        Rule-based summarization fallback
        
        Extracts key sentences based on position and content
        """
        import re
        from collections import Counter
        
        # Clean text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not sentences:
            return text[:200] + "..." if len(text) > 200 else text
        
        # Score sentences
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'have', 'from',
                    'are', 'was', 'were', 'has', 'had', 'but', 'not', 'you', 'your',
                    'they', 'their', 'there', 'what', 'which', 'when', 'where', 'who',
                    'published', 'conference', 'paper', 'iclr'}
        
        # Count word frequencies
        all_words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        word_freq = Counter([w for w in all_words if w not in stopwords])
        
        # Score each sentence
        scored_sentences = []
        for i, sent in enumerate(sentences):
            words = re.findall(r'\b[a-z]{3,}\b', sent.lower())
            score = sum(word_freq.get(w, 0) for w in words if w not in stopwords)
            
            # Boost first and second sentences
            if i == 0:
                score *= 1.5
            elif i == 1:
                score *= 1.2
            
            scored_sentences.append((sent, score))
        
        # Sort by score
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select sentences based on depth
        if depth == 0:
            num_sentences = min(3, len(scored_sentences))
        elif depth == 1:
            num_sentences = min(2, len(scored_sentences))
        else:
            num_sentences = 1
        
        # Get top sentences and maintain order
        selected = scored_sentences[:num_sentences]
        selected_texts = [s[0] for s in selected]
        
        # Reorder by original position
        ordered = []
        for sent in sentences:
            if sent in selected_texts:
                ordered.append(sent)
                if len(ordered) >= num_sentences:
                    break
        
        summary = ' '.join(ordered)
        
        # Ensure reasonable length
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        return summary
    
    def _clean_summary(self, summary: str) -> str:
        """Clean up generated summary"""
        import re
        
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Remove leftover prompt fragments
        fragments = ['Summary:', 'Overview:', 'Key point:', 'Key Information:',
                    '<|assistant|>', '<|user|>', '<|system|>']
        for frag in fragments:
            summary = summary.replace(frag, '')
        
        # Ensure ends with punctuation
        summary = summary.strip()
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Limit length
        words = summary.split()
        if len(words) > 100:
            summary = ' '.join(words[:100]) + '...'
        
        return summary.strip()
    
    def batch_summarize(self, texts: List[str], depths: List[int], 
                        batch_size: int = 1) -> List[str]:
        """Generate multiple summaries"""
        print(f"\n📝 Generating {len(texts)} summaries...")
        
        summaries = []
        for i, (text, depth) in enumerate(zip(texts, depths)):
            print(f"   [{i+1}/{len(texts)}] Depth {depth}...", end=" ")
            summary = self.generate_summary(text, depth)
            summaries.append(summary)
            preview = summary[:50] + "..." if len(summary) > 50 else summary
            print(f"Done: {preview}")
        
        print(f"✅ Generated {len(summaries)} summaries")
        return summaries
    
    def test_summarization(self):
        """Test the summarization capability"""
        print("\n🧪 Testing summarization...")
        
        test_text = """
        Retrieval-Augmented Generation (RAG) has become a popular approach for enhancing 
        large language models with external knowledge. However, traditional RAG systems 
        treat documents as flat collections of chunks, ignoring the inherent hierarchical 
        structure of documents. Our approach, RAPTOR, introduces a novel method for 
        document indexing and retrieval. We cluster related chunks using Gaussian Mixture 
        Models, generate abstractive summaries at multiple levels, and build a tree 
        structure that captures document semantics. Experiments show that RAPTOR 
        outperforms traditional RAG by 25-40% on complex queries.
        """
        
        print(f"Test text: {test_text[:100]}...")
        
        for depth in [0, 1, 2]:
            summary = self.generate_summary(test_text, depth)
            print(f"\n   Depth {depth}: {summary}")
        
        print("\n✅ Summarization test complete!")
        return True
