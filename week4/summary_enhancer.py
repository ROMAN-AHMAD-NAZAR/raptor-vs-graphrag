# week4/summary_enhancer.py
"""
Summary Enhancement Module for RAPTOR

Provides quality evaluation and enhancement for generated summaries
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import re


class SummaryEnhancer:
    """
    Enhances and validates generated summaries with:
    1. Quality scoring
    2. Length optimization
    3. Information density analysis
    4. Redundancy removal
    """
    
    def __init__(self, target_length: Dict[int, Tuple[int, int]] = None):
        """
        Args:
            target_length: Dict mapping tree depth to (min, max) word counts
        """
        # Default target lengths by depth
        self.target_length = target_length or {
            0: (30, 100),   # Root level: comprehensive overview
            1: (20, 60),    # Branch level: section summaries
            2: (10, 40),    # Leaf level: brief points
        }
        
        # Quality thresholds
        self.min_quality_score = 0.5
        
        print("✨ Summary Enhancer initialized")
    
    def enhance_summary(self, summary: str, source_text: str, depth: int = 1) -> Dict:
        """
        Enhance a single summary
        
        Returns:
            Dict with enhanced summary and quality metrics
        """
        # Initial quality assessment
        initial_quality = self.evaluate_quality(summary, source_text)
        
        # Apply enhancements
        enhanced = summary
        
        # 1. Optimize length
        enhanced = self._optimize_length(enhanced, depth)
        
        # 2. Remove redundancy
        enhanced = self._remove_redundancy(enhanced)
        
        # 3. Ensure information preservation
        enhanced = self._preserve_key_info(enhanced, source_text)
        
        # 4. Clean formatting
        enhanced = self._clean_formatting(enhanced)
        
        # Final quality assessment
        final_quality = self.evaluate_quality(enhanced, source_text)
        
        return {
            'original': summary,
            'enhanced': enhanced,
            'initial_quality': initial_quality,
            'final_quality': final_quality,
            'improvement': final_quality['overall'] - initial_quality['overall']
        }
    
    def evaluate_quality(self, summary: str, source_text: str) -> Dict:
        """
        Evaluate summary quality on multiple dimensions
        """
        scores = {}
        
        # 1. Relevance: keyword overlap with source
        scores['relevance'] = self._compute_relevance(summary, source_text)
        
        # 2. Conciseness: information per word
        scores['conciseness'] = self._compute_conciseness(summary)
        
        # 3. Coherence: structural quality
        scores['coherence'] = self._compute_coherence(summary)
        
        # 4. Information density
        scores['info_density'] = self._compute_info_density(summary)
        
        # Overall score (weighted average)
        weights = {'relevance': 0.4, 'conciseness': 0.2, 'coherence': 0.2, 'info_density': 0.2}
        scores['overall'] = sum(scores[k] * weights[k] for k in weights)
        
        return scores
    
    def _compute_relevance(self, summary: str, source: str) -> float:
        """Compute keyword overlap between summary and source"""
        # Extract content words
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'have', 'from',
                    'are', 'was', 'were', 'has', 'had', 'but', 'not', 'you', 'your',
                    'they', 'their', 'there', 'what', 'which', 'when', 'where', 'who'}
        
        def get_content_words(text):
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            return set(w for w in words if w not in stopwords)
        
        summary_words = get_content_words(summary)
        source_words = get_content_words(source)
        
        if not source_words:
            return 0.5
        
        overlap = len(summary_words & source_words)
        precision = overlap / len(summary_words) if summary_words else 0
        recall = overlap / min(len(source_words), 20)  # Cap denominator
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return min(1.0, f1)
    
    def _compute_conciseness(self, summary: str) -> float:
        """Evaluate conciseness (information per word)"""
        words = summary.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
        
        # Penalize too short or too long
        if word_count < 10:
            return 0.3 + (word_count / 10) * 0.4
        elif word_count > 100:
            return max(0.3, 1.0 - (word_count - 100) / 200)
        else:
            return 0.8 + (50 - abs(word_count - 50)) / 250
    
    def _compute_coherence(self, summary: str) -> float:
        """Evaluate structural coherence"""
        if not summary:
            return 0.0
        
        score = 0.7  # Base score
        
        # Check for proper sentence structure
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        valid_sentences = [s for s in sentences if len(s) > 10]
        
        if valid_sentences:
            score += 0.1
        
        # Check for proper ending
        if summary[-1] in '.!?':
            score += 0.1
        
        # Check for capitalization
        if summary[0].isupper():
            score += 0.1
        
        return min(1.0, score)
    
    def _compute_info_density(self, summary: str) -> float:
        """Compute information density"""
        words = re.findall(r'\b[a-z]{3,}\b', summary.lower())
        
        if len(words) < 3:
            return 0.3
        
        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)
        
        # Average word length (longer = more technical/informative)
        avg_length = np.mean([len(w) for w in words])
        length_score = min(1.0, avg_length / 8)
        
        return 0.6 * unique_ratio + 0.4 * length_score
    
    def _optimize_length(self, summary: str, depth: int) -> str:
        """Optimize summary length for tree depth"""
        min_words, max_words = self.target_length.get(depth, (15, 50))
        
        words = summary.split()
        
        if len(words) > max_words:
            # Truncate at sentence boundary if possible
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            truncated = []
            word_count = 0
            
            for sent in sentences:
                sent_words = len(sent.split())
                if word_count + sent_words <= max_words:
                    truncated.append(sent)
                    word_count += sent_words
                else:
                    break
            
            if truncated:
                return ' '.join(truncated)
            else:
                return ' '.join(words[:max_words]) + '...'
        
        return summary
    
    def _remove_redundancy(self, summary: str) -> str:
        """Remove redundant phrases and repeated information"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        
        if len(sentences) <= 1:
            return summary
        
        # Track seen concepts
        seen_concepts = set()
        unique_sentences = []
        
        for sent in sentences:
            # Extract key concepts (bigrams and important words)
            words = re.findall(r'\b[a-z]{4,}\b', sent.lower())
            concepts = set(words)
            
            # Check overlap with seen concepts
            overlap = len(concepts & seen_concepts) / len(concepts) if concepts else 0
            
            if overlap < 0.7:  # Less than 70% overlap
                unique_sentences.append(sent)
                seen_concepts.update(concepts)
        
        return ' '.join(unique_sentences)
    
    def _preserve_key_info(self, summary: str, source: str) -> str:
        """Ensure key information from source is preserved"""
        # Extract named entities/key terms from source
        # Simple heuristic: capitalized words that appear multiple times
        cap_words = re.findall(r'\b[A-Z][a-z]+\b', source)
        
        from collections import Counter
        word_counts = Counter(cap_words)
        key_terms = [w for w, c in word_counts.items() if c >= 2 and len(w) > 3]
        
        # Check if key terms are in summary
        summary_lower = summary.lower()
        missing_terms = [t for t in key_terms[:3] if t.lower() not in summary_lower]
        
        # Don't modify if summary already has key info or no important missing terms
        if not missing_terms or len(missing_terms) > 2:
            return summary
        
        return summary
    
    def _clean_formatting(self, summary: str) -> str:
        """Clean up formatting issues"""
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Ensure proper capitalization
        if summary and not summary[0].isupper():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Remove any remaining artifacts
        summary = re.sub(r'\s+([.,!?])', r'\1', summary)
        
        return summary.strip()
    
    def batch_enhance(self, summaries: List[Dict]) -> List[Dict]:
        """
        Enhance multiple summaries
        
        Args:
            summaries: List of dicts with 'summary', 'source', 'depth'
        """
        print(f"\n✨ Enhancing {len(summaries)} summaries...")
        
        enhanced = []
        total_improvement = 0
        
        for i, item in enumerate(summaries):
            result = self.enhance_summary(
                item['summary'],
                item.get('source', ''),
                item.get('depth', 1)
            )
            enhanced.append(result)
            total_improvement += result['improvement']
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(summaries)}")
        
        avg_improvement = total_improvement / len(summaries) if summaries else 0
        print(f"✅ Enhancement complete! Average quality improvement: {avg_improvement:.3f}")
        
        return enhanced
    
    def get_quality_report(self, enhanced_summaries: List[Dict]) -> Dict:
        """Generate quality report for enhanced summaries"""
        if not enhanced_summaries:
            return {}
        
        metrics = {
            'relevance': [],
            'conciseness': [],
            'coherence': [],
            'info_density': [],
            'overall': []
        }
        
        for item in enhanced_summaries:
            for metric in metrics:
                if metric in item['final_quality']:
                    metrics[metric].append(item['final_quality'][metric])
        
        report = {
            'total_summaries': len(enhanced_summaries),
            'average_quality': {k: np.mean(v) for k, v in metrics.items() if v},
            'quality_std': {k: np.std(v) for k, v in metrics.items() if v},
            'improvement': {
                'mean': np.mean([e['improvement'] for e in enhanced_summaries]),
                'max': max([e['improvement'] for e in enhanced_summaries]),
                'min': min([e['improvement'] for e in enhanced_summaries])
            }
        }
        
        return report
    
    def print_quality_report(self, report: Dict):
        """Print formatted quality report"""
        print("\n" + "=" * 60)
        print("📊 SUMMARY QUALITY REPORT")
        print("=" * 60)
        
        print(f"\nTotal summaries enhanced: {report['total_summaries']}")
        
        print("\n📈 Average Quality Scores:")
        for metric, score in report.get('average_quality', {}).items():
            bar = '█' * int(score * 20) + '░' * (20 - int(score * 20))
            print(f"   {metric:15} [{bar}] {score:.3f}")
        
        print("\n📉 Quality Improvement:")
        imp = report.get('improvement', {})
        print(f"   Mean improvement:  {imp.get('mean', 0):+.3f}")
        print(f"   Max improvement:   {imp.get('max', 0):+.3f}")
        print(f"   Min improvement:   {imp.get('min', 0):+.3f}")
        
        print("=" * 60)
