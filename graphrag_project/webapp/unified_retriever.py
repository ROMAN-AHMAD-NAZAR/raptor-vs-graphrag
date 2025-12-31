# webapp/unified_retriever.py
"""
Unified Retriever for RAPTOR and GraphRAG
Provides a common interface for both retrieval systems
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import numpy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class RetrievalResult:
    """Standard result format for both systems"""
    rank: int
    content: str
    score: float
    source: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'rank': self.rank,
            'content': self.content,
            'score': float(self.score),
            'source': self.source,
            'metadata': self.metadata
        }


@dataclass
class SystemMetrics:
    """Metrics for a retrieval system"""
    query_time_ms: float
    num_results: int
    avg_score: float
    max_score: float
    min_score: float
    coverage: float
    
    def to_dict(self) -> Dict:
        return {
            'query_time_ms': round(self.query_time_ms, 2),
            'num_results': self.num_results,
            'avg_score': round(self.avg_score, 4),
            'max_score': round(self.max_score, 4),
            'min_score': round(self.min_score, 4),
            'coverage': round(self.coverage, 4)
        }


class RAPTORRetriever:
    """
    RAPTOR retrieval system wrapper
    Implements hierarchical tree-based retrieval
    """
    
    def __init__(self, tree_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.tree_path = tree_path
        self.tree_data = None
        self.chunks = []
        self.summaries = []
        self.embeddings_cache = {}
        
        # Lazy load model
        self._model = None
        self.has_model = False
        
        # Load existing tree if available
        if tree_path:
            self._load_tree(tree_path)
    
    @property
    def model(self):
        """Lazy load the sentence transformer model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self.has_model = True
            except ImportError:
                self.has_model = False
                self.logger.warning("sentence-transformers not available for RAPTOR")
        return self._model
    
    def _load_tree(self, path: str):
        """Load RAPTOR tree from file"""
        try:
            tree_file = Path(path)
            if tree_file.exists():
                with open(tree_file, 'r', encoding='utf-8') as f:
                    self.tree_data = json.load(f)
                self._extract_nodes()
                self.logger.info(f"Loaded RAPTOR tree with {len(self.chunks)} chunks")
        except Exception as e:
            self.logger.error(f"Failed to load RAPTOR tree: {e}")
    
    def _extract_nodes(self):
        """Extract chunks and summaries from tree"""
        if not self.tree_data:
            return
        
        # Extract from different possible tree structures
        if 'nodes' in self.tree_data:
            for node in self.tree_data['nodes']:
                if node.get('type') == 'chunk':
                    self.chunks.append(node.get('content', node.get('text', '')))
                elif node.get('type') == 'summary':
                    self.summaries.append(node.get('content', node.get('text', '')))
        
        if 'chunks' in self.tree_data:
            self.chunks.extend(self.tree_data['chunks'])
        
        if 'summaries' in self.tree_data:
            self.summaries.extend(self.tree_data['summaries'])
    
    def index_document(self, text: str, chunk_size: int = 500):
        """Index a document for RAPTOR retrieval"""
        # Split into chunks
        self.chunks = self._chunk_text(text, chunk_size)
        
        # Create hierarchical summaries (simplified)
        self.summaries = self._create_summaries(self.chunks)
        
        # Cache embeddings
        if self.has_model:
            all_texts = self.chunks + self.summaries
            embeddings = self.model.encode(all_texts, normalize_embeddings=True)
            for i, text in enumerate(all_texts):
                self.embeddings_cache[text] = embeddings[i]
        
        self.logger.info(f"Indexed {len(self.chunks)} chunks and {len(self.summaries)} summaries")
        return len(self.chunks)
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 5):  # ~5 chars per word average
            chunk_words = words[i:i + chunk_size // 5]
            if chunk_words:
                chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def _create_summaries(self, chunks: List[str]) -> List[str]:
        """Create hierarchical summaries (simplified version)"""
        summaries = []
        
        # Level 1: Combine every 3 chunks
        for i in range(0, len(chunks), 3):
            group = chunks[i:i+3]
            summary = f"Summary of sections {i+1}-{min(i+3, len(chunks))}: " + " ".join(group)[:300] + "..."
            summaries.append(summary)
        
        # Level 2: Combine level 1 summaries
        if len(summaries) > 3:
            level2_summaries = []
            for i in range(0, len(summaries), 3):
                group = summaries[i:i+3]
                summary = f"High-level summary: " + " ".join(group)[:200] + "..."
                level2_summaries.append(summary)
            summaries.extend(level2_summaries)
        
        return summaries
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant content using RAPTOR"""
        start_time = time.time()
        
        if not self.has_model:
            # Fallback to keyword matching
            return self._keyword_retrieve(query, top_k)
        
        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # Search through all texts (chunks + summaries)
        all_texts = self.chunks + self.summaries
        results = []
        
        for i, text in enumerate(all_texts):
            if text in self.embeddings_cache:
                text_embedding = self.embeddings_cache[text]
            else:
                text_embedding = self.model.encode(text, normalize_embeddings=True)
                self.embeddings_cache[text] = text_embedding
            
            # Calculate cosine similarity
            similarity = float(np.dot(query_embedding, text_embedding))
            
            source = "chunk" if i < len(self.chunks) else "summary"
            results.append({
                'text': text,
                'score': similarity,
                'source': source,
                'level': 0 if source == 'chunk' else 1
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert to RetrievalResult
        final_results = []
        for i, r in enumerate(results[:top_k]):
            final_results.append(RetrievalResult(
                rank=i + 1,
                content=r['text'][:500],  # Truncate for display
                score=r['score'],
                source=f"RAPTOR ({r['source']})",
                metadata={'level': r['level']}
            ))
        
        return final_results
    
    def _keyword_retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Fallback keyword-based retrieval"""
        query_terms = set(query.lower().split())
        all_texts = self.chunks + self.summaries
        
        results = []
        for i, text in enumerate(all_texts):
            text_terms = set(text.lower().split())
            overlap = len(query_terms & text_terms)
            score = overlap / len(query_terms) if query_terms else 0
            
            source = "chunk" if i < len(self.chunks) else "summary"
            results.append({
                'text': text,
                'score': score,
                'source': source
            })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        final_results = []
        for i, r in enumerate(results[:top_k]):
            final_results.append(RetrievalResult(
                rank=i + 1,
                content=r['text'][:500],
                score=r['score'],
                source=f"RAPTOR ({r['source']})",
                metadata={}
            ))
        
        return final_results


class GraphRAGRetriever:
    """
    GraphRAG retrieval system wrapper
    Implements knowledge graph-based retrieval
    """
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        self.logger = logging.getLogger(__name__)
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.driver = None
        self.entities = []
        self.relationships = []
        self.chunks = []
        self.embeddings_cache = {}
        
        # Lazy load model
        self._model = None
        self.has_model = False
        
        # Try to connect to Neo4j
        if neo4j_uri:
            self._connect_neo4j()
    
    @property
    def model(self):
        """Lazy load the sentence transformer model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
                self.has_model = True
            except ImportError:
                self.has_model = False
                self.logger.warning("sentence-transformers not available for GraphRAG")
        return self._model
    
    def _connect_neo4j(self):
        """Connect to Neo4j database"""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            self.driver.verify_connectivity()
            self.logger.info("Connected to Neo4j")
        except Exception as e:
            self.logger.warning(f"Could not connect to Neo4j: {e}")
            self.driver = None
    
    def index_document(self, text: str, chunk_size: int = 500):
        """Index a document for GraphRAG retrieval"""
        # Split into chunks
        self.chunks = self._chunk_text(text, chunk_size)
        
        # Extract entities (simplified NER)
        self.entities = self._extract_entities(text)
        
        # Extract relationships (simplified)
        self.relationships = self._extract_relationships(text, self.entities)
        
        # Cache embeddings for entities
        if self.has_model:
            entity_texts = [e['name'] + ' ' + e.get('type', '') for e in self.entities]
            if entity_texts:
                embeddings = self.model.encode(entity_texts, normalize_embeddings=True)
                for i, e in enumerate(self.entities):
                    self.embeddings_cache[e['name']] = embeddings[i]
            
            # Also embed chunks
            chunk_embeddings = self.model.encode(self.chunks, normalize_embeddings=True)
            for i, chunk in enumerate(self.chunks):
                self.embeddings_cache[f"chunk_{i}"] = chunk_embeddings[i]
        
        self.logger.info(f"Indexed {len(self.entities)} entities and {len(self.relationships)} relationships")
        return len(self.entities)
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 5):
            chunk_words = words[i:i + chunk_size // 5]
            if chunk_words:
                chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text (simplified NER)"""
        entities = []
        
        # Common entity patterns
        import re
        
        # Technical terms (capitalized phrases)
        cap_pattern = r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b'
        matches = re.findall(cap_pattern, text)
        
        seen = set()
        for match in matches:
            if len(match) > 2 and match not in seen:
                seen.add(match)
                entities.append({
                    'name': match,
                    'type': 'CONCEPT',
                    'confidence': 0.7
                })
        
        # Look for specific patterns
        patterns = {
            'METHOD': r'\b(algorithm|method|approach|technique|model|system)\b',
            'METRIC': r'\b(accuracy|precision|recall|F1|NDCG|score|rate)\b',
            'DATASET': r'\b(dataset|corpus|benchmark|collection)\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.lower() not in [e['name'].lower() for e in entities]:
                    entities.append({
                        'name': match.capitalize(),
                        'type': entity_type,
                        'confidence': 0.6
                    })
        
        return entities[:50]  # Limit entities
    
    def _extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict]:
        """Extract relationships between entities (simplified)"""
        relationships = []
        
        entity_names = [e['name'].lower() for e in entities]
        sentences = text.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            found_entities = []
            
            for i, name in enumerate(entity_names):
                if name in sentence_lower:
                    found_entities.append(entities[i])
            
            # Create relationships between co-occurring entities
            for i, e1 in enumerate(found_entities):
                for e2 in found_entities[i+1:]:
                    relationships.append({
                        'source': e1['name'],
                        'target': e2['name'],
                        'type': 'RELATED_TO',
                        'confidence': 0.5
                    })
        
        return relationships[:100]  # Limit relationships
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant content using GraphRAG"""
        if not self.has_model:
            return self._keyword_retrieve(query, top_k)
        
        # Encode query
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        results = []
        
        # Search entities
        for entity in self.entities:
            if entity['name'] in self.embeddings_cache:
                entity_embedding = self.embeddings_cache[entity['name']]
                similarity = float(np.dot(query_embedding, entity_embedding))
                
                # Get related entities
                related = [r for r in self.relationships 
                          if r['source'] == entity['name'] or r['target'] == entity['name']]
                
                context = f"Entity: {entity['name']} ({entity['type']})"
                if related:
                    relations_str = ", ".join([
                        f"{r['source']} -> {r['target']}" for r in related[:3]
                    ])
                    context += f"\nRelationships: {relations_str}"
                
                results.append({
                    'content': context,
                    'score': similarity,
                    'source': 'entity',
                    'entity': entity
                })
        
        # Also search chunks
        for i, chunk in enumerate(self.chunks):
            cache_key = f"chunk_{i}"
            if cache_key in self.embeddings_cache:
                chunk_embedding = self.embeddings_cache[cache_key]
                similarity = float(np.dot(query_embedding, chunk_embedding))
                
                results.append({
                    'content': chunk[:500],
                    'score': similarity * 0.8,  # Slightly lower weight for chunks
                    'source': 'chunk',
                    'entity': None
                })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert to RetrievalResult
        final_results = []
        for i, r in enumerate(results[:top_k]):
            final_results.append(RetrievalResult(
                rank=i + 1,
                content=r['content'],
                score=r['score'],
                source=f"GraphRAG ({r['source']})",
                metadata={'entity': r.get('entity')}
            ))
        
        return final_results
    
    def _keyword_retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Fallback keyword-based retrieval"""
        query_terms = set(query.lower().split())
        results = []
        
        # Search entities
        for entity in self.entities:
            entity_terms = set(entity['name'].lower().split())
            overlap = len(query_terms & entity_terms)
            score = overlap / len(query_terms) if query_terms else 0
            
            if score > 0:
                results.append({
                    'content': f"Entity: {entity['name']} ({entity['type']})",
                    'score': score,
                    'source': 'entity'
                })
        
        # Search chunks
        for chunk in self.chunks:
            chunk_terms = set(chunk.lower().split())
            overlap = len(query_terms & chunk_terms)
            score = (overlap / len(query_terms)) * 0.8 if query_terms else 0
            
            if score > 0:
                results.append({
                    'content': chunk[:500],
                    'score': score,
                    'source': 'chunk'
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        final_results = []
        for i, r in enumerate(results[:top_k]):
            final_results.append(RetrievalResult(
                rank=i + 1,
                content=r['content'],
                score=r['score'],
                source=f"GraphRAG ({r['source']})",
                metadata={}
            ))
        
        return final_results


class UnifiedRetriever:
    """
    Unified interface for comparing RAPTOR and GraphRAG
    """
    
    def __init__(self, config: Dict = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize both retrievers
        self.raptor = RAPTORRetriever(
            tree_path=self.config.get('raptor_tree_path')
        )
        
        self.graphrag = GraphRAGRetriever(
            neo4j_uri=self.config.get('neo4j_uri'),
            neo4j_user=self.config.get('neo4j_user'),
            neo4j_password=self.config.get('neo4j_password')
        )
        
        self.document_hash = None
        self.logger.info("UnifiedRetriever initialized")
    
    def index_document(self, text: str, chunk_size: int = 500) -> Dict:
        """Index document in both systems"""
        # Create hash to check if document changed
        doc_hash = hashlib.md5(text.encode()).hexdigest()
        
        if doc_hash == self.document_hash:
            return {
                'status': 'already_indexed',
                'raptor': {'chunks': len(self.raptor.chunks), 'summaries': len(self.raptor.summaries)},
                'graphrag': {'entities': len(self.graphrag.entities), 'relationships': len(self.graphrag.relationships)}
            }
        
        self.document_hash = doc_hash
        
        # Index in RAPTOR
        raptor_chunks = self.raptor.index_document(text, chunk_size)
        
        # Index in GraphRAG
        graphrag_entities = self.graphrag.index_document(text, chunk_size)
        
        return {
            'status': 'indexed',
            'raptor': {'chunks': raptor_chunks, 'summaries': len(self.raptor.summaries)},
            'graphrag': {'entities': graphrag_entities, 'relationships': len(self.graphrag.relationships)}
        }
    
    def compare_retrieval(self, query: str, top_k: int = 5) -> Dict:
        """
        Run retrieval on both systems and compare results
        """
        comparison = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'raptor': {},
            'graphrag': {},
            'comparison': {}
        }
        
        # RAPTOR retrieval
        start_time = time.time()
        raptor_results = self.raptor.retrieve(query, top_k)
        raptor_time = (time.time() - start_time) * 1000
        
        comparison['raptor'] = {
            'results': [r.to_dict() for r in raptor_results],
            'metrics': self._calculate_metrics(raptor_results, raptor_time)
        }
        
        # GraphRAG retrieval
        start_time = time.time()
        graphrag_results = self.graphrag.retrieve(query, top_k)
        graphrag_time = (time.time() - start_time) * 1000
        
        comparison['graphrag'] = {
            'results': [r.to_dict() for r in graphrag_results],
            'metrics': self._calculate_metrics(graphrag_results, graphrag_time)
        }
        
        # Compare systems
        comparison['comparison'] = self._compare_results(
            comparison['raptor'],
            comparison['graphrag']
        )
        
        return comparison
    
    def _calculate_metrics(self, results: List[RetrievalResult], query_time: float) -> Dict:
        """Calculate metrics for retrieval results"""
        if not results:
            return SystemMetrics(
                query_time_ms=query_time,
                num_results=0,
                avg_score=0,
                max_score=0,
                min_score=0,
                coverage=0
            ).to_dict()
        
        scores = [r.score for r in results]
        
        return SystemMetrics(
            query_time_ms=query_time,
            num_results=len(results),
            avg_score=sum(scores) / len(scores),
            max_score=max(scores),
            min_score=min(scores),
            coverage=len([s for s in scores if s > 0.3]) / len(scores)
        ).to_dict()
    
    def _compare_results(self, raptor_data: Dict, graphrag_data: Dict) -> Dict:
        """Compare results from both systems"""
        raptor_metrics = raptor_data['metrics']
        graphrag_metrics = graphrag_data['metrics']
        
        comparison = {
            'speed_winner': 'RAPTOR' if raptor_metrics['query_time_ms'] < graphrag_metrics['query_time_ms'] else 'GraphRAG',
            'speed_difference_ms': abs(raptor_metrics['query_time_ms'] - graphrag_metrics['query_time_ms']),
            'score_winner': 'RAPTOR' if raptor_metrics['avg_score'] > graphrag_metrics['avg_score'] else 'GraphRAG',
            'score_difference': abs(raptor_metrics['avg_score'] - graphrag_metrics['avg_score']),
            'coverage_winner': 'RAPTOR' if raptor_metrics['coverage'] > graphrag_metrics['coverage'] else 'GraphRAG',
            'raptor_improvement': {
                'vs_baseline_speed': 0,  # Placeholder
                'vs_baseline_score': 0
            },
            'graphrag_improvement': {
                'vs_baseline_speed': 0,
                'vs_baseline_score': 0
            }
        }
        
        # Calculate improvements
        if graphrag_metrics['query_time_ms'] > 0:
            comparison['raptor_improvement']['vs_graphrag_speed'] = round(
                ((graphrag_metrics['query_time_ms'] - raptor_metrics['query_time_ms']) / 
                 graphrag_metrics['query_time_ms']) * 100, 2
            )
        
        if raptor_metrics['avg_score'] > 0:
            comparison['graphrag_improvement']['vs_raptor_score'] = round(
                ((graphrag_metrics['avg_score'] - raptor_metrics['avg_score']) / 
                 raptor_metrics['avg_score']) * 100, 2
            )
        
        # Overall recommendation
        raptor_wins = sum([
            1 if comparison['speed_winner'] == 'RAPTOR' else 0,
            1 if comparison['score_winner'] == 'RAPTOR' else 0,
            1 if comparison['coverage_winner'] == 'RAPTOR' else 0
        ])
        
        comparison['overall_winner'] = 'RAPTOR' if raptor_wins >= 2 else 'GraphRAG'
        comparison['recommendation'] = self._generate_recommendation(comparison)
        
        return comparison
    
    def _generate_recommendation(self, comparison: Dict) -> str:
        """Generate recommendation based on comparison"""
        winner = comparison['overall_winner']
        
        if winner == 'RAPTOR':
            return (
                f"RAPTOR performed better overall. It excels at providing "
                f"comprehensive context through hierarchical summarization. "
                f"Best for complex reasoning queries."
            )
        else:
            return (
                f"GraphRAG performed better overall. It excels at precise "
                f"entity retrieval and relationship queries. "
                f"Best for fact-finding and entity-focused queries."
            )
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            'raptor': {
                'chunks_indexed': len(self.raptor.chunks),
                'summaries_created': len(self.raptor.summaries),
                'has_embeddings': self.raptor.has_model
            },
            'graphrag': {
                'entities_extracted': len(self.graphrag.entities),
                'relationships_found': len(self.graphrag.relationships),
                'chunks_indexed': len(self.graphrag.chunks),
                'has_embeddings': self.graphrag.has_model
            },
            'document_indexed': self.document_hash is not None
        }
