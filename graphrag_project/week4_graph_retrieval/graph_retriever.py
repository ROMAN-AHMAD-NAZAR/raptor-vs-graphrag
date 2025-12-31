# week4_graph_retrieval/graph_retriever.py
"""
Core Graph-Based Retrieval Engine for GraphRAG
Implements semantic, graph traversal, and hybrid retrieval strategies
"""

from typing import List, Dict, Tuple, Any, Optional
import logging
from collections import defaultdict, deque
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RetrievalResult:
    """Structured result from graph retrieval"""
    node_id: str
    node_name: str
    node_type: str
    score: float
    retrieval_method: str
    context: str
    evidence: List[str] = field(default_factory=list)
    path: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'node_name': self.node_name,
            'node_type': self.node_type,
            'score': float(self.score),
            'retrieval_method': self.retrieval_method,
            'context': self.context,
            'evidence': self.evidence,
            'path': self.path
        }


class GraphRetriever:
    """
    Core graph-based retrieval engine
    Implements multiple retrieval strategies:
    1. Semantic similarity search
    2. Graph traversal search
    3. Hybrid approach
    """
    
    def __init__(self, neo4j_manager, embedding_manager):
        self.neo4j = neo4j_manager
        self.embedder = embedding_manager
        self.logger = logging.getLogger(__name__)
        
        # Cache for performance
        self._cached_entities = None
        self.relationship_cache = {}
        
        self.logger.info("✅ GraphRetriever initialized")
    
    def retrieve(self, query: str, top_k: int = 10, 
                strategy: str = "hybrid") -> List[RetrievalResult]:
        """
        Main retrieval method with multiple strategies
        
        Args:
            query: The search query
            top_k: Number of results to return
            strategy: One of "semantic", "graph", or "hybrid"
        
        Returns:
            List of RetrievalResult objects
        """
        self.logger.info(f"🔍 Retrieving for query: '{query[:50]}...'")
        self.logger.info(f"   Strategy: {strategy}, Top-K: {top_k}")
        
        if strategy == "semantic":
            return self._semantic_retrieval(query, top_k)
        elif strategy == "graph":
            return self._graph_traversal_retrieval(query, top_k)
        elif strategy == "hybrid":
            return self._hybrid_retrieval(query, top_k)
        else:
            self.logger.warning(f"Unknown strategy: {strategy}, using hybrid")
            return self._hybrid_retrieval(query, top_k)
    
    def _semantic_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Retrieve based on semantic similarity to query"""
        self.logger.info("   Using semantic similarity retrieval")
        
        # Get all entity nodes
        nodes = self._get_all_entities()
        
        if not nodes:
            self.logger.warning("No entities found in graph")
            return []
        
        # Create text representations for embedding
        node_texts = []
        node_info = []
        
        for node in nodes:
            node_text = self._create_node_embedding_text(node)
            node_texts.append(node_text)
            node_info.append({
                'id': node['id'],
                'name': node.get('properties', {}).get('name', ''),
                'type': node.get('properties', {}).get('type', '')
            })
        
        # Find similar nodes
        similar_nodes = self.embedder.find_similar_nodes(query, node_texts, top_k * 2)
        
        # Convert to RetrievalResult objects
        results = []
        for idx, score, _ in similar_nodes:
            if idx < len(node_info):
                info = node_info[idx]
                
                # Get context for the node
                context = self._get_node_context(info['id'])
                
                result = RetrievalResult(
                    node_id=info['id'],
                    node_name=info['name'],
                    node_type=info['type'],
                    score=float(score),
                    retrieval_method="semantic_similarity",
                    context=context,
                    evidence=[f"Semantic similarity: {score:.3f}"]
                )
                results.append(result)
        
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
    
    def _graph_traversal_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Retrieve by traversing the graph from seed nodes
        Finds connected concepts related to the query
        """
        self.logger.info("   Using graph traversal retrieval")
        
        # Step 1: Find seed nodes using semantic similarity
        seed_nodes = self._find_seed_nodes(query, max_seeds=5)
        
        if not seed_nodes:
            self.logger.warning("No seed nodes found for traversal")
            return []
        
        # Step 2: Traverse graph from seed nodes
        visited_nodes = set()
        node_scores = defaultdict(float)
        node_paths = defaultdict(list)
        
        for seed_id, seed_score in seed_nodes:
            self._traverse_from_node(
                seed_id, 
                seed_score,
                visited_nodes,
                node_scores,
                node_paths,
                max_depth=3,
                decay_factor=0.7
            )
        
        # Step 3: Convert to results
        results = []
        for node_id, score in sorted(node_scores.items(), key=lambda x: x[1], reverse=True):
            if len(results) >= top_k:
                break
            
            node_info = self._get_node_info(node_id)
            if node_info:
                # Get the best path to this node
                best_path = node_paths.get(node_id, [])
                
                result = RetrievalResult(
                    node_id=node_id,
                    node_name=node_info.get('name', ''),
                    node_type=node_info.get('type', ''),
                    score=float(score),
                    retrieval_method="graph_traversal",
                    context=self._get_node_context(node_id),
                    evidence=[f"Reachable from {len(seed_nodes)} seed nodes"],
                    path=best_path
                )
                results.append(result)
        
        return results
    
    def _hybrid_retrieval(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining semantic and graph approaches
        """
        self.logger.info("   Using hybrid retrieval")
        
        # Get semantic results
        semantic_results = self._semantic_retrieval(query, top_k)
        
        # Get graph traversal results
        graph_results = self._graph_traversal_retrieval(query, top_k)
        
        # Combine and re-rank
        all_results = {}
        
        # Add semantic results
        for result in semantic_results:
            all_results[result.node_id] = {
                'result': result,
                'semantic_score': result.score,
                'graph_score': 0.0
            }
        
        # Add graph results (or update existing)
        for result in graph_results:
            if result.node_id in all_results:
                # Node already found semantically, boost its score
                all_results[result.node_id]['graph_score'] = result.score
                all_results[result.node_id]['result'].score *= 1.2  # Boost
                all_results[result.node_id]['result'].retrieval_method = "hybrid"
                all_results[result.node_id]['result'].evidence.append(
                    f"Also found via graph traversal: {result.score:.3f}"
                )
            else:
                all_results[result.node_id] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'graph_score': result.score
                }
        
        # Calculate combined scores
        combined_results = []
        for data in all_results.values():
            result = data['result']
            
            # Weighted combination
            semantic_weight = 0.6
            graph_weight = 0.4
            
            combined_score = (
                data['semantic_score'] * semantic_weight + 
                data['graph_score'] * graph_weight
            )
            
            result.score = float(combined_score)
            combined_results.append(result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results[:top_k]
    
    def _find_seed_nodes(self, query: str, max_seeds: int = 5) -> List[Tuple[str, float]]:
        """Find starting nodes for graph traversal"""
        nodes = self._get_all_entities()
        
        if not nodes:
            return []
        
        # Get node texts for embedding
        node_texts = []
        node_ids = []
        
        for node in nodes:
            node_text = self._create_node_embedding_text(node)
            node_texts.append(node_text)
            node_ids.append(node['id'])
        
        # Find similar nodes
        similar_indices = self.embedder.find_similar_nodes(
            query, node_texts, max_seeds * 2
        )
        
        seeds = []
        for idx, score, _ in similar_indices:
            if idx < len(node_ids) and score > 0.3:  # Minimum similarity threshold
                seeds.append((node_ids[idx], score))
                if len(seeds) >= max_seeds:
                    break
        
        return seeds
    
    def _traverse_from_node(self, start_id: str, start_score: float,
                           visited_nodes: set, node_scores: dict,
                           node_paths: dict, max_depth: int = 3,
                           decay_factor: float = 0.7):
        """
        Depth-limited graph traversal with score propagation
        """
        queue = deque([(start_id, start_score, 0, [])])  # (node_id, score, depth, path)
        
        while queue:
            current_id, current_score, depth, path = queue.popleft()
            
            if depth > max_depth or current_id in visited_nodes:
                continue
            
            visited_nodes.add(current_id)
            
            # Update node score (keep maximum)
            if current_score > node_scores[current_id]:
                node_scores[current_id] = current_score
                node_paths[current_id] = path.copy()
            
            # Get neighbors
            neighbors = self._get_neighbors(current_id)
            
            for neighbor_id, rel_type, rel_confidence in neighbors:
                if neighbor_id not in visited_nodes:
                    # Calculate propagated score with decay
                    neighbor_score = current_score * decay_factor * rel_confidence
                    
                    # Create new path
                    new_path = path + [{
                        'from': current_id,
                        'to': neighbor_id,
                        'relationship': rel_type,
                        'confidence': rel_confidence
                    }]
                    
                    queue.append((neighbor_id, neighbor_score, depth + 1, new_path))
    
    def _get_all_entities(self) -> List[Dict]:
        """Get all entity nodes from Neo4j"""
        if self._cached_entities is not None:
            return self._cached_entities
        
        query = """
        MATCH (e:Entity)
        RETURN e.id as id, properties(e) as properties
        ORDER BY e.confidence DESC
        LIMIT 1000
        """
        
        try:
            results = self.neo4j.execute_cypher(query)
            entities = [{'id': r['id'], 'properties': r['properties']} 
                       for r in results if r['id']]
            
            # Cache for performance
            self._cached_entities = entities
            self.logger.info(f"   Loaded {len(entities)} entities from graph")
            return entities
            
        except Exception as e:
            self.logger.error(f"Failed to get entities: {e}")
            return []
    
    def _get_node_info(self, node_id: str) -> Optional[Dict]:
        """Get information about a specific node"""
        query = """
        MATCH (e:Entity {id: $node_id})
        RETURN e.name as name, e.type as type, e.confidence as confidence
        """
        
        try:
            results = self.neo4j.execute_cypher(query, {'node_id': node_id})
            if results:
                return results[0]
        except Exception as e:
            self.logger.error(f"Failed to get node info {node_id}: {e}")
        
        return None
    
    def _get_neighbors(self, node_id: str) -> List[Tuple[str, str, float]]:
        """Get neighboring nodes with relationship information"""
        if node_id in self.relationship_cache:
            return self.relationship_cache[node_id]
        
        query = """
        MATCH (e:Entity {id: $node_id})-[r]->(neighbor:Entity)
        RETURN neighbor.id as neighbor_id, 
               type(r) as relationship_type,
               r.confidence as confidence
        UNION
        MATCH (e:Entity {id: $node_id})<-[r]-(neighbor:Entity)
        RETURN neighbor.id as neighbor_id, 
               type(r) as relationship_type,
               r.confidence as confidence
        """
        
        try:
            results = self.neo4j.execute_cypher(query, {'node_id': node_id})
            neighbors = []
            
            for r in results:
                if r.get('neighbor_id'):
                    neighbors.append((
                        r['neighbor_id'],
                        r.get('relationship_type', 'RELATED'),
                        r.get('confidence', 0.5) or 0.5
                    ))
            
            # Cache for performance
            self.relationship_cache[node_id] = neighbors
            return neighbors
            
        except Exception as e:
            self.logger.error(f"Failed to get neighbors for {node_id}: {e}")
            return []
    
    def _create_node_embedding_text(self, node: Dict) -> str:
        """Create text representation of node for embedding"""
        props = node.get('properties', {})
        
        parts = []
        
        # Name is most important
        if props.get('name'):
            parts.append(props['name'])
        
        # Type information
        if props.get('type'):
            parts.append(f"type: {props['type']}")
        
        # Additional context
        if props.get('mention'):
            parts.append(f"context: {props['mention']}")
        
        return " ".join(parts) if parts else str(node.get('id', 'unknown'))
    
    def _get_node_context(self, node_id: str) -> str:
        """Get contextual information about a node"""
        query = """
        MATCH (e:Entity {id: $node_id})
        OPTIONAL MATCH (e)-[r]->(other:Entity)
        RETURN e.name as name, e.type as type,
               collect(DISTINCT {type: type(r), target: other.name})[0..3] as relationships
        """
        
        try:
            results = self.neo4j.execute_cypher(query, {'node_id': node_id})
            if results:
                result = results[0]
                
                context_parts = []
                if result.get('name') and result.get('type'):
                    context_parts.append(f"{result['name']} ({result['type']})")
                
                if result.get('relationships'):
                    for rel in result['relationships']:
                        if rel and rel.get('target'):
                            context_parts.append(f"{rel.get('type', 'RELATED')} {rel['target']}")
                
                return " | ".join(context_parts) if context_parts else ""
                
        except Exception as e:
            self.logger.error(f"Failed to get context for {node_id}: {e}")
        
        return ""
    
    def clear_cache(self):
        """Clear internal caches"""
        self._cached_entities = None
        self.relationship_cache = {}
        self.logger.info("🗑️ GraphRetriever cache cleared")
    
    def explain_retrieval(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate explanation of retrieval process for debugging"""
        explanation = []
        explanation.append(f"Query: '{query}'")
        explanation.append(f"Retrieved {len(results)} results:")
        explanation.append("=" * 60)
        
        for i, result in enumerate(results[:5], 1):
            explanation.append(f"\n{i}. {result.node_name} [{result.node_type}]")
            explanation.append(f"   Score: {result.score:.4f}")
            explanation.append(f"   Method: {result.retrieval_method}")
            
            if result.evidence:
                explanation.append(f"   Evidence:")
                for evidence in result.evidence[:3]:
                    explanation.append(f"     • {evidence}")
            
            if result.path:
                explanation.append(f"   Path to node ({len(result.path)} hops):")
                for step in result.path[:3]:  # Show first 3 steps
                    explanation.append(f"     {step['from']} --[{step['relationship']}]--> {step['to']}")
        
        # Statistics
        if results:
            method_counts = {}
            for result in results:
                method = result.retrieval_method
                method_counts[method] = method_counts.get(method, 0) + 1
            
            explanation.append("\n" + "=" * 60)
            explanation.append("📊 Retrieval Statistics:")
            explanation.append(f"   Total results: {len(results)}")
            
            for method, count in method_counts.items():
                percentage = (count / len(results)) * 100
                explanation.append(f"   {method}: {count} ({percentage:.1f}%)")
            
            avg_score = sum(r.score for r in results) / len(results)
            explanation.append(f"   Average score: {avg_score:.4f}")
        
        return "\n".join(explanation)
