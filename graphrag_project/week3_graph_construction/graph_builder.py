# week3_graph_construction/graph_builder.py
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime

class GraphBuilder:
    """
    Builds knowledge graph from extracted entities and relationships
    """
    
    def __init__(self, neo4j_manager, output_dir: Path):
        self.neo4j = neo4j_manager
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_extraction_results(self, results_path: Path) -> Dict:
        """
        Load entities and relationships from Week 2 results
        """
        self.logger.info(f"📂 Loading extraction results from {results_path}")
        
        try:
            results_path = Path(results_path)
            
            if results_path.suffix == '.pkl':
                with open(results_path, 'rb') as f:
                    data = pickle.load(f)
            elif results_path.suffix == '.json':
                with open(results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {results_path.suffix}")
            
            self.logger.info(f"✅ Loaded {len(data.get('entities', []))} entities "
                           f"and {len(data.get('relationships', []))} relationships")
            return data
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load results: {e}")
            return {}
    
    def build_knowledge_graph(self, extraction_data: Dict) -> Dict:
        """
        Build complete knowledge graph in Neo4j
        Returns construction statistics
        """
        stats = {
            'start_time': datetime.now().isoformat(),
            'entities_processed': 0,
            'relationships_processed': 0,
            'chunks_processed': 0,
            'errors': []
        }
        
        # Step 1: Create schema
        self.logger.info("🏗️  Creating graph schema...")
        schema_stats = self.neo4j.create_schema()
        stats['schema'] = schema_stats
        
        # Step 2: Create entity nodes (batch for performance)
        entities = extraction_data.get('entities', [])
        self.logger.info(f"📝 Creating {len(entities)} entity nodes...")
        
        entity_stats = self.neo4j.batch_create_entities(entities)
        stats['entities_processed'] = entity_stats['created']
        stats['entity_errors'] = entity_stats['errors']
        
        # Step 3: Create relationship nodes
        relationships = extraction_data.get('relationships', [])
        self.logger.info(f"🔗 Creating {len(relationships)} relationships...")
        
        relationship_stats = self.neo4j.batch_create_relationships(relationships)
        stats['relationships_processed'] = relationship_stats['created']
        stats['relationship_errors'] = relationship_stats['errors']
        
        # Step 4: Create chunk nodes and link entities
        chunks = extraction_data.get('chunks', [])
        if chunks:
            self.logger.info(f"📄 Creating {len(chunks)} chunk nodes...")
            
            chunk_stats = self._create_chunk_nodes(chunks, entities)
            stats['chunks_processed'] = chunk_stats['created']
            stats['chunk_errors'] = chunk_stats['errors']
        
        # Step 5: Get final graph statistics
        self.logger.info("📊 Collecting graph statistics...")
        graph_stats = self.neo4j.get_graph_statistics()
        stats['graph_statistics'] = graph_stats
        
        stats['end_time'] = datetime.now().isoformat()
        stats['duration_seconds'] = (
            datetime.fromisoformat(stats['end_time']) - 
            datetime.fromisoformat(stats['start_time'])
        ).total_seconds()
        
        return stats
    
    def _create_chunk_nodes(self, chunks: List[Dict], entities: List[Dict]) -> Dict:
        """
        Create chunk nodes and link entities to them
        """
        stats = {'created': 0, 'errors': []}
        
        # Group entities by chunk_id
        entities_by_chunk = {}
        for entity in entities:
            chunk_id = entity.get('chunk_id', 0)
            if chunk_id not in entities_by_chunk:
                entities_by_chunk[chunk_id] = []
            entities_by_chunk[chunk_id].append(entity)
        
        # Process each chunk
        for chunk in chunks:
            try:
                # Create chunk node
                chunk_id = chunk.get('chunk_id', chunk.get('id', 0))
                chunk_node_id = f"chunk_{chunk_id}"
                
                # Check if chunk already exists
                existing = self.neo4j.execute_cypher(
                    "MATCH (c:Chunk {id: $id}) RETURN c.id",
                    {'id': chunk_node_id}
                )
                
                if not existing:
                    success = self.neo4j.create_chunk_node(chunk)
                    if success:
                        stats['created'] += 1
                
                # Link entities to this chunk
                chunk_entities = entities_by_chunk.get(chunk_id, [])
                for entity in chunk_entities:
                    self.neo4j.link_entity_to_chunk(
                        entity['id'],
                        chunk_node_id
                    )
                    
            except Exception as e:
                stats['errors'].append(str(e))
                self.logger.error(f"Failed to process chunk {chunk.get('id', 'unknown')}: {e}")
        
        return stats
    
    def export_graph_data(self, filename: str = "knowledge_graph.json") -> Path:
        """
        Export graph data for visualization and analysis
        """
        output_path = self.output_dir / filename
        
        self.logger.info(f"💾 Exporting graph to {output_path}...")
        
        # Get comprehensive graph data
        query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.id as node_id,
               labels(n) as labels,
               properties(n) as properties,
               type(r) as rel_type,
               r.id as rel_id,
               properties(r) as rel_properties,
               m.id as target_id
        ORDER BY node_id
        """
        
        try:
            results = self.neo4j.execute_cypher(query)
            
            # Reformat for better structure
            graph_data = {
                'nodes': {},
                'relationships': [],
                'statistics': self.neo4j.get_graph_statistics()
            }
            
            for record in results:
                node_id = record['node_id']
                if node_id and node_id not in graph_data['nodes']:
                    graph_data['nodes'][node_id] = {
                        'id': node_id,
                        'labels': record['labels'],
                        'properties': record['properties']
                    }
                
                if record.get('rel_id'):
                    graph_data['relationships'].append({
                        'id': record['rel_id'],
                        'type': record['rel_type'],
                        'source': record['node_id'],
                        'target': record['target_id'],
                        'properties': record['rel_properties']
                    })
            
            # Convert nodes dict to list
            graph_data['nodes'] = list(graph_data['nodes'].values())
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, default=str)
            
            self.logger.info(f"✅ Graph data exported: {len(graph_data['nodes'])} nodes, "
                           f"{len(graph_data['relationships'])} relationships")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export graph data: {e}")
            return None
    
    def create_research_paper_graph(self) -> Dict:
        """
        Create a sample research paper graph for demonstration
        """
        sample_data = {
            'entities': [
                {
                    'id': 'ent_raptor',
                    'name': 'RAPTOR',
                    'type': 'CONCEPT',
                    'mention': 'RAPTOR',
                    'chunk_id': 0,
                    'confidence': 0.95,
                    'start_pos': 0,
                    'end_pos': 6
                },
                {
                    'id': 'ent_graphrag',
                    'name': 'GraphRAG',
                    'type': 'CONCEPT',
                    'mention': 'GraphRAG',
                    'chunk_id': 0,
                    'confidence': 0.95,
                    'start_pos': 10,
                    'end_pos': 18
                },
                {
                    'id': 'ent_rag',
                    'name': 'Traditional RAG',
                    'type': 'CONCEPT',
                    'mention': 'traditional RAG',
                    'chunk_id': 1,
                    'confidence': 0.85,
                    'start_pos': 20,
                    'end_pos': 35
                },
                {
                    'id': 'ent_hierarchical',
                    'name': 'Hierarchical Clustering',
                    'type': 'METHOD',
                    'mention': 'hierarchical clustering',
                    'chunk_id': 2,
                    'confidence': 0.88,
                    'start_pos': 0,
                    'end_pos': 22
                },
                {
                    'id': 'ent_knowledge_graph',
                    'name': 'Knowledge Graph',
                    'type': 'CONCEPT',
                    'mention': 'knowledge graph',
                    'chunk_id': 2,
                    'confidence': 0.92,
                    'start_pos': 25,
                    'end_pos': 40
                },
                {
                    'id': 'ent_recall',
                    'name': 'Recall@K',
                    'type': 'METRIC',
                    'mention': 'recall at K',
                    'chunk_id': 3,
                    'confidence': 0.90,
                    'start_pos': 0,
                    'end_pos': 11
                },
                {
                    'id': 'ent_precision',
                    'name': 'Precision@K',
                    'type': 'METRIC',
                    'mention': 'precision at K',
                    'chunk_id': 3,
                    'confidence': 0.90,
                    'start_pos': 15,
                    'end_pos': 29
                },
                {
                    'id': 'ent_embeddings',
                    'name': 'Embeddings',
                    'type': 'METHOD',
                    'mention': 'embeddings',
                    'chunk_id': 4,
                    'confidence': 0.88,
                    'start_pos': 0,
                    'end_pos': 10
                },
                {
                    'id': 'ent_llm',
                    'name': 'Large Language Model',
                    'type': 'CONCEPT',
                    'mention': 'LLM',
                    'chunk_id': 4,
                    'confidence': 0.92,
                    'start_pos': 12,
                    'end_pos': 15
                },
                {
                    'id': 'ent_nq_dataset',
                    'name': 'Natural Questions',
                    'type': 'DATASET',
                    'mention': 'NQ dataset',
                    'chunk_id': 5,
                    'confidence': 0.85,
                    'start_pos': 0,
                    'end_pos': 10
                }
            ],
            'relationships': [
                {
                    'id': 'rel_comp1',
                    'source_id': 'ent_raptor',
                    'target_id': 'ent_graphrag',
                    'type': 'COMPARES',
                    'mention': 'RAPTOR compared to GraphRAG',
                    'chunk_id': 0,
                    'confidence': 0.75
                },
                {
                    'id': 'rel_out1',
                    'source_id': 'ent_raptor',
                    'target_id': 'ent_rag',
                    'type': 'OUTPERFORMS',
                    'mention': 'RAPTOR outperforms traditional RAG',
                    'chunk_id': 1,
                    'confidence': 0.82
                },
                {
                    'id': 'rel_out2',
                    'source_id': 'ent_graphrag',
                    'target_id': 'ent_rag',
                    'type': 'OUTPERFORMS',
                    'mention': 'GraphRAG outperforms traditional RAG',
                    'chunk_id': 1,
                    'confidence': 0.80
                },
                {
                    'id': 'rel_uses1',
                    'source_id': 'ent_raptor',
                    'target_id': 'ent_hierarchical',
                    'type': 'USES',
                    'mention': 'RAPTOR uses hierarchical clustering',
                    'chunk_id': 2,
                    'confidence': 0.90
                },
                {
                    'id': 'rel_uses2',
                    'source_id': 'ent_graphrag',
                    'target_id': 'ent_knowledge_graph',
                    'type': 'USES',
                    'mention': 'GraphRAG uses knowledge graph',
                    'chunk_id': 2,
                    'confidence': 0.92
                },
                {
                    'id': 'rel_eval1',
                    'source_id': 'ent_raptor',
                    'target_id': 'ent_recall',
                    'type': 'EVALUATED_ON',
                    'mention': 'evaluated on recall',
                    'chunk_id': 3,
                    'confidence': 0.78
                },
                {
                    'id': 'rel_eval2',
                    'source_id': 'ent_graphrag',
                    'target_id': 'ent_precision',
                    'type': 'EVALUATED_ON',
                    'mention': 'evaluated on precision',
                    'chunk_id': 3,
                    'confidence': 0.78
                },
                {
                    'id': 'rel_uses3',
                    'source_id': 'ent_raptor',
                    'target_id': 'ent_embeddings',
                    'type': 'USES',
                    'mention': 'uses embeddings',
                    'chunk_id': 4,
                    'confidence': 0.85
                },
                {
                    'id': 'rel_uses4',
                    'source_id': 'ent_rag',
                    'target_id': 'ent_llm',
                    'type': 'USES',
                    'mention': 'RAG uses LLM',
                    'chunk_id': 4,
                    'confidence': 0.88
                },
                {
                    'id': 'rel_tested1',
                    'source_id': 'ent_raptor',
                    'target_id': 'ent_nq_dataset',
                    'type': 'TESTED_ON',
                    'mention': 'tested on Natural Questions',
                    'chunk_id': 5,
                    'confidence': 0.80
                },
                {
                    'id': 'rel_tested2',
                    'source_id': 'ent_graphrag',
                    'target_id': 'ent_nq_dataset',
                    'type': 'TESTED_ON',
                    'mention': 'tested on Natural Questions',
                    'chunk_id': 5,
                    'confidence': 0.80
                }
            ],
            'chunks': [
                {'chunk_id': 0, 'text': 'RAPTOR and GraphRAG are advanced retrieval augmented generation systems.', 'source': 'research_paper'},
                {'chunk_id': 1, 'text': 'Both systems outperform traditional RAG approaches in complex reasoning tasks.', 'source': 'research_paper'},
                {'chunk_id': 2, 'text': 'RAPTOR uses hierarchical clustering while GraphRAG leverages knowledge graphs.', 'source': 'research_paper'},
                {'chunk_id': 3, 'text': 'Evaluation metrics include recall@K and precision@K measures.', 'source': 'research_paper'},
                {'chunk_id': 4, 'text': 'Embeddings and LLM integration are core components of RAG systems.', 'source': 'research_paper'},
                {'chunk_id': 5, 'text': 'Both methods were tested on Natural Questions dataset.', 'source': 'research_paper'}
            ]
        }
        
        return self.build_knowledge_graph(sample_data)
