# week3_graph_construction/neo4j_manager.py
from neo4j import GraphDatabase, Result
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import json

class Neo4jGraphManager:
    """
    Advanced Neo4j manager for GraphRAG
    Handles graph creation, queries, and management
    """
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j connection
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = None
        self.logger = logging.getLogger(__name__)
        
        self.connect()
    
    def connect(self) -> bool:
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
                database=self.database
            )
            
            # Test connection
            self.driver.verify_connectivity()
            self.logger.info(f"✅ Connected to Neo4j at {self.uri}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Neo4j: {e}")
            return False
    
    def clear_database(self, confirm: bool = False) -> bool:
        """Clear all data from database"""
        if not confirm:
            self.logger.warning("Clear database requires confirmation")
            return False
        
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("🗑️  Database cleared")
                return True
        except Exception as e:
            self.logger.error(f"Failed to clear database: {e}")
            return False
    
    def create_schema(self) -> Dict:
        """
        Create optimized schema for research knowledge graph
        Returns schema statistics
        """
        schema_queries = [
            # Create constraints for uniqueness
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            
            # Create indexes for performance
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            "CREATE INDEX chunk_id_index IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)",
            
            # Create research-specific constraints
            "CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT metric_id_unique IF NOT EXISTS FOR (m:Metric) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
        ]
        
        stats = {}
        try:
            with self.driver.session() as session:
                for query in schema_queries:
                    try:
                        result = session.run(query)
                        stats[query[:50]] = "Created"
                    except Exception as e:
                        stats[query[:50]] = f"Skipped: {str(e)[:50]}"
            
            self.logger.info("✅ Graph schema created")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to create schema: {e}")
            return {}
    
    def create_entity_node(self, entity: Dict) -> bool:
        """
        Create an entity node in Neo4j
        """
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name = $name,
            e.type = $type,
            e.mention = $mention,
            e.chunk_id = $chunk_id,
            e.confidence = $confidence,
            e.start_pos = $start_pos,
            e.end_pos = $end_pos,
            e.created_at = datetime()
        RETURN e.id as entity_id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    id=entity['id'],
                    name=entity['name'],
                    type=entity['type'],
                    mention=entity.get('mention', entity['name']),
                    chunk_id=entity.get('chunk_id', 0),
                    confidence=entity.get('confidence', 0.8),
                    start_pos=entity.get('start_pos', 0),
                    end_pos=entity.get('end_pos', 0)
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to create entity {entity['id']}: {e}")
            return False
    
    def create_relationship(self, relationship: Dict, 
                          source_entity: Dict = None, 
                          target_entity: Dict = None) -> bool:
        """
        Create a relationship between entities
        """
        # Clean relationship type for Neo4j
        rel_type = relationship['type'].replace(" ", "_").replace("-", "_").upper()
        
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{rel_type}]->(target)
        SET r.id = $rel_id,
            r.mention = $mention,
            r.chunk_id = $chunk_id,
            r.confidence = $confidence,
            r.created_at = datetime()
        RETURN r.id as rel_id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    source_id=relationship['source_id'],
                    target_id=relationship['target_id'],
                    rel_id=relationship['id'],
                    mention=relationship.get('mention', ''),
                    chunk_id=relationship.get('chunk_id', 0),
                    confidence=relationship.get('confidence', 0.7)
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to create relationship {relationship['id']}: {e}")
            return False
    
    def create_chunk_node(self, chunk: Dict) -> bool:
        """
        Create a chunk node that contains entities
        """
        query = """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.chunk_id = $chunk_id,
            c.source = $source,
            c.created_at = datetime()
        RETURN c.id as chunk_id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    id=f"chunk_{chunk.get('chunk_id', chunk.get('id', 0))}",
                    text=chunk.get('text', '')[:500],  # Limit text size
                    chunk_id=chunk.get('chunk_id', chunk.get('id', 0)),
                    source=chunk.get('source', 'unknown')
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to create chunk {chunk.get('chunk_id', 'unknown')}: {e}")
            return False
    
    def link_entity_to_chunk(self, entity_id: str, chunk_id: str) -> bool:
        """
        Link entity to its source chunk
        """
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (c:Chunk {id: $chunk_id})
        MERGE (e)-[:APPEARS_IN]->(c)
        RETURN e.id as entity_id
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(query, entity_id=entity_id, chunk_id=chunk_id)
                return True
        except Exception as e:
            self.logger.error(f"Failed to link entity {entity_id} to chunk {chunk_id}: {e}")
            return False
    
    def batch_create_entities(self, entities: List[Dict]) -> Dict:
        """
        Create multiple entities in batch for performance
        """
        stats = {
            'total': len(entities),
            'created': 0,
            'failed': 0,
            'errors': []
        }
        
        batch_size = 100
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            
            query = """
            UNWIND $entities as entity
            MERGE (e:Entity {id: entity.id})
            SET e.name = entity.name,
                e.type = entity.type,
                e.mention = entity.mention,
                e.chunk_id = entity.chunk_id,
                e.confidence = entity.confidence,
                e.start_pos = entity.start_pos,
                e.end_pos = entity.end_pos,
                e.created_at = datetime()
            RETURN count(e) as count
            """
            
            try:
                with self.driver.session() as session:
                    # Prepare entities with defaults
                    prepared_entities = []
                    for entity in batch:
                        prepared_entities.append({
                            'id': entity['id'],
                            'name': entity['name'],
                            'type': entity.get('type', 'UNKNOWN'),
                            'mention': entity.get('mention', entity['name']),
                            'chunk_id': entity.get('chunk_id', 0),
                            'confidence': entity.get('confidence', 0.8),
                            'start_pos': entity.get('start_pos', 0),
                            'end_pos': entity.get('end_pos', 0)
                        })
                    
                    result = session.run(query, entities=prepared_entities)
                    count = result.single()['count']
                    stats['created'] += count
            
            except Exception as e:
                stats['failed'] += len(batch)
                stats['errors'].append(str(e))
                self.logger.error(f"Batch creation failed for batch {i//batch_size}: {e}")
        
        return stats
    
    def batch_create_relationships(self, relationships: List[Dict]) -> Dict:
        """
        Create multiple relationships in batch
        """
        stats = {
            'total': len(relationships),
            'created': 0,
            'failed': 0,
            'errors': []
        }
        
        # Group relationships by type for efficiency
        rels_by_type = {}
        for rel in relationships:
            rel_type = rel['type'].replace(" ", "_").replace("-", "_").upper()
            if rel_type not in rels_by_type:
                rels_by_type[rel_type] = []
            rels_by_type[rel_type].append(rel)
        
        for rel_type, type_rels in rels_by_type.items():
            query = f"""
            UNWIND $relationships as rel
            MATCH (source:Entity {{id: rel.source_id}})
            MATCH (target:Entity {{id: rel.target_id}})
            MERGE (source)-[r:{rel_type}]->(target)
            SET r.id = rel.id,
                r.mention = rel.mention,
                r.chunk_id = rel.chunk_id,
                r.confidence = rel.confidence,
                r.created_at = datetime()
            RETURN count(r) as count
            """
            
            try:
                # Prepare relationships with defaults
                prepared_rels = []
                for rel in type_rels:
                    prepared_rels.append({
                        'id': rel['id'],
                        'source_id': rel['source_id'],
                        'target_id': rel['target_id'],
                        'mention': rel.get('mention', ''),
                        'chunk_id': rel.get('chunk_id', 0),
                        'confidence': rel.get('confidence', 0.7)
                    })
                
                with self.driver.session() as session:
                    result = session.run(query, relationships=prepared_rels)
                    count = result.single()['count']
                    stats['created'] += count
                    
            except Exception as e:
                stats['failed'] += len(type_rels)
                stats['errors'].append(str(e))
                self.logger.error(f"Failed to create {rel_type} relationships: {e}")
        
        return stats
    
    def get_graph_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the graph
        """
        queries = {
            'node_count': "MATCH (n) RETURN count(n) as count",
            'entity_count': "MATCH (n:Entity) RETURN count(n) as count",
            'relationship_count': "MATCH ()-[r]->() RETURN count(r) as count",
            'entity_types': """
                MATCH (n:Entity) 
                RETURN n.type as type, count(*) as count 
                ORDER BY count DESC
            """,
            'relationship_types': """
                MATCH ()-[r]->() 
                RETURN type(r) as type, count(*) as count 
                ORDER BY count DESC
            """,
            'chunk_count': "MATCH (c:Chunk) RETURN count(c) as count",
            'average_degree': """
                MATCH (n:Entity)
                WITH n, size((n)--()) as degree
                RETURN avg(degree) as avg_degree, max(degree) as max_degree
            """
        }
        
        stats = {}
        try:
            with self.driver.session() as session:
                for key, query in queries.items():
                    try:
                        result = session.run(query)
                        records = list(result)
                        if records:
                            if key in ['entity_types', 'relationship_types']:
                                stats[key] = [dict(record) for record in records]
                            else:
                                stats[key] = dict(records[0])
                        else:
                            stats[key] = None
                    except Exception as e:
                        stats[key] = f"Failed: {str(e)[:50]}"
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get graph statistics: {e}")
            return {}
    
    def execute_cypher(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        Execute custom Cypher query
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"Failed to execute query: {e}")
            return []
    
    def export_graph(self, output_path: str) -> bool:
        """
        Export graph data to JSON
        """
        try:
            # Get all nodes
            nodes_query = """
            MATCH (n)
            RETURN n.id as id, labels(n) as labels, properties(n) as properties
            """
            
            # Get all relationships
            rels_query = """
            MATCH (s)-[r]->(t)
            RETURN s.id as source, t.id as target, type(r) as type, properties(r) as properties
            """
            
            with self.driver.session() as session:
                nodes_result = session.run(nodes_query)
                rels_result = session.run(rels_query)
                
                graph_data = {
                    'nodes': [dict(record) for record in nodes_result],
                    'relationships': [dict(record) for record in rels_result]
                }
                
                # Save to file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, default=str)
                
                self.logger.info(f"✅ Graph exported to {output_path}")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to export graph: {e}")
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("🔌 Neo4j connection closed")
