# graphrag_project/config.py
"""
Configuration for GraphRAG Paper Project
"""
from pathlib import Path
import os


class GraphRAGConfig:
    """Configuration settings for the GraphRAG research project"""
    
    # Base directories
    PROJECT_ROOT = Path(__file__).parent
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    DATA_DIR = PROJECT_ROOT / "data"
    
    # Neo4j Connection Settings
    # Default Neo4j Desktop settings - update if different
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # OpenAI API (for entity extraction in Week 2)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Embedding model settings
    EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI model
    EMBEDDING_DIMENSION = 1536
    
    # Local embedding model (for graph retrieval - no API needed)
    LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
    LOCAL_EMBEDDING_DIMENSION = 384
    
    # Entity extraction settings
    ENTITY_TYPES = [
        "CONCEPT",      # AI/ML concepts like RAPTOR, GraphRAG, RAG
        "METHOD",       # Techniques like hierarchical clustering, embeddings
        "METRIC",       # Evaluation metrics like Recall@K, F1
        "PERSON",       # Researchers, authors
        "DATASET",      # NQ, HotpotQA, etc.
        "CONFERENCE",   # NeurIPS, ICML, etc.
        "TOOL",         # Libraries, frameworks
        "ALGORITHM",    # Specific algorithms
        "YEAR"          # Publication years
    ]
    
    RELATIONSHIP_TYPES = [
        "USES",         # X uses Y
        "OUTPERFORMS",  # X outperforms Y
        "COMPARES",     # X is compared to Y
        "CITES",        # X cites Y
        "EVALUATES",    # X is evaluated on Y
        "EXTENDS",      # X extends Y
        "TESTED_ON",    # X tested on dataset Y
        "EVALUATED_ON", # X evaluated on metric Y
        "BASED_ON"      # X is based on Y
    ]
    
    # Chunking settings
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Graph construction settings
    MIN_ENTITY_CONFIDENCE = 0.5
    MIN_RELATIONSHIP_CONFIDENCE = 0.4
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        (cls.OUTPUT_DIR / "entities").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "graphs").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "visualizations").mkdir(exist_ok=True)
        (cls.OUTPUT_DIR / "embeddings").mkdir(exist_ok=True)
        
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n📋 GraphRAG Configuration:")
        print(f"   Project Root: {cls.PROJECT_ROOT}")
        print(f"   Output Dir: {cls.OUTPUT_DIR}")
        print(f"   Neo4j URI: {cls.NEO4J_URI}")
        print(f"   Neo4j User: {cls.NEO4J_USER}")
        print(f"   Neo4j Database: {cls.NEO4J_DATABASE}")
        print(f"   Entity Types: {len(cls.ENTITY_TYPES)}")
        print(f"   Relationship Types: {len(cls.RELATIONSHIP_TYPES)}")


# Create alias for backward compatibility
Config = GraphRAGConfig


if __name__ == "__main__":
    GraphRAGConfig.ensure_directories()
    GraphRAGConfig.print_config()
