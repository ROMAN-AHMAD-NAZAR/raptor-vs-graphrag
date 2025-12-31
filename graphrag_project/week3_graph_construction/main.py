# week3_graph_construction/main.py
"""
WEEK 3: Knowledge Graph Construction in Neo4j
Build the actual knowledge graph from extracted entities and relationships
"""

import sys
from pathlib import Path
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import GraphRAGConfig
from week3_graph_construction.neo4j_manager import Neo4jGraphManager
from week3_graph_construction.graph_builder import GraphBuilder
from week3_graph_construction.graph_visualizer import GraphVisualizer


def setup_logging(output_dir: Path):
    """Setup logging configuration"""
    log_file = output_dir / 'graph_construction.log'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def find_latest_extraction_results(output_dir: Path) -> Path:
    """Find the most recent extraction results from Week 2"""
    entities_dir = output_dir / "entities"
    
    if not entities_dir.exists():
        print("⚠️  No entities directory found. Will create sample graph.")
        return None
    
    # Look for .pkl files
    pkl_files = list(entities_dir.glob("extraction_results_*.pkl"))
    
    if pkl_files:
        latest = max(pkl_files, key=lambda x: x.stat().st_mtime)
        return latest
    
    # Look for .json files
    json_files = list(entities_dir.glob("*.json"))
    if json_files:
        latest = max(json_files, key=lambda x: x.stat().st_mtime)
        print(f"⚠️  Using JSON file: {latest.name}")
        return latest
    
    return None


def main():
    """Main Week 3: Graph Construction"""
    print("=" * 70)
    print("WEEK 3: KNOWLEDGE GRAPH CONSTRUCTION IN NEO4J")
    print("=" * 70)
    
    # Load config
    config = GraphRAGConfig()
    
    # Setup
    logger = setup_logging(config.OUTPUT_DIR)
    
    # Step 1: Initialize Neo4j connection
    print("\n1️⃣  Connecting to Neo4j...")
    print(f"   URI: {config.NEO4J_URI}")
    print(f"   User: {config.NEO4J_USER}")
    print(f"   Database: {config.NEO4J_DATABASE}")
    
    neo4j_manager = Neo4jGraphManager(
        uri=config.NEO4J_URI,
        user=config.NEO4J_USER,
        password=config.NEO4J_PASSWORD,
        database=config.NEO4J_DATABASE
    )
    
    if not neo4j_manager.driver:
        print("\n❌ Failed to connect to Neo4j. Please check:")
        print("   1. Is Neo4j Desktop running? (look for green play button)")
        print("   2. Is the database started?")
        print("   3. Are credentials correct in config.py?")
        print(f"\n   URI: {config.NEO4J_URI}")
        print(f"   User: {config.NEO4J_USER}")
        print("\n💡 To install Neo4j Desktop:")
        print("   1. Download from https://neo4j.com/download/")
        print("   2. Create a new project and database")
        print("   3. Set password to 'password123' or update config.py")
        return
    
    print("   ✅ Connected to Neo4j!")
    
    # Step 2: Find extraction results
    print("\n2️⃣  Loading extraction results from Week 2...")
    
    results_path = find_latest_extraction_results(config.OUTPUT_DIR)
    
    # Initialize graph builder
    graphs_dir = config.OUTPUT_DIR / "graphs"
    graph_builder = GraphBuilder(neo4j_manager, graphs_dir)
    
    if not results_path:
        print("   No extraction results found. Creating sample research paper graph...")
        
        # Clear database first for clean demo
        print("\n⚠️  Clearing existing database for sample demo...")
        neo4j_manager.clear_database(confirm=True)
        
        # Create sample graph
        construction_stats = graph_builder.create_research_paper_graph()
        
    else:
        print(f"   Found: {results_path.name}")
        
        # Load extraction data
        extraction_data = graph_builder.load_extraction_results(results_path)
        
        if not extraction_data:
            print("❌ Failed to load extraction data")
            neo4j_manager.close()
            return
        
        print(f"   Entities: {len(extraction_data.get('entities', []))}")
        print(f"   Relationships: {len(extraction_data.get('relationships', []))}")
        print(f"   Chunks: {len(extraction_data.get('chunks', []))}")
        
        # Build knowledge graph
        print("\n3️⃣  Building knowledge graph...")
        construction_stats = graph_builder.build_knowledge_graph(extraction_data)
    
    # Print construction results
    print(f"\n📊 CONSTRUCTION STATISTICS:")
    print(f"   Entities created: {construction_stats.get('entities_processed', 0)}")
    print(f"   Relationships created: {construction_stats.get('relationships_processed', 0)}")
    print(f"   Chunks processed: {construction_stats.get('chunks_processed', 0)}")
    print(f"   Duration: {construction_stats.get('duration_seconds', 0):.2f} seconds")
    
    # Step 4: Export graph data
    print("\n4️⃣  Exporting graph data...")
    
    export_path = graph_builder.export_graph_data("knowledge_graph.json")
    if export_path:
        print(f"   ✅ Graph exported to: {export_path}")
    
    # Step 5: Create visualizations
    print("\n5️⃣  Creating visualizations for research paper...")
    
    visualizations_dir = config.OUTPUT_DIR / "visualizations"
    visualizer = GraphVisualizer(visualizations_dir)
    
    # Load exported graph data
    if export_path:
        graph_data = visualizer.load_graph_data(export_path)
        
        if graph_data:
            print("   Creating visualizations...")
            
            figures = visualizer.create_paper_figures(graph_data)
            
            print(f"\n📁 VISUALIZATIONS CREATED:")
            for fig_name, fig_path in figures.items():
                if fig_path:
                    print(f"   ✓ {fig_name}: {fig_path.name}")
                else:
                    print(f"   ✗ {fig_name}: skipped (missing dependencies)")
    
    # Step 6: Get final graph statistics
    print("\n6️⃣  Final graph statistics:")
    
    final_stats = neo4j_manager.get_graph_statistics()
    
    if final_stats:
        print(f"\n📈 GRAPH METRICS:")
        print(f"   Total nodes: {final_stats.get('node_count', {}).get('count', 'N/A')}")
        print(f"   Entity nodes: {final_stats.get('entity_count', {}).get('count', 'N/A')}")
        print(f"   Total relationships: {final_stats.get('relationship_count', {}).get('count', 'N/A')}")
        print(f"   Chunk nodes: {final_stats.get('chunk_count', {}).get('count', 'N/A')}")
        
        # Average degree
        avg_degree = final_stats.get('average_degree', {})
        if isinstance(avg_degree, dict):
            print(f"   Average degree: {avg_degree.get('avg_degree', 'N/A'):.2f}" if avg_degree.get('avg_degree') else "")
            print(f"   Max degree: {avg_degree.get('max_degree', 'N/A')}")
        
        # Entity type distribution
        entity_types = final_stats.get('entity_types', [])
        if entity_types and isinstance(entity_types, list):
            print(f"\n🏷️  ENTITY TYPE DISTRIBUTION:")
            for item in entity_types[:8]:  # Top 8
                print(f"   {item.get('type', 'unknown')}: {item.get('count', 0)}")
        
        # Relationship type distribution
        rel_types = final_stats.get('relationship_types', [])
        if rel_types and isinstance(rel_types, list):
            print(f"\n🔗 RELATIONSHIP TYPE DISTRIBUTION:")
            for item in rel_types[:8]:  # Top 8
                print(f"   {item.get('type', 'unknown')}: {item.get('count', 0)}")
    
    # Close connection
    neo4j_manager.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ WEEK 3 COMPLETE!")
    
    print(f"\n📁 OUTPUTS:")
    print(f"   Graph data: {graphs_dir}")
    print(f"   Visualizations: {visualizations_dir}")
    print(f"   Log file: {config.OUTPUT_DIR / 'graph_construction.log'}")
    
    print(f"\n📊 PAPER METRICS COLLECTED:")
    print(f"   ✓ Knowledge graph with {construction_stats.get('entities_processed', 0)} entities")
    print(f"   ✓ {construction_stats.get('relationships_processed', 0)} relationships")
    print(f"   ✓ {construction_stats.get('chunks_processed', 0)} document chunks")
    print(f"   ✓ Multiple visualization formats (PNG, HTML, Plotly)")
    
    print(f"\n🔍 NEXT STEPS:")
    print(f"   1. Open Neo4j Browser: http://localhost:7474")
    print(f"   2. Run: MATCH (n) RETURN n LIMIT 25")
    print(f"   3. View visualizations in: {visualizations_dir}")
    
    print(f"\n🚀 NEXT: Week 4 - Graph-Based Retrieval")
    print("=" * 70)


if __name__ == "__main__":
    main()
