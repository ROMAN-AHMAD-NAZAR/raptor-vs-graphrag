# week3_graph_construction/graph_visualizer.py
import json
from pathlib import Path
from typing import Dict, List, Any
import logging

# Optional imports - will be handled gracefully if not available
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from pyvis.network import Network
    HAS_PYVIS = True
except ImportError:
    HAS_PYVIS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class GraphVisualizer:
    """
    Create visualizations of the knowledge graph for research paper
    """
    
    # Node colors by type
    TYPE_COLORS = {
        'CONCEPT': '#FF6B6B',
        'METHOD': '#4ECDC4',
        'METRIC': '#FFD166',
        'PERSON': '#06D6A0',
        'DATASET': '#118AB2',
        'CONFERENCE': '#EF476F',
        'TOOL': '#073B4C',
        'ALGORITHM': '#7209B7',
        'YEAR': '#F72585',
        'UNKNOWN': '#6C757D'
    }
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log available libraries
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check and log available visualization libraries"""
        deps = {
            'networkx': HAS_NETWORKX,
            'matplotlib': HAS_MATPLOTLIB,
            'pyvis': HAS_PYVIS,
            'plotly': HAS_PLOTLY
        }
        
        missing = [name for name, available in deps.items() if not available]
        if missing:
            self.logger.warning(f"Missing optional libraries: {', '.join(missing)}")
            self.logger.info("Install with: pip install networkx matplotlib pyvis plotly")
    
    def load_graph_data(self, graph_json_path: Path) -> Dict:
        """Load graph data from JSON export"""
        try:
            with open(graph_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load graph data: {e}")
            return {}
    
    def create_networkx_graph(self, graph_data: Dict):
        """Convert graph data to NetworkX graph"""
        if not HAS_NETWORKX:
            self.logger.error("NetworkX not available. Install with: pip install networkx")
            return None
        
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data.get('nodes', []):
            node_id = node.get('id', 'unknown')
            props = node.get('properties', {})
            node_type = props.get('type', 'UNKNOWN')
            
            G.add_node(
                node_id,
                label=props.get('name', node_id),
                type=node_type,
                confidence=props.get('confidence', 0.5),
                size=10 + (props.get('confidence', 0.5) * 20)
            )
        
        # Add edges
        for rel in graph_data.get('relationships', []):
            source = rel.get('source')
            target = rel.get('target')
            rel_type = rel.get('type', 'RELATED_TO')
            props = rel.get('properties', {})
            
            if source and target and source in G and target in G:
                G.add_edge(
                    source,
                    target,
                    label=rel_type,
                    weight=props.get('confidence', 0.5)
                )
        
        return G
    
    def create_matplotlib_visualization(self, G, filename: str = "graph.png") -> Path:
        """Create static matplotlib visualization"""
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            self.logger.error("matplotlib and networkx required for this visualization")
            return None
        
        plt.figure(figsize=(16, 12))
        
        # Get node positions
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
        
        # Draw nodes by type
        for node_type in set(nx.get_node_attributes(G, 'type').values()):
            nodes_of_type = [node for node, attr in G.nodes(data=True) 
                           if attr.get('type') == node_type]
            
            if nodes_of_type:
                sizes = [G.nodes[node].get('size', 15) * 30 for node in nodes_of_type]
                
                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodes_of_type,
                    node_color=self.TYPE_COLORS.get(node_type, '#6C757D'),
                    node_size=sizes,
                    alpha=0.8,
                    label=node_type
                )
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            width=1,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1'
        )
        
        # Draw labels
        node_labels = {node: data.get('label', node)[:15] 
                      for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, node_labels, font_size=8, font_weight='bold')
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, alpha=0.7)
        
        plt.title('Knowledge Graph of Research Concepts', fontsize=16, pad=20)
        plt.legend(title='Entity Types', loc='upper right', fontsize=9)
        plt.axis('off')
        
        # Add statistics text
        stats_text = f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
        plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=10, style='italic')
        
        # Save figure
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✅ Matplotlib visualization saved: {output_path}")
        return output_path
    
    def create_pyvis_interactive(self, G, filename: str = "graph.html") -> Path:
        """Create interactive HTML visualization with PyVis"""
        if not HAS_PYVIS or not HAS_NETWORKX:
            self.logger.error("pyvis and networkx required for this visualization")
            return None
        
        # Create network
        net = Network(
            height="800px",
            width="100%",
            bgcolor="#222222",
            font_color="white",
            directed=True
        )
        
        # Set physics options for better layout
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)
        
        # Add nodes
        for node, data in G.nodes(data=True):
            node_type = data.get('type', 'UNKNOWN')
            node_color = self.TYPE_COLORS.get(node_type, '#6C757D')
            
            net.add_node(
                node,
                label=data.get('label', node),
                color=node_color,
                size=data.get('size', 15),
                title=f"""
                <b>Type:</b> {node_type}<br>
                <b>Name:</b> {data.get('label', node)}<br>
                <b>Confidence:</b> {data.get('confidence', 'N/A'):.2f}
                """
            )
        
        # Add edges
        for source, target, data in G.edges(data=True):
            net.add_edge(
                source,
                target,
                label=data.get('label', 'RELATES_TO'),
                title=f"Relationship: {data.get('label', 'RELATES_TO')}",
                color='#ADB5BD',
                width=1 + (data.get('weight', 0.5) * 2),
                arrows={'to': {'enabled': True, 'scaleFactor': 0.5}}
            )
        
        # Save as HTML
        output_path = self.output_dir / filename
        net.save_graph(str(output_path))
        
        self.logger.info(f"✅ Interactive visualization saved: {output_path}")
        return output_path
    
    def create_plotly_visualization(self, graph_data: Dict, filename: str = "plotly_graph.html") -> Path:
        """Create interactive Plotly visualization"""
        if not HAS_PLOTLY:
            self.logger.error("plotly required for this visualization")
            return None
        
        import numpy as np
        
        # Extract nodes and edges
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('relationships', [])
        
        if not nodes:
            self.logger.warning("No nodes to visualize")
            return None
        
        # Calculate positions in a circle
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        node_labels = []
        
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / n_nodes
            radius = 10
            node_x.append(radius * np.cos(angle))
            node_y.append(radius * np.sin(angle))
            
            props = node.get('properties', {})
            node_type = props.get('type', 'UNKNOWN')
            name = props.get('name', node.get('id', 'Unknown'))
            
            node_text.append(f"{name}<br>Type: {node_type}")
            node_labels.append(name[:12] if len(name) > 12 else name)
            node_color.append(self._get_color_for_type(node_type, hex=False))
            node_size.append(15 + props.get('confidence', 0.5) * 20)
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        
        # Create node id to index mapping
        node_id_to_idx = {node.get('id'): i for i, node in enumerate(nodes)}
        
        for edge in edges:
            source_idx = node_id_to_idx.get(edge.get('source'), -1)
            target_idx = node_id_to_idx.get(edge.get('target'), -1)
            
            if source_idx != -1 and target_idx != -1:
                edge_x.extend([node_x[source_idx], node_x[target_idx], None])
                edge_y.extend([node_y[source_idx], node_y[target_idx], None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Relationships'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="top center",
            textfont=dict(size=9),
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')
            ),
            name='Entities'
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Research Knowledge Graph Visualization',
                font=dict(size=18)
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='white',
            plot_bgcolor='white',
            annotations=[
                dict(
                    text=f"Nodes: {len(nodes)} | Relationships: {len(edges)}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.5, y=-0.02,
                    font=dict(size=12)
                )
            ]
        )
        
        # Save figure
        output_path = self.output_dir / filename
        fig.write_html(str(output_path))
        
        self.logger.info(f"✅ Plotly visualization saved: {output_path}")
        return output_path
    
    def _get_color_for_type(self, node_type: str, hex: bool = True) -> str:
        """Get color for node type"""
        color = self.TYPE_COLORS.get(node_type, self.TYPE_COLORS['UNKNOWN'])
        
        if hex:
            return color
        else:
            # Convert hex to rgb for Plotly
            color = color.lstrip('#')
            return f'rgb({int(color[0:2], 16)}, {int(color[2:4], 16)}, {int(color[4:6], 16)})'
    
    def create_statistics_chart(self, graph_data: Dict, filename: str = "statistics.png") -> Path:
        """Create statistics chart for paper"""
        if not HAS_MATPLOTLIB:
            self.logger.error("matplotlib required for this visualization")
            return None
        
        # Extract node types
        node_types = {}
        for node in graph_data.get('nodes', []):
            props = node.get('properties', {})
            node_type = props.get('type', 'UNKNOWN')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Extract relationship types
        rel_types = {}
        for rel in graph_data.get('relationships', []):
            rel_type = rel.get('type', 'UNKNOWN')
            rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Node types bar chart
        if node_types:
            sorted_node_types = sorted(node_types.items(), key=lambda x: x[1], reverse=True)
            types_list = [t[0] for t in sorted_node_types]
            counts_list = [t[1] for t in sorted_node_types]
            colors = [self.TYPE_COLORS.get(t, '#6C757D') for t in types_list]
            
            axes[0].barh(types_list, counts_list, color=colors)
            axes[0].set_title('Entity Type Distribution', fontsize=14, pad=15)
            axes[0].set_xlabel('Count', fontsize=12)
            axes[0].invert_yaxis()
            
            # Add count labels
            for i, (type_name, count) in enumerate(sorted_node_types):
                axes[0].text(count + 0.1, i, str(count), va='center', fontsize=10)
        
        # Relationship types bar chart
        if rel_types:
            sorted_rel_types = sorted(rel_types.items(), key=lambda x: x[1], reverse=True)
            types_list = [t[0] for t in sorted_rel_types]
            counts_list = [t[1] for t in sorted_rel_types]
            
            axes[1].barh(types_list, counts_list, color='#FF6B6B')
            axes[1].set_title('Relationship Type Distribution', fontsize=14, pad=15)
            axes[1].set_xlabel('Count', fontsize=12)
            axes[1].invert_yaxis()
            
            # Add count labels
            for i, (type_name, count) in enumerate(sorted_rel_types):
                axes[1].text(count + 0.1, i, str(count), va='center', fontsize=10)
        
        plt.suptitle('Knowledge Graph Statistics', fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"✅ Statistics chart saved: {output_path}")
        return output_path
    
    def create_paper_figures(self, graph_data: Dict) -> Dict:
        """Create all figures needed for research paper"""
        figures = {}
        
        # Create NetworkX graph if possible
        G = None
        if HAS_NETWORKX:
            G = self.create_networkx_graph(graph_data)
        
        # 1. Main graph visualization (matplotlib)
        if G and HAS_MATPLOTLIB:
            figures['main_graph'] = self.create_matplotlib_visualization(
                G, filename="paper_figure1_graph.png"
            )
        
        # 2. Interactive version (pyvis)
        if G and HAS_PYVIS:
            figures['interactive_graph'] = self.create_pyvis_interactive(
                G, filename="paper_figure2_interactive.html"
            )
        
        # 3. Plotly version
        if HAS_PLOTLY:
            figures['plotly_graph'] = self.create_plotly_visualization(
                graph_data, filename="paper_figure3_plotly.html"
            )
        
        # 4. Statistics visualization
        if HAS_MATPLOTLIB:
            figures['statistics_chart'] = self.create_statistics_chart(
                graph_data, filename="paper_figure4_statistics.png"
            )
        
        return figures
