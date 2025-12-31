# week3/tree_builder.py
import numpy as np
import json
from collections import defaultdict


class TreeNode:
    """A node in the RAPTOR tree"""
    def __init__(self, node_id, text, embedding, depth=0, 
                 is_summary=False, chunk_indices=None):
        self.id = node_id
        self.text = text
        self.embedding = embedding
        self.depth = depth
        self.is_summary = is_summary
        self.chunk_indices = chunk_indices or []  # Original chunks this node represents
        self.children = []
        self.parent = None
    
    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
    
    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'text': self.text[:200] + "..." if len(self.text) > 200 else self.text,
            'depth': self.depth,
            'is_summary': self.is_summary,
            'num_children': len(self.children),
            'chunk_indices': self.chunk_indices,
            'children': [child.id for child in self.children]
        }
    
    def __repr__(self):
        return f"TreeNode(id={self.id}, depth={self.depth}, summary={self.is_summary})"


class RaptorTreeBuilder:
    def __init__(self):
        self.root = None
        self.nodes = {}
        self.next_node_id = 0
        self.max_depth = 3
    
    def build_tree(self, chunks, embeddings, cluster_labels, depth=0):
        """
        Build hierarchical tree from clusters
        
        Strategy:
        1. At each level, group chunks by cluster
        2. For each cluster, create a summary node
        3. Connect original chunks as children
        4. Stop when at max depth
        """
        print(f"\n🌳 Building tree at depth {depth}...")
        print(f"   Processing {len(chunks)} chunks")
        
        # Group by cluster
        cluster_groups = defaultdict(list)
        for i, (chunk, emb, label) in enumerate(zip(chunks, embeddings, cluster_labels)):
            cluster_groups[label].append({
                'index': i,
                'chunk': chunk,
                'embedding': emb
            })
        
        print(f"   Found {len(cluster_groups)} clusters")
        
        parent_nodes = []
        
        for cluster_id in sorted(cluster_groups.keys()):
            items = cluster_groups[cluster_id]
            
            # Create summary text (placeholder - Week 4 will use LLM)
            cluster_chunks = [item['chunk'] for item in items]
            summary_text = self._create_simple_summary(cluster_chunks, cluster_id)
            
            # Average embedding for the cluster
            cluster_embeddings = np.array([item['embedding'] for item in items])
            avg_embedding = np.mean(cluster_embeddings, axis=0)
            
            # Create parent node
            parent_node = TreeNode(
                node_id=f"D{depth}_C{cluster_id}",
                text=summary_text,
                embedding=avg_embedding,
                depth=depth,
                is_summary=True,
                chunk_indices=[item['index'] for item in items]
            )
            
            # Create leaf nodes for each chunk in this cluster
            for item in items:
                leaf_node = TreeNode(
                    node_id=f"L{depth+1}_{item['index']}",
                    text=item['chunk'],
                    embedding=item['embedding'],
                    depth=depth + 1,
                    is_summary=False,
                    chunk_indices=[item['index']]
                )
                parent_node.add_child(leaf_node)
                self.nodes[leaf_node.id] = leaf_node
            
            parent_nodes.append(parent_node)
            self.nodes[parent_node.id] = parent_node
        
        print(f"✅ Built {len(parent_nodes)} cluster nodes at depth {depth}")
        
        return parent_nodes
    
    def _create_simple_summary(self, chunks, cluster_id):
        """
        Create a simple summary (placeholder for Week 4's LLM summaries)
        """
        # Extract key sentences from first few chunks
        previews = []
        for chunk in chunks[:3]:
            # Get first sentence or first 100 chars
            sentences = chunk.split('.')
            if sentences:
                preview = sentences[0][:100]
                previews.append(preview)
        
        summary = f"[Cluster {cluster_id}] " + " | ".join(previews)
        return summary[:500]  # Limit length
    
    def set_root(self, cluster_nodes):
        """Set the root of the tree (connects all top-level clusters)"""
        if len(cluster_nodes) == 0:
            print("⚠️ No cluster nodes to set as root")
            return None
        
        if len(cluster_nodes) == 1:
            self.root = cluster_nodes[0]
            self.root.depth = 0
            # Adjust children depths
            for child in self.root.children:
                child.depth = 1
        else:
            # Create a root that connects all top-level clusters
            root_text = f"[Document Root] This document contains {len(cluster_nodes)} main topics"
            root_embedding = np.mean([node.embedding for node in cluster_nodes], axis=0)
            
            self.root = TreeNode(
                node_id="ROOT",
                text=root_text,
                embedding=root_embedding,
                depth=0,
                is_summary=True,
                chunk_indices=[]
            )
            
            # Collect all chunk indices
            all_indices = []
            for node in cluster_nodes:
                node.depth = 1
                for child in node.children:
                    child.depth = 2
                self.root.add_child(node)
                all_indices.extend(node.chunk_indices)
            
            self.root.chunk_indices = all_indices
        
        self.nodes[self.root.id] = self.root
        
        print(f"\n🌲 Tree root set: {self.root.id}")
        print(f"   Total nodes in tree: {len(self.nodes)}")
        
        return self.root
    
    def get_tree_stats(self):
        """Get statistics about the tree"""
        if not self.nodes:
            return {}
        
        depths = [node.depth for node in self.nodes.values()]
        summaries = sum(1 for node in self.nodes.values() if node.is_summary)
        leaves = sum(1 for node in self.nodes.values() if not node.is_summary)
        
        return {
            'total_nodes': len(self.nodes),
            'summary_nodes': summaries,
            'leaf_nodes': leaves,
            'max_depth': max(depths),
            'min_depth': min(depths)
        }
    
    def print_tree(self, node=None, indent=0):
        """Print tree structure"""
        if node is None:
            node = self.root
        
        if node is None:
            print("⚠️ Tree is empty")
            return
        
        prefix = "  " * indent
        node_type = "📝" if node.is_summary else "📄"
        children_info = f" ({len(node.children)} children)" if node.children else ""
        print(f"{prefix}{node_type} [{node.id}] Depth {node.depth}{children_info}")
        
        # Show text preview
        text_preview = node.text[:60].replace('\n', ' ')
        print(f"{prefix}   └─ \"{text_preview}...\"")
        
        for child in node.children[:5]:  # Limit display
            self.print_tree(child, indent + 1)
        
        if len(node.children) > 5:
            print(f"{prefix}   ... and {len(node.children) - 5} more children")
    
    def save_tree(self, filename):
        """Save tree structure to file"""
        tree_data = {
            'root_id': self.root.id if self.root else None,
            'stats': self.get_tree_stats(),
            'nodes': {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Tree saved to {filename}")
        print(f"   Total nodes: {len(self.nodes)}")
        
        return tree_data
    
    def get_all_nodes_at_depth(self, depth):
        """Get all nodes at a specific depth"""
        return [node for node in self.nodes.values() if node.depth == depth]
    
    def get_path_to_root(self, node_id):
        """Get path from a node to the root"""
        if node_id not in self.nodes:
            return []
        
        path = []
        current = self.nodes[node_id]
        while current:
            path.append(current)
            current = current.parent
        
        return path
