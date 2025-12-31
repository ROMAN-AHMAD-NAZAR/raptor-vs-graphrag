# week4/tree_enricher.py
"""
Tree Enricher Module for RAPTOR

Adds intelligent summaries to tree nodes at all levels
"""

import pickle
import json
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np


class TreeEnricher:
    """
    Enriches RAPTOR tree with summaries at all levels
    
    Process:
    1. Load tree from Week 3
    2. Generate summaries for leaf nodes (original chunks)
    3. Generate summaries for branch nodes (cluster centers)
    4. Generate summary for root node (full document)
    """
    
    def __init__(self, summarization_engine, summary_enhancer=None):
        """
        Args:
            summarization_engine: SummarizationEngine instance
            summary_enhancer: Optional SummaryEnhancer for quality improvement
        """
        self.summarizer = summarization_engine
        self.enhancer = summary_enhancer
        
        print("🌲 Tree Enricher initialized")
    
    def load_tree(self, tree_path: str) -> Dict:
        """Load tree structure from file"""
        print(f"\n📂 Loading tree from: {tree_path}")
        
        with open(tree_path, 'rb') as f:
            tree_data = pickle.load(f)
        
        print(f"✅ Loaded tree with {len(tree_data.get('nodes', []))} nodes")
        return tree_data
    
    def enrich_tree(self, tree_data: Dict, chunks: List[str] = None) -> Dict:
        """
        Add summaries to all tree nodes
        
        Args:
            tree_data: Tree structure from Week 3
            chunks: Original text chunks (for leaf summaries)
        """
        print("\n🔮 Enriching tree with summaries...")
        
        nodes = tree_data.get('nodes', [])
        if not nodes:
            print("⚠️ No nodes found in tree!")
            return tree_data
        
        # Group nodes by depth
        depth_groups = {}
        for node in nodes:
            depth = node.get('depth', 0)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node)
        
        max_depth = max(depth_groups.keys())
        print(f"   Tree depth: {max_depth}")
        print(f"   Nodes per level: {[(d, len(n)) for d, n in sorted(depth_groups.items())]}")
        
        # Process from leaves to root (bottom-up)
        enriched_nodes = []
        node_summaries = {}  # Store summaries by node_id
        
        for depth in sorted(depth_groups.keys(), reverse=True):
            level_nodes = depth_groups[depth]
            print(f"\n   📝 Processing depth {depth} ({len(level_nodes)} nodes)...")
            
            for i, node in enumerate(level_nodes):
                # Get text to summarize
                if depth == max_depth:
                    # Leaf nodes: use chunk text if available
                    if chunks and 'chunk_indices' in node:
                        indices = node['chunk_indices']
                        text = ' '.join([chunks[i] for i in indices if i < len(chunks)])
                    else:
                        text = node.get('text', node.get('representative_text', ''))
                else:
                    # Branch/root nodes: combine child summaries
                    child_ids = node.get('children', [])
                    child_summaries = [node_summaries.get(cid, '') for cid in child_ids]
                    child_summaries = [s for s in child_summaries if s]
                    
                    if child_summaries:
                        text = ' '.join(child_summaries)
                    else:
                        text = node.get('text', node.get('representative_text', ''))
                
                if not text:
                    text = f"Node {node.get('node_id', i)} at depth {depth}"
                
                # Generate summary
                summary = self.summarizer.generate_summary(text, depth)
                
                # Enhance if enhancer available
                if self.enhancer:
                    result = self.enhancer.enhance_summary(summary, text, depth)
                    summary = result['enhanced']
                
                # Store summary
                node_id = node.get('node_id', i)
                node_summaries[node_id] = summary
                
                # Add to node
                node['summary'] = summary
                node['summary_source'] = 'generated'
                enriched_nodes.append(node)
                
                if (i + 1) % 5 == 0 or i == len(level_nodes) - 1:
                    print(f"      [{i + 1}/{len(level_nodes)}] completed")
        
        # Update tree data
        tree_data['nodes'] = enriched_nodes
        tree_data['enrichment_status'] = 'complete'
        tree_data['summary_count'] = len(node_summaries)
        
        print(f"\n✅ Tree enriched with {len(node_summaries)} summaries!")
        
        return tree_data
    
    def save_enriched_tree(self, tree_data: Dict, output_path: str):
        """Save enriched tree to file"""
        # Save as pickle
        pickle_path = output_path if output_path.endswith('.pkl') else output_path + '.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(tree_data, f)
        print(f"💾 Saved enriched tree: {pickle_path}")
        
        # Also save JSON for readability
        json_path = pickle_path.replace('.pkl', '.json')
        
        # Convert numpy arrays for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(i) for i in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        json_data = convert_for_json(tree_data)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved JSON version: {json_path}")
    
    def create_summary_index(self, tree_data: Dict) -> Dict:
        """Create searchable index of summaries"""
        print("\n📚 Creating summary index...")
        
        index = {
            'by_depth': {},
            'by_node_id': {},
            'all_summaries': []
        }
        
        for node in tree_data.get('nodes', []):
            depth = node.get('depth', 0)
            node_id = node.get('node_id', 0)
            summary = node.get('summary', '')
            
            if summary:
                entry = {
                    'node_id': node_id,
                    'depth': depth,
                    'summary': summary,
                    'children': node.get('children', [])
                }
                
                if depth not in index['by_depth']:
                    index['by_depth'][depth] = []
                index['by_depth'][depth].append(entry)
                index['by_node_id'][node_id] = entry
                index['all_summaries'].append(entry)
        
        print(f"✅ Created index with {len(index['all_summaries'])} summaries")
        return index
    
    def print_tree_summary(self, tree_data: Dict):
        """Print summary of enriched tree"""
        print("\n" + "=" * 60)
        print("🌲 ENRICHED TREE SUMMARY")
        print("=" * 60)
        
        nodes = tree_data.get('nodes', [])
        depth_groups = {}
        for node in nodes:
            d = node.get('depth', 0)
            if d not in depth_groups:
                depth_groups[d] = []
            depth_groups[d].append(node)
        
        for depth in sorted(depth_groups.keys()):
            level_nodes = depth_groups[depth]
            print(f"\n📊 Depth {depth} ({len(level_nodes)} nodes):")
            
            for i, node in enumerate(level_nodes[:3]):  # Show first 3
                summary = node.get('summary', 'No summary')
                preview = summary[:80] + "..." if len(summary) > 80 else summary
                print(f"   Node {node.get('node_id', i)}: {preview}")
            
            if len(level_nodes) > 3:
                print(f"   ... and {len(level_nodes) - 3} more nodes")
        
        print("\n" + "=" * 60)


class TreeNavigator:
    """
    Navigate and query the enriched tree
    """
    
    def __init__(self, tree_data: Dict):
        self.tree = tree_data
        self.nodes_by_id = {n.get('node_id', i): n 
                           for i, n in enumerate(tree_data.get('nodes', []))}
        print(f"🧭 Navigator initialized with {len(self.nodes_by_id)} nodes")
    
    def get_root_summary(self) -> str:
        """Get the root (top-level) summary"""
        root_nodes = [n for n in self.tree.get('nodes', []) if n.get('depth') == 0]
        if root_nodes:
            return root_nodes[0].get('summary', 'No root summary')
        return 'No root found'
    
    def get_level_summaries(self, depth: int) -> List[str]:
        """Get all summaries at a specific depth"""
        return [n.get('summary', '') for n in self.tree.get('nodes', [])
                if n.get('depth') == depth]
    
    def get_node_with_children(self, node_id: int) -> Dict:
        """Get a node and its children's summaries"""
        node = self.nodes_by_id.get(node_id, {})
        children_ids = node.get('children', [])
        children = [self.nodes_by_id.get(cid, {}) for cid in children_ids]
        
        return {
            'node': node,
            'summary': node.get('summary', ''),
            'children': [{
                'node_id': c.get('node_id'),
                'summary': c.get('summary', '')
            } for c in children]
        }
    
    def trace_path(self, node_id: int) -> List[Dict]:
        """Trace path from root to specific node"""
        path = []
        current = self.nodes_by_id.get(node_id)
        
        while current:
            path.insert(0, {
                'node_id': current.get('node_id'),
                'depth': current.get('depth'),
                'summary': current.get('summary', '')
            })
            parent_id = current.get('parent')
            current = self.nodes_by_id.get(parent_id) if parent_id else None
        
        return path
    
    def search_summaries(self, query: str) -> List[Dict]:
        """Simple keyword search in summaries"""
        query_lower = query.lower()
        results = []
        
        for node in self.tree.get('nodes', []):
            summary = node.get('summary', '').lower()
            if query_lower in summary:
                results.append({
                    'node_id': node.get('node_id'),
                    'depth': node.get('depth'),
                    'summary': node.get('summary', ''),
                    'match_score': summary.count(query_lower)
                })
        
        return sorted(results, key=lambda x: x['match_score'], reverse=True)
