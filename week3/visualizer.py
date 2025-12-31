# week3/visualizer.py
import matplotlib.pyplot as plt
import numpy as np
import os


class TreeVisualizer:
    def __init__(self):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_clusters_2d(self, reduced_data, labels, chunks, save_path=None):
        """
        Visualize clusters in 2D
        """
        print("\n🎨 Creating cluster visualization...")
        
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot with cluster colors
        unique_labels = np.unique(labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                reduced_data[mask, 0],
                reduced_data[mask, 1],
                c=[colors[i]],
                label=f'Cluster {label} ({mask.sum()} chunks)',
                alpha=0.7,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add cluster centroid label
            centroid = reduced_data[mask].mean(axis=0)
            plt.annotate(
                f'C{label}',
                centroid,
                fontsize=14,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle="circle,pad=0.3", facecolor="yellow", alpha=0.8)
            )
        
        plt.legend(loc='upper right', fontsize=9)
        plt.title('Document Chunks Clustered by Semantic Similarity', fontsize=16)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        num_clusters = len(unique_labels)
        plt.figtext(0.5, 0.01, 
                   f'Total chunks: {len(chunks)} | Clusters: {num_clusters} | '
                   f'Average cluster size: {len(chunks)/num_clusters:.1f} chunks',
                   ha='center', fontsize=10, style='italic')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Cluster plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_tree_structure(self, tree_builder, save_path=None):
        """
        Visualize the tree structure using matplotlib
        """
        print("\n🌿 Creating tree visualization...")
        
        if tree_builder.root is None:
            print("⚠️ Tree is empty, cannot visualize")
            return
        
        # Collect nodes by depth
        nodes_by_depth = {}
        for node in tree_builder.nodes.values():
            if node.depth not in nodes_by_depth:
                nodes_by_depth[node.depth] = []
            nodes_by_depth[node.depth].append(node)
        
        max_depth = max(nodes_by_depth.keys())
        
        # Calculate positions
        positions = {}
        for depth in range(max_depth + 1):
            nodes = nodes_by_depth.get(depth, [])
            for i, node in enumerate(nodes):
                x = (i + 0.5) / len(nodes) if nodes else 0.5
                y = 1 - depth / (max_depth + 1)
                positions[node.id] = (x, y)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Draw edges first
        for node in tree_builder.nodes.values():
            if node.id in positions:
                x1, y1 = positions[node.id]
                for child in node.children:
                    if child.id in positions:
                        x2, y2 = positions[child.id]
                        ax.plot([x1, x2], [y1, y2], 'gray', linewidth=1, alpha=0.5, zorder=1)
        
        # Draw nodes
        for node in tree_builder.nodes.values():
            if node.id in positions:
                x, y = positions[node.id]
                
                if node.is_summary:
                    color = 'lightblue'
                    marker = 'o'
                    size = 800
                else:
                    color = 'lightgreen'
                    marker = 's'
                    size = 400
                
                ax.scatter([x], [y], c=color, s=size, marker=marker, 
                          edgecolors='black', linewidth=1, zorder=2)
                
                # Add label
                label = node.id.split('_')[0] if '_' in node.id else node.id
                ax.annotate(label, (x, y), ha='center', va='center', 
                           fontsize=8, fontweight='bold', zorder=3)
        
        # Add depth labels
        for depth in range(max_depth + 1):
            y = 1 - depth / (max_depth + 1)
            ax.text(-0.05, y, f'Depth {depth}', ha='right', va='center', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_title('RAPTOR Tree Structure', fontsize=18, pad=20)
        ax.axis('off')
        
        # Add legend
        import matplotlib.patches as mpatches
        summary_patch = mpatches.Patch(color='lightblue', label='Summary Node')
        leaf_patch = mpatches.Patch(color='lightgreen', label='Leaf (Original Chunk)')
        ax.legend(handles=[summary_patch, leaf_patch], loc='upper left')
        
        # Add stats
        stats = tree_builder.get_tree_stats()
        info_text = f"Total: {stats['total_nodes']} nodes | Summaries: {stats['summary_nodes']} | Leaves: {stats['leaf_nodes']}"
        ax.text(0.5, -0.05, info_text, ha='center', transform=ax.transAxes, fontsize=10)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Tree visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_cluster_sizes(self, labels, save_path=None):
        """
        Bar chart of cluster sizes
        """
        print("\n📊 Creating cluster size chart...")
        
        unique, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(unique, counts, color=plt.cm.tab20(np.linspace(0, 1, len(unique))))
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Chunks', fontsize=12)
        plt.title('Cluster Size Distribution', fontsize=14)
        plt.xticks(unique)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Cluster size chart saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_html_visualization(self, reduced_data, labels, chunks, save_path):
        """
        Create a simple interactive HTML visualization
        """
        print("\n🔄 Creating interactive HTML visualization...")
        
        # Generate HTML with embedded JavaScript
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>RAPTOR Cluster Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        #chart { width: 100%; height: 600px; border: 1px solid #ccc; }
        .tooltip { 
            position: absolute; 
            background: white; 
            border: 1px solid #333; 
            padding: 10px; 
            border-radius: 5px;
            max-width: 400px;
            display: none;
        }
        .legend { margin-top: 20px; }
        .legend-item { display: inline-block; margin-right: 20px; }
        .legend-color { width: 20px; height: 20px; display: inline-block; margin-right: 5px; }
    </style>
</head>
<body>
    <h1>RAPTOR Document Clusters</h1>
    <p>Hover over points to see chunk content. Colors represent different clusters.</p>
    <canvas id="chart"></canvas>
    <div class="tooltip" id="tooltip"></div>
    <div class="legend" id="legend"></div>
    
    <script>
        const data = DATA_PLACEHOLDER;
        const canvas = document.getElementById('chart');
        const ctx = canvas.getContext('2d');
        const tooltip = document.getElementById('tooltip');
        
        // Set canvas size
        canvas.width = canvas.offsetWidth;
        canvas.height = 600;
        
        // Colors for clusters
        const colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
                       '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4'];
        
        // Scale data to canvas
        const padding = 50;
        const xMin = Math.min(...data.map(d => d.x));
        const xMax = Math.max(...data.map(d => d.x));
        const yMin = Math.min(...data.map(d => d.y));
        const yMax = Math.max(...data.map(d => d.y));
        
        function scaleX(x) { return padding + (x - xMin) / (xMax - xMin) * (canvas.width - 2*padding); }
        function scaleY(y) { return canvas.height - padding - (y - yMin) / (yMax - yMin) * (canvas.height - 2*padding); }
        
        // Draw points
        function draw() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            data.forEach(point => {
                ctx.beginPath();
                ctx.arc(scaleX(point.x), scaleY(point.y), 8, 0, Math.PI * 2);
                ctx.fillStyle = colors[point.cluster % colors.length];
                ctx.fill();
                ctx.strokeStyle = '#333';
                ctx.stroke();
            });
        }
        
        draw();
        
        // Tooltip on hover
        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            let found = false;
            for (const point of data) {
                const px = scaleX(point.x);
                const py = scaleY(point.y);
                const dist = Math.sqrt((mouseX - px)**2 + (mouseY - py)**2);
                
                if (dist < 15) {
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.pageX + 10) + 'px';
                    tooltip.style.top = (e.pageY + 10) + 'px';
                    tooltip.innerHTML = `<strong>Chunk ${point.id} (Cluster ${point.cluster})</strong><br>${point.text}`;
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                tooltip.style.display = 'none';
            }
        });
        
        // Legend
        const legend = document.getElementById('legend');
        const clusters = [...new Set(data.map(d => d.cluster))].sort((a,b) => a-b);
        clusters.forEach(c => {
            const count = data.filter(d => d.cluster === c).length;
            legend.innerHTML += `<div class="legend-item">
                <span class="legend-color" style="background: ${colors[c % colors.length]}"></span>
                Cluster ${c} (${count} chunks)
            </div>`;
        });
    </script>
</body>
</html>
"""
        
        # Prepare data
        data_points = []
        for i, (point, label, chunk) in enumerate(zip(reduced_data, labels, chunks)):
            preview = chunk[:150].replace('"', '\\"').replace('\n', ' ')
            data_points.append({
                'id': i,
                'x': float(point[0]),
                'y': float(point[1]),
                'cluster': int(label),
                'text': preview + '...' if len(chunk) > 150 else preview
            })
        
        import json
        html_content = html_content.replace('DATA_PLACEHOLDER', json.dumps(data_points))
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Interactive HTML saved to {save_path}")
        print(f"   Open in browser to explore clusters!")
        
        return save_path
