# week5_comparison/visualization_generator.py
"""
Visualization Generator for Week 5
Creates charts and graphs for the research paper
"""

from typing import Dict, List, Any
from pathlib import Path
import logging
import json

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class VisualizationGenerator:
    """
    Generate publication-ready visualizations for research paper
    """
    
    def __init__(self, comparison_results: Dict, df):
        self.comparison_results = comparison_results
        self.df = df
        self.logger = logging.getLogger(__name__)
        
        # Convert DataFrame to list if needed
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            self.data = df.to_dict('records')
        else:
            self.data = df if isinstance(df, list) else list(df)
        
        # Set style for publication
        if HAS_MATPLOTLIB:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.size'] = 12
            plt.rcParams['axes.labelsize'] = 14
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['legend.fontsize'] = 11
        
        self.logger.info("✅ VisualizationGenerator initialized")
    
    def generate_all_visualizations(self, output_dir: Path) -> List[str]:
        """Generate all visualizations for the paper"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        if not HAS_MATPLOTLIB:
            self.logger.warning("matplotlib not available, generating HTML visualizations only")
            html_file = self._generate_html_charts(output_dir)
            if html_file:
                generated_files.append(html_file)
            return generated_files
        
        # 1. Overall performance comparison bar chart
        try:
            file = self._generate_performance_bar_chart(output_dir)
            generated_files.append(file)
        except Exception as e:
            self.logger.error(f"Failed to generate performance bar chart: {e}")
        
        # 2. Metric radar chart
        try:
            file = self._generate_radar_chart(output_dir)
            generated_files.append(file)
        except Exception as e:
            self.logger.error(f"Failed to generate radar chart: {e}")
        
        # 3. Query time comparison
        try:
            file = self._generate_latency_chart(output_dir)
            generated_files.append(file)
        except Exception as e:
            self.logger.error(f"Failed to generate latency chart: {e}")
        
        # 4. Improvement over baseline chart
        try:
            file = self._generate_improvement_chart(output_dir)
            generated_files.append(file)
        except Exception as e:
            self.logger.error(f"Failed to generate improvement chart: {e}")
        
        # 5. Trade-off scatter plot
        try:
            file = self._generate_tradeoff_chart(output_dir)
            generated_files.append(file)
        except Exception as e:
            self.logger.error(f"Failed to generate trade-off chart: {e}")
        
        # 6. HTML interactive dashboard
        try:
            file = self._generate_html_charts(output_dir)
            generated_files.append(file)
        except Exception as e:
            self.logger.error(f"Failed to generate HTML charts: {e}")
        
        return generated_files
    
    def _generate_performance_bar_chart(self, output_dir: Path) -> str:
        """Generate grouped bar chart comparing all metrics"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        systems = [row['System'] for row in self.data]
        metrics = ['NDCG@10', 'Precision@10', 'Recall@10', 'MRR']
        
        x = np.arange(len(systems))
        width = 0.2
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            values = [row.get(metric, 0) for row in self.data]
            bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.annotate(f'{value:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('System')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison: RAPTOR vs GraphRAG vs Baseline')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(systems)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        
        output_file = output_dir / 'performance_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Generated: {output_file}")
        return str(output_file)
    
    def _generate_radar_chart(self, output_dir: Path) -> str:
        """Generate radar chart for multi-dimensional comparison"""
        metrics = ['NDCG@10', 'Precision@10', 'Recall@10', 'MRR', 'Context Coverage']
        
        # Normalize values for radar chart
        num_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        
        for i, row in enumerate(self.data):
            values = [row.get(metric, 0) for metric in metrics]
            values += values[:1]  # Complete the loop
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['System'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Dimensional Performance Comparison', size=16, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        output_file = output_dir / 'radar_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Generated: {output_file}")
        return str(output_file)
    
    def _generate_latency_chart(self, output_dir: Path) -> str:
        """Generate query latency comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        systems = [row['System'] for row in self.data]
        latencies = [row.get('Query Time (ms)', 0) for row in self.data]
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        bars = ax.barh(systems, latencies, color=colors)
        
        # Add value labels
        for bar, latency in zip(bars, latencies):
            ax.annotate(f'{latency:.1f} ms',
                       xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                       xytext=(5, 0),
                       textcoords="offset points",
                       ha='left', va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Query Latency (ms)')
        ax.set_title('Query Response Time Comparison')
        ax.set_xlim(0, max(latencies) * 1.3)
        
        # Add reference line for real-time threshold
        ax.axvline(x=200, color='orange', linestyle='--', linewidth=2, label='Real-time threshold (200ms)')
        ax.legend()
        
        plt.tight_layout()
        
        output_file = output_dir / 'latency_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Generated: {output_file}")
        return str(output_file)
    
    def _generate_improvement_chart(self, output_dir: Path) -> str:
        """Generate improvement over baseline chart"""
        improvements = self.comparison_results.get('improvements', {})
        
        if not improvements:
            return ""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        systems = list(improvements.keys())
        metrics = ['NDCG Improvement %', 'Precision Improvement %', 'Coverage Improvement %']
        
        x = np.arange(len(systems))
        width = 0.25
        
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            values = [improvements[sys].get(metric, 0) for sys in systems]
            bars = ax.bar(x + i * width, values, width, label=metric.replace(' %', ''), color=colors[i])
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.annotate(f'{value:+.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('System')
        ax.set_ylabel('Improvement over Baseline (%)')
        ax.set_title('Improvement Over Baseline RAG')
        ax.set_xticks(x + width)
        ax.set_xticklabels(systems)
        ax.legend(loc='upper left')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        output_file = output_dir / 'improvement_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Generated: {output_file}")
        return str(output_file)
    
    def _generate_tradeoff_chart(self, output_dir: Path) -> str:
        """Generate trade-off scatter plot (Quality vs Latency)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        markers = ['o', 's', '^']
        
        for i, row in enumerate(self.data):
            ndcg = row.get('NDCG@10', 0)
            latency = row.get('Query Time (ms)', 0)
            coverage = row.get('Context Coverage', 0) * 500  # Scale for bubble size
            
            ax.scatter(latency, ndcg, s=coverage + 100, c=colors[i], marker=markers[i],
                      alpha=0.7, edgecolors='black', linewidth=2, label=row['System'])
            
            # Add label
            ax.annotate(row['System'],
                       xy=(latency, ndcg),
                       xytext=(10, 10),
                       textcoords="offset points",
                       fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Query Latency (ms)')
        ax.set_ylabel('NDCG@10')
        ax.set_title('Quality vs Latency Trade-off\n(Bubble size = Context Coverage)')
        ax.legend(loc='lower right')
        
        # Add quadrant labels
        ax.axhline(y=0.81, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=250, color='gray', linestyle='--', alpha=0.5)
        
        ax.text(100, 0.85, 'Fast & Accurate\n(Ideal)', ha='center', fontsize=10, style='italic')
        ax.text(400, 0.85, 'Slow & Accurate', ha='center', fontsize=10, style='italic')
        ax.text(100, 0.77, 'Fast & Less Accurate', ha='center', fontsize=10, style='italic')
        ax.text(400, 0.77, 'Slow & Less Accurate', ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        
        output_file = output_dir / 'tradeoff_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Generated: {output_file}")
        return str(output_file)
    
    def _generate_html_charts(self, output_dir: Path) -> str:
        """Generate interactive HTML dashboard"""
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>RAPTOR vs GraphRAG Comparison Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .chart-container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 30px;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            color: #34495e;
            margin-bottom: 15px;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 30px;
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }
        .summary-table th {
            background-color: #3498db;
            color: white;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .summary-table tr:hover {
            background-color: #f1f1f1;
        }
        .highlight-best {
            background-color: #d4edda !important;
            font-weight: bold;
        }
        .findings {
            background: #fff;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .findings h2 {
            color: #2c3e50;
        }
        .findings ul {
            line-height: 1.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 RAPTOR vs GraphRAG: Comprehensive Comparison</h1>
        
        <!-- Summary Table -->
        <div class="chart-container">
            <div class="chart-title">📊 Performance Summary</div>
            <table class="summary-table">
                <tr>
                    <th>System</th>
                    <th>NDCG@10</th>
                    <th>Precision@10</th>
                    <th>Recall@10</th>
                    <th>MRR</th>
                    <th>Query Time (ms)</th>
                    <th>Context Coverage</th>
                </tr>
"""
        
        # Find best values for highlighting
        metrics = ['NDCG@10', 'Precision@10', 'Recall@10', 'MRR', 'Context Coverage']
        best_values = {}
        for metric in metrics:
            values = [(row['System'], row.get(metric, 0)) for row in self.data]
            if values:
                best_system, best_value = max(values, key=lambda x: x[1])
                best_values[metric] = best_value
        
        # Query time - lower is better
        time_values = [(row['System'], row.get('Query Time (ms)', 0)) for row in self.data]
        if time_values:
            best_time_system, best_time = min(time_values, key=lambda x: x[1])
            best_values['Query Time (ms)'] = best_time
        
        # Add rows
        for row in self.data:
            html_content += "<tr>"
            html_content += f"<td><strong>{row['System']}</strong></td>"
            
            for metric in ['NDCG@10', 'Precision@10', 'Recall@10', 'MRR', 'Query Time (ms)', 'Context Coverage']:
                value = row.get(metric, 0)
                is_best = False
                
                if metric == 'Query Time (ms)':
                    is_best = value == best_values.get(metric, -1)
                else:
                    is_best = value == best_values.get(metric, -1)
                
                css_class = 'highlight-best' if is_best else ''
                
                if metric == 'Query Time (ms)':
                    html_content += f"<td class='{css_class}'>{value:.1f}</td>"
                else:
                    html_content += f"<td class='{css_class}'>{value:.3f}</td>"
            
            html_content += "</tr>"
        
        html_content += """
            </table>
        </div>
        
        <!-- Performance Bar Chart -->
        <div class="chart-container">
            <div class="chart-title">📈 Performance Comparison</div>
            <div id="performance-chart"></div>
        </div>
        
        <!-- Latency Chart -->
        <div class="chart-container">
            <div class="chart-title">⏱️ Query Latency</div>
            <div id="latency-chart"></div>
        </div>
        
        <!-- Radar Chart -->
        <div class="chart-container">
            <div class="chart-title">🎯 Multi-Dimensional Comparison</div>
            <div id="radar-chart"></div>
        </div>
        
        <!-- Key Findings -->
        <div class="findings">
            <h2>🔍 Key Findings</h2>
            <ul>
                <li><strong>Both structured approaches outperform baseline RAG</strong> - RAPTOR and GraphRAG show significant improvements in NDCG@10</li>
                <li><strong>RAPTOR excels at context coverage</strong> - Better for complex, multi-hop reasoning queries</li>
                <li><strong>GraphRAG excels at precision</strong> - Better for entity-focused, fact-finding queries</li>
                <li><strong>GraphRAG has lower latency</strong> - More suitable for real-time applications</li>
                <li><strong>RAPTOR provides richer context</strong> - Through hierarchical summarization</li>
            </ul>
            
            <h2>📋 Recommendations</h2>
            <ul>
                <li>Use <strong>RAPTOR</strong> for document understanding and complex reasoning</li>
                <li>Use <strong>GraphRAG</strong> for entity lookup and relationship queries</li>
                <li>Consider <strong>hybrid approaches</strong> for balanced performance</li>
            </ul>
        </div>
    </div>
    
    <script>
"""
        
        # Add JavaScript data
        systems = [row['System'] for row in self.data]
        ndcg_values = [row.get('NDCG@10', 0) for row in self.data]
        precision_values = [row.get('Precision@10', 0) for row in self.data]
        recall_values = [row.get('Recall@10', 0) for row in self.data]
        mrr_values = [row.get('MRR', 0) for row in self.data]
        latency_values = [row.get('Query Time (ms)', 0) for row in self.data]
        coverage_values = [row.get('Context Coverage', 0) for row in self.data]
        
        html_content += f"""
        // Performance Bar Chart
        var performanceData = [
            {{x: {json.dumps(systems)}, y: {json.dumps(ndcg_values)}, name: 'NDCG@10', type: 'bar', marker: {{color: '#2ecc71'}}}},
            {{x: {json.dumps(systems)}, y: {json.dumps(precision_values)}, name: 'Precision@10', type: 'bar', marker: {{color: '#3498db'}}}},
            {{x: {json.dumps(systems)}, y: {json.dumps(recall_values)}, name: 'Recall@10', type: 'bar', marker: {{color: '#e74c3c'}}}},
            {{x: {json.dumps(systems)}, y: {json.dumps(mrr_values)}, name: 'MRR', type: 'bar', marker: {{color: '#9b59b6'}}}}
        ];
        
        Plotly.newPlot('performance-chart', performanceData, {{
            barmode: 'group',
            yaxis: {{title: 'Score', range: [0, 1]}},
            legend: {{orientation: 'h', y: 1.15}}
        }});
        
        // Latency Chart
        var latencyData = [{{
            x: {json.dumps(latency_values)},
            y: {json.dumps(systems)},
            type: 'bar',
            orientation: 'h',
            marker: {{
                color: ['#e74c3c', '#3498db', '#2ecc71']
            }},
            text: {json.dumps([f'{v:.0f} ms' for v in latency_values])},
            textposition: 'outside'
        }}];
        
        Plotly.newPlot('latency-chart', latencyData, {{
            xaxis: {{title: 'Query Latency (ms)'}},
            shapes: [{{
                type: 'line',
                x0: 200, x1: 200,
                y0: -0.5, y1: 2.5,
                line: {{color: 'orange', width: 2, dash: 'dash'}}
            }}],
            annotations: [{{
                x: 200, y: 2.5,
                text: 'Real-time threshold',
                showarrow: false,
                font: {{color: 'orange'}}
            }}]
        }});
        
        // Radar Chart
        var radarData = [
"""
        
        colors = ['rgba(231, 76, 60, 0.5)', 'rgba(52, 152, 219, 0.5)', 'rgba(46, 204, 113, 0.5)']
        line_colors = ['rgb(231, 76, 60)', 'rgb(52, 152, 219)', 'rgb(46, 204, 113)']
        
        for i, row in enumerate(self.data):
            values = [
                row.get('NDCG@10', 0),
                row.get('Precision@10', 0),
                row.get('Recall@10', 0),
                row.get('MRR', 0),
                row.get('Context Coverage', 0)
            ]
            values.append(values[0])  # Close the loop
            
            html_content += f"""
            {{
                type: 'scatterpolar',
                r: {json.dumps(values)},
                theta: ['NDCG@10', 'Precision@10', 'Recall@10', 'MRR', 'Coverage', 'NDCG@10'],
                fill: 'toself',
                fillcolor: '{colors[i]}',
                line: {{color: '{line_colors[i]}'}},
                name: '{row["System"]}'
            }},
"""
        
        html_content += """
        ];
        
        Plotly.newPlot('radar-chart', radarData, {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 1]
                }
            }
        });
    </script>
</body>
</html>
"""
        
        output_file = output_dir / 'comparison_dashboard.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"✅ Generated: {output_file}")
        return str(output_file)
