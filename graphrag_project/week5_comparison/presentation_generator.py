# week5_comparison/presentation_generator.py
"""
Presentation Generator for Week 5
Creates PowerPoint-style presentation content for research paper
"""

from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class PresentationGenerator:
    """
    Generate presentation slides for research paper defense/conference
    """
    
    def __init__(self, comparison_results: Dict, df):
        self.comparison_results = comparison_results
        self.df = df
        self.timestamp = datetime.now().strftime("%B %d, %Y")
        
        # Convert DataFrame to list if needed
        if HAS_PANDAS and isinstance(df, pd.DataFrame):
            self.data = df.to_dict('records')
        else:
            self.data = df if isinstance(df, list) else list(df)
    
    def _get_metric(self, system: str, metric: str) -> float:
        """Get metric value for a system"""
        for row in self.data:
            if row['System'] == system:
                return row.get(metric, 0)
        return 0
    
    def _calculate_improvement(self, value: float, baseline: float) -> float:
        """Calculate percentage improvement over baseline"""
        if baseline > 0:
            return ((value - baseline) / baseline) * 100
        return 0
    
    def generate_html_presentation(self, output_dir: Path) -> str:
        """Generate HTML presentation using reveal.js style"""
        
        raptor_ndcg = self._get_metric('RAPTOR', 'NDCG@10')
        graphrag_ndcg = self._get_metric('GraphRAG', 'NDCG@10')
        baseline_ndcg = self._get_metric('Baseline RAG', 'NDCG@10')
        
        raptor_improvement = self._calculate_improvement(raptor_ndcg, baseline_ndcg)
        graphrag_improvement = self._calculate_improvement(graphrag_ndcg, baseline_ndcg)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>RAPTOR vs GraphRAG - Research Presentation</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            min-height: 100vh;
        }}
        
        .slide {{
            min-height: 100vh;
            padding: 60px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}
        
        .slide-title {{
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }}
        
        h1 {{
            font-size: 3.5em;
            margin-bottom: 20px;
            background: linear-gradient(45deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        h2 {{
            font-size: 2.5em;
            margin-bottom: 40px;
            color: #00d2ff;
        }}
        
        h3 {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #3a7bd5;
        }}
        
        .subtitle {{
            font-size: 1.5em;
            color: #888;
            margin-bottom: 30px;
        }}
        
        .content {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 40px;
            margin-top: 30px;
        }}
        
        .three-column {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }}
        
        .card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
        }}
        
        .card-raptor {{
            border-left: 4px solid #e74c3c;
        }}
        
        .card-graphrag {{
            border-left: 4px solid #3498db;
        }}
        
        .card-baseline {{
            border-left: 4px solid #95a5a6;
        }}
        
        .metric {{
            font-size: 3em;
            font-weight: bold;
            color: #00d2ff;
        }}
        
        .metric-label {{
            font-size: 1.2em;
            color: #888;
        }}
        
        .improvement {{
            color: #2ecc71;
            font-size: 1.2em;
        }}
        
        .bullet-list {{
            list-style: none;
            padding: 0;
        }}
        
        .bullet-list li {{
            padding: 15px 0;
            padding-left: 40px;
            position: relative;
            font-size: 1.3em;
            line-height: 1.6;
        }}
        
        .bullet-list li:before {{
            content: "→";
            position: absolute;
            left: 0;
            color: #00d2ff;
        }}
        
        .table-container {{
            overflow-x: auto;
            margin-top: 30px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 1.1em;
        }}
        
        th, td {{
            padding: 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        th {{
            background: rgba(0, 210, 255, 0.3);
            font-weight: bold;
        }}
        
        tr:nth-child(even) {{
            background: rgba(255,255,255,0.05);
        }}
        
        .highlight {{
            background: rgba(46, 204, 113, 0.3) !important;
            font-weight: bold;
        }}
        
        .navigation {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
        }}
        
        .nav-button {{
            background: rgba(0, 210, 255, 0.3);
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }}
        
        .nav-button:hover {{
            background: rgba(0, 210, 255, 0.5);
        }}
        
        .slide-number {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            color: #888;
            font-size: 1em;
        }}
        
        .emoji {{
            font-size: 3em;
            margin-bottom: 20px;
        }}
        
        .comparison-arrow {{
            font-size: 2em;
            color: #00d2ff;
            margin: 20px 0;
        }}
        
        .quote {{
            font-style: italic;
            font-size: 1.5em;
            color: #888;
            border-left: 4px solid #00d2ff;
            padding-left: 20px;
            margin: 30px 0;
        }}
    </style>
</head>
<body>
    <!-- Slide 1: Title -->
    <div class="slide slide-title" id="slide-1">
        <div class="emoji">🔬</div>
        <h1>RAPTOR vs GraphRAG</h1>
        <div class="subtitle">A Comparative Analysis of Structured RAG Approaches</div>
        <p style="margin-top: 50px; color: #666;">{self.timestamp}</p>
        <p style="color: #888; margin-top: 20px;">Research Paper Presentation</p>
    </div>
    
    <!-- Slide 2: The Problem -->
    <div class="slide" id="slide-2">
        <div class="content">
            <h2>🎯 The Problem with Traditional RAG</h2>
            <div class="two-column">
                <div class="card">
                    <h3>Traditional RAG</h3>
                    <ul class="bullet-list">
                        <li>Flat document chunking</li>
                        <li>Loss of document structure</li>
                        <li>Limited context understanding</li>
                        <li>Poor performance on complex queries</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>The Solution</h3>
                    <ul class="bullet-list">
                        <li>Structure-aware retrieval</li>
                        <li>Preserve document relationships</li>
                        <li>Hierarchical or graph-based approaches</li>
                        <li>Better reasoning capabilities</li>
                    </ul>
                </div>
            </div>
            <div class="quote">
                "Document structure matters - we need smarter retrieval."
            </div>
        </div>
    </div>
    
    <!-- Slide 3: The Contenders -->
    <div class="slide" id="slide-3">
        <div class="content">
            <h2>🥊 The Contenders</h2>
            <div class="three-column">
                <div class="card card-baseline">
                    <h3>Baseline RAG</h3>
                    <div class="emoji">📄</div>
                    <ul class="bullet-list" style="font-size: 0.9em;">
                        <li>Flat chunking</li>
                        <li>Semantic similarity</li>
                        <li>Simple & fast</li>
                        <li>Limited context</li>
                    </ul>
                </div>
                <div class="card card-raptor">
                    <h3>RAPTOR</h3>
                    <div class="emoji">🌲</div>
                    <ul class="bullet-list" style="font-size: 0.9em;">
                        <li>Hierarchical trees</li>
                        <li>Recursive summarization</li>
                        <li>GMM clustering</li>
                        <li>Multi-level retrieval</li>
                    </ul>
                </div>
                <div class="card card-graphrag">
                    <h3>GraphRAG</h3>
                    <div class="emoji">🕸️</div>
                    <ul class="bullet-list" style="font-size: 0.9em;">
                        <li>Knowledge graphs</li>
                        <li>Entity extraction</li>
                        <li>Relationship mapping</li>
                        <li>Graph traversal</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Slide 4: Results Overview -->
    <div class="slide" id="slide-4">
        <div class="content">
            <h2>📊 Results at a Glance</h2>
            <div class="table-container">
                <table>
                    <tr>
                        <th>System</th>
                        <th>NDCG@10</th>
                        <th>Precision@10</th>
                        <th>Recall@10</th>
                        <th>MRR</th>
                        <th>Query Time</th>
                    </tr>
"""
        
        # Add table rows
        for row in self.data:
            ndcg = row.get('NDCG@10', 0)
            is_best_ndcg = ndcg == max(r.get('NDCG@10', 0) for r in self.data)
            css_class = 'highlight' if is_best_ndcg else ''
            
            html_content += f"""
                    <tr class="{css_class}">
                        <td><strong>{row['System']}</strong></td>
                        <td>{row.get('NDCG@10', 0):.3f}</td>
                        <td>{row.get('Precision@10', 0):.3f}</td>
                        <td>{row.get('Recall@10', 0):.3f}</td>
                        <td>{row.get('MRR', 0):.3f}</td>
                        <td>{row.get('Query Time (ms)', 0):.0f}ms</td>
                    </tr>
"""
        
        html_content += f"""
                </table>
            </div>
        </div>
    </div>
    
    <!-- Slide 5: Key Metrics -->
    <div class="slide" id="slide-5">
        <div class="content">
            <h2>🏆 Key Results</h2>
            <div class="three-column">
                <div class="card card-baseline">
                    <h3>Baseline RAG</h3>
                    <div class="metric">{baseline_ndcg:.3f}</div>
                    <div class="metric-label">NDCG@10</div>
                    <p style="margin-top: 20px;">Reference baseline</p>
                </div>
                <div class="card card-raptor">
                    <h3>RAPTOR</h3>
                    <div class="metric">{raptor_ndcg:.3f}</div>
                    <div class="metric-label">NDCG@10</div>
                    <div class="improvement">+{raptor_improvement:.1f}% improvement</div>
                </div>
                <div class="card card-graphrag">
                    <h3>GraphRAG</h3>
                    <div class="metric">{graphrag_ndcg:.3f}</div>
                    <div class="metric-label">NDCG@10</div>
                    <div class="improvement">+{graphrag_improvement:.1f}% improvement</div>
                </div>
            </div>
            <div class="comparison-arrow">↑ Both structured approaches significantly outperform baseline ↑</div>
        </div>
    </div>
    
    <!-- Slide 6: RAPTOR Strengths -->
    <div class="slide" id="slide-6">
        <div class="content">
            <h2>🌲 RAPTOR Strengths</h2>
            <div class="two-column">
                <div class="card card-raptor">
                    <h3>Best At</h3>
                    <ul class="bullet-list">
                        <li>Context coverage: <strong>{self._get_metric('RAPTOR', 'Context Coverage'):.3f}</strong></li>
                        <li>Complex reasoning queries</li>
                        <li>Multi-hop document understanding</li>
                        <li>Capturing hierarchical themes</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Trade-offs</h3>
                    <ul class="bullet-list">
                        <li>Higher query latency: <strong>{self._get_metric('RAPTOR', 'Query Time (ms)'):.0f}ms</strong></li>
                        <li>Longer build time</li>
                        <li>More computational resources</li>
                    </ul>
                </div>
            </div>
            <div class="quote">
                "RAPTOR excels when you need comprehensive document understanding."
            </div>
        </div>
    </div>
    
    <!-- Slide 7: GraphRAG Strengths -->
    <div class="slide" id="slide-7">
        <div class="content">
            <h2>🕸️ GraphRAG Strengths</h2>
            <div class="two-column">
                <div class="card card-graphrag">
                    <h3>Best At</h3>
                    <ul class="bullet-list">
                        <li>Precision: <strong>{self._get_metric('GraphRAG', 'Precision@10'):.3f}</strong></li>
                        <li>Entity-focused queries</li>
                        <li>Relationship extraction</li>
                        <li>Fact-finding tasks</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>Trade-offs</h3>
                    <ul class="bullet-list">
                        <li>Faster queries: <strong>{self._get_metric('GraphRAG', 'Query Time (ms)'):.0f}ms</strong></li>
                        <li>May miss broader context</li>
                        <li>Dependent on NER quality</li>
                    </ul>
                </div>
            </div>
            <div class="quote">
                "GraphRAG shines for precise entity and relationship queries."
            </div>
        </div>
    </div>
    
    <!-- Slide 8: Recommendations -->
    <div class="slide" id="slide-8">
        <div class="content">
            <h2>🎯 Recommendations</h2>
            <div class="two-column">
                <div class="card card-raptor">
                    <h3>Choose RAPTOR When:</h3>
                    <ul class="bullet-list">
                        <li>Need comprehensive context</li>
                        <li>Complex reasoning required</li>
                        <li>Document structure matters</li>
                        <li>Latency is not critical</li>
                    </ul>
                </div>
                <div class="card card-graphrag">
                    <h3>Choose GraphRAG When:</h3>
                    <ul class="bullet-list">
                        <li>Entity lookup is primary</li>
                        <li>Relationship queries common</li>
                        <li>Real-time response needed</li>
                        <li>Precision over recall</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Slide 9: Future Work -->
    <div class="slide" id="slide-9">
        <div class="content">
            <h2>🔮 Future Work</h2>
            <ul class="bullet-list" style="max-width: 800px; margin: 0 auto;">
                <li>Hybrid approaches combining RAPTOR trees with GraphRAG relationships</li>
                <li>Adaptive retrieval based on query type</li>
                <li>End-to-end training of structured RAG components</li>
                <li>Scaling to larger document collections</li>
                <li>Domain-specific optimizations</li>
            </ul>
        </div>
    </div>
    
    <!-- Slide 10: Conclusion -->
    <div class="slide slide-title" id="slide-10">
        <div class="content">
            <div class="emoji">🎉</div>
            <h2>Conclusion</h2>
            <div class="three-column" style="margin-top: 40px;">
                <div class="card" style="text-align: center;">
                    <div class="metric">{raptor_improvement:.1f}%</div>
                    <div class="metric-label">RAPTOR Improvement</div>
                </div>
                <div class="card" style="text-align: center;">
                    <div class="metric">{graphrag_improvement:.1f}%</div>
                    <div class="metric-label">GraphRAG Improvement</div>
                </div>
                <div class="card" style="text-align: center;">
                    <div class="metric">✓</div>
                    <div class="metric-label">Structure Matters</div>
                </div>
            </div>
            <div class="quote" style="margin-top: 40px;">
                "Both RAPTOR and GraphRAG significantly outperform traditional RAG,<br>
                validating the importance of structure-aware retrieval."
            </div>
            <p style="margin-top: 50px; color: #666;">Thank you! Questions?</p>
        </div>
    </div>
    
    <div class="navigation">
        <button class="nav-button" onclick="previousSlide()">← Previous</button>
        <button class="nav-button" onclick="nextSlide()">Next →</button>
    </div>
    
    <div class="slide-number" id="slide-counter">Slide 1 / 10</div>
    
    <script>
        let currentSlide = 1;
        const totalSlides = 10;
        
        function showSlide(n) {{
            if (n < 1) n = totalSlides;
            if (n > totalSlides) n = 1;
            currentSlide = n;
            
            document.getElementById('slide-' + n).scrollIntoView({{ behavior: 'smooth' }});
            document.getElementById('slide-counter').textContent = 'Slide ' + n + ' / ' + totalSlides;
        }}
        
        function nextSlide() {{
            showSlide(currentSlide + 1);
        }}
        
        function previousSlide() {{
            showSlide(currentSlide - 1);
        }}
        
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'ArrowRight' || e.key === ' ') {{
                nextSlide();
            }} else if (e.key === 'ArrowLeft') {{
                previousSlide();
            }}
        }});
    </script>
</body>
</html>
"""
        
        output_file = output_dir / 'presentation.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    def generate_slide_notes(self, output_dir: Path) -> str:
        """Generate speaker notes for the presentation"""
        raptor_ndcg = self._get_metric('RAPTOR', 'NDCG@10')
        graphrag_ndcg = self._get_metric('GraphRAG', 'NDCG@10')
        baseline_ndcg = self._get_metric('Baseline RAG', 'NDCG@10')
        
        raptor_improvement = self._calculate_improvement(raptor_ndcg, baseline_ndcg)
        graphrag_improvement = self._calculate_improvement(graphrag_ndcg, baseline_ndcg)
        
        notes = f"""# Presentation Speaker Notes

## Slide 1: Title
- Welcome everyone to this presentation on comparing structured RAG approaches
- Today we'll be looking at RAPTOR and GraphRAG - two state-of-the-art methods for improving retrieval-augmented generation

## Slide 2: The Problem
- Traditional RAG has a fundamental limitation: flat chunking
- When we split documents into chunks, we lose the inherent structure
- This matters because documents have hierarchies, relationships, and context that span across chunks
- The solution? Structure-aware retrieval

## Slide 3: The Contenders
- We compare three systems:
  1. Baseline RAG - the standard approach everyone knows
  2. RAPTOR - uses hierarchical trees and recursive summarization
  3. GraphRAG - builds knowledge graphs from entities and relationships
- Each has a different philosophy on how to preserve structure

## Slide 4: Results Overview
- This table shows our key metrics
- Notice that both RAPTOR and GraphRAG outperform baseline across most metrics
- The winner depends on what you're optimizing for

## Slide 5: Key Results
- Let me highlight the NDCG@10 scores specifically
- Baseline: {baseline_ndcg:.3f}
- RAPTOR: {raptor_ndcg:.3f} (+{raptor_improvement:.1f}%)
- GraphRAG: {graphrag_ndcg:.3f} (+{graphrag_improvement:.1f}%)
- Both structured approaches show significant improvements

## Slide 6: RAPTOR Strengths
- RAPTOR shines in context coverage
- The hierarchical structure captures both detail and big-picture themes
- Best for complex reasoning where you need to understand the whole document
- Trade-off: Higher latency due to tree traversal

## Slide 7: GraphRAG Strengths
- GraphRAG excels at precision
- Entity-centric approach is perfect for fact-finding
- Graph traversal enables relationship-based queries
- Trade-off: May miss broader context not captured in entities

## Slide 8: Recommendations
- No one-size-fits-all answer
- RAPTOR for document understanding, complex reasoning
- GraphRAG for entity lookup, relationship queries, real-time needs
- Consider hybrid approaches for best of both worlds

## Slide 9: Future Work
- Exciting opportunities ahead
- Hybrid systems that combine RAPTOR's hierarchy with GraphRAG's relationships
- Adaptive systems that choose the best approach per query
- More research on scaling and domain adaptation

## Slide 10: Conclusion
- Key takeaway: Structure matters in RAG
- Both approaches validate this with significant improvements
- Choose based on your specific use case
- Questions welcome!

---
Generated: {self.timestamp}
"""
        
        output_file = output_dir / 'speaker_notes.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(notes)
        
        return str(output_file)
