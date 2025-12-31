# RAPTOR vs GraphRAG: A Comparative Study

A comprehensive research project comparing **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) and **GraphRAG** (Graph-based Retrieval Augmented Generation) systems.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This project implements and compares two advanced Retrieval-Augmented Generation (RAG) architectures:

| Feature | RAPTOR | GraphRAG |
|---------|--------|----------|
| **Structure** | Hierarchical Tree | Knowledge Graph |
| **Approach** | Bottom-up clustering & summarization | Entity & relationship extraction |
| **Strength** | Document-level understanding | Entity-centric retrieval |
| **Best For** | Complex reasoning questions | Factual entity queries |

## 🚀 Features

- **Dynamic Document Input**: Upload PDF, DOCX, or paste text directly
- **Real-time Comparison**: Side-by-side comparison of both systems
- **Metrics Dashboard**: Query time, accuracy scores, coverage metrics
- **Interactive Visualization**: Charts comparing retrieval performance
- **Query History**: Track and analyze past comparisons

## 📁 Project Structure

```
graphrag_project/
├── week3_graph_construction/    # Neo4j graph building
│   ├── neo4j_manager.py
│   ├── graph_builder.py
│   └── graph_visualizer.py
├── week4_graph_retrieval/       # Retrieval strategies
│   ├── embedding_manager.py
│   ├── graph_retriever.py
│   ├── hybrid_retriever.py
│   └── evaluation.py
├── week5_comparison/            # Analysis & reporting
│   ├── comparison_engine.py
│   ├── paper_generator.py
│   └── visualization_generator.py
└── webapp/                      # Web application
    ├── app.py                   # Flask backend
    ├── unified_retriever.py     # RAPTOR & GraphRAG implementations
    ├── templates/
    │   └── index.html
    └── static/
        ├── css/style.css
        └── js/app.js
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Neo4j (optional, for full GraphRAG features)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/raptor-vs-graphrag.git
cd raptor-vs-graphrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the web application
cd webapp
python app.py
```

### Access the Application
Open your browser to: `http://localhost:5000`

## 📦 Dependencies

```
flask>=2.3.0
flask-cors>=4.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
PyPDF2>=3.0.0
python-docx>=1.0.0
pdfplumber>=0.10.0
neo4j>=5.0.0 (optional)
```

## 🎮 Usage

### Web Interface

1. **Load Document**: Paste text or upload PDF/DOCX file
2. **Index**: Click "Index Document" to process with both systems
3. **Query**: Enter your question and click "Compare"
4. **Analyze**: View side-by-side results with metrics

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/index` | POST | Index a document |
| `/api/query` | POST | Run comparison query |
| `/api/stats` | GET | Get system statistics |
| `/api/health` | GET | Health check |

### Example API Usage

```python
import requests

# Index a document
response = requests.post('http://localhost:5000/api/index', 
    json={'text': 'Your document text here...'})

# Run a query
response = requests.post('http://localhost:5000/api/query',
    json={'query': 'What is RAPTOR?', 'top_k': 5})
print(response.json())
```

## 📊 Evaluation Metrics

- **Average Score**: Mean similarity score across results
- **Max Score**: Highest similarity achieved
- **Query Time**: Processing time in milliseconds
- **Coverage**: Percentage of relevant content captured
- **NDCG**: Normalized Discounted Cumulative Gain
- **Precision@k**: Relevant results in top k

## 🔬 Research Background

### RAPTOR
RAPTOR uses recursive clustering and summarization to build a tree structure from documents, enabling retrieval at multiple abstraction levels.

**Key Techniques:**
- Gaussian Mixture Models (GMM) for clustering
- UMAP for dimensionality reduction
- LLM-based summarization

### GraphRAG
GraphRAG constructs knowledge graphs by extracting entities and relationships, enabling entity-centric retrieval and multi-hop reasoning.

**Key Techniques:**
- Named Entity Recognition (NER)
- Relationship extraction
- Graph traversal algorithms

## 📈 Results Summary

| Metric | RAPTOR | GraphRAG |
|--------|--------|----------|
| Document Understanding | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Entity Queries | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Complex Reasoning | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [RAPTOR Paper](https://arxiv.org/abs/2401.18059) - Original RAPTOR research
- [GraphRAG by Microsoft](https://github.com/microsoft/graphrag) - GraphRAG implementation
- [Sentence Transformers](https://www.sbert.net/) - Embedding models

---
⭐ Star this repo if you find it helpful!

## 🚀 Quick Start: Week 3

### 1. Prerequisites

- **Neo4j Desktop** installed and running
  - Download: https://neo4j.com/download/
  - Create a new database with password: `password123`
  - Or update `config.py` with your credentials

### 2. Install Dependencies

```bash
cd D:\Raptor\graphrag_project
pip install -r requirements.txt
```

### 3. Run Week 3

```bash
# Option 1: Double-click the batch file
run_week3.bat

# Option 2: Run Python directly
python week3_graph_construction/main.py
```

## 📊 Week 3 Outputs

After running, you'll have:

1. **Knowledge Graph in Neo4j** - Query at http://localhost:7474
2. **Exported Graph Data** - `outputs/graphs/knowledge_graph.json`
3. **Visualizations**:
   - `paper_figure1_graph.png` - Static graph image
   - `paper_figure2_interactive.html` - Interactive PyVis visualization
   - `paper_figure3_plotly.html` - Plotly interactive chart
   - `paper_figure4_statistics.png` - Statistics bar charts

## 🔍 Exploring Your Graph in Neo4j

Open Neo4j Browser at http://localhost:7474 and try these queries:

```cypher
-- View all nodes
MATCH (n) RETURN n LIMIT 50

-- Find all CONCEPT entities
MATCH (e:Entity {type: 'CONCEPT'}) RETURN e.name, e.confidence

-- Find relationships from RAPTOR
MATCH (e1:Entity {name: 'RAPTOR'})-[r]->(e2:Entity)
RETURN e1.name, type(r), e2.name

-- Count entity types
MATCH (e:Entity)
RETURN e.type, count(*) as count
ORDER BY count DESC

-- Find paths between concepts
MATCH p = shortestPath((a:Entity {name: 'RAPTOR'})-[*..5]-(b:Entity {name: 'GraphRAG'}))
RETURN p
```

## 📈 Metrics for Your Paper

Week 3 provides these metrics for your research paper:

| Metric | Value |
|--------|-------|
| Total Entities | ~10-150 |
| Total Relationships | ~10-100 |
| Entity Types | 8-10 |
| Relationship Types | 5-9 |
| Graph Density | Calculated |
| Average Node Degree | Calculated |

## 🔮 Next Steps

- **Week 4**: Graph-Based Retrieval
- **Week 5**: Comparison with RAPTOR

## ⚠️ Troubleshooting

### Neo4j Connection Failed
1. Ensure Neo4j Desktop is running
2. Start your database (green play button)
3. Check credentials in `config.py`
4. Default: `bolt://localhost:7687`, user: `neo4j`, password: `password123`

### Missing Visualization Libraries
```bash
pip install networkx matplotlib pyvis plotly
```

### No Entities from Week 2
The script will create a sample research paper graph for demonstration.

---
*GraphRAG Paper Project - Week 3: Graph Construction*
