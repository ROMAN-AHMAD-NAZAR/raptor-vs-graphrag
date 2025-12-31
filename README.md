# 🌲 RAPTOR vs GraphRAG: Complete Comparison Study

A comprehensive research project implementing and comparing **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) and **GraphRAG** (Graph-based Retrieval Augmented Generation) systems.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![RAPTOR](https://img.shields.io/badge/RAPTOR-ICLR%202024-red.svg)
![GraphRAG](https://img.shields.io/badge/GraphRAG-Microsoft-purple.svg)

## 🎯 Project Overview

This project provides complete implementations of two state-of-the-art RAG architectures:

| Feature | RAPTOR | GraphRAG |
|---------|--------|----------|
| **Structure** | Hierarchical Tree | Knowledge Graph |
| **Approach** | Bottom-up clustering & summarization | Entity & relationship extraction |
| **Strength** | Document-level understanding | Entity-centric retrieval |
| **Best For** | Complex reasoning questions | Factual entity queries |

## 📊 Performance Results

| Metric | RAPTOR | GraphRAG | Normal RAG |
|--------|--------|----------|------------|
| NDCG@10 | **0.814** | 0.798 | 0.802 |
| Context Coverage | **0.15** | 0.12 | 0.05 |
| Entity Queries | 0.72 | **0.89** | 0.65 |
| Complex Reasoning | **0.85** | 0.71 | 0.58 |

## 🏗️ Complete Project Structure

```
raptor-vs-graphrag/
│
├── 🌲 RAPTOR IMPLEMENTATION
│   ├── week1/                      # Document Processing
│   │   ├── step1_loader.py        # PDF + TXT loading
│   │   ├── step2_processor.py     # Text cleaning & chunking
│   │   ├── document_processor.py  # Unified processor
│   │   └── main.py                # Week 1 pipeline
│   │
│   ├── week2/                      # Embeddings & Optimization
│   │   ├── embedder.py            # TextEmbedder + FAISSIndex
│   │   ├── embeddings_manager.py  # Semantic validation
│   │   ├── chunk_optimizer.py     # Similarity-based merging
│   │   ├── embedding_storage.py   # Multi-format storage
│   │   └── main.py                # Week 2 pipeline
│   │
│   ├── week3/                      # Clustering & Tree Building
│   │   ├── dimensionality_reducer.py  # UMAP 384D→50D
│   │   ├── hierarchical_clusterer.py  # GMM clustering
│   │   ├── tree_builder.py        # RAPTOR tree structure
│   │   ├── visualizer.py          # Cluster visualization
│   │   └── main.py                # Week 3 pipeline
│   │
│   ├── week4/                      # Intelligent Summarization
│   │   ├── summarization_engine.py    # TinyLlama / rule-based
│   │   ├── summary_enhancer.py    # Quality improvement
│   │   ├── tree_enricher.py       # Add summaries to tree
│   │   └── main.py                # Week 4 pipeline
│   │
│   └── week5/                      # Storage & Retrieval
│       ├── qdrant_manager.py      # Vector database
│       ├── raptor_retriever.py    # Hierarchical search
│       ├── rag_baseline.py        # Flat retrieval baseline
│       ├── evaluator.py           # Performance metrics
│       ├── demo_app.py            # Interactive demo
│       └── main.py                # Week 5 pipeline
│
├── 📊 GRAPHRAG IMPLEMENTATION
│   └── graphrag_project/
│       ├── week3_graph_construction/   # Neo4j Graph Building
│       │   ├── neo4j_manager.py       # Database connection
│       │   ├── graph_builder.py       # Graph construction
│       │   ├── graph_visualizer.py    # Visualization
│       │   └── main.py
│       │
│       ├── week4_graph_retrieval/      # Retrieval Strategies
│       │   ├── embedding_manager.py   # Embeddings
│       │   ├── graph_retriever.py     # Graph-based search
│       │   ├── hybrid_retriever.py    # Combined approach
│       │   ├── evaluation.py          # Metrics
│       │   └── main.py
│       │
│       └── week5_comparison/           # Analysis & Reporting
│           ├── comparison_engine.py   # RAPTOR vs GraphRAG
│           ├── paper_generator.py     # Research paper output
│           ├── visualization_generator.py
│           └── main.py
│
├── 🌐 WEB APPLICATION
│   └── graphrag_project/webapp/
│       ├── app.py                 # Flask backend
│       ├── unified_retriever.py   # Both systems unified
│       ├── templates/
│       │   └── index.html         # Modern UI
│       └── static/
│           ├── css/style.css      # Dark theme
│           └── js/app.js          # Interactive frontend
│
├── 📁 SUPPORTING FILES
│   ├── raptor-system/             # Core RAPTOR system
│   │   └── src/
│   │       ├── clustering/
│   │       ├── document_processing/
│   │       ├── embeddings/
│   │       ├── evaluation/
│   │       └── summarization/
│   │
│   ├── data/                      # Sample documents
│   │   └── test_document.txt
│   │
│   ├── raptor_api.py              # Production REST API
│   ├── requirements.txt           # Dependencies
│   └── PROJECT_SUMMARY.md         # Detailed summary
│
└── tests/                         # Unit tests
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ROMAN-AHMAD-NAZAR/raptor-vs-graphrag.git
cd raptor-vs-graphrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Application

```bash
cd graphrag_project/webapp
python app.py
# Open http://localhost:5000
```

### Run RAPTOR Pipeline

```bash
# Week 1: Document Processing
cd week1 && python main.py

# Week 2: Embeddings
cd week2 && python main.py

# Week 3: Clustering & Tree Building
cd week3 && python main.py

# Week 4: Summarization
cd week4 && python main.py

# Week 5: Retrieval & Evaluation
cd week5 && python main.py
```

### Run GraphRAG Pipeline

```bash
cd graphrag_project

# Week 3: Graph Construction (requires Neo4j)
python -m week3_graph_construction.main

# Week 4: Graph Retrieval
python -m week4_graph_retrieval.main

# Week 5: Comparison
python -m week5_comparison.main
```

## 📦 Dependencies

### Core Dependencies
```
sentence-transformers>=2.2.0
numpy>=1.24.0
scikit-learn>=1.3.0
umap-learn>=0.5.0
faiss-cpu>=1.7.0
```

### Web Application
```
flask>=2.3.0
flask-cors>=4.0.0
PyPDF2>=3.0.0
python-docx>=1.0.0
pdfplumber>=0.10.0
```

### GraphRAG (Optional)
```
neo4j>=5.0.0
```

## 🎮 Web Application Features

- **📄 Dynamic Document Input**: Upload PDF, DOCX, or paste text
- **⚡ Real-time Comparison**: Side-by-side RAPTOR vs GraphRAG
- **📊 Metrics Dashboard**: Query time, accuracy, coverage
- **📈 Interactive Charts**: Visual score comparison
- **📜 Query History**: Track past comparisons

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/index` | POST | Index a document |
| `/api/query` | POST | Run comparison query |
| `/api/stats` | GET | Get system statistics |
| `/api/health` | GET | Health check |
| `/api/sample-document` | GET | Get sample document |
| `/api/sample-queries` | GET | Get sample queries |

## 🔬 Research Background

### RAPTOR (ICLR 2024)
RAPTOR uses recursive clustering and summarization to build a hierarchical tree structure from documents. Key innovations:

- **Gaussian Mixture Models (GMM)** for soft clustering
- **UMAP** for dimensionality reduction (384D → 50D)
- **Multi-level summarization** using LLMs
- **Tree-based retrieval** at multiple abstraction levels

### GraphRAG (Microsoft)
GraphRAG constructs knowledge graphs by extracting entities and relationships:

- **Named Entity Recognition (NER)** for entity extraction
- **Relationship extraction** between entities
- **Graph traversal** for multi-hop reasoning
- **Community detection** for topic clustering

## 📈 Key Achievements

### RAPTOR Implementation
- ✅ Processed 728 chunks → 101 optimized (86% reduction)
- ✅ Built 104-node hierarchical tree
- ✅ Silhouette score: 0.616 (excellent clustering)
- ✅ 3x better context coverage vs flat RAG

### GraphRAG Implementation
- ✅ Neo4j knowledge graph construction
- ✅ Entity and relationship extraction
- ✅ Hybrid retrieval (graph + semantic)
- ✅ Multi-hop reasoning support

### Comparison Web App
- ✅ Real-time side-by-side comparison
- ✅ PDF/DOCX file upload support
- ✅ Interactive metrics visualization
- ✅ Query history tracking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [RAPTOR Paper (ICLR 2024)](https://arxiv.org/abs/2401.18059)
- [GraphRAG by Microsoft](https://github.com/microsoft/graphrag)
- [Sentence Transformers](https://www.sbert.net/)
- [UMAP](https://umap-learn.readthedocs.io/)
- [Neo4j](https://neo4j.com/)

## 📧 Contact

**Roman Ahmad** - [@ROMAN-AHMAD-NAZAR](https://github.com/ROMAN-AHMAD-NAZAR)

Project Link: [https://github.com/ROMAN-AHMAD-NAZAR/raptor-vs-graphrag](https://github.com/ROMAN-AHMAD-NAZAR/raptor-vs-graphrag)

---
⭐ **Star this repo if you find it helpful!**
