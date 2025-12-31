# RAPTOR vs GraphRAG Web Application

A dynamic web interface for comparing RAPTOR and GraphRAG retrieval systems with real-time metrics.

## Features

- **Dynamic Document Input**: Paste text directly or upload files (.txt, .md)
- **Real-time Comparison**: Compare RAPTOR and GraphRAG on the same queries
- **Metrics Dashboard**: View accuracy, query time, coverage, and more
- **Interactive Charts**: Visual comparison of retrieval scores
- **Query History**: Track past queries and their results
- **Winner Tracking**: See which system performs better overall

## Quick Start

### Option 1: Run the batch script (Windows)
```bash
cd webapp
run_webapp.bat
```

### Option 2: Run with Python directly
```bash
cd webapp
pip install -r requirements.txt
python app.py
```

### Option 3: Run from project root
```bash
cd graphrag_project
python -m webapp.app
```

## Access the Application

Once running, open your browser to:
```
http://localhost:5000
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main comparison UI |
| `/api/health` | GET | Health check |
| `/api/index` | POST | Index a document |
| `/api/query` | POST | Run comparison query |
| `/api/stats` | GET | Get statistics |
| `/api/history` | GET | Get query history |
| `/api/clear` | POST | Clear all data |
| `/api/sample-document` | GET | Get sample document |
| `/api/sample-queries` | GET | Get sample queries |

## Usage Guide

### 1. Index a Document

**Option A: Paste Text**
1. Enter or paste your document text in the "Document Input" area
2. Click "Index Document"

**Option B: Upload File**
1. Click "Upload File" button
2. Select a .txt or .md file
3. Click "Index Document"

**Option C: Use Sample**
1. Click "Load Sample Document" for a pre-loaded example

### 2. Run Queries

1. Enter your query in the "Query Input" field
2. Select the number of results (Top 3, 5, or 10)
3. Click "Compare" or press Enter

### 3. Analyze Results

The results section shows:
- **Winner Banner**: Which system won this query
- **Metrics Cards**: Detailed metrics for each system
  - Average Score
  - Max Score
  - Number of Results
  - Query Time (ms)
  - Coverage (%)
- **Score Chart**: Bar chart comparing scores by rank
- **Results List**: Side-by-side comparison of retrieved content

### 4. Track Progress

- Header stats show total wins for each system
- Query history shows all past comparisons
- Click any history item to re-run that query

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Avg Score** | Mean similarity score across all results |
| **Max Score** | Highest similarity score achieved |
| **Query Time** | Time taken to process the query (ms) |
| **Coverage** | Percentage of relevant content captured |
| **Num Results** | Number of results returned |

## System Architecture

```
webapp/
├── __init__.py           # Package initialization
├── app.py                # Flask web server
├── unified_retriever.py  # RAPTOR & GraphRAG implementations
├── requirements.txt      # Python dependencies
├── run_webapp.bat        # Windows startup script
├── README.md             # This file
├── templates/
│   └── index.html        # Main HTML template
└── static/
    ├── css/
    │   └── style.css     # Application styles
    └── js/
        └── app.js        # Frontend JavaScript
```

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Charts**: Chart.js
- **Icons**: Font Awesome
- **Fonts**: Inter (Google Fonts)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)

## Configuration

Environment variables (optional):
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123
```

## Troubleshooting

### "No document indexed" error
- Make sure to click "Index Document" after entering text

### Slow first query
- The first query loads the embedding model (~100MB download)
- Subsequent queries will be faster

### Neo4j connection errors
- GraphRAG features require Neo4j running locally
- Install from: https://neo4j.com/download/

### Missing dependencies
```bash
pip install flask flask-cors sentence-transformers numpy scikit-learn
```

## License

MIT License - See project root for details
