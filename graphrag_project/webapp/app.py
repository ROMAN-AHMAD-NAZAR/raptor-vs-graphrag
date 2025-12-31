# webapp/app.py
"""
Flask Web Application for RAPTOR vs GraphRAG Comparison
Provides REST API and serves frontend
"""

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime
import io

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename


def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    
    # Try PyPDF2 first
    try:
        import PyPDF2
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"PyPDF2 failed: {e}")
    
    # Try pdfplumber as fallback
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"pdfplumber failed: {e}")
    
    # Try pdfminer as last resort
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        text = pdfminer_extract(file_path)
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"pdfminer failed: {e}")
    
    return text


def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except ImportError:
        logging.warning("python-docx not installed. Install with: pip install python-docx")
        return ""
    except Exception as e:
        logging.warning(f"DOCX extraction failed: {e}")
        return ""


def extract_text_from_doc(file_path):
    """Extract text from DOC file (old Word format)"""
    # Try textract
    try:
        import textract
        text = textract.process(file_path).decode('utf-8')
        return text
    except ImportError:
        pass
    except Exception as e:
        logging.warning(f"textract failed: {e}")
    
    # Try antiword via subprocess (if installed)
    try:
        import subprocess
        result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
    except Exception as e:
        logging.warning(f"antiword failed: {e}")
    
    return ""


def extract_text_from_file(file_path):
    """Extract text from various file formats"""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()
    
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.docx':
        return extract_text_from_docx(file_path)
    elif ext == '.doc':
        return extract_text_from_doc(file_path)
    elif ext in ['.txt', '.md', '.text', '.markdown']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    else:
        # Try reading as text
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except:
            return ""

from webapp.unified_retriever import UnifiedRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'uploads'
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize unified retriever
retriever = UnifiedRetriever({
    'raptor_tree_path': None,  # Will be set dynamically
    'neo4j_uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    'neo4j_user': os.getenv('NEO4J_USER', 'neo4j'),
    'neo4j_password': os.getenv('NEO4J_PASSWORD', 'password123')
})

# Store query history
query_history = []


@app.route('/')
def index():
    """Serve the main comparison page"""
    return render_template('index.html')


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'systems': retriever.get_system_stats()
    })


@app.route('/api/index', methods=['POST'])
def index_document():
    """
    Index a document for both RAPTOR and GraphRAG
    
    Accepts:
    - JSON body with 'text' field
    - File upload with 'file' field
    """
    try:
        text = None
        
        # Check for JSON body
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        
        # Check for file upload
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename:
                filename = secure_filename(file.filename)
                filepath = app.config['UPLOAD_FOLDER'] / filename
                file.save(filepath)
                
                # Extract text based on file type
                text = extract_text_from_file(filepath)
                
                if not text.strip():
                    return jsonify({
                        'error': f'Could not extract text from {filename}. For PDF files, install: pip install PyPDF2 pdfplumber. For DOCX files, install: pip install python-docx'
                    }), 400
        
        # Check for form data
        elif request.form.get('text'):
            text = request.form.get('text')
        
        if not text:
            return jsonify({
                'error': 'No text provided. Send JSON with "text" field or upload a file.'
            }), 400
        
        # Index the document
        chunk_size = int(request.args.get('chunk_size', 500))
        result = retriever.index_document(text, chunk_size)
        
        logger.info(f"Indexed document: {result}")
        
        return jsonify({
            'success': True,
            'message': 'Document indexed successfully',
            'details': result,
            'document_length': len(text)
        })
        
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """
    Run a query on both systems and compare results
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query_text = data['query']
        top_k = int(data.get('top_k', 5))
        
        # Run comparison
        result = retriever.compare_retrieval(query_text, top_k)
        
        # Store in history
        query_history.append({
            'query': query_text,
            'timestamp': result['timestamp'],
            'raptor_score': result['raptor']['metrics']['avg_score'],
            'graphrag_score': result['graphrag']['metrics']['avg_score'],
            'winner': result['comparison']['overall_winner']
        })
        
        # Keep only last 50 queries
        if len(query_history) > 50:
            query_history.pop(0)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get system statistics and query history"""
    try:
        stats = retriever.get_system_stats()
        
        # Calculate aggregate metrics from history
        if query_history:
            raptor_wins = sum(1 for q in query_history if q['winner'] == 'RAPTOR')
            graphrag_wins = len(query_history) - raptor_wins
            
            avg_raptor_score = sum(q['raptor_score'] for q in query_history) / len(query_history)
            avg_graphrag_score = sum(q['graphrag_score'] for q in query_history) / len(query_history)
        else:
            raptor_wins = graphrag_wins = 0
            avg_raptor_score = avg_graphrag_score = 0
        
        return jsonify({
            'system_stats': stats,
            'query_history': {
                'total_queries': len(query_history),
                'raptor_wins': raptor_wins,
                'graphrag_wins': graphrag_wins,
                'avg_raptor_score': round(avg_raptor_score, 4),
                'avg_graphrag_score': round(avg_graphrag_score, 4)
            },
            'recent_queries': query_history[-10:]  # Last 10 queries
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/history')
def get_history():
    """Get full query history"""
    return jsonify({
        'history': query_history,
        'total': len(query_history)
    })


@app.route('/api/clear', methods=['POST'])
def clear_data():
    """Clear indexed data and history"""
    global query_history
    query_history = []
    
    # Reinitialize retriever
    global retriever
    retriever = UnifiedRetriever({})
    
    return jsonify({
        'success': True,
        'message': 'Data cleared successfully'
    })


@app.route('/api/sample-document')
def get_sample_document():
    """Get a sample document for testing"""
    sample_text = """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
    
    Abstract: Retrieval-augmented language models can better adapt to changes in world 
    state and incorporate long-tail knowledge. However, most existing methods retrieve 
    only short contiguous chunks from a retrieval corpus, limiting holistic understanding 
    of the overall document context. We introduce RAPTOR, a novel approach that constructs 
    a recursive tree structure from documents. This enables retrieval at different levels 
    of abstraction, from specific details to high-level summaries.
    
    RAPTOR uses a bottom-up approach: it first segments documents into chunks, then 
    recursively clusters and summarizes these chunks to create a tree structure. During 
    retrieval, the system can access any level of this tree, providing both detailed 
    information and broader context.
    
    Key contributions:
    1. A recursive clustering algorithm using Gaussian Mixture Models (GMM)
    2. UMAP dimensionality reduction for efficient clustering
    3. Multi-level summarization using large language models
    4. Tree-based retrieval that outperforms flat retrieval methods
    
    Experimental results show that RAPTOR achieves state-of-the-art performance on 
    question answering benchmarks including QuALITY, QASPER, and NarrativeQA. The 
    hierarchical structure enables RAPTOR to answer both specific factual questions 
    and complex reasoning questions that require understanding document-level context.
    
    GraphRAG: Graph-based Retrieval Augmented Generation
    
    GraphRAG is an alternative approach that constructs knowledge graphs from documents.
    Instead of hierarchical trees, GraphRAG extracts entities and relationships to build
    a graph structure. This enables:
    
    1. Entity-centric retrieval: Find specific entities mentioned in documents
    2. Relationship traversal: Follow connections between concepts
    3. Multi-hop reasoning: Answer questions requiring multiple inference steps
    
    Key differences from RAPTOR:
    - Structure: Knowledge graph vs hierarchical tree
    - Focus: Entities and relationships vs document chunks and summaries
    - Query type: Entity-focused vs document-understanding
    
    Both approaches demonstrate significant improvements over traditional RAG systems
    that use simple flat chunking. The choice between RAPTOR and GraphRAG depends on
    the specific use case and query types.
    
    Evaluation Metrics:
    - NDCG (Normalized Discounted Cumulative Gain): Measures ranking quality
    - Precision@k: Fraction of relevant results in top k
    - Recall@k: Fraction of relevant documents retrieved
    - MRR (Mean Reciprocal Rank): Position of first relevant result
    - Context Coverage: How much relevant context is captured
    
    Conclusion: Structured RAG approaches like RAPTOR and GraphRAG significantly 
    outperform traditional flat RAG. RAPTOR excels at comprehensive document 
    understanding, while GraphRAG excels at precise entity retrieval.
    """
    
    return jsonify({
        'text': sample_text.strip(),
        'description': 'Sample document about RAPTOR and GraphRAG for testing'
    })


@app.route('/api/sample-queries')
def get_sample_queries():
    """Get sample queries for testing"""
    queries = [
        "What is RAPTOR and how does it work?",
        "Compare RAPTOR with GraphRAG",
        "What metrics are used to evaluate retrieval systems?",
        "Explain hierarchical clustering in RAPTOR",
        "What are the key differences between RAPTOR and GraphRAG?",
        "How does GraphRAG extract entities?",
        "What is the purpose of UMAP in RAPTOR?",
        "Which approach is better for complex reasoning?",
        "What benchmarks were used for evaluation?",
        "How does tree-based retrieval work?"
    ]
    
    return jsonify({'queries': queries})


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("RAPTOR vs GraphRAG Comparison Web Application")
    print("=" * 60)
    print()
    print("Starting server at http://localhost:5000")
    print()
    print("API Endpoints:")
    print("  GET  /                    - Main comparison UI")
    print("  GET  /api/health          - Health check")
    print("  POST /api/index           - Index a document")
    print("  POST /api/query           - Run comparison query")
    print("  GET  /api/stats           - Get statistics")
    print("  GET  /api/sample-document - Get sample document")
    print("  GET  /api/sample-queries  - Get sample queries")
    print()
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
