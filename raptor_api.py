# raptor_api.py
"""
Production-Ready RAPTOR REST API

Run with: uvicorn raptor_api:app --reload
"""

import sys
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

# Add week5 to path
sys.path.insert(0, str(Path(__file__).parent / "week5"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results")
    use_raptor: bool = Field(default=True, description="Use RAPTOR (True) or normal RAG (False)")


class SearchResult(BaseModel):
    id: int
    score: float
    text: str
    depth: int
    is_summary: bool
    source: str


class QueryResponse(BaseModel):
    query: str
    method: str
    results: List[SearchResult]
    summary_count: int
    chunk_count: int


class CompareResponse(BaseModel):
    query: str
    raptor_results: List[SearchResult]
    rag_results: List[SearchResult]
    raptor_avg_score: float
    rag_avg_score: float
    improvement_percent: float


# Global instances
raptor_retriever = None
rag_baseline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAPTOR on startup"""
    global raptor_retriever, rag_baseline
    
    print("🚀 Initializing RAPTOR API...")
    
    try:
        from qdrant_manager import QdrantManager
        from raptor_retriever import RaptorRetriever
        from rag_baseline import RAGBaseline
        from sentence_transformers import SentenceTransformer
        
        # Load data
        with open("outputs/week3_results.pkl", 'rb') as f:
            week3_data = pickle.load(f)
        
        chunks = week3_data['chunks']
        embeddings = week3_data['embeddings']
        
        with open("outputs/summaries/enriched_tree.pkl", 'rb') as f:
            tree_data = pickle.load(f)
        
        # Build retrieval tree
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        retrieval_tree = {'embeddings': [], 'texts': [], 'metadata': []}
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            retrieval_tree['embeddings'].append(embedding)
            retrieval_tree['texts'].append(chunk)
            retrieval_tree['metadata'].append({
                'id': f'chunk_{i}', 'depth': 2, 'is_summary': False, 'num_children': 0
            })
        
        for node in tree_data.get('nodes', []):
            if node.get('depth', 0) < 2:
                summary = node.get('summary', node.get('text', ''))
                if summary:
                    retrieval_tree['embeddings'].append(embedder.encode(summary))
                    retrieval_tree['texts'].append(summary)
                    retrieval_tree['metadata'].append({
                        'id': node.get('id', 'summary'),
                        'depth': node.get('depth', 0),
                        'is_summary': True,
                        'num_children': node.get('num_children', 0)
                    })
        
        retrieval_tree['embeddings'] = np.array(retrieval_tree['embeddings'])
        
        # Initialize Qdrant
        qdrant = QdrantManager(":memory:")
        qdrant.create_collection(vector_size=384)
        qdrant.store_raptor_tree(retrieval_tree)
        qdrant.store_normal_rag(chunks, embeddings)
        
        raptor_retriever = RaptorRetriever(qdrant)
        rag_baseline = RAGBaseline(qdrant)
        
        print("✅ RAPTOR API ready!")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        raise
    
    yield
    
    print("👋 Shutting down RAPTOR API")


# Create app
app = FastAPI(
    title="RAPTOR API",
    description="Recursive Abstractive Processing for Tree-Organized Retrieval",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "healthy",
        "service": "RAPTOR API",
        "endpoints": ["/query", "/compare", "/health"]
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "raptor_ready": raptor_retriever is not None,
        "rag_ready": rag_baseline is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Search using RAPTOR or normal RAG
    """
    if raptor_retriever is None or rag_baseline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if request.use_raptor:
        results = raptor_retriever.hierarchical_search(request.query, request.top_k)
        method = "RAPTOR (Hierarchical)"
    else:
        results = rag_baseline.search(request.query, request.top_k)
        method = "Normal RAG (Flat)"
    
    formatted_results = [
        SearchResult(
            id=r.get('id', 0),
            score=r.get('score', 0),
            text=r.get('text', ''),
            depth=r.get('depth', 0),
            is_summary=r.get('is_summary', False),
            source=method
        )
        for r in results
    ]
    
    return QueryResponse(
        query=request.query,
        method=method,
        results=formatted_results,
        summary_count=sum(1 for r in results if r.get('is_summary', False)),
        chunk_count=sum(1 for r in results if not r.get('is_summary', False))
    )


@app.post("/compare", response_model=CompareResponse)
async def compare(query: str, top_k: int = 5):
    """
    Compare RAPTOR vs Normal RAG side-by-side
    """
    if raptor_retriever is None or rag_baseline is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    raptor_results = raptor_retriever.hierarchical_search(query, top_k)
    rag_results = rag_baseline.search(query, top_k)
    
    raptor_avg = sum(r.get('score', 0) for r in raptor_results) / max(1, len(raptor_results))
    rag_avg = sum(r.get('score', 0) for r in rag_results) / max(1, len(rag_results))
    
    improvement = ((raptor_avg - rag_avg) / rag_avg * 100) if rag_avg > 0 else 0
    
    return CompareResponse(
        query=query,
        raptor_results=[
            SearchResult(
                id=r.get('id', 0), score=r.get('score', 0), text=r.get('text', ''),
                depth=r.get('depth', 0), is_summary=r.get('is_summary', False), source="RAPTOR"
            ) for r in raptor_results
        ],
        rag_results=[
            SearchResult(
                id=r.get('id', 0), score=r.get('score', 0), text=r.get('text', ''),
                depth=r.get('depth', 0), is_summary=r.get('is_summary', False), source="RAG"
            ) for r in rag_results
        ],
        raptor_avg_score=raptor_avg,
        rag_avg_score=rag_avg,
        improvement_percent=improvement
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
