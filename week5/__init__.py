# week5/__init__.py
"""
Week 5: Storage, Retrieval & Performance Comparison

The FINAL week - bringing everything together!

Components:
- QdrantManager: Vector database storage for RAPTOR tree
- RaptorRetriever: Hierarchical multi-level retrieval
- RAGBaseline: Traditional flat retrieval for comparison
- PerformanceEvaluator: Metrics and comparison tools
- RAPTORDemo: Interactive demonstration
"""

from .qdrant_manager import QdrantManager
from .raptor_retriever import RaptorRetriever
from .rag_baseline import RAGBaseline
from .evaluator import PerformanceEvaluator
from .demo_app import RAPTORDemo

__all__ = [
    'QdrantManager',
    'RaptorRetriever',
    'RAGBaseline',
    'PerformanceEvaluator',
    'RAPTORDemo'
]
