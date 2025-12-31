# week4/__init__.py
"""
Week 4: Intelligent Summarization for RAPTOR

Components:
- SummarizationEngine: Generate abstractive summaries using transformers
- SummaryEnhancer: Evaluate and improve summary quality
- TreeEnricher: Add summaries to RAPTOR tree nodes
- TreeNavigator: Navigate and query the enriched tree
"""

from .summarization_engine import SummarizationEngine
from .summary_enhancer import SummaryEnhancer
from .tree_enricher import TreeEnricher, TreeNavigator

__all__ = [
    'SummarizationEngine',
    'SummaryEnhancer', 
    'TreeEnricher',
    'TreeNavigator'
]
