# week5_comparison/__init__.py
"""
Week 5: The Final Showdown - RAPTOR vs GraphRAG
Comprehensive comparison and paper generation module
"""

from .results_loader import ResultsLoader
from .comparison_engine import ComparisonEngine, StatisticalTest
from .paper_generator import PaperGenerator
from .visualization_generator import VisualizationGenerator
from .presentation_generator import PresentationGenerator

__all__ = [
    'ResultsLoader',
    'ComparisonEngine',
    'StatisticalTest',
    'PaperGenerator',
    'VisualizationGenerator',
    'PresentationGenerator'
]
