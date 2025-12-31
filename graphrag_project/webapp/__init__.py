# webapp/__init__.py
"""
RAPTOR vs GraphRAG Comparison Web Application
Dynamic user-based comparison with real-time metrics

This module provides a web interface for dynamically comparing
RAPTOR and GraphRAG retrieval systems with:
- Dynamic document input (paste text or upload files)
- Real-time query comparison
- Metrics visualization
- Query history tracking

Components:
- unified_retriever: RAPTOR and GraphRAG retrieval implementations
- app: Flask web server with REST API
- templates/: HTML templates
- static/: CSS and JavaScript files
"""

__version__ = "1.0.0"
__author__ = "RAPTOR Research Team"
