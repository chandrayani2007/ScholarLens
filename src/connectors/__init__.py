"""Scholarly source connectors package."""
from src.connectors.base import BaseConnector
from src.connectors.arxiv import ArxivConnector
from src.connectors.pubmed import EuropePmcConnector
from src.connectors.semantic_scholar import SemanticScholarConnector

__all__ = [
    "BaseConnector",
    "ArxivConnector",
    "EuropePmcConnector",
    "SemanticScholarConnector",
]
