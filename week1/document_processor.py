# week1/document_processor.py
import os
import sys
import nltk

print("=== RAPTOR Implementation - Week 1 ===")
print("Setting up document processing...")

# Download NLTK data
print("\nDownloading NLTK data...")
nltk.download('punkt', quiet=True)
print("NLTK data downloaded.")

# Check Python version
print(f"\nPython version: {sys.version}")

# Check current directory
print(f"Current directory: {os.getcwd()}")
print(f"Virtual environment: {sys.prefix}")