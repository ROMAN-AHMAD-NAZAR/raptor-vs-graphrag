# week5/demo_app.py
"""
Interactive Demo Application for RAPTOR

Showcases the difference between hierarchical and flat retrieval
"""

import sys
from typing import Dict, List


class RAPTORDemo:
    """
    Interactive demo to showcase RAPTOR capabilities
    
    Features:
    - Sample queries
    - Custom queries
    - Side-by-side comparison
    - Batch evaluation
    """
    
    def __init__(self, raptor_retriever, rag_baseline):
        """
        Initialize demo
        
        Args:
            raptor_retriever: RaptorRetriever instance
            rag_baseline: RAGBaseline instance
        """
        self.raptor = raptor_retriever
        self.rag = rag_baseline
        
        print(f"\n🚀 Starting RAPTOR Demo")
        print(f"=" * 50)
    
    def run_demo(self):
        """Run interactive demo"""
        print(f"\nWelcome to the RAPTOR Demo!")
        print(f"Compare hierarchical retrieval vs traditional RAG")
        
        # Sample queries based on common questions
        sample_queries = [
            "What is RAPTOR?",
            "How does hierarchical retrieval work?",
            "What are the advantages over normal RAG?",
            "How are documents clustered?",
            "What is abstractive summarization?"
        ]
        
        while True:
            print(f"\n" + "=" * 50)
            print(f"Choose an option:")
            print(f"1. Use sample query")
            print(f"2. Enter custom query")
            print(f"3. Compare on all sample queries")
            print(f"4. Exit")
            
            choice = input("\nYour choice (1-4): ").strip()
            
            if choice == "1":
                self._sample_query_demo(sample_queries)
            elif choice == "2":
                self._custom_query_demo()
            elif choice == "3":
                self._batch_comparison(sample_queries)
            elif choice == "4":
                print(f"\n👋 Thanks for trying RAPTOR!")
                break
            else:
                print(f"❌ Invalid choice. Please try again.")
    
    def _sample_query_demo(self, sample_queries: List[str]):
        """Demo with sample queries"""
        print(f"\n📝 Sample queries:")
        for i, query in enumerate(sample_queries, 1):
            print(f"  {i}. {query}")
        
        try:
            choice = int(input(f"\nSelect query (1-{len(sample_queries)}): "))
            if 1 <= choice <= len(sample_queries):
                query = sample_queries[choice - 1]
                self._compare_query(query)
            else:
                print(f"❌ Invalid choice")
        except ValueError:
            print(f"❌ Please enter a number")
    
    def _custom_query_demo(self):
        """Demo with custom query"""
        query = input("\nEnter your query: ").strip()
        if query:
            self._compare_query(query)
        else:
            print(f"❌ Please enter a query")
    
    def _compare_query(self, query: str):
        """Compare RAPTOR vs RAG for a single query"""
        print(f"\n🔍 QUERY: '{query}'")
        print(f"\n" + "-" * 40)
        
        # RAPTOR retrieval
        print(f"\n🌳 RAPTOR (Hierarchical Retrieval):")
        raptor_results = self.raptor.hierarchical_search(query, top_k=5)
        
        for i, result in enumerate(raptor_results, 1):
            source = "📝 Summary" if result.get('is_summary', False) else "📄 Chunk"
            depth = result.get('depth', 0)
            score = result.get('score', 0)
            text = result.get('text', '')[:120]
            print(f"\n{i}. {source} (Depth {depth}, Score: {score:.3f})")
            print(f"   {text}...")
        
        # RAG retrieval
        print(f"\n" + "-" * 40)
        print(f"\n📊 Normal RAG (Flat Retrieval):")
        rag_results = self.rag.search(query, top_k=5)
        
        for i, result in enumerate(rag_results, 1):
            score = result.get('score', 0)
            text = result.get('text', '')[:120]
            print(f"\n{i}. Chunk (Score: {score:.3f})")
            print(f"   {text}...")
        
        # Comparison
        self._print_comparison(raptor_results, rag_results)
    
    def _print_comparison(self, raptor_results: List[Dict], rag_results: List[Dict]):
        """Print comparison between results"""
        print(f"\n" + "=" * 50)
        print(f"\n📈 COMPARISON:")
        
        # Analyze result types
        raptor_summaries = sum(1 for r in raptor_results if r.get('is_summary', False))
        raptor_chunks = len(raptor_results) - raptor_summaries
        
        raptor_depths = set(r.get('depth', 0) for r in raptor_results)
        
        print(f"\n🌳 RAPTOR returned:")
        print(f"   • {raptor_summaries} summaries, {raptor_chunks} chunks")
        print(f"   • Depth levels: {sorted(raptor_depths)}")
        print(f"   • Provides hierarchical context")
        print(f"   • Multi-level understanding")
        
        print(f"\n📊 Normal RAG returned:")
        print(f"   • {len(rag_results)} chunks only")
        print(f"   • No hierarchical context")
        print(f"   • Flat similarity search")
        
        # Quality comparison
        if raptor_results and rag_results:
            avg_raptor_score = sum(r.get('score', 0) for r in raptor_results) / len(raptor_results)
            avg_rag_score = sum(r.get('score', 0) for r in rag_results) / len(rag_results)
            
            print(f"\n📊 Average relevance scores:")
            print(f"   RAPTOR: {avg_raptor_score:.3f}")
            print(f"   Normal RAG: {avg_rag_score:.3f}")
            
            if avg_rag_score > 0:
                improvement = ((avg_raptor_score - avg_rag_score) / avg_rag_score) * 100
                if improvement > 0:
                    print(f"   🚀 RAPTOR improvement: {improvement:+.1f}%")
                else:
                    print(f"   📉 Difference: {improvement:.1f}%")
    
    def _batch_comparison(self, queries: List[str]):
        """Compare on multiple queries"""
        print(f"\n📊 Batch Comparison on {len(queries)} queries...")
        
        total_raptor_score = 0
        total_rag_score = 0
        
        for query in queries:
            print(f"\n  Query: '{query}'")
            
            raptor_results = self.raptor.hierarchical_search(query, top_k=5)
            rag_results = self.rag.search(query, top_k=5)
            
            # Quick stats
            raptor_summaries = sum(1 for r in raptor_results if r.get('is_summary', False))
            raptor_avg = sum(r.get('score', 0) for r in raptor_results) / max(1, len(raptor_results))
            rag_avg = sum(r.get('score', 0) for r in rag_results) / max(1, len(rag_results))
            
            total_raptor_score += raptor_avg
            total_rag_score += rag_avg
            
            print(f"    RAPTOR: {len(raptor_results)} results ({raptor_summaries} summaries), avg score: {raptor_avg:.3f}")
            print(f"    RAG: {len(rag_results)} results, avg score: {rag_avg:.3f}")
        
        print(f"\n" + "=" * 50)
        print(f"📊 BATCH RESULTS:")
        
        avg_raptor = total_raptor_score / len(queries)
        avg_rag = total_rag_score / len(queries)
        
        print(f"   Average RAPTOR score: {avg_raptor:.3f}")
        print(f"   Average RAG score: {avg_rag:.3f}")
        
        if avg_rag > 0:
            improvement = ((avg_raptor - avg_rag) / avg_rag) * 100
            print(f"   Overall improvement: {improvement:+.1f}%")
        
        print(f"\n✅ Comparison complete!")
        print(f"\n💡 Key insight: RAPTOR provides context through summaries,")
        print(f"   while RAG only returns isolated chunks.")
    
    def quick_demo(self, query: str = "What is RAPTOR?"):
        """Run a quick non-interactive demo"""
        print(f"\n🎯 Quick Demo: '{query}'")
        self._compare_query(query)
