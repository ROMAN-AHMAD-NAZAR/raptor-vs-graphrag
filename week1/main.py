"""Week 1 Main: Document Processing Pipeline"""
import os
import sys
import glob

# Add week1 directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from step1_loader import DocumentLoader, load_all_documents
from step2_processor import TextProcessor


def main():
    print("=== WEEK 1: Document Processing ===\n")
    
    # Get the project root directory (parent of week1)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    output_path = os.path.join(project_root, "outputs", "week1_chunks.pkl")
    
    # Find all documents (PDF and TXT)
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    all_files = pdf_files + txt_files
    
    if not all_files:
        print(f"No documents found in {data_dir}")
        return
    
    print(f"Found {len(all_files)} document(s) in {data_dir}")
    for f in all_files:
        print(f"  - {os.path.basename(f)}")
    print()
    
    # Ensure outputs directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load all documents
    all_pages = []
    for file_path in all_files:
        loader = DocumentLoader(file_path)
        pages = loader.load()
        all_pages.extend(pages)
    
    print(f"\nTotal pages loaded: {len(all_pages)}")
    
    # 2. Process each page
    processor = TextProcessor(chunk_size=100, chunk_overlap=20)
    all_chunks = []
    chunk_metadata = []  # Store metadata for each chunk
    
    for page in all_pages:
        page_num = page['page_number']
        if isinstance(page_num, int) and page_num == 0:
            page_num = 1  # Fix 0-indexed pages
        
        chunks = processor.create_chunks(page['page_content'])
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                'chunk_id': len(all_chunks) - 1,
                'source': page['source'],
                'page_number': page_num,
                'chunk_index': i
            })
        
        print(f"Page {page_num} ({os.path.basename(page['source'])}): {len(chunks)} chunks")
    
    # 3. Save results
    import pickle
    results = {
        'chunks': all_chunks,
        'metadata': chunk_metadata
    }
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total chunks created: {len(all_chunks)}")
    if all_chunks:
        print(f"Average words per chunk: {sum(len(c.split()) for c in all_chunks)/len(all_chunks):.1f}")
    print(f"Results saved to: {output_path}")
    
    # Show sample chunk
    if all_chunks:
        print(f"\nSample chunk:")
        print(f"{all_chunks[0]}")


if __name__ == "__main__":
    main()
