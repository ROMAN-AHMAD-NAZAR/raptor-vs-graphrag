"""Document loaders for text and PDF files"""
import os


class SimpleTextLoader:
    """Simple text loader for .txt files"""
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        print(f"Loading text document: {self.file_path}")
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Simulate multiple pages for testing
        pages = []
        paragraphs = text.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            if para.strip():  # Skip empty paragraphs
                pages.append({
                    'page_content': para.strip(),
                    'page_number': i,
                    'source': self.file_path
                })
        
        print(f"Loaded {len(pages)} 'pages' (paragraphs)")
        return pages


class PDFLoader:
    """PDF document loader using PyPDF2"""
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load(self):
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF loading. Install with: pip install PyPDF2")
        
        print(f"Loading PDF document: {self.file_path}")
        
        reader = PdfReader(self.file_path)
        pages = []
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({
                    'page_content': text.strip(),
                    'page_number': i + 1,  # 1-indexed for PDFs
                    'source': self.file_path
                })
        
        print(f"Loaded {len(pages)} pages from PDF")
        return pages


class DocumentLoader:
    """Universal document loader that auto-detects file type"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.extension = os.path.splitext(file_path)[1].lower()
    
    def load(self):
        if self.extension == '.pdf':
            loader = PDFLoader(self.file_path)
        elif self.extension in ['.txt', '.md', '.text']:
            loader = SimpleTextLoader(self.file_path)
        else:
            # Default to text loader
            print(f"Unknown extension '{self.extension}', treating as text file")
            loader = SimpleTextLoader(self.file_path)
        
        return loader.load()


def load_all_documents(directory_path, extensions=None):
    """Load all documents from a directory"""
    if extensions is None:
        extensions = ['.pdf', '.txt', '.md']
    
    all_pages = []
    
    for filename in os.listdir(directory_path):
        ext = os.path.splitext(filename)[1].lower()
        if ext in extensions:
            file_path = os.path.join(directory_path, filename)
            loader = DocumentLoader(file_path)
            pages = loader.load()
            all_pages.extend(pages)
    
    return all_pages
