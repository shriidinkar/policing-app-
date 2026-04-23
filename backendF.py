import os
from pathlib import Path
from typing import List, Dict
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx
import io

app = Flask(__name__)
CORS(app)

class RAGSystem:
    def __init__(
        self,
        qdrant_url: str,
        qdrant_api_key: str,
        ollama_url: str = "http://localhost:11434",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.ollama_url = ollama_url
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
    
    def _sanitize_collection_name(self, filename: str) -> str:
        """Convert filename to valid collection name"""
        # Remove extension and special characters
        name = Path(filename).stem
        # Replace special chars with underscore
        name = ''.join(c if c.isalnum() else '_' for c in name)
        # Ensure it starts with a letter
        if not name[0].isalpha():
            name = 'doc_' + name
        return name.lower()
    
    def _clean_response(self, text: str) -> str:
        """Clean response text from special characters"""
        import re
        
        # Remove ALL box/square/bullet Unicode characters (U+25A0 to U+25FF range)
        text = re.sub(r'[\u2500-\u25FF]', '', text)
        
        # Remove other common bullets and symbols
        text = re.sub(r'[•◦▪▫■□▸▹►▻◆◇○●✓✔✗✘]', '', text)
        
        # Remove multiple consecutive dashes or hyphens that might be from lists
        text = re.sub(r'-{2,}', '', text)
        
        # Remove emoji-like characters that might not display well
        text = re.sub(r'[\U0001F300-\U0001F9FF]', '', text)
        
        # Normalize whitespace - multiple spaces/newlines to single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean text from special characters and formatting symbols"""
        import re
        
        # Remove ALL box/square/bullet Unicode characters
        text = re.sub(r'[\u2500-\u25FF]', '', text)
        
        # Remove other common bullets and symbols
        text = re.sub(r'[•◦▪▫■□▸▹►▻◆◇○●✓✔✗✘]', '', text)
        
        # Remove multiple consecutive dashes
        text = re.sub(r'-{2,}', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _read_txt(self, file_content: bytes) -> str:
        text = file_content.decode('utf-8')
        return self._clean_text(text)
    
    def _read_pdf(self, file_content: bytes) -> str:
        text = ""
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        for page in pdf_reader.pages:
            text += page.extract_text()
        return self._clean_text(text)
    
    def _read_docx(self, file_content: bytes) -> str:
        doc = docx.Document(io.BytesIO(file_content))
        text = "\n".join([para.text for para in doc.paragraphs])
        return self._clean_text(text)
    
    def read_document(self, file_content: bytes, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        
        if ext == '.txt':
            return self._read_txt(file_content)
        elif ext == '.pdf':
            return self._read_pdf(file_content)
        elif ext == '.docx':
            return self._read_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
            
            if i + chunk_size >= len(words):
                break
        
        return chunks
    
    def create_collection(self, collection_name: str):
        """Create a new collection"""
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            return True
        except Exception as e:
            # Collection might already exist
            return False
    
    def add_document(self, file_content: bytes, filename: str, chunk_size: int = 500, overlap: int = 50) -> Dict:
        """Add document to its own collection"""
        collection_name = self._sanitize_collection_name(filename)
        
        # Create collection for this document
        self.create_collection(collection_name)
        
        # Read and process document
        text = self.read_document(file_content, filename)
        chunks = self.chunk_text(text, chunk_size, overlap)
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False)
        
        # Upload chunks to collection
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        "text": chunk,
                        "source": filename,
                        "chunk_id": idx
                    }
                )
            )
        
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return {
            "collection_name": collection_name,
            "filename": filename,
            "num_chunks": len(chunks)
        }
    
    def get_all_documents(self) -> List[Dict]:
        """Get all collections (documents)"""
        try:
            collections = self.qdrant_client.get_collections().collections
            documents = []
            
            for collection in collections:
                # Get collection info to find original filename
                points, _ = self.qdrant_client.scroll(
                    collection_name=collection.name,
                    limit=1
                )
                
                if points:
                    filename = points[0].payload.get("source", collection.name)
                else:
                    filename = collection.name
                
                documents.append({
                    "collection_name": collection.name,
                    "filename": filename
                })
            
            return documents
        except Exception as e:
            return []
    
    def delete_document(self, collection_name: str) -> bool:
        """Delete a collection (document)"""
        try:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            return False
    
    def retrieve_context(self, query: str, collection_name: str, top_k: int = 3) -> List[Dict]:
        """Retrieve context from a specific collection"""
        query_embedding = self.embedding_model.encode(query).tolist()
        
        search_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        contexts = []
        for result in search_results:
            contexts.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "score": result.score
            })
        
        return contexts
    
    def generate_answer(self, query: str, contexts: List[Dict], model: str = "llama3.2") -> str:
        """Generate answer using Ollama"""
        context_text = "\n\n".join([
            f"Context {i+1}:\n{ctx['text']}" 
            for i, ctx in enumerate(contexts)
        ])
        
        prompt = f"""You are a precise assistant. Answer the question based ONLY on the provided context. 

RULES:
1. Give a complete but concise answer (3-5 sentences)
2. Use simple plain text - NO bullet points, NO lists, NO special characters
3. Make sure your answer is complete and ends properly
4. If the answer is not in the context, say "I cannot find this information in the provided document"

Context:
{context_text}

Question: {query}

Complete Answer:"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.5,
                "num_predict": 200  # Allow more tokens for complete answers
            }
        )
        
        if response.status_code == 200:
            answer = response.json()["response"]
            # Clean the response
            cleaned = self._clean_response(answer)
            return cleaned
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
    
    def ask(self, question: str, collection_name: str, model: str = "llama3.2", top_k: int = 3) -> Dict:
        """Ask a question about a specific document"""
        contexts = self.retrieve_context(question, collection_name, top_k)
        answer = self.generate_answer(question, contexts, model)
        
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts
        }


# Initialize RAG system
# Replace these with your Qdrant Cloud credentials
QDRANT_URL = "https://802e618e-42b8-4602-afe4-c10487799d7c.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.pNfSHkYcem7Ms_red4y0HgYqBPFYf6jOOxt2n5vpjWM"

rag = RAGSystem(
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)


# API Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_document():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        file_content = file.read()
        result = rag.add_document(file_content, file.filename)
        
        return jsonify({
            "success": True,
            "message": f"Document '{file.filename}' uploaded successfully",
            "data": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    try:
        documents = rag.get_all_documents()
        return jsonify({"documents": documents})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/documents/<collection_name>', methods=['DELETE'])
def delete_document(collection_name):
    try:
        success = rag.delete_document(collection_name)
        if success:
            return jsonify({"success": True, "message": "Document deleted successfully"})
        else:
            return jsonify({"error": "Failed to delete document"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')
        collection_name = data.get('collection_name')
        model = data.get('model', 'llama3.2')
        top_k = data.get('top_k', 3)
        
        if not question or not collection_name:
            return jsonify({"error": "Question and collection_name are required"}), 400
        
        result = rag.ask(question, collection_name, model, top_k)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting RAG Backend Server...")
    print("Make sure to update QDRANT_URL and QDRANT_API_KEY in the code")
    app.run(debug=True, port=5000)