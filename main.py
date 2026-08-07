import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import io
import PyPDF2
import docx
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# ==========================================
# CONFIGURATION (Loaded from .env)
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Initialize Qdrant & Embedding Model 
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

class SearchRequest(BaseModel):
    text: str

# ==========================================
# FRONTEND (HTML / CSS / JS)
# ==========================================
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Doc Vector Search</title>
    <style>
        /* Base font size for the whole page */
        body { font-family: system-ui; margin: 0; padding: 20px; padding-bottom: 220px; background: #f4f4f9; font-size: 18px; }
        
        h2 { font-size: 28px; margin-bottom: 10px; }
        .upload-section { margin-bottom: 20px; }
        
        input[type="file"] { font-size: 18px; padding: 5px; cursor: pointer; }
        
        /* Document display box */
        #content-display { white-space: pre-wrap; line-height: 1.8; border: 1px solid #ddd; padding: 25px; background: white; height: 50vh; overflow-y: auto; border-radius: 8px; font-size: 20px; }
        
        /* Scaled up the selection bar layout */
        .selection-bar { position: fixed; bottom: 0; left: 0; width: 100%; background: white; border-top: 2px solid #007bff; padding: 20px; box-shadow: 0 -4px 12px rgba(0,0,0,0.1); display: flex; flex-direction: column; align-items: center; gap: 15px; z-index: 1000; box-sizing: border-box; }
        
        #selected-text { width: 90%; padding: 15px; border: 1px dashed #aaa; background: #fafafa; min-height: 30px; text-align: center; border-radius: 4px; color: #555; font-size: 22px; }
        
        /* New Manual Input Field */
        #manual-input { width: 90%; padding: 15px; border: 1px solid #ccc; border-radius: 6px; font-size: 20px; box-sizing: border-box; transition: border-color 0.2s; }
        #manual-input:focus { outline: none; border-color: #007bff; box-shadow: 0 0 5px rgba(0,123,255,0.3); }
        
        button { padding: 15px 32px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 6px; font-weight: bold; font-size: 20px; transition: background 0.2s; }
        button:hover { background: #0056b3; }
        #result-filename { font-weight: bold; color: #28a745; min-height: 24px; font-size: 24px; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="upload-section">
        <h2>Upload Document</h2>
        <input type="file" id="file-upload" accept=".txt,.pdf,.doc,.docx" />
    </div>
    
    <div id="content-display">Your document text will appear here. Highlight any text to search.</div>

    <div class="selection-bar">
        <div id="selected-text">No text selected yet...</div>
        <input type="text" id="manual-input" placeholder="Or type your search query here and press Enter..." />
        <button id="search-btn">Search Vector</button>
        <div id="result-filename"></div>
    </div>

    <script>
        document.getElementById('file-upload').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            document.getElementById('content-display').textContent = "Extracting text...";
            const formData = new FormData();
            formData.append('file', file);
            
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            document.getElementById('content-display').textContent = data.text;
        });

        // Handle highlighting text from the document
        document.getElementById('content-display').addEventListener('mouseup', () => {
            const selection = window.getSelection().toString().trim();
            if (selection) {
                document.getElementById('selected-text').textContent = selection;
                document.getElementById('manual-input').value = ""; // Clear manual input to avoid confusion
                document.getElementById('result-filename').textContent = "";
            }
        });

        // Handle manual text entry and "Enter" key press
        document.getElementById('manual-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault(); // Prevent default form submission behavior
                const manualText = e.target.value.trim();
                
                if (manualText) {
                    // Override the selected text display
                    document.getElementById('selected-text').textContent = manualText;
                    document.getElementById('result-filename').textContent = "";
                    
                    // Automatically trigger the search button
                    document.getElementById('search-btn').click();
                }
            }
        });

        // The main search execution logic
        document.getElementById('search-btn').addEventListener('click', async () => {
            const text = document.getElementById('selected-text').textContent;
            if (!text || text === 'No text selected yet...') return alert('Please highlight or type text first.');

            const btn = document.getElementById('search-btn');
            btn.textContent = "Searching...";
            document.getElementById('result-filename').textContent = "";
            
            const res = await fetch('/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();
            
            btn.textContent = "Search Vector";

            if (data.image_name) {
                document.getElementById('result-filename').textContent = "Matched Target File: " + data.image_name;
            } else {
                alert(data.error || 'No match found.');
            }
        });
    </script>
</body>
</html>
"""

# ==========================================
# API ROUTES
# ==========================================
@app.get("/")
def serve_frontend():
    return HTMLResponse(content=html_content)

@app.post("/upload")
async def parse_document(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename.lower()
    text = ""
    
    try:
        if filename.endswith('.txt'):
            text = content.decode('utf-8', errors='ignore')
        elif filename.endswith('.pdf'):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif filename.endswith('.docx') or filename.endswith('.doc'):
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = "Unsupported file format."
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
        
    return {"text": text}

@app.post("/search")
def search_and_map_image(req: SearchRequest):
    # 1. Embed the highlighted/typed text
    vectors = list(embedding_model.embed([req.text]))
    query_vector = vectors[0].tolist()
    
    # 2. Search Qdrant using the new query_points API
    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=1
    )
    
    # query_points returns a Response object containing a list of 'points'
    hits = response.points
    
    if not hits:
        return {"error": "No close vectors found in Qdrant."}
        
    matched_text = hits[0].payload.get("text", "")
    if not matched_text:
        return {"error": "Vector matched, but payload had no 'text' field."}
        
    # 3. Extract the first word
    first_word = matched_text.split()[0]
    image_name = f"{first_word}.png"
    
    return {
        "matched_text": matched_text,
        "image_name": image_name
    }