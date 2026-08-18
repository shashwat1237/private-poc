import os
import io
import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import PyPDF2
import docx
import spacy
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams # NEW: Required for creating collections
from fastembed import TextEmbedding
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# ==========================================
# CONFIGURATION & INITIALIZATION
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "medical_docs") 

# Supabase configuration for public bucket
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://xyz.supabase.co")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "public-bucket")

# 1. Initialize Qdrant Client
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- NEW: Create the collection if it doesn't exist ---
if not qdrant.collection_exists(collection_name=QDRANT_COLLECTION):
    print(f"Collection '{QDRANT_COLLECTION}' not found. Creating it now...")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"✅ Collection '{QDRANT_COLLECTION}' created successfully!")
# --------------------------------------------------------

# 2. Initialize FastEmbed Embedding Model
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 3. Initialize Clinical NLP Model
warnings.filterwarnings("ignore", category=FutureWarning)
nlp = spacy.load("en_ner_bc5cdr_md")

# ==========================================
# SCHEMAS
# ==========================================
class SearchRequest(BaseModel):
    text: str

class ProcessRequest(BaseModel):
    text: str

# ==========================================
# NLP PROCESSING LOGIC
# ==========================================
def hyperlink_medical_terms(text: str) -> dict:
    if not text:
        return {"html": "", "count": 0}
        
    doc = nlp(text)
    entity_count = len(doc.ents)
    
    if not doc.ents:
        return {"html": text.replace("\n", "<br>"), "count": 0}

    output_parts = []
    last_idx = 0

    for ent in doc.ents:
        output_parts.append(text[last_idx:ent.start_char])
        hyperlink = f'<span class="med-entity" onclick="triggerEntitySearch(this.innerText)" title="Run Vector Search for {ent.text}">{ent.text}</span>'
        output_parts.append(hyperlink)
        last_idx = ent.end_char

    output_parts.append(text[last_idx:])
    final_html = "".join(output_parts).replace("\n", "<br>")
    
    return {"html": final_html, "count": entity_count}

# ==========================================
# FRONTEND (HTML / CSS / JS)
# ==========================================
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical NLP Terminal</title>
    <style>
        /* Increased padding-bottom from 280px to 450px to make room for the larger image */
        body { background-color: #f4f7f6; font-family: 'Inter', system-ui, -apple-system, sans-serif; margin: 0; padding: 20px; padding-bottom: 450px; color: #2d3748; }
        
        /* Dashboard Header */
        .dashboard-header { border-bottom: 2px solid #e2e8f0; margin-bottom: 20px; padding-bottom: 10px; }
        .dashboard-header h1 { margin: 0; font-size: 28px; color: #1a202c; }
        .dashboard-header p { margin: 5px 0 0 0; color: #718096; font-size: 16px; }

        /* Main Layout */
        .container { display: flex; gap: 20px; height: 50vh; }
        .col { flex: 1; display: flex; flex-direction: column; }
        .col h3 { margin-top: 0; margin-bottom: 10px; font-size: 18px; color: #4a5568; }

        /* Controls */
        input[type="file"] { margin-bottom: 15px; font-size: 16px; padding: 8px; cursor: pointer; border: 1px solid #cbd5e0; border-radius: 6px; background: white;}
        textarea { flex: 1; padding: 15px; font-size: 16px; border: 1px solid #cbd5e0; border-radius: 8px; resize: none; font-family: inherit; line-height: 1.6; }
        textarea:focus { outline: none; border-color: #3182ce; box-shadow: 0 0 0 1px #3182ce; }

        /* Analyzed Output Box */
        #content-display { flex: 1; background-color: #ffffff; border: 1px solid #e0e6ed; border-radius: 8px; padding: 24px; overflow-y: auto; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); font-size: 16px; line-height: 1.8; }
        
        /* Medical Entity Chips */
        .med-entity { background-color: #e6f2ff; color: #0056b3; border: 1px solid #b3d7ff; padding: 2px 8px; border-radius: 12px; cursor: pointer; font-weight: 600; font-size: 0.95em; transition: all 0.2s ease-in-out; display: inline-block; }
        .med-entity:hover { background-color: #0056b3; color: #ffffff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .status-text { font-size: 14px; color: #4a5568; margin-bottom: 10px; font-weight: bold; }

        /* Bottom Search Pipeline Panel */
        .selection-bar { position: fixed; bottom: 0; left: 0; width: 100%; background: white; border-top: 3px solid #3182ce; padding: 20px; box-shadow: 0 -4px 12px rgba(0,0,0,0.05); display: flex; flex-direction: column; align-items: center; gap: 15px; box-sizing: border-box; z-index: 1000; }
        .selection-bar h4 { margin: 0; color: #4a5568; font-size: 16px; text-transform: uppercase; letter-spacing: 1px; }
        
        .search-controls { display: flex; gap: 10px; width: 80%; max-width: 800px; }
        #manual-input { flex: 1; padding: 12px 15px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 18px; transition: border-color 0.2s; }
        #manual-input:focus { outline: none; border-color: #3182ce; }
        
        button { padding: 12px 24px; cursor: pointer; background: #3182ce; color: white; border: none; border-radius: 6px; font-weight: bold; font-size: 18px; transition: background 0.2s; }
        button:hover { background: #2b6cb0; }
        
        /* UPDATED IMAGE DISPLAY LAYOUT */
        #result-container { display: flex; flex-direction: column; align-items: center; gap: 10px; }
        #result-display { font-weight: bold; font-size: 20px; }
        
        /* Greatly increased image size. Changed object-fit to 'contain' so nothing gets cut off */
        #result-image { 
            display: none; 
            max-height: 200px; 
            max-width: 300px; 
            border-radius: 8px; 
            border: 1px solid #cbd5e0; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
            object-fit: contain; 
            background-color: #f8f9fa;
        }
        
        .success { color: #38a169; }
        .error { color: #e53e3e; }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>⚕️ Clinical NLP Terminal</h1>
        <p>Automated Entity Recognition & Supabase Visual Mapping</p>
    </div>

    <input type="file" id="file-upload" accept=".txt,.pdf,.doc,.docx" />

    <div class="container">
        <div class="col">
            <h3>📝 Source Editor</h3>
            <textarea id="source-editor" placeholder="Type or upload clinical text here..."></textarea>
        </div>
        
        <div class="col">
            <h3>🔍 Entity Extraction</h3>
            <div class="status-text" id="status-counter">Status: Waiting for text...</div>
            <div id="content-display">Your analyzed text will appear here.</div>
        </div>
    </div>

    <!-- BOTTOM: Vector Search Pipeline -->
    <div class="selection-bar">
        <h4>Vector Search Pipeline</h4>
        <div class="search-controls">
            <input type="text" id="manual-input" placeholder="Click a highlighted medical term, or type here..." />
            <button id="search-btn">Search Vector DB</button>
        </div>
        
        <!-- Image rendering container -->
        <div id="result-container">
            <div id="result-display"></div>
            <img id="result-image" src="" alt="Matched File" onerror="this.style.display='none'; document.getElementById('result-display').textContent += ' (Image not found in bucket)';" />
        </div>
    </div>

    <script>
        const editor = document.getElementById('source-editor');
        const display = document.getElementById('content-display');
        const counter = document.getElementById('status-counter');
        const manualInput = document.getElementById('manual-input');
        const searchBtn = document.getElementById('search-btn');
        const resultDisplay = document.getElementById('result-display');
        const resultImage = document.getElementById('result-image');
        
        let typingTimer;

        async function processText() {
            const text = editor.value;
            if (!text.trim()) {
                display.innerHTML = "";
                counter.innerText = "Status: Waiting for text...";
                return;
            }
            
            counter.innerText = "Status: Analyzing medical entities...";
            const res = await fetch('/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();
            
            display.innerHTML = data.html;
            counter.innerText = `Status: Analysis complete. ${data.count} medical entities detected.`;
        }

        editor.addEventListener('keyup', () => {
            clearTimeout(typingTimer);
            typingTimer = setTimeout(processText, 500);
        });

        document.getElementById('file-upload').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            counter.innerText = "Status: Extracting text from document...";
            const formData = new FormData();
            formData.append('file', file);
            
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            
            editor.value = data.text;
            processText();
        });

        function triggerEntitySearch(term) {
            manualInput.value = term;
            searchBtn.click();
        }

        searchBtn.addEventListener('click', async () => {
            const text = manualInput.value.trim();
            if (!text) return alert('Please enter or click a term to search.');

            // Reset state
            searchBtn.textContent = "Searching...";
            resultDisplay.textContent = "";
            resultDisplay.className = "";
            resultImage.style.display = "none";
            resultImage.src = "";
            
            try {
                const res = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                const data = await res.json();
                
                if (data.image_name) {
                    resultDisplay.textContent = "🎯 Matched Target File: " + data.image_name;
                    resultDisplay.className = "success";
                    
                    // Show the image pulled directly from Supabase
                    resultImage.src = data.image_url;
                    resultImage.style.display = "block";
                } else {
                    resultDisplay.textContent = "❌ " + (data.error || 'No match found.');
                    resultDisplay.className = "error";
                }
            } catch (err) {
                resultDisplay.textContent = "❌ Server error connecting to Qdrant.";
                resultDisplay.className = "error";
            } finally {
                searchBtn.textContent = "Search Vector DB";
            }
        });

        manualInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                searchBtn.click();
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

@app.post("/process")
def process_text_nlp(req: ProcessRequest):
    return hyperlink_medical_terms(req.text)

@app.post("/search")
def search_and_map_image(req: SearchRequest):
    try:
        # Embed the incoming clicked text
        vectors = list(embedding_model.embed([req.text]))
        query_vector = vectors[0].tolist()
        
        # Search Qdrant
        response = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=1
        )
        
        hits = response.points
        
        if not hits:
            return {"error": "No close vectors found in Qdrant."}
            
        # Extract the payload directly
        payload = hits[0].payload
        matched_text = payload.get("text", "")
        
        # Grab the exact filename from the metadata uploaded earlier
        image_name = payload.get("filename", "")
        
        if not image_name:
            return {"error": "Vector matched, but payload had no 'filename' field."}
            
        # Construct the Supabase Public URL
        # Format: https://[URL]/storage/v1/object/public/[BUCKET]/[FILENAME]
        base_url = SUPABASE_URL.rstrip('/')
        image_url = f"{base_url}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_name}"
        
        return {
            "matched_text": matched_text,
            "image_name": image_name,
            "image_url": image_url
        }
    except Exception as e:
        return {"error": str(e)}
