import os
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import io
import PyPDF2
import docx
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from dotenv import load_dotenv

# Import our new Medical NLP library
import spacy

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# ==========================================
# CONFIGURATION
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Initialize Qdrant & Embedding Model 
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# ==========================================
# MEDICAL NLP ENGINE (SciSpaCy)
# ==========================================
# Load the pre-trained biomedical model (Detects Diseases and Drugs/Chemicals)
try:
    med_nlp = spacy.load("en_ner_bc5cdr_md")
except OSError:
    raise RuntimeError(
        "Medical NLP model not found. Please install it using:\n"
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
    )

class SearchRequest(BaseModel):
    text: str

def highlight_medical_terms(text: str) -> str:
    """
    Uses AI to dynamically find medical entities and wrap them in HTML tags.
    """
    doc = med_nlp(text)
    
    # We must replace text from the end to the beginning so that changing string 
    # lengths doesn't mess up the character indices of earlier entities.
    entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
    
    highlighted_text = text
    
    for ent in entities:
        # ent.label_ will be "DISEASE" or "CHEMICAL"
        term = ent.text
        start = ent.start_char
        end = ent.end_char
        
        # Wrap the exact detected entity in our terminal node HTML
        replacement = f'<span class="medical-term" data-term="{term}">{term}</span>'
        highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
        
    return highlighted_text

# ==========================================
# FRONTEND (HTML / CSS / JS)
# ==========================================
# (The frontend remains the exact same dark-mode terminal layout)
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedCore Diagnostic Terminal</title>
    <style>
        /* Terminal Color Palette */
        :root {
            --bg-base: #050914;
            --bg-panel: #0a1128;
            --text-main: #94a3b8;
            --text-header: #e2e8f0;
            --accent-cyan: #00f0ff;
            --accent-blue: #3b82f6;
            --accent-red: #ff3366;
            --border-color: #1e293b;
            --font-mono: 'Courier New', Courier, monospace;
            --font-sans: 'Inter', system-ui, sans-serif;
        }

        body { 
            font-family: var(--font-sans); 
            margin: 0; 
            padding: 0;
            background-color: var(--bg-base);
            color: var(--text-main);
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            padding-bottom: 120px;
        }

        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 4px; }

        header {
            width: 100%;
            max-width: 1200px;
            padding: 30px 20px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 20px;
        }

        h1 { 
            font-family: var(--font-mono);
            color: var(--accent-cyan);
            font-size: 24px;
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 2px;
            text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
        }
        
        .blink { animation: blinker 1s linear infinite; }
        @keyframes blinker { 50% { opacity: 0; } }

        .container {
            width: 100%;
            max-width: 1200px;
            padding: 0 20px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .upload-panel { 
            background: var(--bg-panel);
            border: 1px dashed var(--accent-blue);
            border-radius: 8px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .upload-panel:hover {
            border-color: var(--accent-cyan);
            box-shadow: 0 0 15px rgba(0, 240, 255, 0.1);
        }

        .upload-label {
            font-family: var(--font-mono);
            color: var(--accent-cyan);
            font-size: 16px;
            cursor: pointer;
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 240, 255, 0.1);
            border: 1px solid var(--accent-cyan);
            border-radius: 4px;
            transition: all 0.2s;
        }
        
        .upload-label:hover {
            background: var(--accent-cyan);
            color: var(--bg-base);
        }
        
        input[type="file"] { display: none; }

        .terminal-window {
            background: #020617;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }

        .terminal-header {
            background: var(--bg-panel);
            padding: 10px 20px;
            font-family: var(--font-mono);
            font-size: 14px;
            color: var(--accent-blue);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
        }

        #content-display { 
            white-space: pre-wrap; 
            line-height: 1.8; 
            padding: 30px; 
            height: 50vh; 
            overflow-y: auto; 
            font-size: 16px; 
            font-family: var(--font-mono);
        }

        /* High-Tech Medical Terms */
        .medical-term {
            color: var(--accent-cyan);
            background: rgba(0, 240, 255, 0.05);
            border: 1px solid rgba(0, 240, 255, 0.3);
            border-radius: 3px;
            padding: 2px 6px;
            cursor: crosshair;
            font-weight: 600;
            transition: all 0.2s;
        }
        
        .medical-term:hover {
            background: var(--accent-cyan);
            color: var(--bg-base);
            box-shadow: 0 0 10px var(--accent-cyan);
        }

        .telemetry-panel { 
            position: fixed; 
            bottom: 0; 
            left: 0; 
            width: 100%; 
            background: rgba(10, 17, 40, 0.95); 
            backdrop-filter: blur(10px);
            border-top: 1px solid var(--accent-blue); 
            padding: 20px; 
            display: flex; 
            justify-content: space-around;
            align-items: center; 
            z-index: 1000; 
            box-sizing: border-box; 
            font-family: var(--font-mono);
        }

        .panel-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 45%;
        }

        .label {
            font-size: 12px;
            color: var(--text-main);
            letter-spacing: 1px;
            margin-bottom: 5px;
        }

        .value {
            font-size: 20px;
            color: var(--text-header);
            background: #020617;
            padding: 10px 20px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
            width: 100%;
            text-align: center;
            box-sizing: border-box;
            min-height: 48px;
        }

        .value.highlight-cyan { color: var(--accent-cyan); border-color: var(--accent-cyan); box-shadow: 0 0 10px rgba(0, 240, 255, 0.2); }
        .value.highlight-red { color: var(--accent-red); border-color: var(--accent-red); box-shadow: 0 0 10px rgba(255, 51, 102, 0.2); }
    </style>
</head>
<body>

    <header>
        <h1><span class="blink">_</span> SYS.MED_CORE // DIAGNOSTIC TERMINAL</h1>
    </header>

    <div class="container">
        <div class="upload-panel">
            <label class="upload-label" for="file-upload">
                [+] INITIALIZE DOCUMENT SCAN
            </label>
            <input type="file" id="file-upload" accept=".txt,.pdf,.docx" />
            <div id="file-name-display" style="margin-top: 15px; font-family: var(--font-mono); font-size: 14px; color: var(--text-main);">NO FILE SELECTED</div>
        </div>
        
        <div class="terminal-window">
            <div class="terminal-header">
                <span>VIEWER_MODULE</span>
                <span>STATUS: <span id="doc-status" style="color: var(--accent-cyan);">AWAITING_INPUT</span></span>
            </div>
            <div id="content-display">SYSTEM READY. PLEASE UPLOAD PATIENT REPORT TO BEGIN.</div>
        </div>
    </div>

    <div class="telemetry-panel">
        <div class="panel-section">
            <span class="label">TARGET ENTITY</span>
            <span id="selected-text" class="value">--</span>
        </div>
        <div class="panel-section">
            <span class="label">DATABASE MAPPING</span>
            <span id="result-filename" class="value">STANDBY</span>
        </div>
    </div>

    <script>
        // 1. Upload & Parse Document
        document.getElementById('file-upload').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            document.getElementById('file-name-display').textContent = `FILE: ${file.name}`;
            document.getElementById('doc-status').textContent = 'NLP_ENGINE_SCANNING...';
            document.getElementById('doc-status').style.color = '#ff9900';
            document.getElementById('content-display').textContent = ">> EXTRACTING TEXT & RUNNING ENTITY RECOGNITION (NER)...\\n>> THIS MAY TAKE A MOMENT DEPENDING ON DOC SIZE...";
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const res = await fetch('/upload', { method: 'POST', body: formData });
                const data = await res.json();
                
                document.getElementById('content-display').innerHTML = data.html_text;
                document.getElementById('doc-status').textContent = 'SCAN_COMPLETE';
                document.getElementById('doc-status').style.color = 'var(--accent-cyan)';
            } catch (err) {
                document.getElementById('content-display').textContent = ">> ERROR: FAILED TO PROCESS DOCUMENT.";
                document.getElementById('doc-status').textContent = 'SYS_ERROR';
                document.getElementById('doc-status').style.color = 'var(--accent-red)';
            }
        });

        // 2. Handle Clicks on Medical Terms
        document.getElementById('content-display').addEventListener('click', async (e) => {
            if (e.target.classList.contains('medical-term')) {
                const term = e.target.getAttribute('data-term');
                
                const targetBox = document.getElementById('selected-text');
                targetBox.textContent = `> ${term.toUpperCase()}`;
                targetBox.className = 'value highlight-cyan';

                const matchBox = document.getElementById('result-filename');
                matchBox.textContent = "QUERYING VECTOR DB...";
                matchBox.className = 'value';
                
                try {
                    const res = await fetch('/search', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ text: term })
                    });
                    const data = await res.json();
                    
                    if (data.image_name) {
                        matchBox.textContent = `[ MATCH: ${data.image_name} ]`;
                        matchBox.className = 'value highlight-cyan';
                    } else {
                        matchBox.textContent = `[ ERROR: ${data.error || 'NO MATCH'} ]`;
                        matchBox.className = 'value highlight-red';
                    }
                } catch (err) {
                    matchBox.textContent = "[ FATAL: NETWORK ERROR ]";
                    matchBox.className = 'value highlight-red';
                }
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
        elif filename.endswith('.docx'):
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = ">> ERROR: UNSUPPORTED FILE FORMAT."
            
        # Run text through SciSpaCy NLP Engine
        html_text = highlight_medical_terms(text)
        
    except Exception as e:
        html_text = f">> SYSTEM EXCEPTION: {str(e)}"
        
    return {"html_text": html_text}

@app.post("/search")
def search_and_map_image(req: SearchRequest):
    vectors = list(embedding_model.embed([req.text]))
    query_vector = vectors[0].tolist()
    
    response = qdrant.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=1
    )
    
    hits = response.points
    
    if not hits:
        return {"error": "NO VECTOR PROXIMITY"}
        
    matched_text = hits[0].payload.get("text", "")
    if not matched_text:
        return {"error": "NULL PAYLOAD TEXT"}
        
    first_word = re.sub(r'[^a-zA-Z0-9]', '', matched_text.split()[0])
    image_name = f"{first_word.lower()}.png"
    
    return {
        "matched_text": matched_text,
        "image_name": image_name
    }
