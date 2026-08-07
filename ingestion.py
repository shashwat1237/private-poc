import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed import TextEmbedding
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

# Initialize Qdrant with a longer timeout (60 seconds instead of default)
print("Initializing Qdrant and fastembed...")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60.0)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def upload_data(file_path: str):
    # 1. Read the filenames
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(lines)} filenames from {file_path}.")

    # 2. Check and Create collection (Replacing the deprecated recreate_collection)
    print(f"Preparing collection: {QDRANT_COLLECTION}")
    if qdrant.collection_exists(collection_name=QDRANT_COLLECTION):
        print("Collection exists. Deleting it to start fresh...")
        qdrant.delete_collection(collection_name=QDRANT_COLLECTION)
        
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # 3. Embed the lines
    print("Embedding texts locally... This will take a moment.")
    embeddings = list(embedding_model.embed(lines))
    
    # 4. Prepare points
    points = []
    for idx, (text, vector) in enumerate(zip(lines, embeddings)):
        points.append(
            PointStruct(
                id=idx, 
                vector=vector.tolist(),
                payload={"text": text}
            )
        )
    
    # 5. Upload to Qdrant in smaller batches to avoid WriteTimeout
    batch_size = 100 # Reduced from 500
    print("Uploading vectors to Qdrant Cloud...")
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        qdrant.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch
        )
        print(f"Uploaded {min(i + batch_size, len(points))}/{len(points)} points.")
    
    print("✅ Upload complete! Your Qdrant database is ready to be searched.")

if __name__ == "__main__":
    upload_data("my_filenames.txt")