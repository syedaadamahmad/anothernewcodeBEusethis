"""
load_offers_to_mongo.py

Loads CSV offers into MongoDB with vector embeddings for semantic retrieval.
Handles both PO (payment offers) and GO (general offers) in a single collection.

Usage:
    python load_offers_to_mongo.py
"""

import os
import csv
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from pymongo import MongoClient

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

CSV_PATH = os.getenv("UPDATED_DEALS_CSV")
MONGO_URI = os.getenv("MONGO_DB_URI")
DB_NAME = os.getenv("DB_NAME", "smartbhaiDB")
COLLECTION_NAME = "dummy_offers"

# Embeddings (AWS Bedrock - same as your existing setup)
embeddings = BedrockEmbeddings(
    model_id=os.getenv("EMBEDDING_MODEL_ID"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# ============================================================================
# Text Preparation for Embeddings
# ============================================================================

def prepare_offer_text(row: Dict) -> str:
    """
    Create rich semantic text from offer fields for embedding.
    This ensures vector search captures intent like "HDFC credit card discount".
    """
    parts = []
    
    # Core offer content
    if row.get("title"):
        parts.append(f"Title: {row['title']}")
    if row.get("offer"):
        parts.append(f"Offer: {row['offer']}")
    
    # Platform context
    if row.get("platform"):
        parts.append(f"Platform: {row['platform']}")
    
    # Payment context (for PO)
    if row.get("bank"):
        parts.append(f"Bank: {row['bank']}")
    if row.get("payment_mode"):
        parts.append(f"Payment Mode: {row['payment_mode']}")
    
    # Flight type
    if row.get("flight_type"):
        parts.append(f"Flight Type: {row['flight_type']}")
    
    return " | ".join(parts)


# ============================================================================
# Offer Type Classification
# ============================================================================

def classify_offer_type(row: Dict) -> str:
    """
    Determine if offer is PO or GO based on your rules:
    - If bank field is NOT blank → PO
    - If bank field IS blank → GO
    """
    bank = row.get("bank", "").strip()
    
    # Override with explicit offer_type if present
    explicit_type = row.get("offer_type", "").strip().lower()
    if explicit_type in ["po", "go", "gc"]:
        return explicit_type
    
    # Rule-based classification
    if bank:
        return "po"  # Payment offer
    else:
        return "go"  # General offer


# ============================================================================
# Document Builder
# ============================================================================

def build_document(row: Dict, embedding: List[float]) -> Dict:
    """
    Build MongoDB document with all fields + embedding.
    Normalizes dates, handles missing values, adds metadata.
    """
    # Determine offer type
    offer_type = classify_offer_type(row)
    
    # Parse expiry date
    expiry_str = row.get("expiry_date", "").strip()
    expiry_date = None
    if expiry_str:
        try:
            # Try multiple date formats
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
                try:
                    expiry_date = datetime.strptime(expiry_str, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
        except Exception as e:
            print(f"⚠️ Invalid expiry date: {expiry_str} - {e}")
    
    doc = {
        # Core fields
        "platform": row.get("platform", "").strip(),
        "title": row.get("title", "").strip(),
        "offer": row.get("offer", "").strip(),
        "coupon_code": row.get("coupon_code", "").strip(),
        "url": row.get("url", "").strip(),
        
        # Classification
        "offer_type": offer_type,
        
        # Payment fields (only for PO)
        "bank": row.get("bank", "").strip(),
        "payment_mode": row.get("payment_mode", "").strip(),
        
        # Temporal
        "expiry_date": expiry_date,
        
        # Flight type
        "flight_type": row.get("flight_type", "domestic").strip().lower(),
        
        # Embeddings
        "embedding": embedding,
        
        # Metadata
        "indexed_at": datetime.utcnow(),
        "source": "csv_loader"
    }
    
    return doc


# ============================================================================
# Main Loader
# ============================================================================

def load_offers_to_mongo(batch_size: int = 50):
    """
    Load CSV offers into MongoDB with vector embeddings.
    Uses batch processing for efficiency.
    """
    print(f"📂 [LOADER] Reading CSV: {CSV_PATH}")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Clear existing data (optional - comment out to append)
    print(f"🗑️ [MONGO] Clearing existing collection: {COLLECTION_NAME}")
    collection.delete_many({})
    
    # Read CSV
    rows = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"📊 [CSV] Found {len(rows)} offers")
    
    # Process in batches
    documents = []
    total_inserted = 0
    
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        print(f"🔄 [BATCH {i // batch_size + 1}] Processing {len(batch)} offers...")
        
        # Prepare texts for embedding
        texts = [prepare_offer_text(row) for row in batch]
        
        # Generate embeddings (batch API call)
        try:
            batch_embeddings = embeddings.embed_documents(texts)
        except Exception as e:
            print(f"❌ [EMBEDDING ERROR] {e}")
            continue
        
        # Build documents
        for row, embedding in zip(batch, batch_embeddings):
            doc = build_document(row, embedding)
            documents.append(doc)
        
        # Insert batch
        if documents:
            try:
                result = collection.insert_many(documents)
                total_inserted += len(result.inserted_ids)
                print(f"✅ [MONGO] Inserted {len(result.inserted_ids)} documents")
                documents = []  # Clear for next batch
            except Exception as e:
                print(f"❌ [MONGO ERROR] {e}")
    
    print(f"\n🎉 [COMPLETE] Loaded {total_inserted} offers to MongoDB")
    
    # Create vector search index instructions
    print(f"\n📋 [NEXT STEP] Create vector search index in MongoDB Atlas:")
    print(f"   Collection: {DB_NAME}.{COLLECTION_NAME}")
    print(f"   Index name: vector_index")
    print(f"   Configuration:")
    print("""
    {
      "fields": [
        {
          "type": "vector",
          "path": "embedding",
          "numDimensions": 1024,
          "similarity": "cosine"
        },
        {
          "type": "filter",
          "path": "platform"
        },
        {
          "type": "filter",
          "path": "bank"
        },
        {
          "type": "filter",
          "path": "payment_mode"
        },
        {
          "type": "filter",
          "path": "flight_type"
        },
        {
          "type": "filter",
          "path": "offer_type"
        },
        {
          "type": "filter",
          "path": "expiry_date"
        }
      ]
    }
    """)
    
    client.close()


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    print("🚀 SmartBhai Offer Loader - Starting...")
    try:
        load_offers_to_mongo(batch_size=50)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()