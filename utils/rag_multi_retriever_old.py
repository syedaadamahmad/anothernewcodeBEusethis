"""
CORRECTED rag_multi_retriever.py - WORKS WITH MONGODB ATLAS VECTOR SEARCH

This version uses POST-FILTERING after vector search because:
1. MongoDB Atlas Vector Search has LIMITED pre_filter support
2. Complex $regex and nested $and/$or don't work in pre_filter
3. Solution: Retrieve more docs, filter in Python (still fast)

PRESERVES:
- AWS Bedrock embeddings (1024-dim)
- LangChain MongoDB integration
- Your existing collection structure
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from utils import mongoDB
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()

# Initialize embeddings once (YOUR EXISTING CODE)
embeddings = BedrockEmbeddings(
    model_id=os.getenv("EMBEDDING_MODEL_ID"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


# ============================================================================
# ENHANCED RETRIEVAL FUNCTIONS (POST-FILTERING APPROACH)
# ============================================================================

def get_gift_coupons_enhanced(
    query: str,
    platform: Optional[str] = None,
    flight_type: str = "domestic",
    k: int = 10,
    threshold: float = 0.75
) -> Dict:
    """
    ENHANCED gift coupon retrieval with platform filtering.
    Uses POST-FILTERING after vector search for compatibility.
    """
    try:
        mongo_client = mongoDB.get_mongo_client()
        collection = mongoDB.get_collection(mongo_client, "gift_coupons")
        
        if collection is None:
            return {"offers": [], "count": 0, "error": "Collection not found"}
        
        # Simple pre-filter (only offer_type - this works reliably)
        pre_filter = {"offer_type": {"$eq": "gc"}}
        
        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection,
            index_name="vector_index",
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k * 3,  # Retrieve 3x more for post-filtering
                "score_threshold": threshold,
                "pre_filter": pre_filter
            }
        )
        
        docs = retriever.invoke(query)
        
        # POST-FILTER: Apply platform and flight_type filters in Python
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        offers = []
        for doc in docs:
            # Check expiry date
            expiry = doc.metadata.get("expiry_date", "")
            if expiry and expiry < today:
                continue
            
            # Check flight type
            doc_flight_type = doc.metadata.get("flight_type", "").lower()
            if doc_flight_type and doc_flight_type not in [flight_type.lower(), "both", ""]:
                continue
            
            # Check platform (if specified)
            if platform:
                doc_platform = doc.metadata.get("platform", "").lower()
                if platform.lower() not in doc_platform:
                    continue
            
            offer_data = {
                "offer_id": str(doc.metadata.get("_id", "")),
                "platform": doc.metadata.get("platform", ""),
                "title": doc.metadata.get("title", ""),
                "offer": doc.metadata.get("offer", ""),
                "coupon_code": doc.metadata.get("coupon_code", ""),
                "url": doc.metadata.get("url", ""),
                "expiry_date": doc.metadata.get("expiry_date", ""),
                "flight_type": doc.metadata.get("flight_type", ""),
                "discount_type": "flat",
                "offer_type": "gc"
            }
            offers.append(offer_data)
            
            # Stop if we have enough
            if len(offers) >= k:
                break
        
        print(f"✅ [GIFT_COUPONS_ENHANCED] Retrieved {len(offers)} offers")
        return {"offers": offers, "count": len(offers)}
    
    except Exception as e:
        print(f"❌ [GIFT_COUPONS_ENHANCED] Error: {e}")
        return {"offers": [], "count": 0, "error": str(e)}


def get_payment_offers_enhanced(
    query: str,
    bank: str,
    card_type: str,
    platform: Optional[str] = None,
    flight_type: str = "domestic",
    k: int = 10,
    threshold: float = 0.75
) -> Dict:
    """
    ENHANCED payment offer retrieval with platform filtering.
    Uses POST-FILTERING for bank, card_type, platform (more reliable).
    """
    try:
        mongo_client = mongoDB.get_mongo_client()
        collection = mongoDB.get_collection(mongo_client, "payment_offers")
        
        if collection is None:
            return {"offers": [], "count": 0, "error": "Collection not found"}
        
        # Simple pre-filter (only offer_type - this works reliably)
        pre_filter = {"offer_type": {"$eq": "po"}}
        
        vector_store = MongoDBAtlasVectorSearch(
            embedding=embeddings,
            collection=collection,
            index_name="vector_index",
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": k * 5,  # Retrieve 5x more for post-filtering
                "score_threshold": threshold,
                "pre_filter": pre_filter
            }
        )
        
        docs = retriever.invoke(query)
        
        # POST-FILTER: Apply all filters in Python
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        offers = []
        for doc in docs:
            # Check bank (case-insensitive)
            doc_bank = doc.metadata.get("bank", "").lower()
            if bank.lower() not in doc_bank:
                continue
            
            # Check card type (case-insensitive)
            doc_card = doc.metadata.get("payment_mode", "").lower()
            if card_type.lower() not in doc_card:
                continue
            
            # Check expiry date
            expiry = doc.metadata.get("expiry_date", "")
            if expiry and expiry < today:
                continue
            
            # Check flight type
            doc_flight_type = doc.metadata.get("flight_type", "").lower()
            if doc_flight_type and doc_flight_type not in [flight_type.lower(), "both", ""]:
                continue
            
            # Check platform (if specified)
            if platform:
                doc_platform = doc.metadata.get("platform", "").lower()
                if platform.lower() not in doc_platform:
                    continue
            
            offer_data = {
                "offer_id": str(doc.metadata.get("_id", "")),
                "platform": doc.metadata.get("platform", ""),
                "title": doc.metadata.get("title", ""),
                "offer": doc.metadata.get("offer", ""),
                "coupon_code": doc.metadata.get("coupon_code", ""),
                "bank": doc.metadata.get("bank", ""),
                "card_type": doc.metadata.get("payment_mode", ""),
                "url": doc.metadata.get("url", ""),
                "expiry_date": doc.metadata.get("expiry_date", ""),
                "flight_type": doc.metadata.get("flight_type", ""),
                "emi": doc.metadata.get("emi", 0),
                "discount_type": "percentage",
                "offer_type": "po"
            }
            offers.append(offer_data)
            
            # Stop if we have enough
            if len(offers) >= k:
                break
        
        print(f"✅ [PAYMENT_OFFERS_ENHANCED] Retrieved {len(offers)} offers for {bank} {card_type}")
        return {"offers": offers, "count": len(offers)}
    
    except Exception as e:
        print(f"❌ [PAYMENT_OFFERS_ENHANCED] Error: {e}")
        return {"offers": [], "count": 0, "error": str(e)}


def get_general_offers_enhanced(
    query: str,
    platform: Optional[str] = None,
    flight_type: str = "domestic",
    k: int = 10
) -> Dict:
    """
    ENHANCED general offer retrieval with platform filtering.
    Uses MongoDB aggregation (no vector search for general offers).
    """
    try:
        mongo_client = mongoDB.get_mongo_client()
        collection = mongoDB.get_collection(mongo_client, "general_offers")
        
        if collection is None:
            return {"offers": [], "count": 0, "error": "Collection not found"}
        
        # Build query
        today = datetime.utcnow().strftime("%Y-%m-%d")
        
        match_conditions = [
            {"offer_type": {"$eq": "go"}},
            {"$or": [
                {"offer": {"$regex": query, "$options": "i"}},
                {"title": {"$regex": query, "$options": "i"}},
                {"platform": {"$regex": query, "$options": "i"}}
            ]},
            {"$or": [
                {"expiry_date": {"$gte": today}},
                {"expiry_date": {"$exists": False}},
                {"expiry_date": ""}
            ]},
            {"$or": [
                {"flight_type": flight_type},
                {"flight_type": "both"},
                {"flight_type": {"$exists": False}}
            ]}
        ]
        
        # Add platform filter if specified
        if platform:
            match_conditions.append({
                "platform": {"$regex": platform, "$options": "i"}
            })
        
        pipeline = [
            {"$match": {"$and": match_conditions}},
            {"$limit": k}
        ]
        
        docs = list(collection.aggregate(pipeline))
        
        offers = []
        for doc in docs:
            offer_data = {
                "offer_id": str(doc.get("_id", "")),
                "platform": doc.get("platform", ""),
                "title": doc.get("title", ""),
                "offer": doc.get("offer", ""),
                "coupon_code": doc.get("coupon_code", ""),
                "url": doc.get("url", ""),
                "expiry_date": doc.get("expiry_date", ""),
                "flight_type": doc.get("flight_type", ""),
                "discount_type": "flat",
                "offer_type": "go"
            }
            offers.append(offer_data)
        
        print(f"✅ [GENERAL_OFFERS_ENHANCED] Retrieved {len(offers)} offers")
        return {"offers": offers, "count": len(offers)}
    
    except Exception as e:
        print(f"❌ [GENERAL_OFFERS_ENHANCED] Error: {e}")
        return {"offers": [], "count": 0, "error": str(e)}


def get_available_banks(platform: Optional[str] = None) -> List[str]:
    """Get list of unique banks that have payment offers."""
    try:
        mongo_client = mongoDB.get_mongo_client()
        collection = mongoDB.get_collection(mongo_client, "payment_offers")
        
        if collection is None:
            return []
        
        # Build query
        query = {"offer_type": {"$eq": "po"}}
        if platform:
            query["platform"] = {"$regex": platform, "$options": "i"}
        
        banks = collection.distinct("bank", query)
        banks = [b for b in banks if b and b.strip()]  # Remove empty/None
        banks.sort()
        
        print(f"✅ [BANKS] Found {len(banks)} banks: {banks}")
        return banks
    
    except Exception as e:
        print(f"❌ [BANKS] Error: {e}")
        return []


# ============================================================================
# BACKWARD COMPATIBILITY - Keep your existing functions
# ============================================================================

def get_gift_coupons(query: str, k: int = 10, threshold: float = 0.75) -> Dict:
    """Original function - calls enhanced version with defaults."""
    return get_gift_coupons_enhanced(query, platform=None, k=k, threshold=threshold)


def get_payment_offers(query: str, bank: str, card_type: str, k: int = 10, threshold: float = 0.75) -> Dict:
    """Original function - calls enhanced version with defaults."""
    return get_payment_offers_enhanced(query, bank, card_type, platform=None, k=k, threshold=threshold)


def get_general_offers(query: str, k: int = 10) -> Dict:
    """Original function - calls enhanced version with defaults."""
    return get_general_offers_enhanced(query, platform=None, k=k)
















































# # rag_multi_retriever.py deployed on github
# import os
# from datetime import datetime
# from typing import List, Dict, Optional
# from dotenv import load_dotenv
# from utils import mongoDB
# from langchain_aws import BedrockEmbeddings
# from langchain_mongodb import MongoDBAtlasVectorSearch

# load_dotenv()

# # Initialize embeddings once
# embeddings = BedrockEmbeddings(
#     model_id=os.getenv("EMBEDDING_MODEL_ID"),
#     region_name=os.getenv("AWS_DEFAULT_REGION"),
#     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
# )

# def get_gift_coupons(query: str, k: int = 10, threshold: float = 0.75) -> Dict:
#     """
#     Retrieve gift coupons using vector similarity search.
#     """
#     try:
#         mongo_client = mongoDB.get_mongo_client()
#         collection = mongoDB.get_collection(mongo_client, "gift_coupons")
        
#         if collection is None:
#             return {"offers": [], "count": 0, "collection": "gift_coupons", "error": "Collection not found"}
        
#         vector_store = MongoDBAtlasVectorSearch(
#             embedding=embeddings,
#             collection=collection,
#             index_name="vector_index",
#         )
        
#         retriever = vector_store.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={"k": k, "score_threshold": threshold}
#         )
        
#         docs = retriever.invoke(query)
        
#         offers = []
#         for doc in docs:
#             offer_data = {
#                 "platform": doc.metadata.get("platform", ""),
#                 "title": doc.metadata.get("title", ""),
#                 "offer": doc.metadata.get("offer", ""),
#                 "coupon_code": doc.metadata.get("coupon_code", ""),
#                 "url": doc.metadata.get("url", ""),
#                 "expiry_date": doc.metadata.get("expiry_date", ""),
#                 "flight_type": doc.metadata.get("flight_type", ""),
#                 "discount_type": "flat",  # Inferred from content
#                 "collection": "gift_coupons"
#             }
#             offers.append(offer_data)
        
#         print(f"✅ [GIFT_COUPONS] Retrieved {len(offers)} offers")
#         return {"offers": offers, "count": len(offers), "collection": "gift_coupons"}
    
#     except Exception as e:
#         print(f"❌ [GIFT_COUPONS] Error: {e}")
#         return {"offers": [], "count": 0, "collection": "gift_coupons", "error": str(e)}


# def get_payment_offers(query: str, bank: str, card_type: str, k: int = 10, threshold: float = 0.75) -> Dict:
#     """
#     Retrieve payment offers filtered by bank and card type using vector search.
#     """
#     try:
#         mongo_client = mongoDB.get_mongo_client()
#         collection = mongoDB.get_collection(mongo_client, "payment_offers")
        
#         if collection is None:
#             return {"offers": [], "count": 0, "collection": "payment_offers", "error": "Collection not found"}
        
#         # Check if vector index exists
#         vector_store = MongoDBAtlasVectorSearch(
#             embedding=embeddings,
#             collection=collection,
#             index_name="vector_index",
#         )
        
#         retriever = vector_store.as_retriever(
#             search_type="similarity_score_threshold",
#             search_kwargs={"k": k, "score_threshold": threshold}
#         )
        
#         docs = retriever.invoke(query)
        
#         # Filter by bank and card_type in metadata
#         offers = []
#         for doc in docs:
#             doc_bank = doc.metadata.get("bank", "").strip().lower()
#             doc_card = doc.metadata.get("payment_mode", "").strip().lower()
            
#             if bank.lower() in doc_bank and card_type.lower() in doc_card:
#                 offer_data = {
#                     "platform": doc.metadata.get("platform", ""),
#                     "title": doc.metadata.get("title", ""),
#                     "offer": doc.metadata.get("offer", ""),
#                     "coupon_code": doc.metadata.get("coupon_code", ""),
#                     "bank": doc.metadata.get("bank", ""),
#                     "card_type": doc.metadata.get("payment_mode", ""),
#                     "url": doc.metadata.get("url", ""),
#                     "expiry_date": doc.metadata.get("expiry_date", ""),
#                     "flight_type": doc.metadata.get("flight_type", ""),
#                     "emi": doc.metadata.get("emi", 0),
#                     "discount_type": "percentage",  # Inferred
#                     "collection": "payment_offers"
#                 }
#                 offers.append(offer_data)
        
#         print(f"✅ [PAYMENT_OFFERS] Retrieved {len(offers)} offers for {bank} {card_type}")
#         return {"offers": offers, "count": len(offers), "collection": "payment_offers"}
    
#     except Exception as e:
#         print(f"❌ [PAYMENT_OFFERS] Error: {e}")
#         return {"offers": [], "count": 0, "collection": "payment_offers", "error": str(e)}


# def get_general_offers(query: str, k: int = 10) -> Dict:
#     """
#     Retrieve general offers using keyword-based MongoDB text search (no vector index).
#     Filters out expired offers.
#     """
#     try:
#         mongo_client = mongoDB.get_mongo_client()
#         collection = mongoDB.get_collection(mongo_client, "general_offers")
        
#         if collection is None:
#             return {"offers": [], "count": 0, "collection": "general_offers", "error": "Collection not found"}
        
#         # Use MongoDB text search as fallback
#         # Assumes a text index exists on 'offer' and 'title' fields
#         today = datetime.utcnow().strftime("%Y-%m-%d")
        
#         pipeline = [
#             {"$match": {
#                 "$and": [
#                     {"$or": [
#                         {"offer": {"$regex": query, "$options": "i"}},
#                         {"title": {"$regex": query, "$options": "i"}},
#                         {"platform": {"$regex": query, "$options": "i"}}
#                     ]},
#                     {"$or": [
#                         {"expiry_date": {"$gte": today}},
#                         {"expiry_date": {"$exists": False}},
#                         {"expiry_date": ""}
#                     ]}
#                 ]
#             }},
#             {"$limit": k}
#         ]
        
#         docs = list(collection.aggregate(pipeline))
        
#         offers = []
#         for doc in docs:
#             offer_data = {
#                 "platform": doc.get("platform", ""),
#                 "title": doc.get("title", ""),
#                 "offer": doc.get("offer", ""),
#                 "coupon_code": doc.get("coupon_code", ""),
#                 "url": doc.get("url", ""),
#                 "expiry_date": doc.get("expiry_date", ""),
#                 "flight_type": doc.get("flight_type", ""),
#                 "discount_type": "flat",
#                 "collection": "general_offers"
#             }
#             offers.append(offer_data)
        
#         print(f"✅ [GENERAL_OFFERS] Retrieved {len(offers)} offers (keyword search)")
#         return {"offers": offers, "count": len(offers), "collection": "general_offers"}
    
#     except Exception as e:
#         print(f"❌ [GENERAL_OFFERS] Error: {e}")
#         # Fallback: return all non-expired offers
#         try:
#             today = datetime.utcnow().strftime("%Y-%m-%d")
#             docs = list(collection.find({
#                 "$or": [
#                     {"expiry_date": {"$gte": today}},
#                     {"expiry_date": {"$exists": False}}
#                 ]
#             }).limit(k))
            
#             offers = []
#             for doc in docs:
#                 offer_data = {
#                     "platform": doc.get("platform", ""),
#                     "title": doc.get("title", ""),
#                     "offer": doc.get("offer", ""),
#                     "coupon_code": doc.get("coupon_code", ""),
#                     "url": doc.get("url", ""),
#                     "expiry_date": doc.get("expiry_date", ""),
#                     "flight_type": doc.get("flight_type", ""),
#                     "discount_type": "flat",
#                     "collection": "general_offers"
#                 }
#                 offers.append(offer_data)
            
#             print(f"⚠️ [GENERAL_OFFERS] Using fallback retrieval: {len(offers)} offers")
#             return {"offers": offers, "count": len(offers), "collection": "general_offers"}
#         except:
#             return {"offers": [], "count": 0, "collection": "general_offers", "error": str(e)}