"""
rag_platform_combo_retriever.py - FIXED VERSION

Key Fix: Uses POST-FILTERING instead of $regex in pre-filters.
MongoDB Atlas Vector Search doesn't support $regex, so we filter in Python after retrieval.
"""

import os
import re
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import load_dotenv
from utils import mongoDB
from langchain_aws import BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.tools import tool

load_dotenv()

# ============================================================================
# Embeddings Setup
# ============================================================================

embeddings = BedrockEmbeddings(
    model_id=os.getenv("EMBEDDING_MODEL_ID"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)


# ============================================================================
# Discount Extraction
# ============================================================================

def extract_discount_value(offer_text: str) -> Dict[str, float]:
    """
    Extract discount from offer text.
    Returns: {"type": "percentage" | "flat", "value": float}
    
    Examples:
    - "Flat 10% off" → {"type": "percentage", "value": 10.0}
    - "₹5,000 off" → {"type": "flat", "value": 5000.0}
    - "Rs. 1000 cashback" → {"type": "flat", "value": 1000.0}
    """
    if not offer_text:
        return {"type": "flat", "value": 0.0}
    
    text = offer_text.lower()
    
    # Check for percentage first
    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
    if pct_match:
        return {"type": "percentage", "value": float(pct_match.group(1))}
    
    # Check for flat amount (₹, Rs, INR)
    flat_patterns = [
        r'[₹]\s*(\d{1,3}(?:,\d{3})*)',  # ₹5,000
        r'rs\.?\s*(\d{1,3}(?:,\d{3})*)',  # Rs. 1000
        r'inr\s*(\d{1,3}(?:,\d{3})*)',  # INR 500
        r'(\d{1,3}(?:,\d{3})*)\s*(?:off|cashback|discount)',  # 1000 off
    ]
    
    for pattern in flat_patterns:
        match = re.search(pattern, text)
        if match:
            value_str = match.group(1).replace(',', '')
            return {"type": "flat", "value": float(value_str)}
    
    return {"type": "flat", "value": 0.0}


# ============================================================================
# Sequential Discount Calculator
# ============================================================================

def calculate_combo_price(base_price: float, offers: List[Dict]) -> Dict:
    """
    Apply offers sequentially: percentage discounts first, then flat.
    
    Logic:
    1. Sort offers: percentage first, then flat
    2. Apply percentage discounts sequentially (compound effect)
    3. Sum and apply flat discounts
    4. Return breakdown
    """
    current_price = base_price
    steps = []
    
    # Separate percentage and flat offers
    pct_offers = []
    flat_offers = []
    
    for offer in offers:
        discount_info = extract_discount_value(offer.get("offer", ""))
        if discount_info["type"] == "percentage":
            pct_offers.append((offer, discount_info["value"]))
        else:
            flat_offers.append((offer, discount_info["value"]))
    
    # Step 1: Apply percentage discounts sequentially
    for offer, pct in pct_offers:
        if pct <= 0:
            continue
        discount_amount = current_price * (pct / 100)
        current_price -= discount_amount
        steps.append({
            "offer": offer.get("title", offer.get("offer", "Unknown")),
            "type": "percentage",
            "value": pct,
            "discount_amount": discount_amount,
            "price_after": current_price
        })
    
    # Step 2: Apply flat discounts (sum them)
    total_flat = sum(val for _, val in flat_offers if val > 0)
    if total_flat > 0:
        current_price -= total_flat
        for offer, flat_val in flat_offers:
            if flat_val > 0:
                steps.append({
                    "offer": offer.get("title", offer.get("offer", "Unknown")),
                    "type": "flat",
                    "value": flat_val,
                    "discount_amount": flat_val,
                    "price_after": current_price
                })
    
    # Ensure non-negative price
    final_price = max(0, current_price)
    total_savings = base_price - final_price
    
    return {
        "original_price": base_price,
        "final_price": final_price,
        "total_savings": total_savings,
        "steps": steps
    }


# ============================================================================
# Platform-Specific Retrievers (FIXED - POST-FILTERING)
# ============================================================================

def get_platform_general_offers(
    platform: str,
    query: str = "discount offer",
    k: int = 5,
    score_threshold: float = 0.05
) -> List[Dict]:
    """Direct MongoDB query - working version."""
    try:
        mongo_client = mongoDB.get_mongo_client()
        collection = mongoDB.get_collection(mongo_client, "dummy_offers")
        
        if collection is None:
            print("❌ [GO_RETRIEVER] Collection not found")
            return []
        
        # Simple exact match query
        query_filter = {
            "platform": platform,  # Exact match, case-sensitive
            "offer_type": "go"
        }
        
        print(f"[DEBUG] GO Query: {query_filter}")
        
        docs = list(collection.find(query_filter).limit(k))
        print(f"[DEBUG] Found {len(docs)} docs")
        
        offers = []
        for doc in docs:
            offers.append({
                "platform": doc.get("platform", ""),
                "title": doc.get("title", ""),
                "offer": doc.get("offer", ""),
                "coupon_code": doc.get("coupon_code", ""),
                "url": doc.get("url", ""),
                "expiry_date": doc.get("expiry_date", ""),
                "flight_type": doc.get("flight_type", ""),
                "offer_type": "go"
            })
        
        print(f"✅ [GO_RETRIEVER] Found {len(offers)} offers for {platform}")
        return offers
    
    except Exception as e:
        print(f"❌ [GO_RETRIEVER] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def get_platform_payment_offers(
    platform: str,
    bank: str,
    card_type: str,
    query: str = "discount offer",
    k: int = 5,
    score_threshold: float = 0.1
) -> List[Dict]:
    """Direct MongoDB query - working version."""
    try:
        mongo_client = mongoDB.get_mongo_client()
        collection = mongoDB.get_collection(mongo_client, "dummy_offers")
        
        if collection is None:
            print("❌ [PO_RETRIEVER] Collection not found")
            return []
        
        # Simple exact match query
        query_filter = {
            "platform": platform,
            "bank": bank,
            "payment_mode": card_type,
            "offer_type": "po"
        }
        
        print(f"[DEBUG] PO Query: {query_filter}")
        
        docs = list(collection.find(query_filter).limit(k))
        print(f"[DEBUG] Found {len(docs)} docs")
        
        offers = []
        for doc in docs:
            offers.append({
                "platform": doc.get("platform", ""),
                "title": doc.get("title", ""),
                "offer": doc.get("offer", ""),
                "coupon_code": doc.get("coupon_code", ""),
                "bank": doc.get("bank", ""),
                "payment_mode": doc.get("payment_mode", ""),
                "url": doc.get("url", ""),
                "expiry_date": doc.get("expiry_date", ""),
                "offer_type": "po"
            })
        
        print(f"✅ [PO_RETRIEVER] Found {len(offers)} offers")
        return offers
    
    except Exception as e:
        print(f"❌ [PO_RETRIEVER] Error: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# Combo Builder
# ============================================================================

def build_platform_combo(
    platform: str,
    base_price: float,
    bank: Optional[str] = None,
    card_type: Optional[str] = None
) -> Dict:
    """
    Build best combo for a specific platform.
    
    Strategy:
    1. Get best GO (highest discount)
    2. Get best PO (if bank + card_type provided)
    3. Calculate combined savings
    4. Return structured combo
    """
    print(f"\n🔍 [COMBO_BUILDER] ENTRY - platform={platform}, bank={bank}, card_type={card_type}")
    print(f"[DEBUG] About to call get_platform_general_offers...")
    print(f"\n🔍 [COMBO_BUILDER] Building combo for {platform} @ ₹{base_price}")
    
    # Step 1: Get general offers
    go_offers = get_platform_general_offers(platform, k=10)
    
    # Step 2: Get payment offers (if bank provided)
    po_offers = []
    if bank and card_type:
        po_offers = get_platform_payment_offers(platform, bank, card_type, k=10)
    
    # Step 3: Select best offers (highest discount from each category)
    selected_offers = []
    
    # Best GO (compare extracted discount values)
    if go_offers:
        best_go = max(
            go_offers,
            key=lambda o: extract_discount_value(o.get("offer", ""))["value"]
        )
        selected_offers.append(best_go)
        print(f"✅ Best GO: {best_go['title']} - {best_go['offer']}")
    
    # Best PO (compare extracted discount values)
    if po_offers:
        best_po = max(
            po_offers,
            key=lambda o: extract_discount_value(o.get("offer", ""))["value"]
        )
        selected_offers.append(best_po)
        print(f"✅ Best PO: {best_po['title']} - {best_po['offer']}")
    
    # Step 4: Calculate combo price
    if not selected_offers:
        print("❌ [COMBO_BUILDER] No offers found")
        return {
            "platform": platform,
            "base_price": base_price,
            "final_price": base_price,
            "total_savings": 0,
            "offers_used": [],
            "breakdown": [],
            "error": "No offers available"
        }
    
    result = calculate_combo_price(base_price, selected_offers)
    
    # Step 5: Format response
    combo = {
        "platform": platform,
        "base_price": result["original_price"],
        "final_price": result["final_price"],
        "total_savings": result["total_savings"],
        "discount_percentage": (result["total_savings"] / result["original_price"] * 100) if result["original_price"] > 0 else 0,
        "offers_used": selected_offers,
        "breakdown": result["steps"]
    }
    
    print(f"🎉 [COMBO_BUILDER] Combo built - Savings: ₹{combo['total_savings']:.2f}")
    return combo

@tool
def combo_tool(platform: str, base_price: float, bank: str = None, card_type: str = None) -> dict:
    """Get best combo offers for a flight booking platform."""
    return build_platform_combo(platform, base_price, bank, card_type)

# ============================================================================
# Test Function
# ============================================================================

def test_combo_builder():
    """Test combo builder with sample data."""
    print("🧪 Testing Platform Combo Builder\n")
    
    print("=" * 60)
    print("Test 1: General Offers Only (EaseMyTrip)")
    combo1 = build_platform_combo(
        platform="EaseMyTrip",
        base_price=7550
    )
    print(f"\nResult: {combo1}")
    
    print("\n" + "=" * 60)
    print("Test 2: General + Payment Offers (MakeMyTrip + HDFC Credit)")
    combo2 = build_platform_combo(
        platform="MakeMyTrip",
        base_price=10000,
        bank="HDFC",
        card_type="credit"
    )
    print(f"\nResult: {combo2}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_combo_builder()