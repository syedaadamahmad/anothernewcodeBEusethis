# offer_orchestrator_tool.py
from typing import Dict, Optional
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
# AFTER (import enhanced versions)
from utils.rag_multi_retriever import (
    get_gift_coupons_enhanced,
    get_payment_offers_enhanced, 
    get_general_offers_enhanced
)
from utils.rag_combo_builder import build_offer_combo, format_combo_for_frontend

@tool
def offer_orchestrator_tool(
    query: str,
    offer_type: str,
    platform: Optional[str] = None,  # ← NEW
    bank: Optional[str] = None,
    card_type: Optional[str] = None,
    base_price: Optional[float] = None,
    build_combo: bool = False
) -> str:
# AFTER (add platform to docstring)
    """
    Orchestrate offer retrieval and combo building for nested flight chat.
    
    Args:
        query: User's search query (e.g., "flight offers", "cashback deals")
        offer_type: One of ["general", "payment", "gift", "combo"]
        platform: Booking platform name (e.g., "MakeMyTrip", "Cleartrip") ← NEW
        bank: Required for payment offers (e.g., "HDFC")
        card_type: Required for payment offers (e.g., "Credit Card", "Debit Card")
        base_price: Flight price for combo calculation
        build_combo: Whether to compute combos
    
    Returns:
        Formatted string response for frontend display
    """
    try:
        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        
        # Step 1: Retrieve based on offer_type
        # AFTER (pass platform)
        if offer_type == "general":
            result = get_general_offers_enhanced(
                query, 
                platform=platform,  # ← NEW
                k=10
            )
            offers = result.get("offers", [])
            
            if not offers:
                return "😔 No general offers found at the moment. Try checking specific platforms like MakeMyTrip or Cleartrip."
            
            # Format offers
            prompt = f"""
            You are a helpful flight deals assistant.
            
            User wants to see general flight offers.
            
            Available offers:
            {offers}
            
            Format these offers in a clear, numbered list with:
            1. **Offer title in bold**
            2. Platform name
            3. Discount details
            4. Coupon code (if available)
            
            Keep it concise and user-friendly.
            """
            response = llm.invoke(prompt)
            return response.content
        
        # AFTER (pass platform)
        elif offer_type == "payment":
            if not bank or not card_type:
                return "⚠️ Please provide your bank name and card type (Credit/Debit) to see payment offers."
            
            result = get_payment_offers_enhanced(
                query, 
                bank, 
                card_type,
                platform=platform,  # ← NEW
                k=10
            )
            offers = result.get("offers", [])
            
            if not offers:
                return f"😔 No {card_type} offers found for {bank} Bank. Try another bank or check general offers."
            
            prompt = f"""
            You are a helpful flight deals assistant.
            
            User wants to see {card_type} offers for {bank} Bank.
            
            Available offers:
            {offers}
            
            Format these offers in a clear, numbered list with:
            1. **Offer title in bold**
            2. Platform and bank name
            3. Discount/cashback details
            4. Coupon code
            
            Keep it concise and user-friendly.
            """
            response = llm.invoke(prompt)
            return response.content
        
        # AFTER (pass platform)
        elif offer_type == "gift":
            result = get_gift_coupons_enhanced(
                query,
                platform=platform,  # ← NEW
                k=10
            )
            offers = result.get("offers", [])
            
            if not offers:
                return "😔 No gift coupons available right now. Check back later!"
            
            prompt = f"""
            You are a helpful flight deals assistant.
            
            User wants to see gift coupons for flights.
            
            Available coupons:
            {offers}
            
            Format these offers in a clear, numbered list with:
            1. **Coupon title in bold**
            2. Platform name
            3. Discount details
            4. Coupon code
            
            Keep it concise and user-friendly.
            """
            response = llm.invoke(prompt)
            return response.content
        
        elif offer_type == "combo" and build_combo:
            if not base_price:
                return "⚠️ Cannot compute combo without flight price. Please provide the booking price."
            
            # Retrieve all three types
            payment_result = get_payment_offers(query, bank or "", card_type or "", k=5) if bank and card_type else {"offers": []}
            general_result = get_general_offers(query, k=5)
            gift_result = get_gift_coupons(query, k=5)
            
            payment_offers = payment_result.get("offers", [])
            general_offers = general_result.get("offers", [])
            gift_coupons = gift_result.get("offers", [])
            
            # Build combos
            combo_data = build_offer_combo(payment_offers, general_offers, gift_coupons, base_price)
            
            if not combo_data.get("combos"):
                return "😔 No combos available. You can still use individual offers!"
            
            # Format for display
            formatted = format_combo_for_frontend(combo_data)
            return formatted
        
        else:
            return "⚠️ Invalid offer type. Choose: general, payment, gift, or combo."
    
    except Exception as e:
        print(f"❌ [OFFER_ORCHESTRATOR] Error: {e}")
        return f"⚠️ Something went wrong while fetching offers. Error: {str(e)}"


@tool
def ask_for_bank_and_card() -> str:
    """
    Helper tool to prompt user for bank and card type.
    """
    return "💳 To show you the best payment offers, I need two things:\n1. Your bank name (e.g., HDFC, ICICI, SBI)\n2. Your card type (Credit or Debit)\n\nPlease share both!"


@tool  
def ask_for_combo_confirmation(base_price: float) -> str:
    """
    Ask user if they want to see combo offers.
    """
    return f"🎁 Would you like me to find a combo deal for your ₹{base_price:.2f} booking? Combos can save you even more by stacking offers together!"