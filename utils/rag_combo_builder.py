# rag_combo_builder.py
import re
from typing import List, Dict, Optional

def extract_discount_value(offer_text: str, discount_type: str = "percentage") -> float:
    """
    Extract numeric discount value from offer text.
    Handles patterns like:
    - "15% off" → 15.0
    - "₹500 off" → 500.0
    - "Flat 10% discount" → 10.0
    """
    if discount_type == "percentage":
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', offer_text)
        if match:
            return float(match.group(1))
    
    # Flat amount
    match = re.search(r'[₹Rs]\s*(\d+(?:,\d+)?)', offer_text)
    if match:
        value_str = match.group(1).replace(',', '')
        return float(value_str)
    
    return 0.0


def calculate_combo_price(base_price: float, offers: List[Dict]) -> Dict:
    """
    Calculate final price after applying multiple offers.
    Logic:
    - Percentage discounts are applied sequentially
    - Flat discounts are summed and subtracted at the end
    """
    current_price = base_price
    flat_discount = 0.0
    applied_steps = []
    
    for offer in offers:
        discount_type = offer.get("discount_type", "flat")
        offer_text = offer.get("offer", "")
        
        if discount_type == "percentage":
            discount_pct = extract_discount_value(offer_text, "percentage")
            if discount_pct > 0:
                reduction = current_price * (discount_pct / 100)
                current_price -= reduction
                applied_steps.append(f"{offer['title']}: {discount_pct}% off (₹{reduction:.2f})")
        else:
            discount_amt = extract_discount_value(offer_text, "flat")
            if discount_amt > 0:
                flat_discount += discount_amt
                applied_steps.append(f"{offer['title']}: ₹{discount_amt} off")
    
    # Apply flat discounts
    current_price -= flat_discount
    
    total_savings = base_price - current_price
    
    return {
        "original_price": base_price,
        "final_price": max(0, current_price),  # Prevent negative prices
        "total_savings": total_savings,
        "applied_steps": applied_steps
    }


def build_offer_combo(
    payment_offers: List[Dict],
    general_offers: List[Dict],
    gift_coupons: List[Dict],
    base_price: float
) -> Dict:
    """
    Build offer combos following rules:
    1. Payment + General (if both exist)
    2. Payment + Gift (if both exist)
    3. NEVER General + Gift
    
    Returns structured combo data with reasoning.
    """
    combos = []
    
    # Combo 1: Payment + General
    if payment_offers and general_offers:
        combo_offers = payment_offers + general_offers
        result = calculate_combo_price(base_price, combo_offers)
        
        combos.append({
            "type": "payment+general",
            "offers": combo_offers,
            "original_price": result["original_price"],
            "final_price": result["final_price"],
            "total_savings": result["total_savings"],
            "reasoning": f"Combined {len(payment_offers)} payment offer(s) with {len(general_offers)} general offer(s). " +
                        f"You save ₹{result['total_savings']:.2f} in total.",
            "breakdown": result["applied_steps"]
        })
    
    # Combo 2: Payment + Gift
    if payment_offers and gift_coupons:
        combo_offers = payment_offers + gift_coupons
        result = calculate_combo_price(base_price, combo_offers)
        
        combos.append({
            "type": "payment+gift",
            "offers": combo_offers,
            "original_price": result["original_price"],
            "final_price": result["final_price"],
            "total_savings": result["total_savings"],
            "reasoning": f"Combined {len(payment_offers)} payment offer(s) with {len(gift_coupons)} gift coupon(s). " +
                        f"You save ₹{result['total_savings']:.2f} in total.",
            "breakdown": result["applied_steps"]
        })
    
    # Find best combo
    if combos:
        best_combo = max(combos, key=lambda x: x["total_savings"])
        return {
            "combos": combos,
            "best_combo": best_combo,
            "recommendation": f"🎉 Best deal: {best_combo['type']} saves you ₹{best_combo['total_savings']:.2f}!"
        }
    else:
        return {
            "combos": [],
            "best_combo": None,
            "recommendation": "No combo available. Try individual offers instead."
        }


def format_combo_for_frontend(combo_data: Dict) -> str:
    """
    Format combo data into user-friendly message for frontend display.
    """
    if not combo_data.get("combos"):
        return "⚠️ No combos available with your current offers."
    
    best = combo_data.get("best_combo")
    if not best:
        return "⚠️ Unable to compute best combo."
    
    lines = [
        f"🎁 **SmartBhai Combo Deal**",
        f"",
        f"✅ **{best['type'].replace('+', ' + ').title()}**",
        f"💰 Original Price: ₹{best['original_price']:.2f}",
        f"🔥 Final Price: **₹{best['final_price']:.2f}**",
        f"💵 You Save: **₹{best['total_savings']:.2f}**",
        f"",
        f"📋 **Applied Offers:**"
    ]
    
    for i, step in enumerate(best.get("breakdown", []), 1):
        lines.append(f"{i}. {step}")
    
    return "\n".join(lines)