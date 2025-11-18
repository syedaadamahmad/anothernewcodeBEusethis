"""
Local Flight Data Loader
Replaces SerpAPI with local JSON file - backwards compatible with existing code.
"""

import json
import os
from typing import List, Dict, Optional
from functools import lru_cache
from langchain_core.tools import tool

# ========================================
# CONFIGURATION
# ========================================

FLIGHTS_JSON_PATH = os.getenv(
    "FLIGHTS_JSON_PATH",
    r"C:\Users\aksha\OneDrive\Desktop\anothernewcodeBEusethis\flights_master.json"
)

AIRLINE_CODE_MAP = {
    "air india": "AI",
    "indigo": "6E",
    "spicejet": "SG",
    "goair": "G8",
    "vistara": "UK",
    "air asia": "I5",
    "akasa": "QP",
    "air india express": "IX",
    "alliance air": "9I",
    "star air": "S5",
    "flybig": "S9",
    "indiaone air": "I7",
    "fly91": "IC",
}

# ========================================
# LOAD FLIGHTS DATA (CACHED IN MEMORY)
# ========================================

@lru_cache(maxsize=1)
def load_flights_data() -> List[Dict]:
    """
    Load flights from JSON file once and cache in memory.
    This runs only once per application lifecycle.
    """
    if not os.path.exists(FLIGHTS_JSON_PATH):
        print(f"❌ [FLIGHTS_LOADER] File not found: {FLIGHTS_JSON_PATH}")
        return []
    
    try:
        with open(FLIGHTS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ [FLIGHTS_LOADER] Loaded {len(data)} flights from JSON")
        return data
    except Exception as e:
        print(f"❌ [FLIGHTS_LOADER] Error loading JSON: {e}")
        return []


# ========================================
# FILTERING LOGIC
# ========================================

def normalize_airline_codes(user_input: Optional[str]) -> Optional[List[str]]:
    """
    Convert airline names/codes to standard 2-letter codes.
    Returns None if 'no preference' or similar.
    """
    if not user_input:
        return None
    
    raw = str(user_input).lower().strip()
    
    # Check for "no preference" tokens
    if any(token in raw for token in ["any", "no preference", "all airlines", "no airline"]):
        return None
    
    # Split comma-separated airlines
    airlines = [a.strip() for a in raw.split(",") if a.strip()]
    codes = []
    
    for airline in airlines:
        # Direct code match (e.g., "AI", "6E")
        if len(airline) <= 3 and airline.isalpha():
            codes.append(airline.upper())
        # Name match from map
        elif airline in AIRLINE_CODE_MAP:
            codes.append(AIRLINE_CODE_MAP[airline])
        else:
            # Fuzzy match (e.g., "air india" in "airindia")
            for name, code in AIRLINE_CODE_MAP.items():
                if name.replace(" ", "") in airline.replace(" ", ""):
                    codes.append(code)
                    break
    
    return list(set(codes)) if codes else None


def normalize_travel_class(user_input: Optional[str]) -> str:
    """
    Normalize travel class to match JSON format.
    """
    if not user_input:
        return "Economy"
    
    raw = str(user_input).lower().strip()
    
    if "first" in raw:
        return "First"
    elif "business" in raw:
        return "Business"
    elif "premium" in raw:
        return "Premium Economy"
    else:
        return "Economy"


def extract_date(datetime_str: str) -> str:
    """
    Extract date from datetime string (e.g., "2025-11-30 06:00" → "2025-11-30")
    """
    if not datetime_str:
        return ""
    return datetime_str.split(" ")[0]


def filter_flights(
    all_flights: List[Dict],
    departure_id: str,
    arrival_id: str,
    departure_date: str,
    include_airlines: Optional[str] = None,
    max_price: Optional[str] = None,
    travel_class: Optional[str] = None,
) -> List[Dict]:
    """
    Filter flights based on user criteria.
    Returns list of flight objects matching the format expected by frontend.
    """
    print(f"🔍 [FILTER] Starting with {len(all_flights)} flights")
    print(f"    Route: {departure_id} → {arrival_id}")
    print(f"    Date: {departure_date}")
    print(f"    Airlines: {include_airlines or 'All'}")
    print(f"    Max Price: ₹{max_price or 'No limit'}")
    print(f"    Class: {travel_class or 'Economy'}")
    
    # Normalize inputs
    airline_codes = normalize_airline_codes(include_airlines)
    normalized_class = normalize_travel_class(travel_class)
    
    # Parse max price
    max_price_int = None
    if max_price:
        try:
            max_price_int = int(str(max_price).replace(",", "").strip())
        except ValueError:
            pass
    
    filtered = []
    
    for flight_obj in all_flights:
        # Your JSON already has the correct structure
        flight_data = flight_obj.get("flight_data", [])
        booking_options = flight_obj.get("booking_options", [])
        
        if not flight_data or not booking_options:
            continue
        
        # Get first flight segment
        first_flight = flight_data[0]
        
        # Filter 1: Route match (departure_id → arrival_id)
        dep_id = first_flight.get("departure_airport", {}).get("id", "").upper()
        arr_id = first_flight.get("arrival_airport", {}).get("id", "").upper()
        
        if dep_id != departure_id.upper() or arr_id != arrival_id.upper():
            continue
        
        # Filter 2: Date match
        dep_time = first_flight.get("departure_airport", {}).get("time", "")
        flight_date = extract_date(dep_time)
        
        if flight_date != departure_date:
            continue
        
        # Filter 3: Airline match (if specified)
        if airline_codes:
            flight_airline = first_flight.get("airline", "")
            
            # Check if any of the specified airline codes match
            airline_matched = False
            for code in airline_codes:
                # Check if airline name contains the code (e.g., "Air India" contains "AI")
                if code.upper() in flight_airline.upper():
                    airline_matched = True
                    break
                # Also check reverse mapping
                for name, mapped_code in AIRLINE_CODE_MAP.items():
                    if mapped_code.upper() == code.upper() and name in flight_airline.lower():
                        airline_matched = True
                        break
            
            if not airline_matched:
                continue
        
        # Filter 4: Travel class match
        flight_class = first_flight.get("travel_class", "Economy")
        if flight_class != normalized_class:
            continue
        
        # Filter 5: Price filter (check booking options)
        if max_price_int:
            # Check if any booking option is within budget
            within_budget = False
            for option in booking_options:
                together = option.get("together", {})
                price = together.get("price")
                if price and int(price) <= max_price_int:
                    within_budget = True
                    break
            
            if not within_budget:
                continue
        
        # Flight passed all filters - include it
        filtered.append(flight_obj)
    
    print(f"✅ [FILTER] {len(filtered)} flights match criteria")
    return filtered


# ========================================
# MAIN TOOL (LANGCHAIN COMPATIBLE)
# ========================================

def _get_flight_with_aggregator_internal(
    departure_id: str,
    arrival_id: str,
    departure_date: str,
    include_airlines: Optional[str] = None,
    max_price: Optional[str] = None,
    travel_class: Optional[str] = None,
) -> List[Dict]:
    """
    Internal implementation so tests can call it directly while the @tool wrapper
    keeps LangChain compatibility.
    """
    print(f"\n🚀 [FLIGHT SEARCH] Starting search...")
    print(f"    Query: {departure_id} → {arrival_id} on {departure_date}")
    print(f"    Airlines: {include_airlines or 'All'}")
    print(f"    Max Price: ₹{max_price or 'No limit'}")
    print(f"    Class: {travel_class or 'Economy'}")

    all_flights = load_flights_data()
    if not all_flights:
        print("❌ [FLIGHT SEARCH] No flights available in database")
        return []

    filtered_flights = filter_flights(
        all_flights=all_flights,
        departure_id=departure_id,
        arrival_id=arrival_id,
        departure_date=departure_date,
        include_airlines=include_airlines,
        max_price=max_price,
        travel_class=travel_class,
    )

    print(f"✅ [FLIGHT SEARCH] Returning {len(filtered_flights)} flights\n")
    return filtered_flights


@tool("get_flight_with_aggregator", description="Search local flight database for matching flights")
def get_flight_with_aggregator(tool_input: dict) -> List[Dict]:
    """
    LangChain-compatible entrypoint. Accepts a single dict (tool_input).
    Internal logic lives in _get_flight_with_aggregator_internal for direct tests.
    """
    departure_id = tool_input.get("departure_id")
    arrival_id = tool_input.get("arrival_id")
    departure_date = tool_input.get("departure_date")
    include_airlines = tool_input.get("include_airlines")
    max_price = tool_input.get("max_price")
    travel_class = tool_input.get("travel_class")

    return _get_flight_with_aggregator_internal(
        departure_id,
        arrival_id,
        departure_date,
        include_airlines,
        max_price,
        travel_class,
    )


# ========================================
# HELPER FOR MAIN.PY STARTUP
# ========================================

def preload_flights():
    """
    Call this in main.py startup to preload flights into memory.
    """
    flights = load_flights_data()
    print(f"📦 [STARTUP] Preloaded {len(flights)} flights into memory")
    return len(flights)


# ========================================
# TESTING
# ========================================

if __name__ == "__main__":
    print("=== Testing Flights Loader ===\n")
    
    # Preload
    count = preload_flights()
    print(f"\nLoaded {count} flights\n")
    
    # Test 1: Basic search
    print("\n--- Test 1: DEL → MAA, Economy, No airline preference ---")
    results = _get_flight_with_aggregator_internal(
        departure_id="DEL",
        arrival_id="MAA",
        departure_date="2025-11-30",
        include_airlines=None,
        max_price="10000",
        travel_class="Economy"
    )
    
    print(f"\n=== Test 1 Results ===")
    print(f"Found {len(results)} flights")
    
    if results:
        first = results[0]
        print(f"\nFirst flight details:")
        print(f"  Airline: {first['flight_data'][0]['airline']}")
        print(f"  Flight: {first['flight_data'][0]['flight_number']}")
        print(f"  Departure: {first['flight_data'][0]['departure_airport']['time']}")
        print(f"  Arrival: {first['flight_data'][0]['arrival_airport']['time']}")
        print(f"  Class: {first['flight_data'][0]['travel_class']}")
        print(f"  Price: ₹{first['booking_options'][0]['together']['price']}")
        print(f"  Booking options: {len(first['booking_options'])}")
    
    # Test 2: Airline filter
    print("\n\n--- Test 2: DEL → MAA, Air India only ---")
    results2 = _get_flight_with_aggregator_internal(
        departure_id="DEL",
        arrival_id="MAA",
        departure_date="2025-11-30",
        include_airlines="Air India",
        max_price=None,
        travel_class="Economy"
    )
    
    print(f"\n=== Test 2 Results ===")
    print(f"Found {len(results2)} Air India flights")
    
    if results2:
        for i, flight in enumerate(results2[:3], 1):
            print(f"\n  Flight {i}:")
            print(f"    Airline: {flight['flight_data'][0]['airline']}")
            print(f"    Price: ₹{flight['booking_options'][0]['together']['price']}")