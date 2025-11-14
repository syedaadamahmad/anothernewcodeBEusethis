# cc
# get_flights.py
import os
import re
import time
import asyncio
from functools import wraps
from dotenv import load_dotenv
from serpapi import GoogleSearch
from langchain_core.tools import tool
from utils.mongoDB import get_api_cache_result, save_api_cache_result, batch_save_cache_results

load_dotenv()

# ----------------- Constants -----------------
AIRLINE_MAP = {
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

ANY_TOKENS = {"any", "any airline", "no preference", "no airline", "all airlines"}
FLIGHT_TYPE = (os.getenv("FLIGHT_TYPE"))

# ----------------- Helpers -----------------
def normalize_price(value):
    if not value:
        return None
    cleaned = re.sub(r"[^\d]", "", str(value))
    return cleaned if cleaned.isdigit() else None


def map_airlines(user_input):
    if not user_input:
        return None
    raw = str(user_input).lower().strip()
    if any(tok in raw for tok in ANY_TOKENS):
        return None
    airlines = [a.strip().lower() for a in raw.split(",") if a.strip()]
    codes = []
    for a in airlines:
        if len(a) <= 3 and a.isalpha():
            codes.append(a.upper())
        elif a in AIRLINE_MAP:
            codes.append(AIRLINE_MAP[a])
        else:
            for k, v in AIRLINE_MAP.items():
                if k.replace(" ", "") in a.replace(" ", ""):
                    codes.append(v)
                    break
    return ",".join(set(codes)) if codes else None


def is_flight_under_budget(flight, max_price):
    if not max_price:
        return True
    cleaned_max_price = normalize_price(max_price)
    if not cleaned_max_price:
        return True

    price_obj = flight.get("price", {})
    price_amount = price_obj.get("amount") if isinstance(price_obj, dict) else flight.get("price_amount")

    if price_amount:
        try:
            return int(str(price_amount).replace(",", "")) <= int(cleaned_max_price)
        except Exception:
            return True
    return True


def safe_get_token(flight):
    """Extract booking token from flight entry."""
    return flight.get("booking_token")


# ----------------- Retry Logic -----------------
def retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"❌ Function {func.__name__} failed after {max_retries} attempts. Final error: {e}")
                        raise e
                    delay = min(initial_delay * (2 ** attempt), max_delay)
                    print(f"⚠️ Retrying {func.__name__} in {delay:.2f}s (Attempt {attempt + 1}/{max_retries}) due to error: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator


# ----------------- Core Fetch Logic -----------------
@retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8)
def get_flights(departure_id, arrival_id, departure_date, max_price=None, include_airlines=None, travel_class=None):
    """
    Fetch flights with intelligent caching.
    Returns tuple: (flights_data, is_from_cache)
    """
    trip_type = FLIGHT_TYPE

    normalized_airlines = map_airlines(include_airlines)
    normalized_price = normalize_price(max_price)

    request_key = {
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "departure_date": departure_date,
        "trip_type": trip_type,
        "max_price": normalized_price,
        "airlines": normalized_airlines,
        "travel_class": travel_class,
    }

    # Check cache first
    cached_result = get_api_cache_result(request_key)
    if cached_result is not None:
        print("✅ [CACHE HIT] Using cached flight data")
        return cached_result, True  # Return data + cache hit flag

    print("❌ [CACHE MISS] Fetching fresh flight data from API")

    params = {
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": os.getenv("SEARCH_ENGINE"),
        "hl": os.getenv("LANGUAGE"),
        "gl": os.getenv("COUNTRY"),
        "currency": os.getenv("CURRENCY"),
        "no_cache": True,
        "departure_id": departure_id,
        "arrival_id": arrival_id,
        "outbound_date": departure_date,
        "type": trip_type,
        "show_hidden": "true",
        "deep_search": "true",
    }

    mapped = map_airlines(include_airlines)
    if mapped:
        params["include_airlines"] = mapped

    cleaned = normalize_price(max_price)
    if cleaned:
        params["max_price"] = cleaned
      # Add travel class (1=Economy, 2=Premium, 3=Business, 4=First)
    if travel_class:
        class_map = {"economy": 1, "premium economy": 2, "premium": 2, "business": 3, "first": 4, "first class": 4}
        travel_class_lower = str(travel_class).lower().strip()
        params["travel_class"] = class_map.get(travel_class_lower, 1)  # Default to Economy

    print("🔎 [API CALL] Fetching flights:", params)
    search = GoogleSearch(params)
    results = search.get_dict()

    best = results.get("best_flights", [])
    other = results.get("other_flights", [])
    outbound_flights = best + other
    print(f"🛫 [API RESPONSE] Found {len(outbound_flights)} flights")

    # Save to cache
    if outbound_flights:
        save_api_cache_result(request_key, outbound_flights, verbose=False)

    return outbound_flights, False  # Return data + cache miss flag


# ----------------- Async Booking Fetch -----------------
async def fetch_booking_options_async(booking_token, departure_date, departure_id, arrival_id):
    """
    Async wrapper for fetching booking options.
    Runs the synchronous GoogleSearch in a thread pool to avoid blocking.
    """
    try:
        params = {
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": os.getenv("SEARCH_ENGINE"),
            "hl": os.getenv("LANGUAGE"),
            "gl": os.getenv("COUNTRY"),
            "currency": os.getenv("CURRENCY"),
            "type": FLIGHT_TYPE,
            "no_cache": True,
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": departure_date,
            "booking_token": booking_token,
            "show_hidden": "true",
            "deep_search": "true",
        }
        
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: GoogleSearch(params).get_dict()
        )
        return result
    except Exception as e:
        print(f"❌ Async booking fetch failed for token {booking_token[:10]}: {e}")
        return None


async def get_booking_from_cache_async(token):
    """
    Async cache lookup for booking options (used when flights were cached).
    """
    booking_key = {"booking_token": token}
    
    # Run cache lookup in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    cached_data = await loop.run_in_executor(
        None,
        lambda: get_api_cache_result(booking_key, verbose=False)
    )
    
    return cached_data


async def enrich_flights_with_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
    """
    When flights ARE cached, check booking cache for each flight.
    This maintains the benefit of caching booking options from previous searches.
    """
    print(f"📋 [CACHE MODE] Checking booking cache for {len(filtered_flights)} flights...")
    
    tasks = []
    for flight in filtered_flights:
        token = safe_get_token(flight)
        if not token:
            continue
        
        # Check cache first
        task = get_booking_from_cache_async(token)
        tasks.append((flight, token, task))
    
    # Wait for all cache checks
    cache_results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
    # Separate cached vs non-cached bookings
    enriched_flights = []
    fetch_tasks = []
    cache_hits = 0
    cache_misses = 0
    
    for (flight, token, _), cached_data in zip(tasks, cache_results):
        if isinstance(cached_data, Exception) or cached_data is None:
            # Cache miss - need to fetch
            cache_misses += 1
            fetch_tasks.append((flight, token, departure_date, departure_id, arrival_id))
        else:
            # Cache hit
            cache_hits += 1
            selected = cached_data.get("selected_flights", [])
            booking_opts = cached_data.get("booking_options", [])
            flight_obj = []
            
            if selected and isinstance(selected, list):
                first = selected[0]
                if isinstance(first, dict) and first.get("flights"):
                    flight_obj = first["flights"]
            
            enriched_flights.append({
                "flight_data": flight_obj,
                "booking_options": booking_opts,
            })
    
    print(f"⏳ Booking cache: {cache_hits} hits, {cache_misses} misses")
    
    # Fetch missing bookings concurrently
    if fetch_tasks:
        print(f"🚀 [API CALLS] Fetching {len(fetch_tasks)} missing booking options...")
        fetch_results = await asyncio.gather(*[
            fetch_booking_options_async(token, dep_date, dep_id, arr_id)
            for _, token, dep_date, dep_id, arr_id in fetch_tasks
        ], return_exceptions=True)
        
        # Process fetched results and save to cache
        cache_batch = []
        for (flight, token, _, _, _), booking_data in zip(fetch_tasks, fetch_results):
            if isinstance(booking_data, Exception) or not booking_data:
                continue
            
            selected = booking_data.get("selected_flights", [])
            booking_opts = booking_data.get("booking_options", [])
            flight_obj = []
            
            if selected and isinstance(selected, list):
                first = selected[0]
                if isinstance(first, dict) and first.get("flights"):
                    flight_obj = first["flights"]
            
            enriched_flights.append({
                "flight_data": flight_obj,
                "booking_options": booking_opts,
            })
            
            # Prepare for batch cache save
            booking_key = {"booking_token": token}
            cache_batch.append((booking_key, booking_data))
        
        # Batch save all new booking data
        if cache_batch:
            batch_save_cache_results(cache_batch)
    
    return enriched_flights


async def enrich_flights_no_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
    """
    When flights are NOT cached, skip booking cache checks entirely.
    Fetch all booking options directly and save them in a batch.
    """
    print(f"🚀 [NO-CACHE MODE] Fetching {len(filtered_flights)} booking options directly (skipping cache checks)...")
    
    tasks = []
    for flight in filtered_flights:
        token = safe_get_token(flight)
        if not token:
            continue
        
        # Directly fetch without cache check
        task = fetch_booking_options_async(token, departure_date, departure_id, arrival_id)
        tasks.append((flight, token, task))
    
    # Fetch all bookings concurrently
    results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
    enriched_flights = []
    cache_batch = []
    successful_fetches = 0
    
    for (flight, token, _), booking_data in zip(tasks, results):
        if isinstance(booking_data, Exception):
            print(f"❌ Task failed: {booking_data}")
            continue
        
        if not booking_data:
            continue
        
        successful_fetches += 1
        
        selected = booking_data.get("selected_flights", [])
        booking_opts = booking_data.get("booking_options", [])
        flight_obj = []
        
        if selected and isinstance(selected, list):
            first = selected[0]
            if isinstance(first, dict) and first.get("flights"):
                flight_obj = first["flights"]

        enriched_flights.append({
            "flight_data": flight_obj,
            "booking_options": booking_opts,
        })
        
        # Prepare for batch cache save
        booking_key = {"booking_token": token}
        cache_batch.append((booking_key, booking_data))
    
    print(f"✅ [API CALLS] Completed {successful_fetches} booking fetches")
    
    # Batch save all booking data to cache
    if cache_batch:
        batch_save_cache_results(cache_batch)
    
    return enriched_flights


# ----------------- Aggregator Tool -----------------
@tool("get_flight_with_aggregator", description="Fetch and enrich one-way flight data with booking options.")
def get_flight_with_aggregator(
    departure_id: str,
    arrival_id: str,
    departure_date: str,
    include_airlines: str | None = None,
    max_price: str | None = None,
    travel_class: str | None = None,  # ← ADD THIS
):
    """
    Main entry point for flight search with intelligent cache-aware booking fetching.
    
    Strategy:
    - If flights are cached → check booking cache for each flight
    - If flights are NOT cached → skip booking cache, fetch all directly
    
    This eliminates wasteful cache checks when we know bookings won't be cached.
    """
    print("🚀 [FLIGHT SEARCH] Starting aggregator")

    # Get flights and check if they were cached
    all_flights, is_cached = get_flights(
        departure_id, arrival_id, departure_date, max_price, include_airlines
    )
    outbound_list = all_flights or []

    # Filter by budget
    filtered = [f for f in outbound_list if is_flight_under_budget(f, max_price)]
    print(f"🔎 [FILTER] {len(outbound_list)} flights → {len(filtered)} within budget")

    if not filtered:
        print("⚠️ No flights found within budget.")
        return []

    # Choose enrichment strategy based on cache status
    try:
        if is_cached:
            # Flights were cached → check booking cache
            enriched_flights = asyncio.run(
                enrich_flights_with_cache_async(filtered, departure_date, departure_id, arrival_id)
            )
        else:
            # Flights were NOT cached → skip booking cache entirely
            enriched_flights = asyncio.run(
                enrich_flights_no_cache_async(filtered, departure_date, departure_id, arrival_id)
            )
    except Exception as e:
        print(f"❌ Async enrichment failed: {e}")
        enriched_flights = []

    print(f"✅ [COMPLETE] Returning {len(enriched_flights)} enriched flights")
    return enriched_flights













































# # get_flights.py with all the filters
# import os
# import re
# import json
# import time
# import asyncio
# from functools import wraps
# from dotenv import load_dotenv
# from serpapi import GoogleSearch
# from langchain_core.tools import tool
# from utils.mongoDB import get_api_cache_result, save_api_cache_result, batch_save_cache_results

# # ----------------- User Context Helper -----------------
# def get_user_context():
#     return {
#         "user_bank": os.getenv("TEST_USER_BANK") or None,
#         "card_type": os.getenv("TEST_CARD_TYPE") or None,
#     }
    
# # ----------------- Combiner Async Wrapper -----------------
# async def run_offer_combiner_async(base_price, user_bank, card_type, platform_name, flight_type):
#     loop = asyncio.get_event_loop()
#     try:
#         print("🔍 Type of offer_combiner_tool:", type(offer_combiner_tool))
#         res = await loop.run_in_executor(
#             None,
#             lambda: offer_combiner_tool.invoke({
#                 "base_price": float(base_price),
#                 "user_bank": user_bank,
#                 "card_type": card_type,
#                 "platform": platform_name,
#                 "flight_type": flight_type,
#                 "query_text": f"{user_bank or ''} {card_type or ''} flight offers on {platform_name}",
#             }),
#         )
#         return res
#     except Exception as e:
#         return {"error": str(e)}

# load_dotenv()

# # ----------------- Constants -----------------
# AIRLINE_MAP = {
#     "air india": "AI", "indigo": "6E", "spicejet": "SG", "goair": "G8",
#     "vistara": "UK", "air asia": "I5", "akasa": "QP", "air india express": "IX",
#     "alliance air": "9I", "star air": "S5", "flybig": "S9", "indiaone air": "I7", "fly91": "IC",
# }

# ANY_TOKENS = {"any", "any airline", "no preference", "no airline", "all airlines"}
# FLIGHT_TYPE = os.getenv("FLIGHT_TYPE")

# # ----------------- Time Mapping -----------------
# TIME_RANGE_MAP = {
#     "morning": "0,11",    # 12:00 AM - 12:00 PM
#     "noon": "12,14",      # 12:00 PM - 3:00 PM
#     "afternoon": "15,17", # 3:00 PM - 6:00 PM
#     "evening": "18,20",   # 6:00 PM - 9:00 PM
#     "night": "21,23",     # 9:00 PM - 12:00 AM
# }

# # ----------------- Travel Class Mapping -----------------
# TRAVEL_CLASS_MAP = {
#     "economy": 1,
#     "premium economy": 2,
#     "premium": 2,
#     "business": 3,
#     "first": 4,
#     "first class": 4,
# }

# # ----------------- Helpers -----------------
# def normalize_price(value):
#     if not value:
#         return None
#     cleaned = re.sub(r"[^\d]", "", str(value))
#     return cleaned if cleaned.isdigit() else None


# def map_airlines(user_input):
#     if not user_input:
#         return None
#     raw = str(user_input).lower().strip()
#     if any(tok in raw for tok in ANY_TOKENS):
#         return None
#     airlines = [a.strip().lower() for a in raw.split(",") if a.strip()]
#     codes = []
#     for a in airlines:
#         if len(a) <= 3 and a.isalpha():
#             codes.append(a.upper())
#         elif a in AIRLINE_MAP:
#             codes.append(AIRLINE_MAP[a])
#         else:
#             for k, v in AIRLINE_MAP.items():
#                 if k.replace(" ", "") in a.replace(" ", ""):
#                     codes.append(v)
#                     break
#     return ",".join(set(codes)) if codes else None

# def get_final_arrival_id(flight):
#     """
#     Extract the final arrival airport ID from flight data.
#     Handles both single-leg and multi-leg flights.
#     """
#     if not flight:
#         return None
    
#     # Check for layovers array (multi-leg flight)
#     layovers = flight.get("layovers", [])
#     if layovers:
#         last_layover = layovers[-1]
#         return last_layover.get("id")
    
#     # Single-leg flight
#     arrival_airport = flight.get("arrival_airport", {})
#     return arrival_airport.get("id")


# def is_flight_under_budget(flight, max_price):
#     """
#     Check if flight is within budget.
#     Handles BOTH raw SerpAPI format and enriched format with booking_options.
#     """
#     if not max_price:
#         return True
    
#     cleaned_max_price = normalize_price(max_price)
#     if not cleaned_max_price:
#         return True
    
#     # Check root-level price (raw SerpAPI format)
#     price = flight.get("price")
#     if price:
#         try:
#             price_int = int(str(price).replace(",", ""))
#             max_int = int(cleaned_max_price)
#             return price_int <= max_int
#         except Exception:
#             pass
    
#     # Check booking_options[0].together.price (enriched format)
#     booking_options = flight.get("booking_options", [])
#     if booking_options and len(booking_options) > 0:
#         first_option = booking_options[0]
#         together = first_option.get("together", {})
#         option_price = together.get("price")
#         if option_price:
#             try:
#                 price_int = int(str(option_price).replace(",", ""))
#                 max_int = int(cleaned_max_price)
#                 return price_int <= max_int
#             except Exception:
#                 pass
    
#     return True


# def parse_passengers(passenger_str):
#     """
#     Parse passenger string into component counts.
#     Format: "adults,children,infants_in_seat,infants_on_lap"
#     Example: "2,1,0,1" → {adults: 2, children: 1, infants_in_seat: 0, infants_on_lap: 1}
#     """
#     try:
#         parts = str(passenger_str).split(",")
#         return {
#             "adults": int(parts[0]) if len(parts) > 0 else 1,
#             "children": int(parts[1]) if len(parts) > 1 else 0,
#             "infants_in_seat": int(parts[2]) if len(parts) > 2 else 0,
#             "infants_on_lap": int(parts[3]) if len(parts) > 3 else 0,
#         }
#     except (ValueError, IndexError):
#         return {"adults": 1, "children": 0, "infants_in_seat": 0, "infants_on_lap": 0}


# def map_outbound_times(time_preference):
#     """
#     Map natural language time preferences to SerpAPI format.
#     Example: "morning" → "0,11"
#     """
#     if not time_preference:
#         return None
#     time_key = str(time_preference).lower().strip()
#     return TIME_RANGE_MAP.get(time_key)


# def map_travel_class(class_name):
#     """
#     Map natural language travel class to SerpAPI numeric format.
#     Example: "Economy" → 1, "Business" → 3
#     """
#     if not class_name:
#         return 1  # Default to Economy
#     class_key = str(class_name).lower().strip()
#     return TRAVEL_CLASS_MAP.get(class_key, 1)


# # ----------------- Rate Limiter -----------------
# def rate_limit(max_calls=90, period=60):
#     timestamps = []
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             now = time.time()
#             timestamps[:] = [t for t in timestamps if now - t < period]
#             if len(timestamps) >= max_calls:
#                 sleep_time = period - (now - timestamps[0])
#                 if sleep_time > 0:
#                     print(f"⏳ Rate limit reached. Sleeping for {sleep_time:.2f}s")
#                     time.sleep(sleep_time)
#                     timestamps[:] = []
#             timestamps.append(time.time())
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator


# from utils.offer_orchestrator_tool import offer_orchestrator_tool as offer_combiner_tool


# # ----------------- Offer Combiner Integration -----------------
# @rate_limit(max_calls=90, period=60)
# def safe_get_token(flight):
#     return flight.get("departure_token") or flight.get("booking_token")


# def get_flights(
#     departure_id, 
#     arrival_id, 
#     departure_date, 
#     max_price=None, 
#     include_airlines=None,
#     passengers="1,0,0,0",
#     outbound_times=None,
#     travel_class="Economy"
# ):
#     """
#     Fetch flights with intelligent caching.
#     Adds support for:
#       - travel_class
#       - passenger counts (adults, children, infants)
#       - outbound_times (e.g., '5,12' for morning)
#     """
#     trip_type = FLIGHT_TYPE
#     normalized_airlines = map_airlines(include_airlines)
#     normalized_price = normalize_price(max_price)

#     request_key = {
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "departure_date": departure_date,
#         "trip_type": trip_type,
#         "max_price": normalized_price,
#         "airlines": normalized_airlines,
#         "passengers": passengers,
#         "outbound_times": outbound_times,
#         "travel_class": travel_class,
#     }

#     cached_result = get_api_cache_result(request_key)
#     if cached_result is not None:
#         print("✅ [CACHE HIT] Using cached flight data")
#         return cached_result, True

#     print("❌ [CACHE MISS] Fetching fresh flight data from API")
#     print("\n" + "="*80)
#     print("📦 [DEBUG] RAW PARAMETERS (Before Conversion):")
#     print("="*80)
#     print(f"  departure_id      : {departure_id}")
#     print(f"  arrival_id        : {arrival_id}")
#     print(f"  departure_date    : {departure_date}")
#     print(f"  include_airlines  : {include_airlines}")
#     print(f"  max_price         : {max_price}")
#     print(f"  passengers        : {passengers}")
#     print(f"  outbound_times    : {outbound_times}")
#     print(f"  travel_class      : {travel_class}")
#     print("="*80 + "\n")

#     params = {
#         "api_key": os.getenv("SERPAPI_API_KEY"),
#         "engine": os.getenv("SEARCH_ENGINE"),
#         "hl": os.getenv("LANGUAGE"),
#         "gl": os.getenv("COUNTRY"),
#         "currency": os.getenv("CURRENCY"),
#         "no_cache": True,
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "outbound_date": departure_date,
#         "type": trip_type,
#         "show_hidden": "true",
#     }

#     # Map airlines
#     mapped = map_airlines(include_airlines)
#     if mapped:
#         params["include_airlines"] = mapped

#     # Map price
#     cleaned = normalize_price(max_price)
#     if cleaned:
#         params["max_price"] = cleaned
#         print(f"[DEBUG] Normalized max_price: {max_price} → {cleaned}")
        
#     # Outbound times and travel class already mapped by model_with_tool
#     if outbound_times:
#         params["outbound_times"] = outbound_times
#         print(f"[DEBUG] Using pre-mapped outbound_times: {outbound_times}")
    
#     if travel_class:
#         params["travel_class"] = travel_class
#         print(f"[DEBUG] Using pre-mapped travel_class: {travel_class}")
    
#     print("=" * 80)
#     # Passengers (parse the string)
#     passenger_counts = parse_passengers(passengers)
#     params["adults"] = passenger_counts["adults"]
#     if passenger_counts["children"] > 0:
#         params["children"] = passenger_counts["children"]
#     if passenger_counts["infants_in_seat"] > 0:
#         params["infants_in_seat"] = passenger_counts["infants_in_seat"]
#     if passenger_counts["infants_on_lap"] > 0:
#         params["infants_on_lap"] = passenger_counts["infants_on_lap"]
    
#     print("[DEBUG] ========================================")
#     print("[DEBUG] SERPAPI CALL PARAMETERS:")
#     print("[DEBUG] ========================================")
    
#     for key, value in params.items():
#         print(f"[DEBUG]   {key:25} = {value}")
#     print("[DEBUG] ========================================\n")
    
#     print("[DEBUG] Calling SerpAPI GoogleSearch...")
#     try:
#         search = GoogleSearch(params)
#         results = search.get_dict()
#         print("[DEBUG] ✅ SerpAPI call successful")
#     except Exception as e:
#         print(f"[ERROR] ❌ SerpAPI call failed: {e}")
#         import traceback
#         traceback.print_exc()
#         return [], False

#     # ========================================
#     # DIAGNOSTIC: Dump raw API response
#     # ========================================
#     try:
#         with open('C:/Users/newbr/OneDrive/Desktop/backendSB_p+g2/SmartBhaiBackend/serpapi_response.json', 'w', encoding='utf-8') as f:
#             json.dump(results, f, indent=2, ensure_ascii=False)
#         print("[DEBUG] ✅ Saved raw API response to serpapi_response.json")
#     except Exception as e:
#         print(f"[DEBUG] ⚠️ Could not save JSON: {e}")

#     best = results.get("best_flights", [])
#     other = results.get("other_flights", [])
#     outbound_flights = best + other
    
#     print(f"\n[DEBUG] ========================================")
#     print(f"[DEBUG] API RESPONSE SUMMARY:")
#     print(f"[DEBUG] ========================================")
#     print(f"[DEBUG]   best_flights    : {len(best)}")
#     print(f"[DEBUG]   other_flights   : {len(other)}")
#     print(f"[DEBUG]   total_flights   : {len(outbound_flights)}")
#     print(f"[DEBUG] ========================================\n")
    
#     # DIAGNOSTIC: Inspect first 3 flights
#     if outbound_flights:
#         print("[DEBUG] ========================================")
#         print("[DEBUG] FIRST 3 FLIGHT STRUCTURES:")
#         print("[DEBUG] ========================================")
#         for idx, flight in enumerate(outbound_flights[:3]):
#             print(f"\n[DEBUG] ✈️ Flight {idx+1}:")
#             print(f"[DEBUG]   Keys: {list(flight.keys())}")
            
#             # Check all possible price locations
#             if 'price' in flight:
#                 print(f"[DEBUG]   price: {flight['price']} (type: {type(flight['price'])})")
#             if 'total_price' in flight:
#                 print(f"[DEBUG]   total_price: {flight['total_price']} (type: {type(flight['total_price'])})")
#             if 'booking_options' in flight:
#                 print(f"[DEBUG]   booking_options: {len(flight['booking_options'])} options")
#                 if flight['booking_options']:
#                     first_option = flight['booking_options'][0]
#                     print(f"[DEBUG]     First option keys: {list(first_option.keys())}")
#                     if 'price' in first_option:
#                         print(f"[DEBUG]     First option price: {first_option['price']}")
#         print("[DEBUG] ========================================\n")

#     if outbound_flights:
#         save_api_cache_result(request_key, outbound_flights, verbose=False)

#     return outbound_flights, False

# # ----------------- Async Booking Fetch -----------------
# async def fetch_booking_options_async(booking_token, departure_date, departure_id, arrival_id):
#     """
#     Async wrapper for fetching booking options.
#     Runs the synchronous GoogleSearch in a thread pool to avoid blocking.
#     """
#     try:
#         params = {
#             "api_key": os.getenv("SERPAPI_API_KEY"),
#             "engine": os.getenv("SEARCH_ENGINE"),
#             "hl": os.getenv("LANGUAGE"),
#             "gl": os.getenv("COUNTRY"),
#             "currency": os.getenv("CURRENCY"),
#             "type": FLIGHT_TYPE,
#             "no_cache": True,
#             "departure_id": departure_id,
#             "arrival_id": arrival_id,
#             "outbound_date": departure_date,
#             "booking_token": booking_token,
#             "show_hidden": "true",
#         }
        
#         # Run in thread pool to avoid blocking the event loop
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None, 
#             lambda: GoogleSearch(params).get_dict()
#         )
#         return result
#     except Exception as e:
#         print(f"❌ Async booking fetch failed for token {booking_token[:10]}: {e}")
#         return None


# async def get_booking_from_cache_async(token):
#     """
#     Async cache lookup for booking options (used when flights were cached).
#     """
#     booking_key = {"booking_token": token}
    
#     # Run cache lookup in thread pool to avoid blocking
#     loop = asyncio.get_event_loop()
#     cached_data = await loop.run_in_executor(
#         None,
#         lambda: get_api_cache_result(booking_key, verbose=False)
#     )
    
#     return cached_data


# async def enrich_flights_with_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     CACHED PATH: Enrich flights with booking options and offers using async cache lookups.
#     """
#     print(f"📦 [CACHE PATH] Enriching {len(filtered_flights)} flights from cache...")
    
#     user_ctx = get_user_context()
#     user_bank = user_ctx.get("user_bank")
#     card_type = user_ctx.get("card_type")
    
#     results = []
#     for idx, flight in enumerate(filtered_flights):
#         token = safe_get_token(flight)
#         if not token:
#             print(f"⚠️  Flight {idx+1}: No token, skipping")
#             continue
        
#         print(f"🔍 Flight {idx+1}/{len(filtered_flights)}: Checking cache for token {token[:10]}...")
        
#         # Async cache lookup
#         booking_data = await get_booking_from_cache_async(token)
#         if not booking_data:
#             print(f"❌ Flight {idx+1}: Cache miss for booking token, skipping")
#             continue
        
#         selected_flights = booking_data.get("selected_flights", [])
#         booking_options = booking_data.get("booking_options", [])
        
#         flight_segments = []
#         if selected_flights and isinstance(selected_flights, list):
#             first_selected = selected_flights[0]
#             if isinstance(first_selected, dict) and first_selected.get("flights"):
#                 flight_segments = first_selected["flights"]
        
#         # Build enriched flight object
#         enriched = {
#             "flight_data": flight_segments,
#             "booking_options": booking_options,
#         }
        
#         # Run offer combiner if we have booking options
#         if booking_options and user_bank and card_type:
#             platform_name = booking_options[0].get("together", {}).get("book_with", "Unknown")
#             base_price = booking_options[0].get("together", {}).get("price", 0)
            
#             if base_price:
#                 print(f"🎟️  Flight {idx+1}: Running offer combiner (cached path)")
#                 offer_result = await run_offer_combiner_async(
#                     base_price, user_bank, card_type, platform_name, "Domestic"
#                 )
                
#                 if offer_result and not offer_result.get("error"):
#                     enriched["platform_offers"] = offer_result
        
#         results.append(enriched)
    
#     print(f"✅ [CACHE PATH] Enriched {len(results)} flights")
#     return results


# async def enrich_flights_no_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     NO-CACHE PATH: Enrich flights with booking options and offers using async API calls.
#     """
#     print(f"🌐 [NO-CACHE PATH] Enriching {len(filtered_flights)} flights from API...")
    
#     user_ctx = get_user_context()
#     user_bank = user_ctx.get("user_bank")
#     card_type = user_ctx.get("card_type")
    
#     # Collect all tokens first
#     tokens = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if token:
#             tokens.append(token)
    
#     if not tokens:
#         print("⚠️  No valid tokens found in filtered flights")
#         return []
    
#     print(f"📞 [ASYNC] Fetching booking options for {len(tokens)} flights concurrently...")
    
#     # Fetch all booking options concurrently
#     booking_tasks = [
#         fetch_booking_options_async(token, departure_date, departure_id, arrival_id)
#         for token in tokens
#     ]
#     booking_results = await asyncio.gather(*booking_tasks, return_exceptions=True)
    
#     # Cache booking results
#     cache_items = []
#     for token, booking_data in zip(tokens, booking_results):
#         if booking_data and not isinstance(booking_data, Exception):
#             cache_items.append({
#                 "key": {"booking_token": token},
#                 "value": booking_data
#             })
    
#     if cache_items:
#         batch_save_cache_results(cache_items, verbose=False)
#         print(f"💾 Cached {len(cache_items)} booking results")
    
#     # Build enriched results
#     results = []
#     for idx, (token, booking_data) in enumerate(zip(tokens, booking_results)):
#         if isinstance(booking_data, Exception):
#             print(f"❌ Flight {idx+1}: Booking fetch exception: {booking_data}")
#             continue
        
#         if not booking_data:
#             print(f"⚠️  Flight {idx+1}: No booking data")
#             continue
        
#         selected_flights = booking_data.get("selected_flights", [])
#         booking_options = booking_data.get("booking_options", [])
        
#         flight_segments = []
#         if selected_flights and isinstance(selected_flights, list):
#             first_selected = selected_flights[0]
#             if isinstance(first_selected, dict) and first_selected.get("flights"):
#                 flight_segments = first_selected["flights"]
        
#         # Build enriched flight object
#         enriched = {
#             "flight_data": flight_segments,
#             "booking_options": booking_options,
#         }
        
#         # Run offer combiner if we have booking options
#         if booking_options and user_bank and card_type:
#             platform_name = booking_options[0].get("together", {}).get("book_with", "Unknown")
#             base_price = booking_options[0].get("together", {}).get("price", 0)
            
#             if base_price:
#                 print(f"🎟️  Flight {idx+1}: Running offer combiner (no-cache path)")
#                 offer_result = await run_offer_combiner_async(
#                     base_price, user_bank, card_type, platform_name, "Domestic"
#                 )
                
#                 if offer_result and not offer_result.get("error"):
#                     enriched["platform_offers"] = offer_result
        
#         results.append(enriched)
    
#     print(f"✅ [NO-CACHE PATH] Enriched {len(results)} flights")
#     return results


# @tool
# async def get_flight_with_aggregator(
#     departure_id: str,
#     arrival_id: str,
#     departure_date: str,
#     include_airlines: str | None = None,
#     max_price: str | None = None,
#     passengers: str = "1,0,0,0",
#     outbound_times: str | None = None,
#     travel_class: int = 1
# ) -> list:
#     """
#     Search for flights with filters.
    
#     Args:
#         departure_id: 3-letter airport code (e.g., "DEL")
#         arrival_id: 3-letter airport code (e.g., "BOM")
#         departure_date: Date in YYYY-MM-DD format
#         include_airlines: Comma-separated airline codes or None
#         max_price: Maximum price in INR or None
#         passengers: Format "adults,children,infants_in_seat,infants_on_lap"
#         outbound_times: Already mapped time range (e.g., "0,11" for morning)
#         travel_class: Already mapped class number (1=Economy, 2=Premium, 3=Business, 4=First)
    
#     Returns:
#         List of flight results with booking options
#     """
#     print("🚀 [FLIGHT SEARCH] Starting aggregator (async)")
#     print("[DEBUG] ========================================")
#     print("[DEBUG] PARAMETERS RECEIVED FROM MODEL_WITH_TOOL:")
#     print("[DEBUG] ========================================")
#     print(f"[DEBUG]   departure_id      : {departure_id} (type: {type(departure_id).__name__})")
#     print(f"[DEBUG]   arrival_id        : {arrival_id} (type: {type(arrival_id).__name__})")
#     print(f"[DEBUG]   departure_date    : {departure_date} (type: {type(departure_date).__name__})")
#     print(f"[DEBUG]   include_airlines  : {include_airlines} (type: {type(include_airlines).__name__})")
#     print(f"[DEBUG]   max_price         : {max_price} (type: {type(max_price).__name__})")
#     print(f"[DEBUG]   passengers        : {passengers} (type: {type(passengers).__name__})")
#     print(f"[DEBUG]   outbound_times    : {outbound_times} (type: {type(outbound_times).__name__})")
#     print(f"[DEBUG]   travel_class      : {travel_class} (type: {type(travel_class).__name__})")
#     print("[DEBUG] ========================================\n")

#     outbound_list, is_cached = await asyncio.to_thread(
#         get_flights, 
#         departure_id, 
#         arrival_id, 
#         departure_date, 
#         max_price, 
#         include_airlines,
#         passengers,
#         outbound_times,
#         travel_class
#     )
#     outbound_list = outbound_list or []
    
#     print(f"[DEBUG] ========================================")
#     print(f"[DEBUG] FILTERING PHASE:")
#     print(f"[DEBUG] ========================================")
#     print(f"[DEBUG]   Input flights    : {len(outbound_list)}")
#     print(f"[DEBUG]   max_price filter : {max_price}")
#     print(f"[DEBUG]   arrival_id check : {arrival_id}")
    
#     filtered = [
#         f for f in outbound_list
#         if is_flight_under_budget(f, max_price)
#         and f.get("arrival_airport", {}).get("id") == arrival_id
#     ]
    
#     print(f"[DEBUG]   Output flights   : {len(filtered)}")
#     print(f"[DEBUG] ========================================\n")

#     if not filtered:
#         print("[DEBUG] ⚠️ No flights found within budget.")
#         return []

#     print(f"[DEBUG] Starting enrichment phase...")
#     try:
#         if is_cached:
#             enriched_flights = await enrich_flights_with_cache_async(filtered, departure_date, departure_id, arrival_id)
#         else:
#             enriched_flights = await enrich_flights_no_cache_async(filtered, departure_date, departure_id, arrival_id)
#     except Exception as e:
#         print(f"[ERROR] ❌ Async enrichment failed: {e}")
#         import traceback
#         traceback.print_exc()
#         enriched_flights = []

#     print(f"\n[DEBUG] ========================================")
#     print(f"[DEBUG] FINAL RESULT:")
#     print(f"[DEBUG] ========================================")
#     print(f"[DEBUG]   Enriched flights : {len(enriched_flights)}")
#     print(f"[DEBUG] ========================================\n")
#     return enriched_flights






















































































# # get_flights.py
# import os
# import re
# import time
# import asyncio
# from functools import wraps
# from dotenv import load_dotenv
# from serpapi import GoogleSearch
# from langchain_core.tools import tool
# from utils.mongoDB import get_api_cache_result, save_api_cache_result, batch_save_cache_results

# load_dotenv()

# # ----------------- Constants -----------------
# AIRLINE_MAP = {
#     "air india": "AI",
#     "indigo": "6E",
#     "spicejet": "SG",
#     "goair": "G8",
#     "vistara": "UK",
#     "air asia": "I5",
#     "akasa": "QP",
#     "air india express": "IX",
#     "alliance air": "9I",
#     "star air": "S5",
#     "flybig": "S9",
#     "indiaone air": "I7",
#     "fly91": "IC",
# }

# ANY_TOKENS = {"any", "any airline", "no preference", "no airline", "all airlines"}
# FLIGHT_TYPE = (os.getenv("FLIGHT_TYPE"))

# # ----------------- Helpers -----------------
# def normalize_price(value):
#     if not value:
#         return None
#     cleaned = re.sub(r"[^\d]", "", str(value))
#     return cleaned if cleaned.isdigit() else None


# def map_airlines(user_input):
#     if not user_input:
#         return None
#     raw = str(user_input).lower().strip()
#     if any(tok in raw for tok in ANY_TOKENS):
#         return None
#     airlines = [a.strip().lower() for a in raw.split(",") if a.strip()]
#     codes = []
#     for a in airlines:
#         if len(a) <= 3 and a.isalpha():
#             codes.append(a.upper())
#         elif a in AIRLINE_MAP:
#             codes.append(AIRLINE_MAP[a])
#         else:
#             for k, v in AIRLINE_MAP.items():
#                 if k.replace(" ", "") in a.replace(" ", ""):
#                     codes.append(v)
#                     break
#     return ",".join(set(codes)) if codes else None


# def is_flight_under_budget(flight, max_price):
#     if not max_price:
#         return True
#     cleaned_max_price = normalize_price(max_price)
#     if not cleaned_max_price:
#         return True

#     price_obj = flight.get("price", {})
#     price_amount = price_obj.get("amount") if isinstance(price_obj, dict) else flight.get("price_amount")

#     if price_amount:
#         try:
#             return int(str(price_amount).replace(",", "")) <= int(cleaned_max_price)
#         except Exception:
#             return True
#     return True


# def safe_get_token(flight):
#     """Extract booking token from flight entry."""
#     return flight.get("booking_token")


# # ----------------- Retry Logic -----------------
# def retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         print(f"❌ Function {func.__name__} failed after {max_retries} attempts. Final error: {e}")
#                         raise e
#                     delay = min(initial_delay * (2 ** attempt), max_delay)
#                     print(f"⚠️ Retrying {func.__name__} in {delay:.2f}s (Attempt {attempt + 1}/{max_retries}) due to error: {e}")
#                     time.sleep(delay)
#         return wrapper
#     return decorator


# # ----------------- Core Fetch Logic -----------------
# @retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8)
# def get_flights(departure_id, arrival_id, departure_date, max_price=None, include_airlines=None, travel_class=None):
#     """
#     Fetch flights with intelligent caching.
#     Returns tuple: (flights_data, is_from_cache)
#     """
#     trip_type = FLIGHT_TYPE

#     normalized_airlines = map_airlines(include_airlines)
#     normalized_price = normalize_price(max_price)

#     request_key = {
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "departure_date": departure_date,
#         "trip_type": trip_type,
#         "max_price": normalized_price,
#         "airlines": normalized_airlines,
#         "travel_class": travel_class,
#     }

#     # Check cache first
#     cached_result = get_api_cache_result(request_key)
#     if cached_result is not None:
#         print("✅ [CACHE HIT] Using cached flight data")
#         return cached_result, True  # Return data + cache hit flag

#     print("❌ [CACHE MISS] Fetching fresh flight data from API")

#     params = {
#         "api_key": os.getenv("SERPAPI_API_KEY"),
#         "engine": os.getenv("SEARCH_ENGINE"),
#         "hl": os.getenv("LANGUAGE"),
#         "gl": os.getenv("COUNTRY"),
#         "currency": os.getenv("CURRENCY"),
#         "no_cache": True,
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "outbound_date": departure_date,
#         "type": trip_type,
#         "show_hidden": "true",
#         "deep_search": "true",
#     }

#     mapped = map_airlines(include_airlines)
#     if mapped:
#         params["include_airlines"] = mapped

#     cleaned = normalize_price(max_price)
#     if cleaned:
#         params["max_price"] = cleaned
#       # Add travel class (1=Economy, 2=Premium, 3=Business, 4=First)
#     if travel_class:
#         class_map = {"economy": 1, "premium economy": 2, "premium": 2, "business": 3, "first": 4, "first class": 4}
#         travel_class_lower = str(travel_class).lower().strip()
#         params["travel_class"] = class_map.get(travel_class_lower, 1)  # Default to Economy

#     print("🔎 [API CALL] Fetching flights:", params)
#     search = GoogleSearch(params)
#     results = search.get_dict()

#     best = results.get("best_flights", [])
#     other = results.get("other_flights", [])
#     outbound_flights = best + other
#     print(f"🛫 [API RESPONSE] Found {len(outbound_flights)} flights")

#     # Save to cache
#     if outbound_flights:
#         save_api_cache_result(request_key, outbound_flights, verbose=False)

#     return outbound_flights, False  # Return data + cache miss flag


# # ----------------- Async Booking Fetch -----------------
# async def fetch_booking_options_async(booking_token, departure_date, departure_id, arrival_id):
#     """
#     Async wrapper for fetching booking options.
#     Runs the synchronous GoogleSearch in a thread pool to avoid blocking.
#     """
#     try:
#         params = {
#             "api_key": os.getenv("SERPAPI_API_KEY"),
#             "engine": os.getenv("SEARCH_ENGINE"),
#             "hl": os.getenv("LANGUAGE"),
#             "gl": os.getenv("COUNTRY"),
#             "currency": os.getenv("CURRENCY"),
#             "type": FLIGHT_TYPE,
#             "no_cache": True,
#             "departure_id": departure_id,
#             "arrival_id": arrival_id,
#             "outbound_date": departure_date,
#             "booking_token": booking_token,
#             "show_hidden": "true",
#             "deep_search": "true",
#         }
        
#         # Run in thread pool to avoid blocking the event loop
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None, 
#             lambda: GoogleSearch(params).get_dict()
#         )
#         return result
#     except Exception as e:
#         print(f"❌ Async booking fetch failed for token {booking_token[:10]}: {e}")
#         return None


# async def get_booking_from_cache_async(token):
#     """
#     Async cache lookup for booking options (used when flights were cached).
#     """
#     booking_key = {"booking_token": token}
    
#     # Run cache lookup in thread pool to avoid blocking
#     loop = asyncio.get_event_loop()
#     cached_data = await loop.run_in_executor(
#         None,
#         lambda: get_api_cache_result(booking_key, verbose=False)
#     )
    
#     return cached_data


# async def enrich_flights_with_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights ARE cached, check booking cache for each flight.
#     This maintains the benefit of caching booking options from previous searches.
#     """
#     print(f"📋 [CACHE MODE] Checking booking cache for {len(filtered_flights)} flights...")
    
#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue
        
#         # Check cache first
#         task = get_booking_from_cache_async(token)
#         tasks.append((flight, token, task))
    
#     # Wait for all cache checks
#     cache_results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
#     # Separate cached vs non-cached bookings
#     enriched_flights = []
#     fetch_tasks = []
#     cache_hits = 0
#     cache_misses = 0
    
#     for (flight, token, _), cached_data in zip(tasks, cache_results):
#         if isinstance(cached_data, Exception) or cached_data is None:
#             # Cache miss - need to fetch
#             cache_misses += 1
#             fetch_tasks.append((flight, token, departure_date, departure_id, arrival_id))
#         else:
#             # Cache hit
#             cache_hits += 1
#             selected = cached_data.get("selected_flights", [])
#             booking_opts = cached_data.get("booking_options", [])
#             flight_obj = []
            
#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]
            
#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#             })
    
#     print(f"⏳ Booking cache: {cache_hits} hits, {cache_misses} misses")
    
#     # Fetch missing bookings concurrently
#     if fetch_tasks:
#         print(f"🚀 [API CALLS] Fetching {len(fetch_tasks)} missing booking options...")
#         fetch_results = await asyncio.gather(*[
#             fetch_booking_options_async(token, dep_date, dep_id, arr_id)
#             for _, token, dep_date, dep_id, arr_id in fetch_tasks
#         ], return_exceptions=True)
        
#         # Process fetched results and save to cache
#         cache_batch = []
#         for (flight, token, _, _, _), booking_data in zip(fetch_tasks, fetch_results):
#             if isinstance(booking_data, Exception) or not booking_data:
#                 continue
            
#             selected = booking_data.get("selected_flights", [])
#             booking_opts = booking_data.get("booking_options", [])
#             flight_obj = []
            
#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]
            
#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#             })
            
#             # Prepare for batch cache save
#             booking_key = {"booking_token": token}
#             cache_batch.append((booking_key, booking_data))
        
#         # Batch save all new booking data
#         if cache_batch:
#             batch_save_cache_results(cache_batch)
    
#     return enriched_flights


# async def enrich_flights_no_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights are NOT cached, skip booking cache checks entirely.
#     Fetch all booking options directly and save them in a batch.
#     """
#     print(f"🚀 [NO-CACHE MODE] Fetching {len(filtered_flights)} booking options directly (skipping cache checks)...")
    
#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue
        
#         # Directly fetch without cache check
#         task = fetch_booking_options_async(token, departure_date, departure_id, arrival_id)
#         tasks.append((flight, token, task))
    
#     # Fetch all bookings concurrently
#     results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
#     enriched_flights = []
#     cache_batch = []
#     successful_fetches = 0
    
#     for (flight, token, _), booking_data in zip(tasks, results):
#         if isinstance(booking_data, Exception):
#             print(f"❌ Task failed: {booking_data}")
#             continue
        
#         if not booking_data:
#             continue
        
#         successful_fetches += 1
        
#         selected = booking_data.get("selected_flights", [])
#         booking_opts = booking_data.get("booking_options", [])
#         flight_obj = []
        
#         if selected and isinstance(selected, list):
#             first = selected[0]
#             if isinstance(first, dict) and first.get("flights"):
#                 flight_obj = first["flights"]

#         enriched_flights.append({
#             "flight_data": flight_obj,
#             "booking_options": booking_opts,
#         })
        
#         # Prepare for batch cache save
#         booking_key = {"booking_token": token}
#         cache_batch.append((booking_key, booking_data))
    
#     print(f"✅ [API CALLS] Completed {successful_fetches} booking fetches")
    
#     # Batch save all booking data to cache
#     if cache_batch:
#         batch_save_cache_results(cache_batch)
    
#     return enriched_flights


# # ----------------- Aggregator Tool -----------------
# @tool("get_flight_with_aggregator", description="Fetch and enrich one-way flight data with booking options.")
# def get_flight_with_aggregator(
#     departure_id: str,
#     arrival_id: str,
#     departure_date: str,
#     include_airlines: str | None = None,
#     max_price: str | None = None,
#     travel_class: str | None = None,  # ← ADD THIS
# ):
#     """
#     Main entry point for flight search with intelligent cache-aware booking fetching.
    
#     Strategy:
#     - If flights are cached → check booking cache for each flight
#     - If flights are NOT cached → skip booking cache, fetch all directly
    
#     This eliminates wasteful cache checks when we know bookings won't be cached.
#     """
#     print("🚀 [FLIGHT SEARCH] Starting aggregator")

#     # Get flights and check if they were cached
#     all_flights, is_cached = get_flights(
#         departure_id, arrival_id, departure_date, max_price, include_airlines
#     )
#     outbound_list = all_flights or []

#     # Filter by budget
#     filtered = [f for f in outbound_list if is_flight_under_budget(f, max_price)]
#     print(f"🔎 [FILTER] {len(outbound_list)} flights → {len(filtered)} within budget")

#     if not filtered:
#         print("⚠️ No flights found within budget.")
#         return []

#     # Choose enrichment strategy based on cache status
#     try:
#         if is_cached:
#             # Flights were cached → check booking cache
#             enriched_flights = asyncio.run(
#                 enrich_flights_with_cache_async(filtered, departure_date, departure_id, arrival_id)
#             )
#         else:
#             # Flights were NOT cached → skip booking cache entirely
#             enriched_flights = asyncio.run(
#                 enrich_flights_no_cache_async(filtered, departure_date, departure_id, arrival_id)
#             )
#     except Exception as e:
#         print(f"❌ Async enrichment failed: {e}")
#         enriched_flights = []

#     print(f"✅ [COMPLETE] Returning {len(enriched_flights)} enriched flights")
#     return enriched_flights
































# # get_flights.py without combo | single-cache-system + travel class
# import os
# import re
# import time
# import asyncio
# from functools import wraps
# from dotenv import load_dotenv
# from serpapi import GoogleSearch
# from langchain_core.tools import tool
# from utils.mongoDB import get_api_cache_result, save_api_cache_result, batch_save_cache_results

# load_dotenv()

# # ----------------- Constants -----------------
# AIRLINE_MAP = {
#     "air india": "AI",
#     "indigo": "6E",
#     "spicejet": "SG",
#     "goair": "G8",
#     "vistara": "UK",
#     "air asia": "I5",
#     "akasa": "QP",
#     "air india express": "IX",
#     "alliance air": "9I",
#     "star air": "S5",
#     "flybig": "S9",
#     "indiaone air": "I7",
#     "fly91": "IC",
# }

# ANY_TOKENS = {"any", "any airline", "no preference", "no airline", "all airlines"}
# FLIGHT_TYPE = (os.getenv("FLIGHT_TYPE"))

# # ----------------- Helpers -----------------
# def normalize_price(value):
#     if not value:
#         return None
#     cleaned = re.sub(r"[^\d]", "", str(value))
#     return cleaned if cleaned.isdigit() else None


# def map_airlines(user_input):
#     if not user_input:
#         return None
#     raw = str(user_input).lower().strip()
#     if any(tok in raw for tok in ANY_TOKENS):
#         return None
#     airlines = [a.strip().lower() for a in raw.split(",") if a.strip()]
#     codes = []
#     for a in airlines:
#         if len(a) <= 3 and a.isalpha():
#             codes.append(a.upper())
#         elif a in AIRLINE_MAP:
#             codes.append(AIRLINE_MAP[a])
#         else:
#             for k, v in AIRLINE_MAP.items():
#                 if k.replace(" ", "") in a.replace(" ", ""):
#                     codes.append(v)
#                     break
#     return ",".join(set(codes)) if codes else None


# def is_flight_under_budget(flight, max_price):
#     if not max_price:
#         return True
#     cleaned_max_price = normalize_price(max_price)
#     if not cleaned_max_price:
#         return True

#     price_obj = flight.get("price", {})
#     price_amount = price_obj.get("amount") if isinstance(price_obj, dict) else flight.get("price_amount")

#     if price_amount:
#         try:
#             return int(str(price_amount).replace(",", "")) <= int(cleaned_max_price)
#         except Exception:
#             return True
#     return True


# def safe_get_token(flight):
#     """Extract booking token from flight entry."""
#     return flight.get("booking_token")


# # ----------------- Retry Logic -----------------
# def retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         print(f"❌ Function {func.__name__} failed after {max_retries} attempts. Final error: {e}")
#                         raise e
#                     delay = min(initial_delay * (2 ** attempt), max_delay)
#                     print(f"⚠️ Retrying {func.__name__} in {delay:.2f}s (Attempt {attempt + 1}/{max_retries}) due to error: {e}")
#                     time.sleep(delay)
#         return wrapper
#     return decorator


# # ----------------- Core Fetch Logic -----------------
# @retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8)
# def get_flights(departure_id, arrival_id, departure_date, max_price=None, include_airlines=None, travel_class=None):
#     """
#     Fetch flights with intelligent caching.
#     Returns tuple: (flights_data, is_from_cache)
#     """
#     trip_type = FLIGHT_TYPE

#     normalized_airlines = map_airlines(include_airlines)
#     normalized_price = normalize_price(max_price)

#     request_key = {
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "departure_date": departure_date,
#         "trip_type": trip_type,
#         "max_price": normalized_price,
#         "airlines": normalized_airlines,
#         "travel_class": travel_class,  # ← ADD THIS
#     }

#     # Check cache first
#     cached_result = get_api_cache_result(request_key)
#     if cached_result is not None:
#         print("✅ [CACHE HIT] Using cached flight data")
#         return cached_result, True  # Return data + cache hit flag

#     print("❌ [CACHE MISS] Fetching fresh flight data from API")

#     params = {
#         "api_key": os.getenv("SERPAPI_API_KEY"),
#         "engine": os.getenv("SEARCH_ENGINE"),
#         "hl": os.getenv("LANGUAGE"),
#         "gl": os.getenv("COUNTRY"),
#         "currency": os.getenv("CURRENCY"),
#         "no_cache": True,
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "outbound_date": departure_date,
#         "type": trip_type,
#         "show_hidden": "true",
#         "deep_search": "true",
#     }

#     mapped = map_airlines(include_airlines)
#     if mapped:
#         params["include_airlines"] = mapped

#     cleaned = normalize_price(max_price)
#     if cleaned:
#         params["max_price"] = cleaned
#      # Add travel class (1=Economy, 2=Premium, 3=Business, 4=First)
#     if travel_class:
#         class_map = {"economy": 1, "premium economy": 2, "premium": 2, "business": 3, "first": 4, "first class": 4}
#         travel_class_lower = str(travel_class).lower().strip()
#         params["travel_class"] = class_map.get(travel_class_lower, 1)  # Default to Economy
    
#     print("🔎 [API CALL] Fetching flights:", params)
#     search = GoogleSearch(params)
#     results = search.get_dict()

#     best = results.get("best_flights", [])
#     other = results.get("other_flights", [])
#     outbound_flights = best + other
#     print(f"🛫 [API RESPONSE] Found {len(outbound_flights)} flights")

#     # Save to cache
#     if outbound_flights:
#         save_api_cache_result(request_key, outbound_flights, verbose=False)

#     return outbound_flights, False  # Return data + cache miss flag


# # ----------------- Async Booking Fetch -----------------
# async def fetch_booking_options_async(booking_token, departure_date, departure_id, arrival_id):
#     """
#     Async wrapper for fetching booking options.
#     Runs the synchronous GoogleSearch in a thread pool to avoid blocking.
#     """
#     try:
#         params = {
#             "api_key": os.getenv("SERPAPI_API_KEY"),
#             "engine": os.getenv("SEARCH_ENGINE"),
#             "hl": os.getenv("LANGUAGE"),
#             "gl": os.getenv("COUNTRY"),
#             "currency": os.getenv("CURRENCY"),
#             "type": FLIGHT_TYPE,
#             "no_cache": True,
#             "departure_id": departure_id,
#             "arrival_id": arrival_id,
#             "outbound_date": departure_date,
#             "booking_token": booking_token,
#             "show_hidden": "true",
#             "deep_search": "true",
#         }
        
#         # Run in thread pool to avoid blocking the event loop
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None, 
#             lambda: GoogleSearch(params).get_dict()
#         )
#         return result
#     except Exception as e:
#         print(f"❌ Async booking fetch failed for token {booking_token[:10]}: {e}")
#         return None


# async def get_booking_from_cache_async(token):
#     """
#     Async cache lookup for booking options (used when flights were cached).
#     """
#     booking_key = {"booking_token": token}
    
#     # Run cache lookup in thread pool to avoid blocking
#     loop = asyncio.get_event_loop()
#     cached_data = await loop.run_in_executor(
#         None,
#         lambda: get_api_cache_result(booking_key, verbose=False)
#     )
    
#     return cached_data


# async def enrich_flights_with_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights ARE cached, check booking cache for each flight.
#     This maintains the benefit of caching booking options from previous searches.
#     """
#     print(f"📋 [CACHE MODE] Checking booking cache for {len(filtered_flights)} flights...")
    
#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue
        
#         # Check cache first
#         task = get_booking_from_cache_async(token)
#         tasks.append((flight, token, task))
    
#     # Wait for all cache checks
#     cache_results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
#     # Separate cached vs non-cached bookings
#     enriched_flights = []
#     fetch_tasks = []
#     cache_hits = 0
#     cache_misses = 0
    
#     for (flight, token, _), cached_data in zip(tasks, cache_results):
#         if isinstance(cached_data, Exception) or cached_data is None:
#             # Cache miss - need to fetch
#             cache_misses += 1
#             fetch_tasks.append((flight, token, departure_date, departure_id, arrival_id))
#         else:
#             # Cache hit
#             cache_hits += 1
#             selected = cached_data.get("selected_flights", [])
#             booking_opts = cached_data.get("booking_options", [])
#             flight_obj = []
            
#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]
            
#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#             })
    
#     print(f"⏳ Booking cache: {cache_hits} hits, {cache_misses} misses")
    
#     # Fetch missing bookings concurrently
#     if fetch_tasks:
#         print(f"🚀 [API CALLS] Fetching {len(fetch_tasks)} missing booking options...")
#         fetch_results = await asyncio.gather(*[
#             fetch_booking_options_async(token, dep_date, dep_id, arr_id)
#             for _, token, dep_date, dep_id, arr_id in fetch_tasks
#         ], return_exceptions=True)
        
#         # Process fetched results and save to cache
#         cache_batch = []
#         for (flight, token, _, _, _), booking_data in zip(fetch_tasks, fetch_results):
#             if isinstance(booking_data, Exception) or not booking_data:
#                 continue
            
#             selected = booking_data.get("selected_flights", [])
#             booking_opts = booking_data.get("booking_options", [])
#             flight_obj = []
            
#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]
            
#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#             })
            
#             # Prepare for batch cache save
#             booking_key = {"booking_token": token}
#             cache_batch.append((booking_key, booking_data))
        
#         # Batch save all new booking data
#         if cache_batch:
#             batch_save_cache_results(cache_batch)
    
#     return enriched_flights


# async def enrich_flights_no_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights are NOT cached, skip booking cache checks entirely.
#     Fetch all booking options directly and save them in a batch.
#     """
#     print(f"🚀 [NO-CACHE MODE] Fetching {len(filtered_flights)} booking options directly (skipping cache checks)...")
    
#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue
        
#         # Directly fetch without cache check
#         task = fetch_booking_options_async(token, departure_date, departure_id, arrival_id)
#         tasks.append((flight, token, task))
    
#     # Fetch all bookings concurrently
#     results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
#     enriched_flights = []
#     cache_batch = []
#     successful_fetches = 0
    
#     for (flight, token, _), booking_data in zip(tasks, results):
#         if isinstance(booking_data, Exception):
#             print(f"❌ Task failed: {booking_data}")
#             continue
        
#         if not booking_data:
#             continue
        
#         successful_fetches += 1
        
#         selected = booking_data.get("selected_flights", [])
#         booking_opts = booking_data.get("booking_options", [])
#         flight_obj = []
        
#         if selected and isinstance(selected, list):
#             first = selected[0]
#             if isinstance(first, dict) and first.get("flights"):
#                 flight_obj = first["flights"]

#         enriched_flights.append({
#             "flight_data": flight_obj,
#             "booking_options": booking_opts,
#         })
        
#         # Prepare for batch cache save
#         booking_key = {"booking_token": token}
#         cache_batch.append((booking_key, booking_data))
    
#     print(f"✅ [API CALLS] Completed {successful_fetches} booking fetches")
    
#     # Batch save all booking data to cache
#     if cache_batch:
#         batch_save_cache_results(cache_batch)
    
#     return enriched_flights


# # ----------------- Aggregator Tool -----------------
# @tool("get_flight_with_aggregator", description="Fetch and enrich one-way flight data with booking options.")
# def get_flight_with_aggregator(
#     departure_id: str,
#     arrival_id: str,
#     departure_date: str,
#     include_airlines: str | None = None,
#     max_price: str | None = None,
#     travel_class: str | None = None,  # ← ADD THIS
# ):
#     """
#     Main entry point for flight search with intelligent cache-aware booking fetching.
    
#     Strategy:
#     - If flights are cached → check booking cache for each flight
#     - If flights are NOT cached → skip booking cache, fetch all directly
    
#     This eliminates wasteful cache checks when we know bookings won't be cached.
#     """
#     print("🚀 [FLIGHT SEARCH] Starting aggregator")

#     # Get flights and check if they were cached
#     all_flights, is_cached = get_flights(
#         departure_id, arrival_id, departure_date, max_price, include_airlines, travel_class
#     )
#     outbound_list = all_flights or []

#     # Filter by budget
#     filtered = [f for f in outbound_list if is_flight_under_budget(f, max_price)]
#     print(f"🔎 [FILTER] {len(outbound_list)} flights → {len(filtered)} within budget")

#     if not filtered:
#         print("⚠️ No flights found within budget.")
#         return []

#     # Choose enrichment strategy based on cache status
#     try:
#         if is_cached:
#             # Flights were cached → check booking cache
#             enriched_flights = asyncio.run(
#                 enrich_flights_with_cache_async(filtered, departure_date, departure_id, arrival_id)
#             )
#         else:
#             # Flights were NOT cached → skip booking cache entirely
#             enriched_flights = asyncio.run(
#                 enrich_flights_no_cache_async(filtered, departure_date, departure_id, arrival_id)
#             )
#     except Exception as e:
#         print(f"❌ Async enrichment failed: {e}")
#         enriched_flights = []

#     print(f"✅ [COMPLETE] Returning {len(enriched_flights)} enriched flights")
#     return enriched_flights






















































# # get_flights.py deployed on github
# import os
# import re
# import time
# import asyncio
# from functools import wraps
# from dotenv import load_dotenv
# from serpapi import GoogleSearch
# from langchain_core.tools import tool
# from utils.mongoDB import get_api_cache_result, save_api_cache_result, batch_save_cache_results

# # ----------------- User Context Helper -----------------
# def get_user_context():
#     """
#     Return a dict: {'user_bank': str|None, 'card_type': str|None}.
#     Default implementation reads env vars (for local testing).
#     You should override/monkeypatch this in tests or replace with request-scoped logic.
#     """
#     return {
#         "user_bank": os.getenv("TEST_USER_BANK") or None,
#         "card_type": os.getenv("TEST_CARD_TYPE") or None,
#     }
    
# # ----------------- Combiner Async Wrapper -----------------
# async def run_offer_combiner_async(base_price, user_bank, card_type, platform_name, flight_type):
#     """
#     Run the (synchronous) offer_combiner_tool in a thread pool so it does not block the event loop.
#     """
#     loop = asyncio.get_event_loop()
#     try:
#         res = await loop.run_in_executor(
#             None,
#             lambda: offer_combiner_tool.invoke({
#                 "base_price": float(base_price),
#                 "user_bank": user_bank,
#                 "card_type": card_type,
#                 "platform": platform_name,
#                 "flight_type": flight_type,
#                 "query_text": f"{user_bank or ''} {card_type or ''} flight offers on {platform_name}",
#             }),
#         )
#         return res
#     except Exception as e:
#         return {"error": str(e)}

# load_dotenv()

# # ----------------- Constants -----------------
# AIRLINE_MAP = {
#     "air india": "AI",
#     "indigo": "6E",
#     "spicejet": "SG",
#     "goair": "G8",
#     "vistara": "UK",
#     "air asia": "I5",
#     "akasa": "QP",
#     "air india express": "IX",
#     "alliance air": "9I",
#     "star air": "S5",
#     "flybig": "S9",
#     "indiaone air": "I7",
#     "fly91": "IC",
# }

# ANY_TOKENS = {"any", "any airline", "no preference", "no airline", "all airlines"}
# FLIGHT_TYPE = (os.getenv("FLIGHT_TYPE"))

# # ----------------- Helpers -----------------
# def normalize_price(value):
#     if not value:
#         return None
#     cleaned = re.sub(r"[^\d]", "", str(value))
#     return cleaned if cleaned.isdigit() else None


# def map_airlines(user_input):
#     if not user_input:
#         return None
#     raw = str(user_input).lower().strip()
#     if any(tok in raw for tok in ANY_TOKENS):
#         return None
#     airlines = [a.strip().lower() for a in raw.split(",") if a.strip()]
#     codes = []
#     for a in airlines:
#         if len(a) <= 3 and a.isalpha():
#             codes.append(a.upper())
#         elif a in AIRLINE_MAP:
#             codes.append(AIRLINE_MAP[a])
#         else:
#             for k, v in AIRLINE_MAP.items():
#                 if k.replace(" ", "") in a.replace(" ", ""):
#                     codes.append(v)
#                     break
#     return ",".join(set(codes)) if codes else None


# # def is_flight_under_budget(flight, max_price):
# #     # ensure max_price is numeric for SerpAPI
# #     cleaned = normalize_price(max_price)
# #     if cleaned:
# #         try:
# #             params["max_price"] = int(cleaned)
# #         except Exception:
# #             params["max_price"] = cleaned  # fallback

# #     # ensure 'type' is numeric int (SerpAPI expects 1 or 2)
# #     try:
# #         params["type"] = int(FLIGHT_TYPE) if FLIGHT_TYPE else 2
# #     except Exception:
# #         params["type"] = 2


# #     price_obj = flight.get("price", {})
# #     price_amount = price_obj.get("amount") if isinstance(price_obj, dict) else flight.get("price_amount")

# #     if price_amount:
# #         try:
# #             return int(str(price_amount).replace(",", "")) <= int(cleaned_max_price)
# #         except Exception:
# #             return True
# #     return True
# def is_flight_under_budget(flight, max_price):
#     """
#     Check if a flight is within the given max_price limit.
#     Returns True if under budget or no limit.
#     """
#     if not max_price:
#         return True

#     cleaned_max_price = normalize_price(max_price)
#     if not cleaned_max_price:
#         return True

#     price_obj = flight.get("price", {})
#     price_amount = (
#         price_obj.get("amount")
#         if isinstance(price_obj, dict)
#         else flight.get("price_amount")
#     )

#     if price_amount:
#         try:
#             return int(str(price_amount).replace(",", "")) <= int(cleaned_max_price)
#         except Exception:
#             return True

#     return True



# def safe_get_token(flight):
#     """Extract booking token from flight entry."""
#     return flight.get("booking_token")


# # ----------------- Retry Logic -----------------
# def retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         print(f"❌ Function {func.__name__} failed after {max_retries} attempts. Final error: {e}")
#                         raise e
#                     delay = min(initial_delay * (2 ** attempt), max_delay)
#                     print(f"⚠️ Retrying {func.__name__} in {delay:.2f}s (Attempt {attempt + 1}/{max_retries}) due to error: {e}")
#                     time.sleep(delay)
#         return wrapper
#     return decorator


# # ----------------- Core Fetch Logic -----------------
# @retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8)
# def get_flights(departure_id, arrival_id, departure_date, max_price=None, include_airlines=None):
#     """
#     Fetch flights with intelligent caching.
#     Returns tuple: (flights_data, is_from_cache)
#     """
#     trip_type = FLIGHT_TYPE

#     normalized_airlines = map_airlines(include_airlines)
#     normalized_price = normalize_price(max_price)

#     request_key = {
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "departure_date": departure_date,
#         "trip_type": trip_type,
#         "max_price": normalized_price,
#         "airlines": normalized_airlines,
#     }

#     # Check cache first
#     cached_result = get_api_cache_result(request_key)
#     if cached_result is not None:
#         print("✅ [CACHE HIT] Using cached flight data")
#         return cached_result, True  # Return data + cache hit flag

#     print("❌ [CACHE MISS] Fetching fresh flight data from API")

#     params = {
#         "api_key": os.getenv("SERPAPI_API_KEY"),
#         "engine": os.getenv("SEARCH_ENGINE"),
#         "hl": os.getenv("LANGUAGE"),
#         "gl": os.getenv("COUNTRY"),
#         "currency": os.getenv("CURRENCY"),
#         "no_cache": True,
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "outbound_date": departure_date,
#         "type": trip_type,
#         "show_hidden": "true",
#         # "deep_search": "true",
#     }

#     mapped = map_airlines(include_airlines)
#     if mapped:
#         params["include_airlines"] = mapped

#     cleaned = normalize_price(max_price)
#     if cleaned:
#         params["max_price"] = cleaned

#     print("🔎 [API CALL] Fetching flights:", params)
#     search = GoogleSearch(params)
#     results = search.get_dict()

#     best = results.get("best_flights", [])
#     other = results.get("other_flights", [])
#     outbound_flights = best + other
#     print(f"🛫 [API RESPONSE] Found {len(outbound_flights)} flights")

#     # Save to cache
#     if outbound_flights:
#         save_api_cache_result(request_key, outbound_flights, verbose=False)

#     return outbound_flights, False  # Return data + cache miss flag


# # ----------------- Async Booking Fetch -----------------
# async def fetch_booking_options_async(booking_token, departure_date, departure_id, arrival_id):
#     """
#     Async wrapper for fetching booking options.
#     Runs the synchronous GoogleSearch in a thread pool to avoid blocking.
#     """
#     try:
#         params = {
#             "api_key": os.getenv("SERPAPI_API_KEY"),
#             "engine": os.getenv("SEARCH_ENGINE"),
#             "hl": os.getenv("LANGUAGE"),
#             "gl": os.getenv("COUNTRY"),
#             "currency": os.getenv("CURRENCY"),
#             "type": FLIGHT_TYPE,
#             "no_cache": True,
#             "departure_id": departure_id,
#             "arrival_id": arrival_id,
#             "outbound_date": departure_date,
#             "booking_token": booking_token,
#             "show_hidden": "true",
#             # "deep_search": "true",
#         }
        
#         # Run in thread pool to avoid blocking the event loop
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None, 
#             lambda: GoogleSearch(params).get_dict()
#         )
#         return result
#     except Exception as e:
#         print(f"❌ Async booking fetch failed for token {booking_token[:10]}: {e}")
#         return None


# async def get_booking_from_cache_async(token):
#     """
#     Async cache lookup for booking options (used when flights were cached).
#     """
#     booking_key = {"booking_token": token}
    
#     # Run cache lookup in thread pool to avoid blocking
#     loop = asyncio.get_event_loop()
#     cached_data = await loop.run_in_executor(
#         None,
#         lambda: get_api_cache_result(booking_key, verbose=False)
#     )
    
#     return cached_data


# async def enrich_flights_with_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights ARE cached, check booking cache for each flight.
#     This maintains the benefit of caching booking options from previous searches.
#     """
#     print(f"📋 [CACHE MODE] Checking booking cache for {len(filtered_flights)} flights...")

#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue

#         # Check cache first
#         task = get_booking_from_cache_async(token)
#         tasks.append((flight, token, task))

#     # Wait for all cache checks
#     cache_results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

#     # Separate cached vs non-cached bookings
#     enriched_flights = []
#     fetch_tasks = []
#     cache_hits = 0
#     cache_misses = 0

#     for (flight, token, _), cached_data in zip(tasks, cache_results):
#         if isinstance(cached_data, Exception) or cached_data is None:
#             # Cache miss - need to fetch
#             cache_misses += 1
#             fetch_tasks.append((flight, token, departure_date, departure_id, arrival_id))
#         else:
#             # Cache hit
#             cache_hits += 1
#             selected = cached_data.get("selected_flights", [])
#             booking_opts = cached_data.get("booking_options", [])
#             flight_obj = []

#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]

#             # ---- OFFER ENGINE ----
#             user_ctx = get_user_context()
#             user_bank = user_ctx.get("user_bank")
#             card_type = user_ctx.get("card_type")

#             # extract base_price / platform robustly from booking_opts
#             base_price = None
#             platform_name = None
#             if booking_opts and isinstance(booking_opts, list) and booking_opts:
#                 first_opt = booking_opts[0]
#                 # price may be nested or directly numeric
#                 price_field = first_opt.get("price") if isinstance(first_opt, dict) else None
#                 if isinstance(price_field, dict):
#                     base_price = price_field.get("amount") or price_field.get("price")
#                 else:
#                     base_price = price_field or first_opt.get("price")
#                 # provider keys vary; check common ones
#                 platform_name = (first_opt.get("book_with") or first_opt.get("provider") or first_opt.get("seller") or "").strip().lower()

#             offer_result = {}
#             final_price = base_price if base_price else None

#             if base_price and platform_name:
#                 try:
#                     offer_result = await run_offer_combiner_async(
#                         base_price, user_bank, card_type, platform_name, FLIGHT_TYPE or "domestic"
#                     )
#                     # offer_result may be dict or StructuredTool output; ensure dict
#                     if isinstance(offer_result, dict):
#                         final_price = offer_result.get("final_price", final_price)
#                     else:
#                         # If langchain returns an object, try attribute
#                         final_price = getattr(offer_result, "final_price", final_price)
#                 except Exception as e:
#                     offer_result = {"error": str(e)}
#                     final_price = base_price

#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#                 "offer_engine": offer_result,
#                 "final_price_with_offers": float(final_price) if final_price is not None else None,
#             })

#     print(f"⏳ Booking cache: {cache_hits} hits, {cache_misses} misses")

#     # Fetch missing bookings concurrently
#     if fetch_tasks:
#         print(f"🚀 [API CALLS] Fetching {len(fetch_tasks)} missing booking options...")
#         fetch_results = await asyncio.gather(*[
#             fetch_booking_options_async(token, dep_date, dep_id, arr_id)
#             for _, token, dep_date, dep_id, arr_id in fetch_tasks
#         ], return_exceptions=True)

#         # Process fetched results and save to cache
#         cache_batch = []
#         for (flight, token, _, _, _), booking_data in zip(fetch_tasks, fetch_results):
#             if isinstance(booking_data, Exception) or not booking_data:
#                 continue

#             selected = booking_data.get("selected_flights", [])
#             booking_opts = booking_data.get("booking_options", [])
#             flight_obj = []

#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]

#             # ---- OFFER ENGINE (same extraction as above) ----
#             user_ctx = get_user_context()
#             user_bank = user_ctx.get("user_bank")
#             card_type = user_ctx.get("card_type")

#             base_price = None
#             platform_name = None
#             if booking_opts and isinstance(booking_opts, list) and booking_opts:
#                 first_opt = booking_opts[0]
#                 price_field = first_opt.get("price") if isinstance(first_opt, dict) else None
#                 if isinstance(price_field, dict):
#                     base_price = price_field.get("amount") or price_field.get("price")
#                 else:
#                     base_price = price_field or first_opt.get("price")
#                 platform_name = (first_opt.get("book_with") or first_opt.get("provider") or first_opt.get("seller") or "").strip().lower()

#             offer_result = {}
#             final_price = base_price if base_price else None

#             if base_price and platform_name:
#                 try:
#                     offer_result = await run_offer_combiner_async(
#                         base_price, user_bank, card_type, platform_name, FLIGHT_TYPE or "domestic"
#                     )
#                     if isinstance(offer_result, dict):
#                         final_price = offer_result.get("final_price", final_price)
#                     else:
#                         final_price = getattr(offer_result, "final_price", final_price)
#                 except Exception as e:
#                     offer_result = {"error": str(e)}
#                     final_price = base_price

#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#                 "offer_engine": offer_result,
#                 "final_price_with_offers": float(final_price) if final_price is not None else None,
#             })

#             # Prepare for batch cache save
#             booking_key = {"booking_token": token}
#             cache_batch.append((booking_key, booking_data))

#         # Batch save all new booking data
#         if cache_batch:
#             batch_save_cache_results(cache_batch)

#     # Sort by final effective price (cheapest first). Missing prices pushed to end.
#     enriched_flights.sort(key=lambda x: (x.get("final_price_with_offers") is None, x.get("final_price_with_offers") or float("inf")))
#     return enriched_flights



# async def enrich_flights_no_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights are NOT cached, skip booking cache checks entirely.
#     Fetch all booking options directly and save them in a batch.
#     """
#     print(f"🚀 [NO-CACHE MODE] Fetching {len(filtered_flights)} booking options directly (skipping cache checks)...")

#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue

#         # Directly fetch without cache check
#         task = fetch_booking_options_async(token, departure_date, departure_id, arrival_id)
#         tasks.append((flight, token, task))

#     # Fetch all bookings concurrently
#     results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

#     enriched_flights = []
#     cache_batch = []
#     successful_fetches = 0

#     for (flight, token, _), booking_data in zip(tasks, results):
#         if isinstance(booking_data, Exception):
#             print(f"❌ Task failed: {booking_data}")
#             continue

#         if not booking_data:
#             continue

#         successful_fetches += 1

#         selected = booking_data.get("selected_flights", [])
#         booking_opts = booking_data.get("booking_options", [])
#         flight_obj = []

#         if selected and isinstance(selected, list):
#             first = selected[0]
#             if isinstance(first, dict) and first.get("flights"):
#                 flight_obj = first["flights"]

#         # ---- OFFER ENGINE ----
#         user_ctx = get_user_context()
#         user_bank = user_ctx.get("user_bank")
#         card_type = user_ctx.get("card_type")

#         base_price = None
#         platform_name = None
#         if booking_opts and isinstance(booking_opts, list) and booking_opts:
#             first_opt = booking_opts[0]
#             price_field = first_opt.get("price") if isinstance(first_opt, dict) else None
#             if isinstance(price_field, dict):
#                 base_price = price_field.get("amount") or price_field.get("price")
#             else:
#                 base_price = price_field or first_opt.get("price")
#             platform_name = (first_opt.get("book_with") or first_opt.get("provider") or first_opt.get("seller") or "").strip().lower()

#         offer_result = {}
#         final_price = base_price if base_price else None

#         if base_price and platform_name:
#             try:
#                 offer_result = await run_offer_combiner_async(
#                     base_price, user_bank, card_type, platform_name, FLIGHT_TYPE or "domestic"
#                 )
#                 if isinstance(offer_result, dict):
#                     final_price = offer_result.get("final_price", final_price)
#                 else:
#                     final_price = getattr(offer_result, "final_price", final_price)
#             except Exception as e:
#                 offer_result = {"error": str(e)}
#                 final_price = base_price

#         enriched_flights.append({
#             "flight_data": flight_obj,
#             "booking_options": booking_opts,
#             "offer_engine": offer_result,
#             "final_price_with_offers": float(final_price) if final_price is not None else None,
#         })

#         # Prepare for batch cache save
#         booking_key = {"booking_token": token}
#         cache_batch.append((booking_key, booking_data))

#     print(f"✅ [API CALLS] Completed {successful_fetches} booking fetches")

#     # Batch save all booking data to cache
#     if cache_batch:
#         batch_save_cache_results(cache_batch)

#     # Sort by final effective price (cheapest first). Missing prices pushed to end.
#     enriched_flights.sort(key=lambda x: (x.get("final_price_with_offers") is None, x.get("final_price_with_offers") or float("inf")))
#     return enriched_flights



# # ----------------- Aggregator Tool (async) -----------------
# async def get_flight_with_aggregator(
#     departure_id: str,
#     arrival_id: str,
#     departure_date: str,
#     include_airlines: str | None = None,
#     max_price: str | None = None,
# ):
#     """
#     Async aggregator:
#     - runs blocking get_flights() in a thread via asyncio.to_thread
#     - calls async enrichment helpers (which are already async)
#     """
#     print("🚀 [FLIGHT SEARCH] Starting aggregator (async)")

#     # run blocking get_flights in threadpool
#     outbound_list, is_cached = await asyncio.to_thread(
#         get_flights, departure_id, arrival_id, departure_date, max_price, include_airlines
#     )
#     outbound_list = outbound_list or []

#     # Filter by budget (this is synchronous and cheap)
#     # filtered = [f for f in outbound_list if is_flight_under_budget(f, max_price)]
#     filtered = [
#     f for f in outbound_list
#     if is_flight_under_budget(f, max_price)
#     and f.get("arrival_airport", {}).get("id") == arrival_id
#     ]

#     print(f"🔎 [FILTER] {len(outbound_list)} flights → {len(filtered)} within budget")

#     if not filtered:
#         print("⚠️ No flights found within budget.")
#         return []

#     try:
#         if is_cached:
#             # cached -> check booking cache asynchronously
#             enriched_flights = await enrich_flights_with_cache_async(filtered, departure_date, departure_id, arrival_id)
#         else:
#             # not cached -> fetch bookings directly asynchronously
#             enriched_flights = await enrich_flights_no_cache_async(filtered, departure_date, departure_id, arrival_id)
#     except Exception as e:
#         print(f"❌ Async enrichment failed: {e}")
#         enriched_flights = []

#     print(f"✅ [COMPLETE] Returning {len(enriched_flights)} enriched flights")
#     return enriched_flights
























































































































# get_flights.py without combo | single-cache-system
# import os
# import re
# import time
# import asyncio
# from functools import wraps
# from dotenv import load_dotenv
# from serpapi import GoogleSearch
# from langchain_core.tools import tool
# from utils.mongoDB import get_api_cache_result, save_api_cache_result, batch_save_cache_results

# load_dotenv()

# # ----------------- Constants -----------------
# AIRLINE_MAP = {
#     "air india": "AI",
#     "indigo": "6E",
#     "spicejet": "SG",
#     "goair": "G8",
#     "vistara": "UK",
#     "air asia": "I5",
#     "akasa": "QP",
#     "air india express": "IX",
#     "alliance air": "9I",
#     "star air": "S5",
#     "flybig": "S9",
#     "indiaone air": "I7",
#     "fly91": "IC",
# }

# ANY_TOKENS = {"any", "any airline", "no preference", "no airline", "all airlines"}
# FLIGHT_TYPE = (os.getenv("FLIGHT_TYPE"))

# # ----------------- Helpers -----------------
# def normalize_price(value):
#     if not value:
#         return None
#     cleaned = re.sub(r"[^\d]", "", str(value))
#     return cleaned if cleaned.isdigit() else None


# def map_airlines(user_input):
#     if not user_input:
#         return None
#     raw = str(user_input).lower().strip()
#     if any(tok in raw for tok in ANY_TOKENS):
#         return None
#     airlines = [a.strip().lower() for a in raw.split(",") if a.strip()]
#     codes = []
#     for a in airlines:
#         if len(a) <= 3 and a.isalpha():
#             codes.append(a.upper())
#         elif a in AIRLINE_MAP:
#             codes.append(AIRLINE_MAP[a])
#         else:
#             for k, v in AIRLINE_MAP.items():
#                 if k.replace(" ", "") in a.replace(" ", ""):
#                     codes.append(v)
#                     break
#     return ",".join(set(codes)) if codes else None


# def is_flight_under_budget(flight, max_price):
#     if not max_price:
#         return True
#     cleaned_max_price = normalize_price(max_price)
#     if not cleaned_max_price:
#         return True

#     price_obj = flight.get("price", {})
#     price_amount = price_obj.get("amount") if isinstance(price_obj, dict) else flight.get("price_amount")

#     if price_amount:
#         try:
#             return int(str(price_amount).replace(",", "")) <= int(cleaned_max_price)
#         except Exception:
#             return True
#     return True


# def safe_get_token(flight):
#     """Extract booking token from flight entry."""
#     return flight.get("booking_token")


# # ----------------- Retry Logic -----------------
# def retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return func(*args, **kwargs)
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         print(f"❌ Function {func.__name__} failed after {max_retries} attempts. Final error: {e}")
#                         raise e
#                     delay = min(initial_delay * (2 ** attempt), max_delay)
#                     print(f"⚠️ Retrying {func.__name__} in {delay:.2f}s (Attempt {attempt + 1}/{max_retries}) due to error: {e}")
#                     time.sleep(delay)
#         return wrapper
#     return decorator


# # ----------------- Core Fetch Logic -----------------
# @retry_with_backoff(max_retries=4, initial_delay=1, max_delay=8)
# def get_flights(departure_id, arrival_id, departure_date, max_price=None, include_airlines=None):
#     """
#     Fetch flights with intelligent caching.
#     Returns tuple: (flights_data, is_from_cache)
#     """
#     trip_type = FLIGHT_TYPE

#     normalized_airlines = map_airlines(include_airlines)
#     normalized_price = normalize_price(max_price)

#     request_key = {
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "departure_date": departure_date,
#         "trip_type": trip_type,
#         "max_price": normalized_price,
#         "airlines": normalized_airlines,
#     }

#     # Check cache first
#     cached_result = get_api_cache_result(request_key)
#     if cached_result is not None:
#         print("✅ [CACHE HIT] Using cached flight data")
#         return cached_result, True  # Return data + cache hit flag

#     print("❌ [CACHE MISS] Fetching fresh flight data from API")

#     params = {
#         "api_key": os.getenv("SERPAPI_API_KEY"),
#         "engine": os.getenv("SEARCH_ENGINE"),
#         "hl": os.getenv("LANGUAGE"),
#         "gl": os.getenv("COUNTRY"),
#         "currency": os.getenv("CURRENCY"),
#         "no_cache": True,
#         "departure_id": departure_id,
#         "arrival_id": arrival_id,
#         "outbound_date": departure_date,
#         "type": trip_type,
#         "show_hidden": "true",
#         "deep_search": "true",
#     }

#     mapped = map_airlines(include_airlines)
#     if mapped:
#         params["include_airlines"] = mapped

#     cleaned = normalize_price(max_price)
#     if cleaned:
#         params["max_price"] = cleaned

#     print("🔎 [API CALL] Fetching flights:", params)
#     search = GoogleSearch(params)
#     results = search.get_dict()

#     best = results.get("best_flights", [])
#     other = results.get("other_flights", [])
#     outbound_flights = best + other
#     print(f"🛫 [API RESPONSE] Found {len(outbound_flights)} flights")

#     # Save to cache
#     if outbound_flights:
#         save_api_cache_result(request_key, outbound_flights, verbose=False)

#     return outbound_flights, False  # Return data + cache miss flag


# # ----------------- Async Booking Fetch -----------------
# async def fetch_booking_options_async(booking_token, departure_date, departure_id, arrival_id):
#     """
#     Async wrapper for fetching booking options.
#     Runs the synchronous GoogleSearch in a thread pool to avoid blocking.
#     """
#     try:
#         params = {
#             "api_key": os.getenv("SERPAPI_API_KEY"),
#             "engine": os.getenv("SEARCH_ENGINE"),
#             "hl": os.getenv("LANGUAGE"),
#             "gl": os.getenv("COUNTRY"),
#             "currency": os.getenv("CURRENCY"),
#             "type": FLIGHT_TYPE,
#             "no_cache": True,
#             "departure_id": departure_id,
#             "arrival_id": arrival_id,
#             "outbound_date": departure_date,
#             "booking_token": booking_token,
#             "show_hidden": "true",
#             "deep_search": "true",
#         }
        
#         # Run in thread pool to avoid blocking the event loop
#         loop = asyncio.get_event_loop()
#         result = await loop.run_in_executor(
#             None, 
#             lambda: GoogleSearch(params).get_dict()
#         )
#         return result
#     except Exception as e:
#         print(f"❌ Async booking fetch failed for token {booking_token[:10]}: {e}")
#         return None


# async def get_booking_from_cache_async(token):
#     """
#     Async cache lookup for booking options (used when flights were cached).
#     """
#     booking_key = {"booking_token": token}
    
#     # Run cache lookup in thread pool to avoid blocking
#     loop = asyncio.get_event_loop()
#     cached_data = await loop.run_in_executor(
#         None,
#         lambda: get_api_cache_result(booking_key, verbose=False)
#     )
    
#     return cached_data


# async def enrich_flights_with_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights ARE cached, check booking cache for each flight.
#     This maintains the benefit of caching booking options from previous searches.
#     """
#     print(f"📋 [CACHE MODE] Checking booking cache for {len(filtered_flights)} flights...")
    
#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue
        
#         # Check cache first
#         task = get_booking_from_cache_async(token)
#         tasks.append((flight, token, task))
    
#     # Wait for all cache checks
#     cache_results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
#     # Separate cached vs non-cached bookings
#     enriched_flights = []
#     fetch_tasks = []
#     cache_hits = 0
#     cache_misses = 0
    
#     for (flight, token, _), cached_data in zip(tasks, cache_results):
#         if isinstance(cached_data, Exception) or cached_data is None:
#             # Cache miss - need to fetch
#             cache_misses += 1
#             fetch_tasks.append((flight, token, departure_date, departure_id, arrival_id))
#         else:
#             # Cache hit
#             cache_hits += 1
#             selected = cached_data.get("selected_flights", [])
#             booking_opts = cached_data.get("booking_options", [])
#             flight_obj = []
            
#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]
            
#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#             })
    
#     print(f"⏳ Booking cache: {cache_hits} hits, {cache_misses} misses")
    
#     # Fetch missing bookings concurrently
#     if fetch_tasks:
#         print(f"🚀 [API CALLS] Fetching {len(fetch_tasks)} missing booking options...")
#         fetch_results = await asyncio.gather(*[
#             fetch_booking_options_async(token, dep_date, dep_id, arr_id)
#             for _, token, dep_date, dep_id, arr_id in fetch_tasks
#         ], return_exceptions=True)
        
#         # Process fetched results and save to cache
#         cache_batch = []
#         for (flight, token, _, _, _), booking_data in zip(fetch_tasks, fetch_results):
#             if isinstance(booking_data, Exception) or not booking_data:
#                 continue
            
#             selected = booking_data.get("selected_flights", [])
#             booking_opts = booking_data.get("booking_options", [])
#             flight_obj = []
            
#             if selected and isinstance(selected, list):
#                 first = selected[0]
#                 if isinstance(first, dict) and first.get("flights"):
#                     flight_obj = first["flights"]
            
#             enriched_flights.append({
#                 "flight_data": flight_obj,
#                 "booking_options": booking_opts,
#             })
            
#             # Prepare for batch cache save
#             booking_key = {"booking_token": token}
#             cache_batch.append((booking_key, booking_data))
        
#         # Batch save all new booking data
#         if cache_batch:
#             batch_save_cache_results(cache_batch)
    
#     return enriched_flights


# async def enrich_flights_no_cache_async(filtered_flights, departure_date, departure_id, arrival_id):
#     """
#     When flights are NOT cached, skip booking cache checks entirely.
#     Fetch all booking options directly and save them in a batch.
#     """
#     print(f"🚀 [NO-CACHE MODE] Fetching {len(filtered_flights)} booking options directly (skipping cache checks)...")
    
#     tasks = []
#     for flight in filtered_flights:
#         token = safe_get_token(flight)
#         if not token:
#             continue
        
#         # Directly fetch without cache check
#         task = fetch_booking_options_async(token, departure_date, departure_id, arrival_id)
#         tasks.append((flight, token, task))
    
#     # Fetch all bookings concurrently
#     results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)
    
#     enriched_flights = []
#     cache_batch = []
#     successful_fetches = 0
    
#     for (flight, token, _), booking_data in zip(tasks, results):
#         if isinstance(booking_data, Exception):
#             print(f"❌ Task failed: {booking_data}")
#             continue
        
#         if not booking_data:
#             continue
        
#         successful_fetches += 1
        
#         selected = booking_data.get("selected_flights", [])
#         booking_opts = booking_data.get("booking_options", [])
#         flight_obj = []
        
#         if selected and isinstance(selected, list):
#             first = selected[0]
#             if isinstance(first, dict) and first.get("flights"):
#                 flight_obj = first["flights"]

#         enriched_flights.append({
#             "flight_data": flight_obj,
#             "booking_options": booking_opts,
#         })
        
#         # Prepare for batch cache save
#         booking_key = {"booking_token": token}
#         cache_batch.append((booking_key, booking_data))
    
#     print(f"✅ [API CALLS] Completed {successful_fetches} booking fetches")
    
#     # Batch save all booking data to cache
#     if cache_batch:
#         batch_save_cache_results(cache_batch)
    
#     return enriched_flights


# # ----------------- Aggregator Tool -----------------
# @tool("get_flight_with_aggregator", description="Fetch and enrich one-way flight data with booking options.")
# def get_flight_with_aggregator(
#     departure_id: str,
#     arrival_id: str,
#     departure_date: str,
#     include_airlines: str | None = None,
#     max_price: str | None = None,
# ):
#     """
#     Main entry point for flight search with intelligent cache-aware booking fetching.
    
#     Strategy:
#     - If flights are cached → check booking cache for each flight
#     - If flights are NOT cached → skip booking cache, fetch all directly
    
#     This eliminates wasteful cache checks when we know bookings won't be cached.
#     """
#     print("🚀 [FLIGHT SEARCH] Starting aggregator")

#     # Get flights and check if they were cached
#     all_flights, is_cached = get_flights(
#         departure_id, arrival_id, departure_date, max_price, include_airlines
#     )
#     outbound_list = all_flights or []

#     # Filter by budget
#     filtered = [f for f in outbound_list if is_flight_under_budget(f, max_price)]
#     print(f"🔎 [FILTER] {len(outbound_list)} flights → {len(filtered)} within budget")

#     if not filtered:
#         print("⚠️ No flights found within budget.")
#         return []

#     # Choose enrichment strategy based on cache status
#     try:
#         if is_cached:
#             # Flights were cached → check booking cache
#             enriched_flights = asyncio.run(
#                 enrich_flights_with_cache_async(filtered, departure_date, departure_id, arrival_id)
#             )
#         else:
#             # Flights were NOT cached → skip booking cache entirely
#             enriched_flights = asyncio.run(
#                 enrich_flights_no_cache_async(filtered, departure_date, departure_id, arrival_id)
#             )
#     except Exception as e:
#         print(f"❌ Async enrichment failed: {e}")
#         enriched_flights = []

#     print(f"✅ [COMPLETE] Returning {len(enriched_flights)} enriched flights")
#     return enriched_flights