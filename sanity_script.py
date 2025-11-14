from serpapi import GoogleSearch
import json
import os
params = {
    "api_key": "52d763ced35a8fceaa3f4684b005c7805ffb8523fe60cdcc4be24b8f0002234a",
    "engine": "google_flights",
    "hl": "en",
    "gl": "in",
    "currency": "INR",
    "departure_id": "DEL",
    "arrival_id": "BOM",
    "outbound_date": "2025-11-28",
    "show_hidden": "true",
    # "deep_search": "true",
    "include_airlines": "AI",
    "max_price": 8000,
    "adults": 1,
    "outbound_times": "0,11",
    "travel_class": 1,
    "type": 2,
    "no_cache": "true"
}
print("🔧 Request parameters:\n", json.dumps(params, indent=2))

try:
    res = GoogleSearch(params).get_dict()
except Exception as e:
    print("❌ Connection or API failure:", e)
    raise SystemExit(1)

# --- Basic response diagnostics ---
print("\n🔎 Response keys:", list(res.keys()))

if "error" in res:
    err = res["error"].lower()
    print(f"❌ SerpAPI returned an error: {res['error']}")
    if "invalid" in err or "unauthorized" in err:
        print("⚠️  The API key is invalid or misconfigured.")
    elif "exceeded" in err or "quota" in err or "limit" in err:
        print("🚫  The API key quota is exhausted or rate-limited.")
    else:
        print("⚠️  General SerpAPI error, check response payload below:")
        print(json.dumps(res, indent=2))
else:
    best = res.get("best_flights", [])
    other = res.get("other_flights", [])
    print(f"✅ API key appears valid. Flights returned:")
    print(f"   • Best flights:  {len(best)}")
    print(f"   • Other flights: {len(other)}")

# --- Optional: check remaining credits if metadata is present ---
meta = res.get("search_metadata", {})
if meta:
    print("\n📊 Search metadata:")
    print(json.dumps(meta, indent=2))