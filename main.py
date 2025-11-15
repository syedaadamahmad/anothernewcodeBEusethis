# cc main.py csv changed to json

from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import json
from dotenv import load_dotenv
from utils import model_with_tool, mongoDB, flights_loader
from utils import rag_platform_combo_retriever

load_dotenv()

app = FastAPI()

# CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup
@app.on_event("startup")
async def startup_event():
    """Preload flights on startup."""
    count = flights_loader.preload_flights()
    print(f"✅ [STARTUP] Flight database ready ({count} flights)")

# Request Models
class ChatRequest(BaseModel):
    chat_history: List[dict]
    flight_context: Optional[dict] = None  # ✅ ADDED for nested chat

class PlatformComboRequest(BaseModel):
    platform: str = Field(..., description="Booking platform")
    base_price: float = Field(..., description="Flight base price")
    bank: Optional[str] = Field(None, description="User's bank")
    card_type: Optional[str] = Field(None, description="credit or debit")

class GetOffersRequest(BaseModel):
    platform: str
    offer_type: str = Field(..., description="'go' or 'po'")
    bank: Optional[str] = None
    card_type: Optional[str] = None
    k: int = Field(5, description="Number of results")

JSON_FILE_PATH = os.getenv(
    "UPDATED_DEALS_JSON",
    "enhanced_combos.json"
)

# ========================================
# EXISTING ENDPOINTS (UNCHANGED)
# ========================================

@app.get("/")
def home():
    return {"message": "SmartBhai Backend - Local Flight Database", "status": "online"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """Chat endpoint using model_with_tool.rag_agent."""
    result = model_with_tool.rag_agent(request.chat_history)
    return JSONResponse(content=result)

@app.get("/get_latest_deals")
def get_latest_deals():
    """Get deals from JSON or MongoDB fallback."""
    EXPECTED_COLUMNS = [
        "platform", "title", "offer", "coupon_code", "bank", "payment_mode",
        "url", "expiry_date", "flight_type", "offer_type"
    ]
    
    # Try JSON first
    try:
        if os.path.exists(JSON_FILE_PATH):
            with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle both array and object with "deals" key
            deals = data if isinstance(data, list) else data.get("deals", [])
            
            # Normalize each deal
            normalized_deals = []
            for deal in deals:
                normalized = {
                    k: (deal.get(k, "") if deal.get(k, None) is not None else "")
                    for k in EXPECTED_COLUMNS
                }
                normalized_deals.append(normalized)
            
            return JSONResponse(content={"deals": normalized_deals})
    except Exception as e:
        print(f"[get_latest_deals] JSON error: {e}")
    
    # Fallback to MongoDB
    try:
        client = mongoDB.connect_db()
        coll = mongoDB.get_collection(client, "flight_coupons")
        if coll is None:
            return JSONResponse(
                content={"deals": [], "error": "No data source"},
                status_code=500
            )
        docs = list(coll.find({}, {"_id": 0, "embedding": 0}))
        normalized_docs = []
        for d in docs:
            normalized = {k: d.get(k, "") for k in EXPECTED_COLUMNS}
            normalized_docs.append(normalized)
        return JSONResponse(content={"deals": normalized_docs})
    except Exception as e:
        print(f"[get_latest_deals] Mongo error: {e}")
        return JSONResponse(
            content={"deals": [], "error": str(e)},
            status_code=500
        )

@app.get("/health")
def health_check():
    """Health check with flight database info."""
    try:
        flights = flights_loader.load_flights_data()
        return {
            "status": "healthy",
            "flights_loaded": len(flights),
            "database": "local_json"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

# ========================================
# COMBO ENDPOINTS (EXISTING)
# ========================================

@app.post("/api/combo")
def get_combo_endpoint(request: PlatformComboRequest):
    """
    Orchestrate platform-specific combo (GO + PO).
    Returns best combo with sequential discount calculation.
    """
    try:
        # Normalize platform casing
        platform = request.platform.lower().strip()

        if request.base_price <= 0:
            return JSONResponse(
                content={"error": "base_price must be positive"},
                status_code=400
            )
        
        card_type = None
        if request.card_type:
            card_type = request.card_type.lower().strip()
            if card_type not in ["credit", "debit"]:
                return JSONResponse(
                    content={"error": "card_type must be 'credit' or 'debit'"},
                    status_code=400
                )
        
        combo = rag_platform_combo_retriever.build_platform_combo(
            platform=request.platform,
            base_price=request.base_price,
            bank=request.bank,
            card_type=card_type
        )
        
        if combo.get("error"):
            return JSONResponse(content={
                "platform": request.platform,
                "base_price": request.base_price,
                "message": combo["error"],
                "offers_available": False
            })
        
        return JSONResponse(content={
            **combo,
            "offers_available": True,
            "message": f"Found {len(combo['offers_used'])} offers"
        })
    
    except Exception as e:
        print(f"❌ [COMBO_ENDPOINT] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.post("/api/offers")
def get_offers_endpoint(request: GetOffersRequest):
    """Get individual offers (GO or PO) for chat."""
    try:
        if request.offer_type == "go":
            offers = rag_platform_combo_retriever.get_platform_general_offers(
                platform=request.platform,
                k=request.k
            )
        elif request.offer_type == "po":
            if not request.bank or not request.card_type:
                return JSONResponse(
                    content={"error": "bank and card_type required"},
                    status_code=400
                )
            offers = rag_platform_combo_retriever.get_platform_payment_offers(
                platform=request.platform,
                bank=request.bank,
                card_type=request.card_type.lower(),
                k=request.k
            )
        else:
            return JSONResponse(
                content={"error": "offer_type must be 'go' or 'po'"},
                status_code=400
            )
        
        return JSONResponse(content={
            "offers": offers,
            "count": len(offers),
            "offer_type": request.offer_type,
            "platform": request.platform
        })
    
    except Exception as e:
        print(f"❌ [OFFERS_ENDPOINT] Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

# ========================================
# ✅ NEW: NESTED CHAT ENDPOINT
# ========================================

@app.post("/chat/flight")
def flight_nested_chat(request: ChatRequest):
    """
    Nested chat for flight card offers.
    Triggers combo_tool in model_with_tool.rag_agent.
    """
    try:
        chat_history = request.chat_history
        flight_context = request.flight_context
        
        # Inject flight context as system message
        if flight_context:
            platform = flight_context.get('platform', 'Unknown')
            base_price = flight_context.get('base_price', 0)
            
            context_msg = {
                "role": "system",
                "content": f"NESTED CHAT: User viewing {platform} at ₹{base_price}. Guide offer discovery using combo_tool."
            }
            chat_history.insert(0, context_msg)
        
        # Call rag_agent (triggers combo_tool)
        result = model_with_tool.rag_agent(chat_history)
        return JSONResponse(content={"content": result["content"]})
        
    except Exception as e:
        print(f"❌ [NESTED_CHAT_ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))





















































# # main.py with all the filters but returns a blank array
# import os
# import csv
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from utils import model_with_tool, mongoDB
# from utils.get_flights import get_flight_with_aggregator
# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any
# import traceback

# app = FastAPI()

# # ✅ CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Ensure MongoDB connection pooling on startup
# @app.on_event("startup")
# async def startup_event():
#     """Initialize MongoDB connection pool on startup."""
#     mongoDB.get_mongo_client()
#     print("✅ MongoDB connection pool initialized")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Close MongoDB connections on shutdown."""
#     mongoDB.close_mongo_connection()
#     print("✅ MongoDB connection pool closed")

# # -------------------------------
# # Request body models
# # -------------------------------
# class ChatMessage(BaseModel):
#     role: str  # "human" or "ai"
#     content: str

# class ChatContext(BaseModel):
#     """Optional context for nested chat flows."""
#     nested_chat: Optional[bool] = False
#     booking_platform: Optional[str] = None
#     base_price: Optional[float] = None
#     flight_type: Optional[str] = "domestic"

# class ChatRequest(BaseModel):
#     chat_history: List[Dict[str, Any]]  # Changed from List[dict] for better typing
#     context: Optional[ChatContext] = None  # NEW: Context for nested offers

# # -------------------------------
# # Routes
# # -------------------------------

# @app.get("/")
# def home():
#     return {
#         "message": "SmartBhai Backend API",
#         "version": "2.0",
#         "endpoints": [
#             "/chat - Main chat endpoint",
#             "/get_latest_deals - Fetch all deals",
#             "/flights/search - Direct flight search",
#             "/health - Health check"
#         ]
#     }

# @app.get("/health")
# def health_check():
#     """Health check endpoint for monitoring."""
#     try:
#         # Test MongoDB connection
#         client = mongoDB.get_mongo_client()
#         if client is not None:
#             client.admin.command('ping')
#             db_status = "connected"
#         else:
#             db_status = "disconnected"
        
#         return {
#             "status": "healthy",
#             "database": db_status
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     """
#     Unified chat endpoint handling:
#     1️⃣ Main flight search chat (SerpAPI queries)
#     2️⃣ Nested offer chat (offer orchestration)
    
#     The context field determines which flow to use:
#     - context.nested_chat=True → Offer orchestration mode
#     - context.nested_chat=False/None → Flight search mode
#     """
#     try:
#         # Extract context if provided
#         context = req.context or ChatContext()
        
#         # Pass context to rag_agent (✅ FIXED: Added await)
#         result = await model_with_tool.rag_agent(
#             chat_history=req.chat_history,
#             nested_chat=context.nested_chat,
#             platform=context.booking_platform,
#             base_price=context.base_price,
#             flight_type=context.flight_type,
#         )

#         return result

#     except Exception as e:
#         print(f"❌ [chat] Error: {e}")
#         traceback.print_exc()
#         return {
#             "content": "⚠️ Something went wrong. Please try again.",
#             "error": str(e),
#             "flight_data": None
#         }

# @app.get("/get_latest_deals")
# def get_latest_deals():
#     """
#     Fetch all deals from MongoDB (or CSV fallback).
#     Returns combined data from all three collections.
#     """
#     EXPECTED_COLUMNS = [
#         "platform", "title", "offer", "coupon_code", "bank",
#         "payment_mode", "url", "expiry_date", "terms_and_conditions", "flight_type"
#     ]
    
#     # CSV fallback (if needed)
#     CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "./data/deals.csv")
    
#     try:
#         if os.path.exists(CSV_FILE_PATH):
#             print("📁 [get_latest_deals] Using CSV fallback")
#             deals = []
#             with open(CSV_FILE_PATH, newline="", encoding="utf-8") as csvfile:
#                 reader = csv.DictReader(csvfile)
#                 for row in reader:
#                     normalized = {k: (row.get(k, "") or "") for k in EXPECTED_COLUMNS}
#                     deals.append(normalized)
#             return JSONResponse(content={"deals": deals, "source": "csv"})
#     except Exception as e:
#         print(f"⚠️ [get_latest_deals] CSV read error: {e}")

#     # MongoDB retrieval
#     try:
#         print("💾 [get_latest_deals] Fetching from MongoDB")
#         client = mongoDB.get_mongo_client()
        
#         all_deals = []
        
#         # Fetch from all three collections
#         for collection_name in ["gift_coupons", "payment_offers", "general_offers"]:
#             coll = mongoDB.get_collection(client, collection_name)
#             if coll:
#                 docs = list(coll.find({}, {"_id": 0, "embedding": 0}))
#                 # Add collection tag
#                 for doc in docs:
#                     doc["collection"] = collection_name
#                 all_deals.extend(docs)
        
#         if not all_deals:
#             raise ValueError("No deals found in any collection")
        
#         # Normalize fields
#         normalized_docs = []
#         for d in all_deals:
#             normalized = {k: d.get(k, "") for k in EXPECTED_COLUMNS}
#             normalized["collection"] = d.get("collection", "unknown")
#             normalized_docs.append(normalized)
        
#         print(f"✅ [get_latest_deals] Retrieved {len(normalized_docs)} total deals")
#         return JSONResponse(content={
#             "deals": normalized_docs,
#             "source": "mongodb",
#             "count": len(normalized_docs)
#         })
        
#     except Exception as e:
#         print(f"❌ [get_latest_deals] MongoDB error: {e}")
#         traceback.print_exc()
#         return JSONResponse(
#             content={"deals": [], "error": str(e), "source": "error"},
#             status_code=500
#         )

# @app.post("/flights/search")
# async def flight_search(req: Request):
#     """
#     Direct flight search endpoint (bypasses chat interface).
    
#     Example JSON body:
#     {
#         "departure_id": "DEL",
#         "arrival_id": "BOM",
#         "departure_date": "2025-11-28",
#         "include_airlines": "AI,6E",
#         "max_price": "10000",
#         "passengers": "2,1,0,0",
#         "outbound_times": "afternoon",
#         "travel_class": "Business"
#     }
#     """
#     try:
#         data = await req.json()
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

#     # Validate required fields
#     required_fields = ["departure_id", "arrival_id", "departure_date"]
#     for field in required_fields:
#         if not data.get(field):
#             raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

#     try:
#         print(f"✈️ [flight_search] Searching flights: {data.get('departure_id')} → {data.get('arrival_id')}")
        
#         # ✅ FIXED: Use ainvoke instead of invoke + added new parameters
#         result = await get_flight_with_aggregator.ainvoke({
#             "departure_id": data["departure_id"],
#             "arrival_id": data["arrival_id"],
#             "departure_date": data["departure_date"],
#             "include_airlines": data.get("include_airlines"),
#             "max_price": data.get("max_price"),
#             "passengers": data.get("passengers", "1,0,0,0"),        # NEW
#             "outbound_times": data.get("outbound_times"),            # NEW
#             "travel_class": data.get("travel_class", "Economy")      # NEW
#         })
        
#         return JSONResponse(content={
#             "results": result,
#             "count": len(result) if result else 0
#         })
        
#     except Exception as e:
#         print(f"❌ [flight_search] Error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/offers/search")
# async def offer_search(req: Request):
#     """
#     Direct offer search endpoint (bypasses chat).
    
#     Example JSON body:
#     {
#         "offer_type": "payment",
#         "bank": "HDFC",
#         "card_type": "Credit Card",
#         "query": "flight cashback"
#     }
#     """
#     try:
#         data = await req.json()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid JSON body")
    
#     offer_type = data.get("offer_type", "general")
    
#     try:
#         from utils.offer_orchestrator_tool import offer_orchestrator_tool
        
#         result = offer_orchestrator_tool.invoke({
#             "query": data.get("query", "flight offers"),
#             "offer_type": offer_type,
#             "bank": data.get("bank"),
#             "card_type": data.get("card_type"),
#             "base_price": data.get("base_price"),
#             "build_combo": data.get("build_combo", False)
#         })
        
#         return JSONResponse(content={"result": result})
        
#     except Exception as e:
#         print(f"❌ [offer_search] Error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# # Optional: Prometheus metrics endpoint
# try:
#     from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
#     from fastapi.responses import Response
    
#     @app.get("/metrics")
#     def metrics():
#         """Expose Prometheus metrics."""
#         return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
# except ImportError:
#     print("⚠️ prometheus_client not installed, /metrics endpoint disabled")




















































# # main.py cheeat code
# from typing import List
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import os
# import csv
# from dotenv import load_dotenv
# from utils import model_with_tool, mongoDB
# from utils import flights_loader  # ← NEW: Import flight loader

# load_dotenv()

# app = FastAPI()

# origins = ["*"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ PRELOAD FLIGHTS ON STARTUP
# @app.on_event("startup")
# async def startup_event():
#     """Preload flights into memory on application startup."""
#     count = flights_loader.preload_flights()
#     print(f"✅ [STARTUP] Flight database ready ({count} flights)")

# # Request body model
# class ChatRequest(BaseModel):
#     chat_history: List[dict]

# CSV_FILE_PATH = os.getenv(
#     "UPDATED_DEALS_CSV",
#     r"D:\New folder\backend_price_airlines\ChatSB-Backend\master_offers.csv"
# )

# @app.get("/")
# def home():
#     return {"message": "SmartBhai Backend - Local Flight Database", "status": "online"}

# @app.post("/chat")
# def chat_endpoint(request: ChatRequest):
#     """
#     Chat endpoint that uses model_with_tool.rag_agent.
#     Now uses local JSON flight database instead of SerpAPI.
#     """
#     result = model_with_tool.rag_agent(request.chat_history)
#     return JSONResponse(content=result)

# @app.get("/get_latest_deals")
# def get_latest_deals():
#     """
#     Primary endpoint for the frontend Book Now flow.
#     - Tries to read CSV_FILE_PATH and return as JSON {"deals": [...]}
#     - If CSV missing or unreadable, falls back to MongoDB collection "flight_coupons"
#     """
#     EXPECTED_COLUMNS = [
#         "platform","title","offer","coupon_code","bank","payment_mode",
#         "url","expiry_date","terms_and_conditions","flight_type"
#     ]
    
#     # 1) Try CSV first
#     try:
#         if os.path.exists(CSV_FILE_PATH):
#             deals = []
#             with open(CSV_FILE_PATH, newline="", encoding="utf-8") as csvfile:
#                 reader = csv.DictReader(csvfile)
#                 for row in reader:
#                     normalized = {
#                         k: (row.get(k, "") if row.get(k, None) is not None else "")
#                         for k in EXPECTED_COLUMNS
#                     }
#                     deals.append(normalized)
#             return JSONResponse(content={"deals": deals})
#     except Exception as e:
#         print(f"[get_latest_deals] Error reading CSV ({CSV_FILE_PATH}): {e}")
    
#     # 2) Fallback to MongoDB
#     try:
#         client = mongoDB.connect_db()
#         coll = mongoDB.get_collection(client, "flight_coupons")
#         if coll is None:
#             return JSONResponse(
#                 content={"deals": [], "error": "No data source available"},
#                 status_code=500
#             )
#         docs = list(coll.find({}, {"_id": 0, "embedding": 0}))
#         normalized_docs = []
#         for d in docs:
#             normalized = {}
#             for k in EXPECTED_COLUMNS:
#                 normalized[k] = d.get(k, "")
#             normalized_docs.append(normalized)
#         return JSONResponse(content={"deals": normalized_docs})
#     except Exception as e:
#         print(f"[get_latest_deals] Mongo fallback error: {e}")
#         return JSONResponse(
#             content={"deals": [], "error": str(e)},
#             status_code=500
#         )

# # ✅ NEW: Health check endpoint shows flight database status
# @app.get("/health")
# def health_check():
#     """Health check with flight database info."""
#     try:
#         flights = flights_loader.load_flights_data()
#         return {
#             "status": "healthy",
#             "flights_loaded": len(flights),
#             "database": "local_json"
#         }
#     except Exception as e:
#         return {
#             "status": "degraded",
#             "error": str(e)
#         }


































# # main.py
# from typing import List
# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import os
# import csv
# from dotenv import load_dotenv
# from utils import model_with_tool, mongoDB

# load_dotenv()

# app = FastAPI()

# origins = ["*"]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Request body model
# class ChatRequest(BaseModel):
#     chat_history: List[dict]  # [{"role": "human", "content": "..."}, {"role": "ai", "content": "..."}]

# CSV_FILE_PATH = os.getenv(
#     "UPDATED_DEALS_CSV",
#     r"D:\New folder\backend_price_airlines\ChatSB-Backend\master_offers.csv"
# )

# @app.get("/")
# def home():
#     return {"message": "its working fine :)"}


# @app.post("/chat")
# def chat_endpoint(request: ChatRequest):
#     """
#     Chat endpoint that uses model_with_tool.rag_agent.
#     Returns both the assistant's message and any structured flight data.
#     """
#     result = model_with_tool.rag_agent(request.chat_history)
#     # result is already a dict: {"content": "...", "flight_data": [...]}
#     return JSONResponse(content=result)


# @app.get("/get_latest_deals")
# def get_latest_deals():
#     """
#     Primary endpoint for the frontend Book Now flow.
#     - Tries to read CSV_FILE_PATH and return as JSON {"deals": [...]}
#     - If CSV missing or unreadable, falls back to MongoDB collection "flight_coupons"
#     - Ensures returned JSON keys match EXPECTED_COLUMNS used throughout the project
#     """
#     EXPECTED_COLUMNS = [
#         "platform","title","offer","coupon_code","bank","payment_mode","url","expiry_date","terms_and_conditions","flight_type"
#         # "platform", "title", "offer", "coupon_code", "bank",
#         # "payment_mode", "emi", "url", "expiry_date",
#         # "current/upcoming", "flight_type"
#     ]

#     # 1) Try CSV first
#     try:
#         if os.path.exists(CSV_FILE_PATH):
#             deals = []
#             with open(CSV_FILE_PATH, newline="", encoding="utf-8") as csvfile:
#                 reader = csv.DictReader(csvfile)
#                 for row in reader:
#                     # normalize to expected columns: if missing columns exist, insert empty string
#                     normalized = {
#                         k: (row.get(k, "") if row.get(k, None) is not None else "")
#                         for k in EXPECTED_COLUMNS
#                     }
#                     deals.append(normalized)
#             return JSONResponse(content={"deals": deals})
#     except Exception as e:
#         # Log but continue to fallback to MongoDB
#         print(f"[get_latest_deals] Error reading CSV ({CSV_FILE_PATH}): {e}")

#     # 2) Fallback to MongoDB
#     try:
#         client = mongoDB.connect_db()
#         coll = mongoDB.get_collection(client, "flight_coupons")
#         if coll is None:
#             return JSONResponse(
#                 content={"deals": [], "error": "No data source available"},
#                 status_code=500
#             )

#         docs = list(coll.find({}, {"_id": 0, "embedding": 0}))
#         # Ensure keys match EXPECTED_COLUMNS: convert keys if necessary or fill missing
#         normalized_docs = []
#         for d in docs:
#             normalized = {}
#             for k in EXPECTED_COLUMNS:
#                 normalized[k] = d.get(k, "")
#             normalized_docs.append(normalized)

#         return JSONResponse(content={"deals": normalized_docs})
#     except Exception as e:
#         print(f"[get_latest_deals] Mongo fallback error: {e}")
#         return JSONResponse(
#             content={"deals": [], "error": str(e)},
#             status_code=500
#         )                  











































# # main.py + travel but shows onl air india
# import os
# import csv
# from fastapi import FastAPI, Request, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from utils import model_with_tool, mongoDB
# from utils.get_flights import get_flight_with_aggregator
# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any
# import traceback

# app = FastAPI()

# # ✅ CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Ensure MongoDB connection pooling on startup
# @app.on_event("startup")
# async def startup_event():
#     """Initialize MongoDB connection pool on startup."""
#     mongoDB.get_mongo_client()
#     print("✅ MongoDB connection pool initialized")

# @app.on_event("shutdown")
# async def shutdown_event():
#     """Close MongoDB connections on shutdown."""
#     mongoDB.close_mongo_connection()
#     print("✅ MongoDB connection pool closed")

# # -------------------------------
# # Request body models
# # -------------------------------
# class ChatMessage(BaseModel):
#     role: str  # "human" or "ai"
#     content: str

# class ChatContext(BaseModel):
#     """Optional context for nested chat flows."""
#     nested_chat: Optional[bool] = False
#     booking_platform: Optional[str] = None
#     base_price: Optional[float] = None
#     flight_type: Optional[str] = "domestic"

# class ChatRequest(BaseModel):
#     chat_history: List[Dict[str, Any]]  # Changed from List[dict] for better typing
#     context: Optional[ChatContext] = None  # NEW: Context for nested offers

# # -------------------------------
# # Routes
# # -------------------------------

# @app.get("/")
# def home():
#     return {
#         "message": "SmartBhai Backend API",
#         "version": "2.0",
#         "endpoints": [
#             "/chat - Main chat endpoint",
#             "/get_latest_deals - Fetch all deals",
#             "/flights/search - Direct flight search",
#             "/health - Health check"
#         ]
#     }

# @app.get("/health")
# def health_check():
#     """Health check endpoint for monitoring."""
#     try:
#         # Test MongoDB connection
#         client = mongoDB.get_mongo_client()
#         if client is not None:
#             client.admin.command('ping')
#             db_status = "connected"
#         else:
#             db_status = "disconnected"
        
#         return {
#             "status": "healthy",
#             "database": db_status
#         }
#     except Exception as e:
#         return {
#             "status": "unhealthy",
#             "error": str(e)
#         }

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     """
#     Unified chat endpoint handling:
#     1️⃣ Main flight search chat (SerpAPI queries)
#     2️⃣ Nested offer chat (offer orchestration)
    
#     The context field determines which flow to use:
#     - context.nested_chat=True → Offer orchestration mode
#     - context.nested_chat=False/None → Flight search mode
#     """
#     try:
#         # Extract context if provided
#         context = req.context or ChatContext()
        
#         # Pass context to rag_agent (✅ FIXED: Added await)
#         result = await model_with_tool.rag_agent(
#             chat_history=req.chat_history,
#             nested_chat=context.nested_chat,
#             platform=context.booking_platform,
#             base_price=context.base_price,
#             flight_type=context.flight_type,
#         )

#         return result

#     except Exception as e:
#         print(f"❌ [chat] Error: {e}")
#         traceback.print_exc()
#         return {
#             "content": "⚠️ Something went wrong. Please try again.",
#             "error": str(e),
#             "flight_data": None
#         }

# @app.get("/get_latest_deals")
# def get_latest_deals():
#     """
#     Fetch all deals from MongoDB (or CSV fallback).
#     Returns combined data from all three collections.
#     """
#     EXPECTED_COLUMNS = [
#         "platform", "title", "offer", "coupon_code", "bank",
#         "payment_mode", "url", "expiry_date", "terms_and_conditions", "flight_type"
#     ]
    
#     # CSV fallback (if needed)
#     CSV_FILE_PATH = os.getenv("CSV_FILE_PATH", "./data/deals.csv")
    
#     try:
#         if os.path.exists(CSV_FILE_PATH):
#             print("📁 [get_latest_deals] Using CSV fallback")
#             deals = []
#             with open(CSV_FILE_PATH, newline="", encoding="utf-8") as csvfile:
#                 reader = csv.DictReader(csvfile)
#                 for row in reader:
#                     normalized = {k: (row.get(k, "") or "") for k in EXPECTED_COLUMNS}
#                     deals.append(normalized)
#             return JSONResponse(content={"deals": deals, "source": "csv"})
#     except Exception as e:
#         print(f"⚠️ [get_latest_deals] CSV read error: {e}")

#     # MongoDB retrieval
#     try:
#         print("💾 [get_latest_deals] Fetching from MongoDB")
#         client = mongoDB.get_mongo_client()
        
#         all_deals = []
        
#         # Fetch from all three collections
#         for collection_name in ["gift_coupons", "payment_offers", "general_offers"]:
#             coll = mongoDB.get_collection(client, collection_name)
#             if coll:
#                 docs = list(coll.find({}, {"_id": 0, "embedding": 0}))
#                 # Add collection tag
#                 for doc in docs:
#                     doc["collection"] = collection_name
#                 all_deals.extend(docs)
        
#         if not all_deals:
#             raise ValueError("No deals found in any collection")
        
#         # Normalize fields
#         normalized_docs = []
#         for d in all_deals:
#             normalized = {k: d.get(k, "") for k in EXPECTED_COLUMNS}
#             normalized["collection"] = d.get("collection", "unknown")
#             normalized_docs.append(normalized)
        
#         print(f"✅ [get_latest_deals] Retrieved {len(normalized_docs)} total deals")
#         return JSONResponse(content={
#             "deals": normalized_docs,
#             "source": "mongodb",
#             "count": len(normalized_docs)
#         })
        
#     except Exception as e:
#         print(f"❌ [get_latest_deals] MongoDB error: {e}")
#         traceback.print_exc()
#         return JSONResponse(
#             content={"deals": [], "error": str(e), "source": "error"},
#             status_code=500
#         )

# @app.post("/flights/search")
# async def flight_search(req: Request):
#     """
#     Direct flight search endpoint (bypasses chat interface).
    
#     Example JSON body:
#     {
#         "departure_id": "DEL",
#         "arrival_id": "BOM",
#         "departure_date": "2025-11-28",
#         "include_airlines": "AI,6E",
#         "max_price": "10000",
#         "travel_class": "Economy"
#     }
#     """
#     try:
#         data = await req.json()
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid JSON body: {str(e)}")

#     # Validate required fields
#     required_fields = ["departure_id", "arrival_id", "departure_date"]
#     for field in required_fields:
#         if not data.get(field):
#             raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

#     try:
#         print(f"✈️ [flight_search] Searching flights: {data.get('departure_id')} → {data.get('arrival_id')}")
        
#         # Call flight search tool with travel_class only
#         result = await get_flight_with_aggregator.ainvoke({
#             "departure_id": data["departure_id"],
#             "arrival_id": data["arrival_id"],
#             "departure_date": data["departure_date"],
#             "include_airlines": data.get("include_airlines"),
#             "max_price": data.get("max_price"),
#             "travel_class": data.get("travel_class", "economy")
#         })
        
#         return JSONResponse(content={
#             "results": result,
#             "count": len(result) if result else 0
#         })
        
#     except Exception as e:
#         print(f"❌ [flight_search] Error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/offers/search")
# async def offer_search(req: Request):
#     """
#     Direct offer search endpoint (bypasses chat).
    
#     Example JSON body:
#     {
#         "offer_type": "payment",
#         "bank": "HDFC",
#         "card_type": "Credit Card",
#         "query": "flight cashback"
#     }
#     """
#     try:
#         data = await req.json()
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid JSON body")
    
#     offer_type = data.get("offer_type", "general")
    
#     try:
#         from utils.offer_orchestrator_tool import offer_orchestrator_tool
        
#         result = offer_orchestrator_tool.invoke({
#             "query": data.get("query", "flight offers"),
#             "offer_type": offer_type,
#             "bank": data.get("bank"),
#             "card_type": data.get("card_type"),
#             "base_price": data.get("base_price"),
#             "build_combo": data.get("build_combo", False)
#         })
        
#         return JSONResponse(content={"result": result})
        
#     except Exception as e:
#         print(f"❌ [offer_search] Error: {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

# # Optional: Prometheus metrics endpoint
# try:
#     from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
#     from fastapi.responses import Response
    
#     @app.get("/metrics")
#     def metrics():
#         """Expose Prometheus metrics."""
#         return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
# except ImportError:
#     print("⚠️ prometheus_client not installed, /metrics endpoint disabled")





























