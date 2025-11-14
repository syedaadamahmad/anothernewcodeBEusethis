# cc
import os
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from datetime import datetime, timedelta
import hashlib
import json
from threading import Lock

load_dotenv()

# --- CACHE CONFIGURATION ---
CACHE_TTL_MINUTES = 4300
# ---------------------------

# Global MongoDB client with connection pooling
_mongo_client = None
_client_lock = Lock()

def get_mongo_client():
    """
    Returns a singleton MongoDB client with connection pooling.
    This avoids creating new connections for every cache operation.
    """
    global _mongo_client
    
    if _mongo_client is not None:
        return _mongo_client
    
    with _client_lock:
        # Double-check after acquiring lock
        if _mongo_client is not None:
            return _mongo_client
            
        uri = os.getenv("MONGO_DB_URI")
        if not uri:
            print("❌ MONGO_DB_URI not set in env")
            return None

        try:
            # Create client with connection pooling enabled
            _mongo_client = MongoClient(
                uri,
                serverSelectionTimeoutMS=5000,
                maxPoolSize=50,  # Allow up to 50 concurrent connections
                minPoolSize=10,  # Keep 10 connections ready
                maxIdleTimeMS=30000,  # Close idle connections after 30s
            )
            _mongo_client.admin.command("ping")
            print("✅ MongoDB client initialized with connection pooling")
            return _mongo_client
        except errors.ServerSelectionTimeoutError as e:
            print("❌ Connection timed out. Check your URI and internet connection.")
            print("Error:", e)
            return None
        except Exception as e:
            print("❌ Failed to connect to MongoDB Atlas.")
            print("Error:", e)
            return None

def connect_db():
    """
    Legacy function for backwards compatibility.
    Now uses the singleton client with connection pooling.
    """
    return get_mongo_client()

def get_collection(client, collection: str):
    """
    Gets a specific collection from the MongoDB client.
    """
    if client is None:
        print("⚠️ No MongoDB client available. Returning None.")
        return None

    db_name = os.getenv("DB_NAME")
    if not db_name:
        print("❌ DB_NAME not set in env")
        return None

    db = client[db_name]
    collection = db[collection]
    return collection

def generate_cache_key(request_params: dict) -> str:
    """
    Generate a deterministic, globally consistent cache key from request parameters.
    
    This ensures that identical requests from different users/terminals will 
    produce the same cache key, enabling true global caching.
    
    Args:
        request_params: Dictionary of request parameters
        
    Returns:
        A deterministic hash string that uniquely identifies this request
    """
    # Sort keys to ensure consistent ordering
    sorted_params = dict(sorted(request_params.items()))
    
    # Convert to JSON string with sorted keys
    json_str = json.dumps(sorted_params, sort_keys=True)
    
    # Generate SHA256 hash for a compact, deterministic key
    cache_key = hashlib.sha256(json_str.encode()).hexdigest()
    
    return cache_key

def get_api_cache_result(request_key: dict, collection_name: str = "api_cache", verbose: bool = True):
    """
    Looks up a cached API response based on the request key.
    
    Now uses connection pooling for efficiency.
    Returns the API response data only if it is fresh (within CACHE_TTL_MINUTES).
    
    Args:
        request_key: Dictionary of request parameters
        collection_name: MongoDB collection name
        verbose: Whether to print cache status messages
    """
    client = get_mongo_client()
    coll = get_collection(client, collection_name)
    
    if coll is None:
        return None

    try:
        # Generate deterministic cache key
        cache_key = generate_cache_key(request_key)
        
        # Find using the hash key (not the nested dict)
        cached_doc = coll.find_one({"cache_key": cache_key})
        
        if cached_doc:
            cached_at = cached_doc.get("cached_at")
            
            # Check Time-To-Live (TTL)
            if cached_at and (datetime.utcnow() - cached_at) < timedelta(minutes=CACHE_TTL_MINUTES):
                age_seconds = (datetime.utcnow() - cached_at).total_seconds()
                if verbose:
                    print(f"⏳ Cache HIT for key: {cache_key[:16]}... (age: {age_seconds:.0f}s)")
                return cached_doc.get("data")
            else:
                if cached_at:
                    age_seconds = (datetime.utcnow() - cached_at).total_seconds()
                    if verbose:
                        print(f"❌ Cache MISS: Data stale (age: {age_seconds:.0f}s > {CACHE_TTL_MINUTES}min TTL)")
                else:
                    if verbose:
                        print(f"❌ Cache MISS: No valid timestamp")
                
                # Cleanup stale document
                coll.delete_one({"_id": cached_doc["_id"]})
                return None
        else:
            if verbose:
                print(f"❌ Cache MISS: No document found for key: {cache_key[:16]}...")
            return None
            
    except Exception as e:
        print(f"❌ Error during cache lookup: {e}")
        return None

def save_api_cache_result(
    request_key: dict,
    api_response_data: dict,
    collection_name: str = "api_cache",
    verbose: bool = True
):
    """
    Saves an API response to the designated cache collection.
    
    Now uses connection pooling for efficiency.
    
    Args:
        request_key: Dictionary of request parameters
        api_response_data: The API response data to cache
        collection_name: MongoDB collection name
        verbose: Whether to print cache status messages
    """
    client = get_mongo_client()
    coll = get_collection(client, collection_name)
    
    # Generate deterministic cache key
    cache_key = generate_cache_key(request_key)
    
    if verbose:
        print(f"💾 [CACHE WRITE] Saving data for key: {cache_key[:16]}...") 

    if coll is None:
        print("❌ Could not get MongoDB collection for caching.")
        return False

    # Use the hash as the unique identifier
    filter_query = {"cache_key": cache_key}

    # Store both the hash key and the original params (for debugging/auditing)
    document = {
        "cache_key": cache_key,
        "request_params": request_key,
        "data": api_response_data,
        "cached_at": datetime.utcnow()
    }
    
    try:
        result = coll.replace_one(filter_query, document, upsert=True)
        
        if verbose:
            if result.upserted_id:
                print(f"✅ Cache inserted (ID: {result.upserted_id})")
            else:
                print(f"✅ Cache updated for key: {cache_key[:16]}...")
        return True
    
    except Exception as e:
        print(f"❌ Failed to save cache: {e}")
        return False

def batch_save_cache_results(cache_entries: list, collection_name: str = "api_cache"):
    """
    Efficiently saves multiple cache entries in a single batch operation.
    
    Args:
        cache_entries: List of tuples [(request_key, api_response_data), ...]
        collection_name: MongoDB collection name
    
    Returns:
        Number of successfully saved entries
    """
    if not cache_entries:
        return 0
        
    client = get_mongo_client()
    coll = get_collection(client, collection_name)
    
    if coll is None:
        return 0
    
    operations = []
    for request_key, api_response_data in cache_entries:
        cache_key = generate_cache_key(request_key)
        
        document = {
            "cache_key": cache_key,
            "request_params": request_key,
            "data": api_response_data,
            "cached_at": datetime.utcnow()
        }
        
        # Create upsert operation
        operations.append({
            "filter": {"cache_key": cache_key},
            "replacement": document,
            "upsert": True
        })
    
    try:
        # Batch write all operations
        from pymongo import ReplaceOne
        bulk_ops = [ReplaceOne(op["filter"], op["replacement"], upsert=op["upsert"]) for op in operations]
        result = coll.bulk_write(bulk_ops, ordered=False)
        
        print(f"✅ Batch saved {result.upserted_count + result.modified_count} cache entries")
        return result.upserted_count + result.modified_count
    except Exception as e:
        print(f"❌ Batch save failed: {e}")
        return 0

def insert_vector_data(collection:str, csv_file:str):
    """
    Helper wrapper to call create_vector_store.insert_csv_with_embeddings
    """
    from utils import create_vector_store
    mongo_client = get_mongo_client()
    coll = get_collection(mongo_client, collection)
    create_vector_store.insert_csv_with_embeddings(csv_file, coll)

def get_all_deals(collection_name: str = "flight_coupons"):
    """
    Return list of deals from MongoDB (no _id, no embedding).
    """
    client = get_mongo_client()
    coll = get_collection(client, collection_name)
    if coll is None:
        return []
    try:
        docs = list(coll.find({}, {"_id": 0, "embedding": 0}))
        return docs
    except Exception as e:
        print(f"[get_all_deals] error: {e}")
        return []

def close_mongo_connection():
    """
    Closes the MongoDB connection pool.
    Should be called when shutting down the application.
    """
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        print("✅ MongoDB connection pool closed")





























































# # mongodb.py use with all flights
# import os
# from dotenv import load_dotenv
# from pymongo import MongoClient, errors
# from datetime import datetime, UTC
# import hashlib
# import json
# from threading import Lock

# load_dotenv()

# # --- CACHE CONFIGURATION ---
# CACHE_TTL_MINUTES = 432000
# # ---------------------------

# # Global MongoDB client with connection pooling
# _mongo_client = None
# _client_lock = Lock()

# def get_mongo_client():
#     """
#     Returns a singleton MongoDB client with connection pooling.
#     This avoids creating new connections for every cache operation.
#     """
#     global _mongo_client
    
#     if _mongo_client is not None:
#         return _mongo_client
    
#     with _client_lock:
#         # Double-check after acquiring lock
#         if _mongo_client is not None:
#             return _mongo_client
            
#         uri = os.getenv("MONGO_DB_URI")
#         if not uri:
#             print("❌ MONGO_DB_URI not set in env")
#             return None

#         try:
#             # Create client with connection pooling enabled
#             _mongo_client = MongoClient(
#                 uri,
#                 serverSelectionTimeoutMS=5000,
#                 maxPoolSize=50,  # Allow up to 50 concurrent connections
#                 minPoolSize=10,  # Keep 10 connections ready
#                 maxIdleTimeMS=30000,  # Close idle connections after 30s
#             )
#             _mongo_client.admin.command("ping")
#             print("✅ MongoDB client initialized with connection pooling")
#             return _mongo_client
#         except errors.ServerSelectionTimeoutError as e:
#             print("❌ Connection timed out. Check your URI and internet connection.")
#             print("Error:", e)
#             return None
#         except Exception as e:
#             print("❌ Failed to connect to MongoDB Atlas.")
#             print("Error:", e)
#             return None

# def connect_db():
#     """
#     Legacy function for backwards compatibility.
#     Now uses the singleton client with connection pooling.
#     """
#     return get_mongo_client()

# def get_collection(client, collection: str):
#     """
#     Gets a specific collection from the MongoDB client.
#     """
#     if client is None:
#         print("⚠️ No MongoDB client available. Returning None.")
#         return None

#     db_name = os.getenv("DB_NAME")
#     if not db_name:
#         print("❌ DB_NAME not set in env")
#         return None

#     db = client[db_name]
#     collection = db[collection]
#     return collection

# def generate_cache_key(request_params: dict) -> str:
#     """
#     Generate a deterministic, globally consistent cache key from request parameters.
    
#     This ensures that identical requests from different users/terminals will 
#     produce the same cache key, enabling true global caching.
    
#     Args:
#         request_params: Dictionary of request parameters
        
#     Returns:
#         A deterministic hash string that uniquely identifies this request
#     """
#     # Sort keys to ensure consistent ordering
#     sorted_params = dict(sorted(request_params.items()))
    
#     # Convert to JSON string with sorted keys
#     json_str = json.dumps(sorted_params, sort_keys=True)
    
#     # Generate SHA256 hash for a compact, deterministic key
#     cache_key = hashlib.sha256(json_str.encode()).hexdigest()
    
#     return cache_key

# def get_api_cache_result(request_key: dict, collection_name: str = "api_cache", verbose: bool = True):
#     """
#     Looks up a cached API response based on the request key.
    
#     Now uses connection pooling for efficiency.
#     Returns the API response data only if it is fresh (within CACHE_TTL_MINUTES).
    
#     Args:
#         request_key: Dictionary of request parameters
#         collection_name: MongoDB collection name
#         verbose: Whether to print cache status messages
#     """
#     client = get_mongo_client()
#     coll = get_collection(client, collection_name)
    
#     if coll is None:
#         return None

#     try:
#         # Generate deterministic cache key
#         cache_key = generate_cache_key(request_key)
        
#         # Find using the hash key (not the nested dict)
#         cached_doc = coll.find_one({"cache_key": cache_key})
        
#         if cached_doc:
#             cached_at = cached_doc.get("cached_at")
            
#             # Check Time-To-Live (TTL)
#             if cached_at and (datetime.now(UTC) - cached_at) < timedelta(minutes=CACHE_TTL_MINUTES):
#                 age_seconds = (datetime.now(UTC) - cached_at).total_seconds()
#                 if verbose:
#                     print(f"⏳ Cache HIT for key: {cache_key[:16]}... (age: {age_seconds:.0f}s)")
#                 return cached_doc.get("data")
#             else:
#                 if cached_at:
#                     age_seconds = (datetime.now(UTC) - cached_at).total_seconds()
#                     if verbose:
#                         print(f"❌ Cache MISS: Data stale (age: {age_seconds:.0f}s > {CACHE_TTL_MINUTES}min TTL)")
#                 else:
#                     if verbose:
#                         print(f"❌ Cache MISS: No valid timestamp")
                
#                 # Cleanup stale document
#                 coll.delete_one({"_id": cached_doc["_id"]})
#                 return None
#         else:
#             if verbose:
#                 print(f"❌ Cache MISS: No document found for key: {cache_key[:16]}...")
#             return None
            
#     except Exception as e:
#         print(f"❌ Error during cache lookup: {e}")
#         return None

# def save_api_cache_result(
#     request_key: dict,
#     api_response_data: dict,
#     collection_name: str = "api_cache",
#     verbose: bool = True
# ):
#     """
#     Saves an API response to the designated cache collection.
    
#     Now uses connection pooling for efficiency.
    
#     Args:
#         request_key: Dictionary of request parameters
#         api_response_data: The API response data to cache
#         collection_name: MongoDB collection name
#         verbose: Whether to print cache status messages
#     """
#     client = get_mongo_client()
#     coll = get_collection(client, collection_name)
    
#     # Generate deterministic cache key
#     cache_key = generate_cache_key(request_key)
    
#     if verbose:
#         print(f"💾 [CACHE WRITE] Saving data for key: {cache_key[:16]}...") 

#     if coll is None:
#         print("❌ Could not get MongoDB collection for caching.")
#         return False

#     # Use the hash as the unique identifier
#     filter_query = {"cache_key": cache_key}

#     # Store both the hash key and the original params (for debugging/auditing)
#     document = {
#         "cache_key": cache_key,
#         "request_params": request_key,
#         "data": api_response_data,
#         "cached_at": datetime.utcnow()
#     }
    
#     try:
#         result = coll.replace_one(filter_query, document, upsert=True)
        
#         if verbose:
#             if result.upserted_id:
#                 print(f"✅ Cache inserted (ID: {result.upserted_id})")
#             else:
#                 print(f"✅ Cache updated for key: {cache_key[:16]}...")
#         return True
    
#     except Exception as e:
#         print(f"❌ Failed to save cache: {e}")
#         return False

# def batch_save_cache_results(cache_entries: list, collection_name: str = "api_cache"):
#     """
#     Efficiently saves multiple cache entries in a single batch operation.
    
#     Args:
#         cache_entries: List of tuples [(request_key, api_response_data), ...]
#         collection_name: MongoDB collection name
    
#     Returns:
#         Number of successfully saved entries
#     """
#     if not cache_entries:
#         return 0
        
#     client = get_mongo_client()
#     coll = get_collection(client, collection_name)
    
#     if coll is None:
#         return 0
    
#     operations = []
#     for request_key, api_response_data in cache_entries:
#         cache_key = generate_cache_key(request_key)
        
#         document = {
#             "cache_key": cache_key,
#             "request_params": request_key,
#             "data": api_response_data,
#             "cached_at": datetime.now(UTC)
#         }
        
#         # Create upsert operation
#         operations.append({
#             "filter": {"cache_key": cache_key},
#             "replacement": document,
#             "upsert": True
#         })
    
#     try:
#         # Batch write all operations
#         from pymongo import ReplaceOne
#         bulk_ops = [ReplaceOne(op["filter"], op["replacement"], upsert=op["upsert"]) for op in operations]
#         result = coll.bulk_write(bulk_ops, ordered=False)
        
#         print(f"✅ Batch saved {result.upserted_count + result.modified_count} cache entries")
#         return result.upserted_count + result.modified_count
#     except Exception as e:
#         print(f"❌ Batch save failed: {e}")
#         return 0

# def insert_vector_data(collection:str, csv_file:str):
#     """
#     Helper wrapper to call create_vector_store.insert_csv_with_embeddings
#     """
#     from utils import create_vector_store
#     mongo_client = get_mongo_client()
#     coll = get_collection(mongo_client, collection)
#     create_vector_store.insert_csv_with_embeddings(csv_file, coll)

# def get_all_deals(collection_name: str = "flight_coupons"):
#     """
#     Return list of deals from MongoDB (no _id, no embedding).
#     """
#     client = get_mongo_client()
#     coll = get_collection(client, collection_name)
#     if coll is None:
#         return []
#     try:
#         docs = list(coll.find({}, {"_id": 0, "embedding": 0}))
#         return docs
#     except Exception as e:
#         print(f"[get_all_deals] error: {e}")
#         return []

# def close_mongo_connection():
#     """
#     Closes the MongoDB connection pool.
#     Should be called when shutting down the application.
#     """
#     global _mongo_client
#     if _mongo_client is not None:
#         _mongo_client.close()
#         _mongo_client = None
#         print("✅ MongoDB connection pool closed")