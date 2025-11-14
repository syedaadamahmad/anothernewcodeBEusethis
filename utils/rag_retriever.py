# utils/rag_retriever.py
import os
from utils import mongoDB
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_aws import BedrockEmbeddings
from langchain.chat_models import init_chat_model
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()

# Embeddings (shared)
embeddings = BedrockEmbeddings(
    model_id=os.getenv("EMBEDDING_MODEL_ID"),
    region_name=os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

mongo_client = mongoDB.connect_db()

def get_retriever(collection_name: str = "general_offers"):
    """
    Build and return an L2 retriever for the given collection.
    Default collection is 'general_offers'.
    """
    coll = mongoDB.get_collection(mongo_client, collection_name)
    if coll is None:
        raise RuntimeError(f"Mongo collection '{collection_name}' not available")

    vector_store = MongoDBAtlasVectorSearch(
        embedding=embeddings,
        collection=coll,
        index_name="vector_index",
    )

    return vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.4}, #0.75
    )

@tool
def rag_tool(query: str, collection_name: str = "general_offers"):
    """
    Retrieve textual offers from the specified collection.
    Default: 'general_offers' (not gift_coupons).
    Use collection_name="gift_coupons" when user explicitly asks for gift coupons.
    """
    try:
        retriever = get_retriever(collection_name)
    except Exception as e:
        return f"Error accessing offers store: {e}"

    docs = retriever.invoke(query)
    print(f"[RAG DEBUG] Retrieved {len(docs)} docs for query: {query}")
    context = "\n".join(
        d.page_content if hasattr(d, "page_content") else d.get("text", "")
        for d in docs
    )
    # Use a simple LLM wrapper to format results as the previous implementation did.
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    prompt = f"""
You are a helpful assistant. You will receive a user query and a set of offers.
Task: Return only the matched offers in short numbered bolded lines.
User query: {query}

Available offers:
{context}

Answer:
"""
    resp = llm.invoke(prompt)
    return resp.content