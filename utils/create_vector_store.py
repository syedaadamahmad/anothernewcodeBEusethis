# cc
#create_vector_store.py
import os
import pandas as pd
from dotenv import load_dotenv
from pymongo.errors import PyMongoError
from langchain_aws import BedrockEmbeddings
from langchain.docstore.document import Document
from langchain_mongodb import MongoDBAtlasVectorSearch

load_dotenv()


def insert_csv_with_embeddings(csv_file: str, collection):
    """
    Reads a CSV, generates embeddings using AWS Bedrock, and inserts
    documents into MongoDB Atlas Vector Search.
    Includes strict error handling and data sanitization.
    """
    if collection is None:
        print("⚠️ Skipping insert because MongoDB connection failed.")
        return

    try:
        # Initialize AWS Bedrock Embeddings
        embeddings = BedrockEmbeddings(
            model_id=os.getenv("EMBEDDING_MODEL_ID"),
            region_name=os.getenv("AWS_DEFAULT_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        # Initialize MongoDB Vector Store
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=embeddings,
            index_name="vector_index",
            relevance_score_fn="cosine",
        )

        # Load and validate CSV
        df = pd.read_csv(csv_file)
        if df.empty:
            print(f"❌ CSV file '{csv_file}' is empty.")
            return

        docs = []
        for _, row in df.iterrows():
            text_to_embed = generate_offer_string(row)
            emi_value = str(row.get("emi", "")).strip().lower()
            emi = 1 if emi_value == "y" else 0

            metadata = {
                "platform": str(row.get("platform", "")).strip(),
                "title": str(row.get("title", "")).strip(),
                "offer": str(row.get("offer", "")).strip(),
                "coupon_code": str(row.get("coupon_code", "")).strip(),
                "bank": str(row.get("bank", "")).strip(),
                "payment_mode": str(row.get("payment_mode", "")).strip(),
                "url": str(row.get("url", "")).strip(),
                # "terms_and_conditions": str(row.get("terms_and_conditions", "")).strip(),
                "emi": emi,
                "flight_type": str(row.get("flight_type", "")).strip(),
            }

            docs.append(Document(page_content=text_to_embed, metadata=metadata))

        # Insert all documents into MongoDB Vector Index
        vector_store.add_documents(documents=docs)
        print(f"✅ Successfully inserted {len(docs)} documents into MongoDB vector index.")

    except FileNotFoundError:
        print(f"❌ CSV file '{csv_file}' not found.")
    except pd.errors.EmptyDataError:
        print(f"❌ CSV file '{csv_file}' is empty or invalid.")
    except PyMongoError as e:
        print("❌ Failed to insert documents into MongoDB.")
        print("Error:", e)
    except Exception as e:
        print("❌ An unexpected error occurred while inserting CSV data.")
        print("Error:", e)


def generate_offer_string(row):
    """
    Generates a descriptive, human-friendly embedding string combining
    platform, title, offer, bank, payment mode, EMI, and flight type.
    """
    platform = str(row.get("platform", "")).strip()
    title = str(row.get("title", "")).strip()
    offers = str(row.get("offer", "")).strip()
    bank = str(row.get("bank", "")).strip()
    payment_mode = str(row.get("payment_mode", "")).strip()
    emi = str(row.get("emi", "")).strip()
    flight_type = str(row.get("flight_type", "")).strip()

    payment_info = ""
    if bank and payment_mode:
        payment_info = f" for customers using {bank} with {payment_mode}"
    elif bank:
        payment_info = f" for customers using {bank}"
    elif payment_mode:
        payment_info = f" for customers with {payment_mode}"

    emi_info = " EMI options are available" if emi.lower() == "y" else ""

    offer_string = (
        f"Take advantage of '{title}' on {platform}, which provides {offers}{payment_info} "
        f"This exclusive deal is valid for {flight_type} flights and lets you save while traveling comfortably."
    )

    return offer_string




































# #create_vector_store.py
# import os
# import pandas as pd
# from dotenv import load_dotenv
# from pymongo.errors import PyMongoError
# from langchain_aws import BedrockEmbeddings
# from langchain.docstore.document import Document
# from langchain_mongodb import MongoDBAtlasVectorSearch

# load_dotenv()


# def insert_csv_with_embeddings(csv_file: str, collection):
#     """
#     Reads a CSV, generates embeddings using AWS Bedrock, and inserts
#     documents into MongoDB Atlas Vector Search.
#     Includes strict error handling and data sanitization.
#     """
#     if collection is None:
#         print("⚠️ Skipping insert because MongoDB connection failed.")
#         return

#     try:
#         # Initialize AWS Bedrock Embeddings
#         embeddings = BedrockEmbeddings(
#             model_id=os.getenv("EMBEDDING_MODEL_ID"),
#             region_name=os.getenv("AWS_DEFAULT_REGION"),
#             aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
#             aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
#         )

#         # Initialize MongoDB Vector Store
#         vector_store = MongoDBAtlasVectorSearch(
#             collection=collection,
#             embedding=embeddings,
#             index_name="vector_index",
#             relevance_score_fn="cosine",
#         )

#         # Load and validate CSV
#         df = pd.read_csv(csv_file)
#         if df.empty:
#             print(f"❌ CSV file '{csv_file}' is empty.")
#             return

#         docs = []
#         for _, row in df.iterrows():
#             text_to_embed = generate_offer_string(row)
#             emi_value = str(row.get("emi", "")).strip().lower()
#             emi = 1 if emi_value == "y" else 0

#             metadata = {
#                 "platform": str(row.get("platform", "")).strip(),
#                 "title": str(row.get("title", "")).strip(),
#                 "offer": str(row.get("offer", "")).strip(),
#                 "coupon_code": str(row.get("coupon_code", "")).strip(),
#                 "bank": str(row.get("bank", "")).strip(),
#                 "payment_mode": str(row.get("payment_mode", "")).strip(),
#                 "url": str(row.get("url", "")).strip(),
#                 # "terms_and_conditions": str(row.get("terms_and_conditions", "")).strip(),
#                 "emi": emi,
#                 "flight_type": str(row.get("flight_type", "")).strip(),
#             }

#             docs.append(Document(page_content=text_to_embed, metadata=metadata))

#         # Insert all documents into MongoDB Vector Index
#         vector_store.add_documents(documents=docs)
#         print(f"✅ Successfully inserted {len(docs)} documents into MongoDB vector index.")

#     except FileNotFoundError:
#         print(f"❌ CSV file '{csv_file}' not found.")
#     except pd.errors.EmptyDataError:
#         print(f"❌ CSV file '{csv_file}' is empty or invalid.")
#     except PyMongoError as e:
#         print("❌ Failed to insert documents into MongoDB.")
#         print("Error:", e)
#     except Exception as e:
#         print("❌ An unexpected error occurred while inserting CSV data.")
#         print("Error:", e)


# def generate_offer_string(row):
#     """
#     Generates a descriptive, human-friendly embedding string combining
#     platform, title, offer, bank, payment mode, EMI, and flight type.
#     """
#     platform = str(row.get("platform", "")).strip()
#     title = str(row.get("title", "")).strip()
#     offers = str(row.get("offer", "")).strip()
#     bank = str(row.get("bank", "")).strip()
#     payment_mode = str(row.get("payment_mode", "")).strip()
#     emi = str(row.get("emi", "")).strip()
#     flight_type = str(row.get("flight_type", "")).strip()

#     payment_info = ""
#     if bank and payment_mode:
#         payment_info = f" for customers using {bank} with {payment_mode}"
#     elif bank:
#         payment_info = f" for customers using {bank}"
#     elif payment_mode:
#         payment_info = f" for customers with {payment_mode}"

#     emi_info = " EMI options are available" if emi.lower() == "y" else ""

#     offer_string = (
#         f"Take advantage of '{title}' on {platform}, which provides {offers}{payment_info} "
#         f"This exclusive deal is valid for {flight_type} flights and lets you save while traveling comfortably."
#     )

#     return offer_string












