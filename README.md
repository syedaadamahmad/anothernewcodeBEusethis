# Flight Coupon Assistant Backend

A FastAPI-based backend service that provides an intelligent flight coupon and discount assistant using RAG (Retrieval-Augmented Generation) with MongoDB Atlas vector search and AWS Bedrock embeddings.

## 🚀 Features

- **Intelligent Flight Deal Assistant**: AI-powered chatbot that helps users find the best flight offers, discounts, and coupons
- **RAG-based Retrieval**: Uses vector embeddings to find relevant offers from a database of flight deals
- **Multi-platform Support**: Works with various booking platforms (MakeMyTrip, Goibibo, EaseMyTrip, etc.)
- **Bank-specific Offers**: Supports offers from different banks (HDFC, ICICI, SBI, etc.)
- **Payment Method Filtering**: Handles both credit and debit card offers
- **Flight Type Support**: Covers both domestic and international flight offers
- **EMI Options**: Includes EMI availability information
- **Conversational Interface**: Natural language processing with follow-up questions

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │────│   RAG Agent      │────│  MongoDB Atlas  │
│   (main.py)     │    │ (model_with_tool)│    │  Vector Store   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CORS Support  │    │  AWS Bedrock     │    │  CSV Data       │
│   Chat Endpoint │    │  Embeddings      │    │  Processing     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
backend/
├── main.py                    # FastAPI application with chat endpoint
├── requirements.txt           # Python dependencies
├── utils/
│   ├── create_vector_store.py # CSV processing and vector store creation
│   ├── model_with_tool.py    # RAG agent with conversational AI
│   ├── mongoDB.py            # MongoDB connection and utilities
│   └── rag_retriever.py     # Vector search and retrieval logic
└── README.md                 # This documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account
- AWS account with Bedrock access
- Google AI API key (for Gemini model)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd backend
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Environment Configuration

Create a `.env` file in the project root with the following variables:

```env
# MongoDB Configuration
MONGO_DB_URI=your_mongodb_atlas_connection_string
DB_NAME=your_database_name

# AWS Bedrock Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=your_aws_region
EMBEDDING_MODEL_ID=amazon.titan-embed-text-v1

# Google AI Configuration (for Gemini model)
GOOGLE_API_KEY=your_google_api_key
```

### Step 5: Database Setup

1. **MongoDB Atlas Setup**:
   - Create a MongoDB Atlas cluster
   - Create a database and collection (e.g., `flight_coupons`)
   - Set up a vector search index named `vector_index`
   - Configure the index with cosine similarity

2. **Vector Index Configuration**:
   ```json
   {
     "fields": [
       {
         "type": "vector",
         "path": "embedding",
         "numDimensions": 1536,
         "similarity": "cosine"
       }
     ]
   }
   ```

### Step 6: Data Population (Optional)

To populate the database with flight offers:

```python
from utils.mongoDB import insert_vector_data

# Insert data from CSV file
insert_vector_data("flight_coupons", "path/to/your/flight_offers.csv")
```

## 🚀 Running the Application

### Development Mode

```bash
fastapi dev main.py
        Or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "message": "its working fine :)"
}
```

#### 2. Chat Endpoint
```http
POST /chat
```

**Request Body:**
```json
{
  "chat_history": [
    {
      "role": "human",
      "content": "I'm looking for HDFC credit card offers on MakeMyTrip for domestic flights"
    }
  ]
}
```

**Response:**
```json
{
  "answer": "Here are the available offers for HDFC credit card on MakeMyTrip for domestic flights:\n\n1. **Get FLAT ₹400 OFF on domestic flights when your transaction is above ₹7,500.**\n2. **Get FLAT ₹500 OFF on domestic flights when your transaction is above ₹10,000.**"
}
```

## 🔧 Configuration Details

### CSV Data Format

The system expects CSV files with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `platform` | Booking platform | "MakeMyTrip", "Goibibo" |
| `title` | Offer title | "HDFC Credit Card Offer" |
| `offer` | Offer description | "Get FLAT ₹400 OFF" |
| `coupon_code` | Coupon code | "HDFC400" |
| `bank` | Bank name | "HDFC", "ICICI" |
| `payment_mode` | Payment method | "credit", "debit" |
| `emi` | EMI availability | "y", "n" |
| `url` | Offer URL | "https://example.com" |
| `flight_type` | Flight type | "domestic", "international" |

### AI Model Configuration

- **LLM**: Google Gemini 2.5 Flash
- **Embeddings**: AWS Bedrock Titan Embeddings
- **Vector Store**: MongoDB Atlas Vector Search
- **Similarity**: Cosine similarity with 0.75 threshold

## 🎯 Usage Examples

### Example 1: Basic Query
```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "chat_history": [
        {
            "role": "human", 
            "content": "Show me ICICI credit card offers"
        }
    ]
})

print(response.json()["answer"])
```

### Example 2: Multi-turn Conversation
```python
chat_history = [
    {"role": "human", "content": "I want flight offers"},
    {"role": "ai", "content": "Got it! Which bank would you like to check offers for?"},
    {"role": "human", "content": "HDFC"},
    {"role": "ai", "content": "Great! Are you looking for credit card or debit card offers?"},
    {"role": "human", "content": "Credit card"},
    {"role": "ai", "content": "Perfect! Which platform are you planning to book on?"},
    {"role": "human", "content": "MakeMyTrip"}
]

response = requests.post("http://localhost:8000/chat", json={
    "chat_history": chat_history
})
```

## 🔍 Key Components

### 1. RAG Agent (`model_with_tool.py`)
- Implements conversational AI with tool calling
- Handles multi-turn conversations
- Manages context and follow-up questions
- Uses Google Gemini 2.5 Flash model

### 2. Vector Store (`create_vector_store.py`)
- Processes CSV data into embeddings
- Generates human-friendly offer descriptions
- Handles metadata extraction and storage

### 3. Retrieval System (`rag_retriever.py`)
- Performs semantic search on vector database
- Filters results by similarity threshold
- Formats responses with offer details

### 4. Database Layer (`mongoDB.py`)
- Manages MongoDB Atlas connections
- Handles error scenarios gracefully
- Provides collection management utilities
