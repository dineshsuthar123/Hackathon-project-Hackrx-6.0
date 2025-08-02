"""
LLM-Powered Intelligent Query-Retrieval System
Main FastAPI application for document analysis and query processing
"""

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
from dotenv import load_dotenv

from src.document_processor import DocumentProcessor
from src.embedding_engine import EmbeddingEngine
from src.llm_handler import LLMHandler
from src.clause_matcher import ClauseMatcher
from src.response_generator import ResponseGenerator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Advanced document analysis system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
HACKRX_API_TOKEN = os.getenv("HACKRX_API_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the API token"""
    if credentials.credentials != HACKRX_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials

# Pydantic models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL or path to the document(s)")
    questions: List[str] = Field(..., description="List of questions to answer")

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="Answers to the provided questions")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata and explanations")

# Initialize components
document_processor = DocumentProcessor()
embedding_engine = EmbeddingEngine()
llm_handler = LLMHandler()
clause_matcher = ClauseMatcher(embedding_engine)
response_generator = ResponseGenerator(llm_handler)

@app.on_event("startup")
async def startup_event():
    """Initialize the application components"""
    logger.info("Starting LLM-Powered Query-Retrieval System...")
    await embedding_engine.initialize()
    logger.info("System initialized successfully")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "LLM-Powered Intelligent Query-Retrieval System",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/api/v1/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "document_processor": "ready",
            "embedding_engine": "ready",
            "llm_handler": "ready",
            "clause_matcher": "ready"
        }
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
) -> QueryResponse:
    """
    Main endpoint for processing document queries
    
    This endpoint:
    1. Downloads and processes the document
    2. Extracts and indexes content using embeddings
    3. Matches queries with relevant clauses
    4. Generates structured responses using LLM
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        
        # Step 1: Process the document
        logger.info("Processing document...")
        document_content = await document_processor.process_document(request.documents)
        
        # Step 2: Create embeddings and index content
        logger.info("Creating embeddings...")
        indexed_content = await embedding_engine.index_content(document_content)
        
        # Step 3: Process each question
        answers = []
        explanations = []
        
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
            
            # Find relevant clauses
            relevant_clauses = await clause_matcher.find_relevant_clauses(
                question, indexed_content
            )
            
            # Generate response
            answer, explanation = await response_generator.generate_response(
                question, relevant_clauses, document_content
            )
            
            answers.append(answer)
            explanations.append(explanation)
        
        # Prepare response
        response = QueryResponse(
            answers=answers,
            metadata={
                "total_questions": len(request.questions),
                "document_processed": True,
                "explanations": explanations,
                "processing_details": {
                    "content_length": len(document_content.get("text", "")),
                    "embedding_dimensions": embedding_engine.get_embedding_dimensions(),
                    "model_used": llm_handler.get_model_name()
                }
            }
        )
        
        logger.info("Request processed successfully")
        return response
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/test")
async def test_endpoint(
    token: str = Depends(verify_token)
):
    """Test endpoint for system validation"""
    return {
        "message": "Test successful",
        "system_status": "operational",
        "components_loaded": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
