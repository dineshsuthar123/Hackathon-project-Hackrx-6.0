# LLM-Powered Intelligent Query-Retrieval System
## 🏆 Hack 6.0 Hackathon Submission

### 📋 Project Overview

This project implements a sophisticated document analysis system that processes large documents (PDFs, DOCX, emails) and answers natural language queries using advanced AI techniques. The system is specifically designed for insurance, legal, HR, and compliance domains.

### 🎯 Problem Statement Solution

**Challenge**: Design an LLM-Powered Intelligent Query-Retrieval System that can process large documents and make contextual decisions for real-world scenarios.

**Our Solution**: A multi-component system that combines:
- Advanced document processing (PDFs, DOCX, emails)
- Semantic search using FAISS embeddings
- GPT-4 powered query understanding
- Explainable AI with decision rationale
- Structured JSON responses

### 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Input Documents│    │   LLM Parser    │    │ Embedding Search│
│   PDF/DOCX/Email│───▶│Extract structured│───▶│ FAISS/Semantic  │
│                 │    │     query       │    │   Retrieval     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   JSON Output   │    │ Logic Evaluation│    │ Clause Matching │
│Structured response│◀──│Decision processing│◀──│Semantic similarity│
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 🔧 Technical Implementation

#### Core Components

1. **Document Processor** (`src/document_processor.py`)
   - Handles PDF, DOCX, TXT, EML, HTML files
   - Downloads documents from URLs
   - Extracts text, tables, and metadata
   - Detects document sections automatically

2. **Embedding Engine** (`src/embedding_engine.py`)
   - Uses sentence-transformers for text embeddings
   - FAISS vector database for similarity search
   - Intelligent text chunking with overlap
   - Optimized for semantic search

3. **LLM Handler** (`src/llm_handler.py`)
   - OpenAI GPT-4 integration
   - Query analysis and intent recognition
   - Token-efficient processing
   - Response validation

4. **Clause Matcher** (`src/clause_matcher.py`)
   - Semantic similarity matching
   - Query expansion with domain synonyms
   - Multi-factor ranking algorithm
   - Confidence scoring

5. **Response Generator** (`src/response_generator.py`)
   - Structured response generation
   - Explainable AI with decision rationale
   - Confidence assessment
   - Quality validation

### 🚀 Installation & Setup

#### Prerequisites
- Python 3.8+
- OpenAI API key

#### Quick Start

1. **Clone and Setup**
   ```bash
   cd "hack-6.0-hackathon"
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Create `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   HACKRX_API_TOKEN=a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36
   ```

3. **Start the System**
   ```bash
   python start.py
   ```
   
   Or manually:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### 📡 API Documentation

#### Base URL
```
http://localhost:8000/api/v1
```

#### Authentication
```
Authorization: Bearer a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36
```

#### Main Endpoint

**POST** `/hackrx/run`

**Request Body:**
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date.",
        "There is a waiting period of thirty-six (36) months for pre-existing diseases.",
        "Yes, the policy covers maternity expenses with 24 months continuous coverage requirement."
    ],
    "metadata": {
        "total_questions": 3,
        "document_processed": true,
        "explanations": [...],
        "processing_details": {...}
    }
}
```

#### Additional Endpoints

- **GET** `/` - Health check
- **GET** `/api/v1/health` - Detailed system status
- **POST** `/api/v1/test` - System validation

### 🧪 Testing

#### Run System Tests
```bash
python tests/test_system.py
```

#### Test API Endpoint
```bash
python test_client.py
```

#### Manual Testing
Visit `http://localhost:8000/docs` for interactive API documentation.

### 🎯 Evaluation Criteria Compliance

#### ✅ Accuracy
- **Semantic Search**: FAISS-powered vector similarity
- **Context Understanding**: GPT-4 analysis with domain knowledge
- **Clause Matching**: Multi-factor ranking with confidence scoring
- **Validation**: Response quality assessment

#### ✅ Token Efficiency
- **Smart Chunking**: Overlapping text segments for context preservation
- **Context Truncation**: Token-aware content limitation
- **Targeted Queries**: Focused embedding searches
- **Cost Estimation**: Built-in token counting

#### ✅ Latency
- **Async Processing**: Non-blocking I/O operations
- **Parallel Operations**: Concurrent query processing
- **Efficient Indexing**: FAISS optimized vector search
- **Caching**: Embedding reuse capabilities

#### ✅ Reusability
- **Modular Design**: Independent, composable components
- **Configurable**: Environment-based configuration
- **Extensible**: Easy to add new document formats
- **API-First**: RESTful interface for integration

#### ✅ Explainability
- **Decision Rationale**: Step-by-step reasoning chain
- **Source Attribution**: Clause-level traceability
- **Confidence Scoring**: Multi-factor confidence assessment
- **Evidence Strength**: Quality metrics for retrieved content

### 🔬 Advanced Features

#### Intelligent Query Expansion
- Domain-specific synonym mapping
- Automatic keyword extraction
- Multi-query similarity search

#### Confidence Assessment
- Evidence strength evaluation
- Source quality scoring
- Response validation metrics

#### Structured Explanations
- Decision process tracking
- Clause contribution analysis
- Reasoning chain generation

### 📊 Performance Metrics

The system tracks:
- Processing time per query
- Token usage and costs
- Confidence scores
- Similarity match quality
- Document processing efficiency

### 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **LLM**: OpenAI GPT-4 Turbo
- **Vector DB**: FAISS (CPU optimized)
- **Embeddings**: sentence-transformers
- **Document Processing**: PyPDF2, pdfplumber, python-docx
- **Authentication**: Bearer token
- **Async**: asyncio, aiofiles

### 🔐 Security & Configuration

- API token authentication
- Environment variable configuration
- Input validation and sanitization
- Error handling and logging

### 📈 Scalability Considerations

- Async processing for concurrent requests
- Modular architecture for horizontal scaling
- FAISS index persistence for restart efficiency
- Configurable resource limits

### 🏁 Hackathon Submission Checklist

- ✅ **Complete System**: All components implemented and integrated
- ✅ **API Compliance**: Matches required endpoint specification
- ✅ **Documentation**: Comprehensive README and code comments
- ✅ **Testing**: System tests and API client included
- ✅ **Evaluation Criteria**: All 5 criteria addressed
- ✅ **Real-world Ready**: Production-quality code structure
- ✅ **Explainable AI**: Detailed decision rationale
- ✅ **Token Optimization**: Efficient LLM usage

### 🎯 Sample Use Cases

1. **Insurance Policy Analysis**
   - "Does this policy cover knee surgery under what conditions?"
   - "What is the waiting period for maternity benefits?"

2. **Legal Contract Review**
   - "What are the termination clauses in this agreement?"
   - "Are there any liability limitations mentioned?"

3. **HR Policy Queries**
   - "What is the vacation policy for new employees?"
   - "Are remote work arrangements permitted?"

4. **Compliance Document Search**
   - "What are the data retention requirements?"
   - "Are there specific audit procedures outlined?"

### 🚀 Future Enhancements

- Multi-language support
- Advanced table/chart analysis
- Integration with external databases
- Real-time document monitoring
- Custom domain model fine-tuning

---

**Built for Hack 6.0 Hackathon** 🏆  
*Intelligent Document Analysis with Explainable AI*
