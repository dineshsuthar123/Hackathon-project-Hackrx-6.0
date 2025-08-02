"""
Utility functions for the LLM-Powered Query-Retrieval System
"""

import logging
import time
import functools
from typing import Any, Callable, Dict, List
import json
import asyncio

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('app.log')
        ]
    )
    return logging.getLogger(__name__)

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.getLogger(func.__module__).info(
                f"{func.__name__} executed in {execution_time:.3f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.getLogger(func.__module__).error(
                f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}"
            )
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logging.getLogger(func.__module__).info(
                f"{func.__name__} executed in {execution_time:.3f} seconds"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logging.getLogger(func.__module__).error(
                f"{func.__name__} failed after {execution_time:.3f} seconds: {str(e)}"
            )
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def validate_json_response(response: Dict[str, Any]) -> bool:
    """Validate JSON response format"""
    required_fields = ["answers"]
    
    if not isinstance(response, dict):
        return False
    
    for field in required_fields:
        if field not in response:
            return False
    
    if not isinstance(response["answers"], list):
        return False
    
    return True

def sanitize_text(text: str) -> str:
    """Sanitize text for processing"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = " ".join(text.split())
    
    # Remove special characters that might interfere with processing
    import re
    text = re.sub(r'[^\w\s\.,!?\-:;()"\']', ' ', text)
    
    return text.strip()

def chunk_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """Chunk text into smaller pieces with overlap"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_length
        
        # Try to break at sentence boundaries
        if end < len(text):
            # Look for sentence endings
            for punct in ['. ', '! ', '? ', '\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct > start + max_length // 2:
                    end = last_punct + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks

def extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from text using simple heuristics"""
    import re
    
    # Common patterns for key phrases
    patterns = [
        r'\b(?:covered|excluded|included|required|must|shall|will|may)\s+[^.!?]*',
        r'\b(?:waiting period|grace period|deductible|premium|coverage)\s+[^.!?]*',
        r'\b(?:\d+\s*(?:months?|days?|years?|%))\b[^.!?]*',
        r'\b(?:medical|surgical|dental|vision|mental health)\s+[^.!?]*'
    ]
    
    key_phrases = []
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            phrase = match.group().strip()
            if len(phrase) > 10 and phrase not in key_phrases:
                key_phrases.append(phrase)
    
    return key_phrases[:10]  # Return top 10 key phrases

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate basic text similarity using word overlap"""
    if not text1 or not text2:
        return 0.0
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1 & words2
    union = words1 | words2
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)

def format_confidence_score(score: float) -> str:
    """Format confidence score as percentage with description"""
    percentage = score * 100
    
    if percentage >= 90:
        return f"{percentage:.1f}% (Very High)"
    elif percentage >= 75:
        return f"{percentage:.1f}% (High)"
    elif percentage >= 60:
        return f"{percentage:.1f}% (Medium)"
    elif percentage >= 40:
        return f"{percentage:.1f}% (Low)"
    else:
        return f"{percentage:.1f}% (Very Low)"

class TokenCounter:
    """Simple token counter for cost estimation"""
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count (rough approximation)"""
        # GPT models approximately use 1 token per 4 characters
        return len(text) // 4
    
    @staticmethod
    def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
        """Estimate API cost based on token usage"""
        # Rough pricing estimates (as of 2024)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }
        
        if model not in pricing:
            model = "gpt-4"
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        
        return input_cost + output_cost
