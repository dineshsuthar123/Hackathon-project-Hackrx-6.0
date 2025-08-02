"""
LLM Handler Module
Manages interactions with OpenAI GPT models for query understanding and response generation
"""

import asyncio
import logging
import openai
import os
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import tiktoken

load_dotenv()

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handles LLM operations for query processing and response generation"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = self.api_key
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the user query to understand intent and extract key information
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary with query analysis results
        """
        try:
            system_prompt = """You are an expert document analyst specializing in insurance, legal, HR, and compliance domains. 
            Analyze the given query and extract key information that will help in document retrieval and response generation.
            
            Your analysis should identify:
            1. Query intent (coverage_inquiry, condition_check, definition_request, procedure_inquiry, etc.)
            2. Key entities (medical procedures, time periods, coverage types, etc.)
            3. Specific requirements or conditions mentioned
            4. Domain context (insurance, legal, HR, compliance)
            5. Expected response type (yes/no, definition, procedure, amount, etc.)
            
            Return your analysis in a structured format."""
            
            user_prompt = f"""Analyze this query: "{query}"
            
            Provide a JSON response with the following structure:
            {{
                "intent": "description of what the user is asking",
                "entities": ["list", "of", "key", "entities"],
                "domain": "primary domain (insurance/legal/hr/compliance)",
                "keywords": ["relevant", "search", "keywords"],
                "response_type": "expected response format",
                "complexity": "simple/moderate/complex"
            }}"""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            
            # Try to extract JSON, fallback to text analysis
            try:
                import json
                analysis = json.loads(analysis_text)
            except:
                # Fallback to basic analysis
                analysis = {
                    "intent": "general_inquiry",
                    "entities": self._extract_basic_entities(query),
                    "domain": "insurance",
                    "keywords": query.split(),
                    "response_type": "descriptive",
                    "complexity": "moderate"
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Return basic analysis as fallback
            return {
                "intent": "general_inquiry",
                "entities": [],
                "domain": "general",
                "keywords": query.split(),
                "response_type": "descriptive",
                "complexity": "moderate"
            }
    
    async def generate_answer(self, 
                            query: str, 
                            relevant_context: List[Dict[str, Any]], 
                            document_metadata: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a comprehensive answer based on query and relevant document context
        
        Args:
            query: User's question
            relevant_context: List of relevant document chunks
            document_metadata: Metadata about the source document
            
        Returns:
            Tuple of (answer, explanation_metadata)
        """
        try:
            # Prepare context from relevant chunks
            context_text = self._prepare_context(relevant_context)
            
            # Check token limits
            if self._count_tokens(context_text) > self.max_tokens // 2:
                context_text = self._truncate_context(context_text, self.max_tokens // 2)
            
            system_prompt = """You are an expert document analyst specializing in insurance policies, legal documents, HR policies, and compliance regulations. 
            
            Your task is to provide accurate, comprehensive answers based on the provided document context. 
            
            Guidelines:
            1. Base your answer ONLY on the provided context
            2. Be specific and cite relevant clauses or sections when possible
            3. If the context doesn't contain enough information, state this clearly
            4. Provide clear, actionable information
            5. Use professional, clear language
            6. Include specific details like amounts, time periods, conditions, etc.
            7. If there are conditions or exceptions, mention them clearly
            
            Format your response as a clear, direct answer that fully addresses the question."""
            
            user_prompt = f"""Document Context:
            {context_text}
            
            Question: {query}
            
            Please provide a comprehensive answer based on the document context provided above. Be specific and include relevant details, conditions, and exceptions."""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens - self._count_tokens(system_prompt + user_prompt)
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Generate explanation metadata
            explanation = await self._generate_explanation(query, relevant_context, answer)
            
            return answer, explanation
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}", {}
    
    async def _generate_explanation(self, 
                                  query: str, 
                                  context_chunks: List[Dict[str, Any]], 
                                  answer: str) -> Dict[str, Any]:
        """Generate explanation metadata for the response"""
        try:
            explanation = {
                "sources_used": len(context_chunks),
                "confidence_indicators": [],
                "key_clauses": [],
                "reasoning": "",
                "context_chunks": []
            }
            
            # Extract key information from context chunks
            for chunk in context_chunks[:3]:  # Top 3 most relevant
                explanation["context_chunks"].append({
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "similarity_score": chunk.get("similarity_score", 0),
                    "source": chunk.get("metadata", {}).get("source", "unknown")
                })
            
            # Generate reasoning
            reasoning_prompt = f"""Based on the query "{query}" and the answer "{answer[:200]}...", 
            provide a brief explanation of the reasoning process used to arrive at this answer.
            Focus on which document sections were most relevant and why."""
            
            reasoning_response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": reasoning_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            explanation["reasoning"] = reasoning_response.choices[0].message.content.strip()
            
            # Identify confidence indicators
            if context_chunks:
                avg_similarity = sum(chunk.get("similarity_score", 0) for chunk in context_chunks) / len(context_chunks)
                if avg_similarity > 0.8:
                    explanation["confidence_indicators"].append("High context relevance")
                elif avg_similarity > 0.6:
                    explanation["confidence_indicators"].append("Moderate context relevance")
                else:
                    explanation["confidence_indicators"].append("Lower context relevance")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return {"reasoning": "Explanation generation failed", "sources_used": len(context_chunks)}
    
    def _prepare_context(self, relevant_context: List[Dict[str, Any]]) -> str:
        """Prepare context text from relevant chunks"""
        if not relevant_context:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(relevant_context):
            chunk_text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "")
            
            # Add source information
            source_info = f"[Source: {source}"
            if "section_name" in metadata:
                source_info += f", Section: {metadata['section_name']}"
            if "page_number" in metadata:
                source_info += f", Page: {metadata['page_number']}"
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{chunk_text}\n")
        
        return "\n".join(context_parts)
    
    def _extract_basic_entities(self, query: str) -> List[str]:
        """Basic entity extraction as fallback"""
        # Common entities in insurance/legal domains
        entities = []
        
        # Medical terms
        medical_terms = ["surgery", "treatment", "procedure", "condition", "disease", "illness", "injury"]
        for term in medical_terms:
            if term.lower() in query.lower():
                entities.append(term)
        
        # Time periods
        time_terms = ["period", "months", "years", "days", "waiting", "grace"]
        for term in time_terms:
            if term.lower() in query.lower():
                entities.append(term)
        
        # Coverage terms
        coverage_terms = ["cover", "coverage", "benefit", "claim", "premium", "deductible"]
        for term in coverage_terms:
            if term.lower() in query.lower():
                entities.append(term)
        
        return entities
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            # Fallback to character-based estimation
            return len(text) // 4
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit within token limits"""
        try:
            tokens = self.tokenizer.encode(context)
            if len(tokens) <= max_tokens:
                return context
            
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
        except:
            # Fallback to character-based truncation
            max_chars = max_tokens * 4
            return context[:max_chars] + "..." if len(context) > max_chars else context
    
    def get_model_name(self) -> str:
        """Get the current model name"""
        return self.model
    
    async def validate_response_quality(self, query: str, answer: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate the quality of the generated response"""
        try:
            validation_prompt = f"""Evaluate the quality of this answer:
            
            Question: {query}
            Answer: {answer}
            
            Rate the answer on:
            1. Accuracy (based on context provided)
            2. Completeness (fully addresses the question)
            3. Clarity (easy to understand)
            4. Specificity (includes relevant details)
            
            Provide a brief assessment and overall score (1-10)."""
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            return {
                "validation_result": response.choices[0].message.content.strip(),
                "context_chunks_used": len(context)
            }
            
        except Exception as e:
            logger.error(f"Error validating response: {str(e)}")
            return {"validation_result": "Validation failed", "context_chunks_used": len(context)}
