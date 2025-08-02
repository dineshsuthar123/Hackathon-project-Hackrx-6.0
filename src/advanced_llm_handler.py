"""
Advanced LLM Handler with Enhanced Accuracy
Multi-model support, better prompting, and response validation
"""

import asyncio
import logging
import openai
import os
from typing import Dict, Any, List, Optional, Tuple, Union
from dotenv import load_dotenv
import tiktoken
import json
import re
from dataclasses import dataclass
from enum import Enum

load_dotenv()

logger = logging.getLogger(__name__)

class QueryType(Enum):
    DEFINITION = "definition"
    COVERAGE = "coverage"
    EXCLUSION = "exclusion"
    PROCEDURE = "procedure"
    FINANCIAL = "financial"
    TEMPORAL = "temporal"
    COMPARISON = "comparison"
    GENERAL = "general"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    entities: List[str]
    keywords: List[str]
    domain: str
    complexity: str
    expected_response_type: str
    confidence: float

class AdvancedLLMHandler:
    """Advanced LLM handler with enhanced accuracy and multi-model support"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.primary_model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        self.fallback_model = "gpt-3.5-turbo"
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4000"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.1"))
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
        # Initialize tokenizers
        try:
            self.primary_tokenizer = tiktoken.encoding_for_model(self.primary_model)
        except KeyError:
            self.primary_tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Enhanced prompts for different query types
        self.system_prompts = {
            QueryType.DEFINITION: """You are an expert document analyst specializing in precise definitions and terminology. 
            Your task is to provide clear, accurate definitions based solely on the provided document context.
            
            Guidelines:
            - Extract exact definitions from the document
            - If multiple definitions exist, provide the most comprehensive one
            - Include any conditions or qualifications mentioned
            - Use the exact terminology from the document
            - If no definition is found, state this clearly""",
            
            QueryType.COVERAGE: """You are an expert insurance and policy analyst specializing in coverage analysis.
            Your task is to determine what is covered based on the provided document context.
            
            Guidelines:
            - Clearly state what IS covered
            - Include any conditions, limits, or requirements
            - Mention any sub-limits or caps
            - Include waiting periods if applicable
            - Be specific about amounts, percentages, and time periods
            - If coverage is conditional, explain the conditions clearly""",
            
            QueryType.EXCLUSION: """You are an expert policy analyst specializing in exclusions and limitations.
            Your task is to identify what is NOT covered based on the provided document context.
            
            Guidelines:
            - Clearly state what is excluded or not covered
            - Include any exceptions to exclusions
            - Mention specific conditions that void coverage
            - Be precise about exclusion periods or circumstances
            - If there are partial exclusions, explain them clearly""",
            
            QueryType.PROCEDURE: """You are an expert process analyst specializing in procedures and workflows.
            Your task is to explain step-by-step processes based on the provided document context.
            
            Guidelines:
            - Provide clear, sequential steps
            - Include all required documents or information
            - Mention time limits or deadlines
            - Include contact information if provided
            - Explain any alternative procedures
            - Be specific about requirements and conditions""",
            
            QueryType.FINANCIAL: """You are an expert financial analyst specializing in costs, premiums, and financial terms.
            Your task is to provide accurate financial information based on the provided document context.
            
            Guidelines:
            - State exact amounts, percentages, and rates
            - Include any conditions that affect costs
            - Mention payment schedules or options
            - Include any discounts or penalties
            - Be precise about currency and units
            - Explain calculation methods if provided""",
            
            QueryType.TEMPORAL: """You are an expert analyst specializing in time-related terms and conditions.
            Your task is to provide accurate information about time periods, deadlines, and schedules.
            
            Guidelines:
            - State exact time periods (days, months, years)
            - Include start and end conditions
            - Mention any grace periods or extensions
            - Explain how time periods are calculated
            - Include any exceptions or special circumstances
            - Be precise about business days vs. calendar days""",
            
            QueryType.GENERAL: """You are an expert document analyst with comprehensive knowledge across multiple domains.
            Your task is to provide accurate, helpful answers based solely on the provided document context.
            
            Guidelines:
            - Base your answer entirely on the provided context
            - Be specific and include relevant details
            - If information is incomplete, state what is available
            - Include any relevant conditions or qualifications
            - Use professional, clear language
            - If the context doesn't contain the answer, state this clearly"""
        }
        
        # Domain-specific knowledge bases
        self.domain_keywords = {
            'insurance': [
                'policy', 'coverage', 'premium', 'deductible', 'claim', 'benefit',
                'exclusion', 'waiting period', 'grace period', 'sum insured',
                'policyholder', 'insured', 'beneficiary', 'rider', 'endorsement'
            ],
            'legal': [
                'contract', 'agreement', 'clause', 'provision', 'terms', 'conditions',
                'liability', 'obligation', 'rights', 'duties', 'breach', 'remedy',
                'jurisdiction', 'governing law', 'dispute resolution'
            ],
            'hr': [
                'employee', 'employer', 'benefits', 'compensation', 'leave', 'vacation',
                'sick leave', 'performance', 'evaluation', 'termination', 'resignation',
                'policy', 'procedure', 'handbook', 'code of conduct'
            ],
            'financial': [
                'revenue', 'income', 'profit', 'loss', 'assets', 'liabilities',
                'equity', 'cash flow', 'investment', 'return', 'risk', 'dividend',
                'interest', 'principal', 'balance sheet', 'income statement'
            ]
        }
    
    async def analyze_query_advanced(self, query: str) -> QueryAnalysis:
        """Advanced query analysis with enhanced accuracy"""
        try:
            # Use LLM for sophisticated query analysis
            analysis_prompt = f"""Analyze this query and provide a structured analysis:

Query: "{query}"

Provide a JSON response with the following structure:
{{
    "query_type": "definition|coverage|exclusion|procedure|financial|temporal|comparison|general",
    "entities": ["list", "of", "key", "entities"],
    "keywords": ["relevant", "search", "keywords"],
    "domain": "insurance|legal|hr|financial|general",
    "complexity": "simple|moderate|complex",
    "expected_response_type": "yes_no|amount|definition|procedure|list|explanation",
    "confidence": 0.95
}}

Consider:
- What type of information is being requested?
- What domain does this query belong to?
- What entities or concepts are mentioned?
- How complex is the query?
- What type of response would best answer this query?"""

            response = await self.client.chat.completions.create(
                model=self.primary_model,
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,
                max_tokens=500
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            try:
                analysis_data = json.loads(analysis_text)
                return QueryAnalysis(
                    query_type=QueryType(analysis_data.get('query_type', 'general')),
                    entities=analysis_data.get('entities', []),
                    keywords=analysis_data.get('keywords', []),
                    domain=analysis_data.get('domain', 'general'),
                    complexity=analysis_data.get('complexity', 'moderate'),
                    expected_response_type=analysis_data.get('expected_response_type', 'explanation'),
                    confidence=analysis_data.get('confidence', 0.8)
                )
            except (json.JSONDecodeError, ValueError):
                # Fallback to rule-based analysis
                return self._fallback_query_analysis(query)
                
        except Exception as e:
            logger.warning(f"Advanced query analysis failed: {e}, using fallback")
            return self._fallback_query_analysis(query)
    
    def _fallback_query_analysis(self, query: str) -> QueryAnalysis:
        """Fallback rule-based query analysis"""
        query_lower = query.lower()
        
        # Determine query type
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'means']):
            query_type = QueryType.DEFINITION
        elif any(word in query_lower for word in ['covered', 'coverage', 'benefit', 'include']):
            query_type = QueryType.COVERAGE
        elif any(word in query_lower for word in ['excluded', 'exclusion', 'not covered', 'limitation']):
            query_type = QueryType.EXCLUSION
        elif any(word in query_lower for word in ['how to', 'process', 'procedure', 'steps']):
            query_type = QueryType.PROCEDURE
        elif any(word in query_lower for word in ['cost', 'price', 'premium', 'amount', 'fee']):
            query_type = QueryType.FINANCIAL
        elif any(word in query_lower for word in ['when', 'period', 'time', 'duration', 'waiting']):
            query_type = QueryType.TEMPORAL
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            query_type = QueryType.COMPARISON
        else:
            query_type = QueryType.GENERAL
        
        # Determine domain
        domain = 'general'
        for domain_name, keywords in self.domain_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                domain = domain_name
                break
        
        # Extract entities and keywords
        words = query.split()
        entities = [word for word in words if len(word) > 3 and word.isalpha()]
        keywords = [word.lower() for word in words if len(word) > 2]
        
        # Determine complexity
        complexity = 'simple' if len(words) <= 8 else 'complex' if len(words) > 15 else 'moderate'
        
        # Expected response type
        if query_lower.startswith(('is', 'does', 'can', 'will')):
            expected_response_type = 'yes_no'
        elif 'how much' in query_lower or 'what is the cost' in query_lower:
            expected_response_type = 'amount'
        elif query_type == QueryType.DEFINITION:
            expected_response_type = 'definition'
        elif query_type == QueryType.PROCEDURE:
            expected_response_type = 'procedure'
        else:
            expected_response_type = 'explanation'
        
        return QueryAnalysis(
            query_type=query_type,
            entities=entities[:5],  # Top 5 entities
            keywords=keywords[:10],  # Top 10 keywords
            domain=domain,
            complexity=complexity,
            expected_response_type=expected_response_type,
            confidence=0.7
        )
    
    async def generate_answer_advanced(self, 
                                     query: str, 
                                     relevant_context: List[Dict[str, Any]], 
                                     document_metadata: Dict[str, Any],
                                     query_analysis: Optional[QueryAnalysis] = None) -> Tuple[str, Dict[str, Any]]:
        """Generate advanced answer with enhanced accuracy"""
        try:
            if not query_analysis:
                query_analysis = await self.analyze_query_advanced(query)
            
            # Prepare enhanced context
            context_text = self._prepare_enhanced_context(relevant_context, query_analysis)
            
            # Check token limits and truncate if necessary
            if self._count_tokens(context_text) > self.max_tokens // 2:
                context_text = self._truncate_context_intelligently(context_text, self.max_tokens // 2, query_analysis)
            
            # Select appropriate system prompt
            system_prompt = self.system_prompts.get(query_analysis.query_type, self.system_prompts[QueryType.GENERAL])
            
            # Create enhanced user prompt
            user_prompt = self._create_enhanced_user_prompt(query, context_text, query_analysis)
            
            # Generate response with primary model
            try:
                response = await self._generate_with_model(
                    self.primary_model, system_prompt, user_prompt, query_analysis
                )
            except Exception as e:
                logger.warning(f"Primary model failed: {e}, trying fallback")
                response = await self._generate_with_model(
                    self.fallback_model, system_prompt, user_prompt, query_analysis
                )
            
            answer = response.choices[0].message.content.strip()
            
            # Validate and enhance the response
            validated_answer = await self._validate_and_enhance_response(
                answer, query, relevant_context, query_analysis
            )
            
            # Generate enhanced explanation
            explanation = await self._generate_enhanced_explanation(
                query, relevant_context, validated_answer, query_analysis
            )
            
            return validated_answer, explanation
            
        except Exception as e:
            logger.error(f"Error generating advanced answer: {str(e)}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}", {}
    
    def _prepare_enhanced_context(self, relevant_context: List[Dict[str, Any]], query_analysis: QueryAnalysis) -> str:
        """Prepare enhanced context based on query analysis"""
        if not relevant_context:
            return ""
        
        # Sort context by relevance and category match
        sorted_context = sorted(
            relevant_context,
            key=lambda x: (
                x.get('enhanced_score', x.get('similarity_score', 0)),
                1.0 if x.get('category') == query_analysis.query_type.value else 0.5
            ),
            reverse=True
        )
        
        context_parts = []
        for i, chunk in enumerate(sorted_context):
            chunk_text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            # Add source information with enhanced details
            source_info = f"[Source {i+1}"
            if metadata.get('section_name'):
                source_info += f", Section: {metadata['section_name']}"
            if metadata.get('content_type'):
                source_info += f", Type: {metadata['content_type']}"
            if chunk.get('category'):
                source_info += f", Category: {chunk['category']}"
            
            # Add confidence indicators
            similarity = chunk.get('similarity_score', 0)
            if similarity > 0.8:
                source_info += ", High Relevance"
            elif similarity > 0.6:
                source_info += ", Medium Relevance"
            
            source_info += "]"
            
            context_parts.append(f"{source_info}\n{chunk_text}\n")
        
        return "\n".join(context_parts)
    
    def _truncate_context_intelligently(self, context: str, max_tokens: int, query_analysis: QueryAnalysis) -> str:
        """Intelligently truncate context based on query analysis"""
        try:
            tokens = self.primary_tokenizer.encode(context)
            if len(tokens) <= max_tokens:
                return context
            
            # Split context into sections
            sections = context.split('[Source')
            if len(sections) <= 1:
                # Fallback to simple truncation
                truncated_tokens = tokens[:max_tokens]
                return self.primary_tokenizer.decode(truncated_tokens)
            
            # Prioritize sections based on query type
            prioritized_sections = []
            remaining_tokens = max_tokens
            
            for section in sections[1:]:  # Skip first empty section
                section_text = '[Source' + section
                section_tokens = len(self.primary_tokenizer.encode(section_text))
                
                if section_tokens <= remaining_tokens:
                    prioritized_sections.append(section_text)
                    remaining_tokens -= section_tokens
                else:
                    # Truncate this section if it's the first one
                    if not prioritized_sections:
                        truncated_tokens = self.primary_tokenizer.encode(section_text)[:remaining_tokens]
                        prioritized_sections.append(self.primary_tokenizer.decode(truncated_tokens))
                    break
            
            return '\n'.join(prioritized_sections)
            
        except Exception as e:
            logger.warning(f"Intelligent truncation failed: {e}, using simple truncation")
            return self._truncate_context(context, max_tokens)
    
    def _create_enhanced_user_prompt(self, query: str, context: str, query_analysis: QueryAnalysis) -> str:
        """Create enhanced user prompt based on query analysis"""
        base_prompt = f"""Document Context:
{context}

Question: {query}

Based on the document context provided above, please provide a comprehensive answer."""
        
        # Add specific instructions based on query type
        if query_analysis.query_type == QueryType.DEFINITION:
            base_prompt += "\n\nProvide the exact definition as stated in the document. If multiple definitions exist, provide the most complete one."
        
        elif query_analysis.query_type == QueryType.COVERAGE:
            base_prompt += "\n\nClearly state what is covered, including any conditions, limits, waiting periods, and specific amounts or percentages."
        
        elif query_analysis.query_type == QueryType.EXCLUSION:
            base_prompt += "\n\nClearly state what is excluded or not covered, including any exceptions to the exclusions."
        
        elif query_analysis.query_type == QueryType.PROCEDURE:
            base_prompt += "\n\nProvide step-by-step instructions, including required documents, deadlines, and contact information."
        
        elif query_analysis.query_type == QueryType.FINANCIAL:
            base_prompt += "\n\nProvide specific amounts, percentages, rates, and any conditions that affect the financial terms."
        
        elif query_analysis.query_type == QueryType.TEMPORAL:
            base_prompt += "\n\nProvide exact time periods, including how they are calculated and any exceptions or extensions."
        
        # Add domain-specific instructions
        if query_analysis.domain == 'insurance':
            base_prompt += "\n\nEnsure you mention any relevant policy conditions, exclusions, or limitations."
        elif query_analysis.domain == 'legal':
            base_prompt += "\n\nEnsure you mention any relevant legal conditions, obligations, or rights."
        
        return base_prompt
    
    async def _generate_with_model(self, model: str, system_prompt: str, user_prompt: str, query_analysis: QueryAnalysis):
        """Generate response with specified model"""
        # Adjust parameters based on query analysis
        temperature = self.temperature
        if query_analysis.query_type in [QueryType.DEFINITION, QueryType.FINANCIAL, QueryType.TEMPORAL]:
            temperature = 0.05  # More deterministic for factual queries
        elif query_analysis.query_type == QueryType.PROCEDURE:
            temperature = 0.1   # Slightly more deterministic for procedures
        
        max_tokens = min(self.max_tokens, 1000)  # Reasonable limit for responses
        if query_analysis.expected_response_type == 'yes_no':
            max_tokens = 200
        elif query_analysis.expected_response_type in ['amount', 'definition']:
            max_tokens = 300
        
        return await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    async def _validate_and_enhance_response(self, 
                                           answer: str, 
                                           query: str, 
                                           context: List[Dict[str, Any]], 
                                           query_analysis: QueryAnalysis) -> str:
        """Validate and enhance the response"""
        # Basic validation
        if not answer or len(answer.strip()) < 10:
            return "I apologize, but I couldn't generate a sufficient response based on the available information."
        
        # Check for hallucination indicators
        if self._detect_potential_hallucination(answer, context):
            logger.warning("Potential hallucination detected, requesting regeneration")
            # Could implement regeneration logic here
        
        # Enhance response based on query type
        enhanced_answer = await self._enhance_response_by_type(answer, query_analysis)
        
        return enhanced_answer
    
    def _detect_potential_hallucination(self, answer: str, context: List[Dict[str, Any]]) -> bool:
        """Detect potential hallucination in the response"""
        # Simple heuristic: check if answer contains specific details not in context
        answer_lower = answer.lower()
        
        # Extract specific numbers, dates, names from answer
        import re
        answer_specifics = set()
        answer_specifics.update(re.findall(r'\d+(?:\.\d+)?%', answer))  # Percentages
        answer_specifics.update(re.findall(r'\$[\d,]+', answer))        # Dollar amounts
        answer_specifics.update(re.findall(r'inr\s*[\d,]+', answer_lower))  # INR amounts
        answer_specifics.update(re.findall(r'\d+\s*(?:days?|months?|years?)', answer_lower))  # Time periods
        
        # Check if these specifics appear in context
        context_text = ' '.join([chunk.get('text', '') for chunk in context]).lower()
        
        for specific in answer_specifics:
            if specific.lower() not in context_text:
                return True  # Potential hallucination
        
        return False
    
    async def _enhance_response_by_type(self, answer: str, query_analysis: QueryAnalysis) -> str:
        """Enhance response based on query type"""
        if query_analysis.query_type == QueryType.FINANCIAL:
            # Ensure financial responses include currency and are properly formatted
            answer = self._format_financial_response(answer)
        
        elif query_analysis.query_type == QueryType.TEMPORAL:
            # Ensure temporal responses are clear about time periods
            answer = self._format_temporal_response(answer)
        
        elif query_analysis.expected_response_type == 'yes_no':
            # Ensure yes/no responses start clearly
            answer = self._format_yes_no_response(answer)
        
        return answer
    
    def _format_financial_response(self, answer: str) -> str:
        """Format financial responses for clarity"""
        # Add currency symbols if missing
        import re
        
        # Find numbers that might be amounts
        numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b', answer)
        
        # This is a simplified implementation
        # In practice, you'd want more sophisticated formatting
        return answer
    
    def _format_temporal_response(self, answer: str) -> str:
        """Format temporal responses for clarity"""
        # Ensure time periods are clearly stated
        return answer
    
    def _format_yes_no_response(self, answer: str) -> str:
        """Format yes/no responses for clarity"""
        answer_lower = answer.lower().strip()
        
        if answer_lower.startswith('yes'):
            return answer
        elif answer_lower.startswith('no'):
            return answer
        elif 'yes' in answer_lower[:50]:
            return f"Yes, {answer}"
        elif 'no' in answer_lower[:50] or 'not' in answer_lower[:50]:
            return f"No, {answer}"
        else:
            return answer
    
    async def _generate_enhanced_explanation(self, 
                                           query: str, 
                                           context: List[Dict[str, Any]], 
                                           answer: str, 
                                           query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """Generate enhanced explanation with detailed metadata"""
        try:
            explanation = {
                "query_analysis": query_analysis.__dict__,
                "sources_used": len(context),
                "confidence_indicators": [],
                "reasoning_chain": [],
                "evidence_strength": self._assess_evidence_strength(context),
                "response_validation": {},
                "context_analysis": {}
            }
            
            # Analyze context quality
            if context:
                avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context) / len(context)
                explanation["context_analysis"] = {
                    "average_similarity": avg_similarity,
                    "total_chunks": len(context),
                    "categories_used": list(set(chunk.get('category', 'unknown') for chunk in context)),
                    "high_confidence_chunks": len([c for c in context if c.get('similarity_score', 0) > 0.8])
                }
                
                # Confidence indicators
                if avg_similarity > 0.8:
                    explanation["confidence_indicators"].append("High semantic similarity with source material")
                if len(context) >= 3:
                    explanation["confidence_indicators"].append("Multiple supporting sources found")
                if query_analysis.confidence > 0.8:
                    explanation["confidence_indicators"].append("High confidence in query understanding")
            
            # Generate reasoning chain
            explanation["reasoning_chain"] = await self._generate_reasoning_chain(
                query, context, answer, query_analysis
            )
            
            # Response validation
            explanation["response_validation"] = {
                "length_appropriate": 50 <= len(answer) <= 1000,
                "contains_specifics": bool(re.search(r'\d+|%|\$|INR', answer)),
                "addresses_query_type": query_analysis.query_type.value in answer.lower(),
                "hallucination_risk": "low"  # Could implement more sophisticated detection
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating enhanced explanation: {str(e)}")
            return {"error": f"Explanation generation failed: {str(e)}"}
    
    async def _generate_reasoning_chain(self, 
                                      query: str, 
                                      context: List[Dict[str, Any]], 
                                      answer: str, 
                                      query_analysis: QueryAnalysis) -> List[str]:
        """Generate detailed reasoning chain"""
        reasoning_steps = []
        
        # Step 1: Query understanding
        reasoning_steps.append(
            f"Identified query as {query_analysis.query_type.value} type in {query_analysis.domain} domain"
        )
        
        # Step 2: Context analysis
        if context:
            reasoning_steps.append(
                f"Found {len(context)} relevant document sections with average similarity {sum(c.get('similarity_score', 0) for c in context) / len(context):.2f}"
            )
        
        # Step 3: Information extraction
        if query_analysis.query_type == QueryType.DEFINITION:
            reasoning_steps.append("Extracted definition from document sections")
        elif query_analysis.query_type == QueryType.COVERAGE:
            reasoning_steps.append("Analyzed coverage terms and conditions")
        elif query_analysis.query_type == QueryType.FINANCIAL:
            reasoning_steps.append("Extracted financial amounts and conditions")
        
        # Step 4: Response synthesis
        reasoning_steps.append("Synthesized information to provide comprehensive answer")
        
        return reasoning_steps
    
    def _assess_evidence_strength(self, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the strength of evidence from context"""
        if not context:
            return {"strength": "none", "score": 0, "description": "No relevant evidence found"}
        
        similarity_scores = [chunk.get('similarity_score', 0) for chunk in context]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Consider enhanced scores if available
        enhanced_scores = [chunk.get('enhanced_score', chunk.get('similarity_score', 0)) for chunk in context]
        avg_enhanced = sum(enhanced_scores) / len(enhanced_scores)
        
        final_score = max(avg_similarity, avg_enhanced)
        
        if final_score >= 0.85:
            strength = "very_strong"
            description = "Multiple highly relevant sections with strong semantic match"
        elif final_score >= 0.7:
            strength = "strong"
            description = "Relevant sections provide good supporting evidence"
        elif final_score >= 0.5:
            strength = "moderate"
            description = "Some relevant information found, moderate confidence"
        else:
            strength = "weak"
            description = "Limited relevant information available"
        
        return {
            "strength": strength,
            "score": final_score,
            "description": description,
            "source_count": len(context),
            "category_diversity": len(set(chunk.get('category', 'unknown') for chunk in context))
        }
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.primary_tokenizer.encode(text))
        except:
            return len(text) // 4  # Fallback estimation
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Simple context truncation fallback"""
        try:
            tokens = self.primary_tokenizer.encode(context)
            if len(tokens) <= max_tokens:
                return context
            
            truncated_tokens = tokens[:max_tokens]
            return self.primary_tokenizer.decode(truncated_tokens)
        except:
            max_chars = max_tokens * 4
            return context[:max_chars] + "..." if len(context) > max_chars else context
    
    def get_model_name(self) -> str:
        """Get the current primary model name"""
        return self.primary_model
    
    async def validate_response_quality_advanced(self, 
                                               query: str, 
                                               answer: str, 
                                               context: List[Dict[str, Any]],
                                               query_analysis: QueryAnalysis) -> Dict[str, Any]:
        """Advanced response quality validation"""
        try:
            validation_prompt = f"""Evaluate the quality of this answer for the given question:
            
            Question Type: {query_analysis.query_type.value}
            Domain: {query_analysis.domain}
            Expected Response: {query_analysis.expected_response_type}
            
            Question: {query}
            Answer: {answer}
            
            Rate the answer on a scale of 1-10 for:
            1. Accuracy (based on context)
            2. Completeness (fully addresses question)
            3. Clarity (easy to understand)
            4. Specificity (includes relevant details)
            5. Appropriateness (matches expected response type)
            
            Provide a brief assessment and overall score."""
            
            response = await self.client.chat.completions.create(
                model=self.fallback_model,  # Use cheaper model for validation
                messages=[
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            return {
                "validation_result": response.choices[0].message.content.strip(),
                "context_chunks_used": len(context),
                "query_analysis": query_analysis.__dict__
            }
            
        except Exception as e:
            logger.error(f"Error in advanced response validation: {str(e)}")
            return {
                "validation_result": "Validation failed",
                "error": str(e),
                "context_chunks_used": len(context)
            }