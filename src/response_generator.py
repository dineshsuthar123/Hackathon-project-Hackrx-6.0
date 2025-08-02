"""
Response Generator Module
Generates structured JSON responses with explainable decision rationale
"""

import logging
import asyncio
from typing import Dict, Any, List, Tuple, Optional
from src.llm_handler import LLMHandler
import json

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates comprehensive responses with explanations and decision rationale"""
    
    def __init__(self, llm_handler: LLMHandler):
        self.llm_handler = llm_handler
        
    async def generate_response(self, 
                              query: str, 
                              relevant_clauses: List[Dict[str, Any]], 
                              document_content: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Generate a comprehensive response with explanation
        
        Args:
            query: User's question
            relevant_clauses: List of relevant document clauses
            document_content: Full document content and metadata
            
        Returns:
            Tuple of (answer, explanation_metadata)
        """
        try:
            logger.info(f"Generating response for query with {len(relevant_clauses)} relevant clauses")
            
            # Step 1: Analyze the query
            query_analysis = await self.llm_handler.analyze_query(query)
            
            # Step 2: Generate the main answer
            answer, explanation = await self.llm_handler.generate_answer(
                query, relevant_clauses, document_content
            )
            
            # Step 3: Enhance explanation with decision rationale
            enhanced_explanation = await self._create_decision_rationale(
                query, query_analysis, relevant_clauses, answer, explanation
            )
            
            # Step 4: Validate response quality
            validation = await self.llm_handler.validate_response_quality(
                query, answer, relevant_clauses
            )
            enhanced_explanation.update(validation)
            
            logger.info("Response generated successfully")
            return answer, enhanced_explanation
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}", {}
    
    async def _create_decision_rationale(self, 
                                       query: str,
                                       query_analysis: Dict[str, Any],
                                       relevant_clauses: List[Dict[str, Any]],
                                       answer: str,
                                       base_explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Create detailed decision rationale for explainable AI"""
        try:
            rationale = {
                **base_explanation,
                "query_analysis": query_analysis,
                "decision_process": [],
                "evidence_strength": self._assess_evidence_strength(relevant_clauses),
                "clause_analysis": [],
                "reasoning_chain": [],
                "confidence_assessment": {}
            }
            
            # Analyze each relevant clause
            for i, clause in enumerate(relevant_clauses[:5]):  # Top 5 clauses
                clause_analysis = await self._analyze_clause_contribution(
                    query, clause, answer
                )
                rationale["clause_analysis"].append({
                    "clause_rank": i + 1,
                    "text_preview": clause.get("text", "")[:150] + "...",
                    "similarity_score": clause.get("similarity_score", 0),
                    "contribution": clause_analysis.get("contribution", ""),
                    "relevance": clause_analysis.get("relevance", ""),
                    "source_info": clause.get("metadata", {})
                })
            
            # Create reasoning chain
            rationale["reasoning_chain"] = await self._create_reasoning_chain(
                query, relevant_clauses, answer
            )
            
            # Assess overall confidence
            rationale["confidence_assessment"] = self._assess_response_confidence(
                query_analysis, relevant_clauses, answer
            )
            
            # Decision process steps
            rationale["decision_process"] = [
                f"1. Analyzed query intent: {query_analysis.get('intent', 'general_inquiry')}",
                f"2. Identified key entities: {', '.join(query_analysis.get('entities', [])[:3])}",
                f"3. Found {len(relevant_clauses)} relevant document sections",
                f"4. Applied domain knowledge for {query_analysis.get('domain', 'general')} context",
                "5. Generated response based on highest-confidence evidence"
            ]
            
            return rationale
            
        except Exception as e:
            logger.error(f"Error creating decision rationale: {str(e)}")
            return base_explanation
    
    async def _analyze_clause_contribution(self, 
                                         query: str, 
                                         clause: Dict[str, Any], 
                                         answer: str) -> Dict[str, Any]:
        """Analyze how a specific clause contributes to the answer"""
        try:
            clause_text = clause.get("text", "")
            
            analysis_prompt = f"""Analyze how this document clause contributes to answering the user's question:
            
            Question: {query}
            
            Clause: {clause_text[:500]}...
            
            Answer generated: {answer[:200]}...
            
            Explain in 1-2 sentences:
            1. How this clause is relevant to the question
            2. What specific information it provides for the answer
            
            Be concise and specific."""
            
            response = await self.llm_handler.client.chat.completions.create(
                model=self.llm_handler.model,
                messages=[
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            return {
                "contribution": analysis_text,
                "relevance": "high" if clause.get("similarity_score", 0) > 0.7 else "medium" if clause.get("similarity_score", 0) > 0.5 else "low"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing clause contribution: {str(e)}")
            return {"contribution": "Analysis unavailable", "relevance": "unknown"}
    
    async def _create_reasoning_chain(self, 
                                    query: str, 
                                    relevant_clauses: List[Dict[str, Any]], 
                                    answer: str) -> List[str]:
        """Create a step-by-step reasoning chain"""
        try:
            if not relevant_clauses:
                return ["No relevant clauses found in document"]
            
            reasoning_steps = []
            
            # Step 1: Document search
            reasoning_steps.append(
                f"Searched document and found {len(relevant_clauses)} relevant sections"
            )
            
            # Step 2: Top clause analysis
            top_clause = relevant_clauses[0] if relevant_clauses else None
            if top_clause:
                similarity = top_clause.get("similarity_score", 0)
                reasoning_steps.append(
                    f"Identified most relevant section with {similarity:.2f} similarity score"
                )
            
            # Step 3: Information extraction
            if "yes" in answer.lower() or "no" in answer.lower():
                reasoning_steps.append("Determined definitive yes/no answer from policy clauses")
            else:
                reasoning_steps.append("Extracted specific details and conditions from policy text")
            
            # Step 4: Context consideration
            has_conditions = any(word in answer.lower() for word in ["provided", "condition", "if", "unless", "except"])
            if has_conditions:
                reasoning_steps.append("Identified relevant conditions and exceptions")
            
            # Step 5: Final synthesis
            reasoning_steps.append("Synthesized information to provide comprehensive answer")
            
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Error creating reasoning chain: {str(e)}")
            return ["Reasoning chain generation failed"]
    
    def _assess_evidence_strength(self, relevant_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the strength of evidence from relevant clauses"""
        if not relevant_clauses:
            return {"strength": "none", "score": 0, "description": "No relevant evidence found"}
        
        # Calculate average similarity score
        similarity_scores = [clause.get("similarity_score", 0) for clause in relevant_clauses]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)
        
        # Assess evidence quality
        if avg_similarity >= 0.8:
            strength = "very_strong"
            description = "Multiple highly relevant clauses with clear information"
        elif avg_similarity >= 0.6:
            strength = "strong"
            description = "Relevant clauses provide good supporting evidence"
        elif avg_similarity >= 0.4:
            strength = "moderate"
            description = "Some relevant information found, may require interpretation"
        else:
            strength = "weak"
            description = "Limited relevant information available"
        
        return {
            "strength": strength,
            "score": avg_similarity,
            "description": description,
            "clause_count": len(relevant_clauses)
        }
    
    def _assess_response_confidence(self, 
                                  query_analysis: Dict[str, Any], 
                                  relevant_clauses: List[Dict[str, Any]], 
                                  answer: str) -> Dict[str, Any]:
        """Assess confidence in the generated response"""
        confidence_factors = []
        confidence_score = 0.5  # Base confidence
        
        # Factor 1: Evidence strength
        if relevant_clauses:
            avg_similarity = sum(clause.get("similarity_score", 0) for clause in relevant_clauses) / len(relevant_clauses)
            confidence_score += (avg_similarity - 0.5) * 0.3
            if avg_similarity > 0.7:
                confidence_factors.append("High similarity with source material")
        
        # Factor 2: Query complexity
        complexity = query_analysis.get("complexity", "moderate")
        if complexity == "simple":
            confidence_score += 0.1
            confidence_factors.append("Straightforward question type")
        elif complexity == "complex":
            confidence_score -= 0.1
            confidence_factors.append("Complex question requiring interpretation")
        
        # Factor 3: Answer definitiveness
        if any(phrase in answer.lower() for phrase in ["yes,", "no,", "the policy covers", "is covered"]):
            confidence_score += 0.1
            confidence_factors.append("Definitive answer available")
        
        # Factor 4: Specific details
        if any(char.isdigit() for char in answer) or any(word in answer.lower() for word in ["months", "days", "years", "%", "amount"]):
            confidence_score += 0.1
            confidence_factors.append("Specific quantitative details provided")
        
        # Factor 5: Uncertainty indicators
        uncertainty_phrases = ["may", "might", "possibly", "unclear", "not specified", "depends"]
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            confidence_score -= 0.2
            confidence_factors.append("Some uncertainty in source material")
        
        # Normalize confidence score
        confidence_score = max(0.1, min(0.95, confidence_score))
        
        # Determine confidence level
        if confidence_score >= 0.8:
            level = "high"
        elif confidence_score >= 0.6:
            level = "medium"
        else:
            level = "low"
        
        return {
            "score": confidence_score,
            "level": level,
            "factors": confidence_factors,
            "recommendation": self._get_confidence_recommendation(level)
        }
    
    def _get_confidence_recommendation(self, confidence_level: str) -> str:
        """Get recommendation based on confidence level"""
        recommendations = {
            "high": "Answer is well-supported by document content and can be relied upon",
            "medium": "Answer is reasonably supported but may benefit from additional verification",
            "low": "Answer has limited support; recommend consulting original document or subject matter expert"
        }
        return recommendations.get(confidence_level, "Unable to assess confidence")
    
    async def generate_structured_json_response(self, 
                                              answers: List[str], 
                                              explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate the final structured JSON response"""
        try:
            response = {
                "answers": answers,
                "metadata": {
                    "total_questions": len(answers),
                    "processing_timestamp": str(asyncio.get_event_loop().time()),
                    "explanations": explanations,
                    "system_info": {
                        "model_used": self.llm_handler.get_model_name(),
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "processing_method": "semantic_search_with_llm"
                    }
                }
            }
            
            # Add aggregate confidence metrics
            if explanations:
                confidence_scores = [
                    exp.get("confidence_assessment", {}).get("score", 0.5) 
                    for exp in explanations
                ]
                response["metadata"]["aggregate_confidence"] = {
                    "average_score": sum(confidence_scores) / len(confidence_scores),
                    "min_score": min(confidence_scores),
                    "max_score": max(confidence_scores)
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating structured JSON response: {str(e)}")
            return {
                "answers": answers,
                "metadata": {
                    "total_questions": len(answers),
                    "error": f"Metadata generation failed: {str(e)}"
                }
            }
