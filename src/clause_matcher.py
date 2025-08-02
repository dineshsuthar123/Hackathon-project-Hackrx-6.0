"""
Clause Matcher Module
Handles semantic matching between queries and document clauses
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from src.embedding_engine import EmbeddingEngine

logger = logging.getLogger(__name__)

class ClauseMatcher:
    """Matches user queries with relevant document clauses using semantic similarity"""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
        self.similarity_threshold = 0.3
        self.max_results = 10
        
    async def find_relevant_clauses(self, 
                                  query: str, 
                                  indexed_content: Dict[str, Any],
                                  top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find document clauses most relevant to the user query
        
        Args:
            query: User's natural language query
            indexed_content: Content that has been indexed with embeddings
            top_k: Number of top results to return
            
        Returns:
            List of relevant clauses with similarity scores and metadata
        """
        try:
            logger.info(f"Finding relevant clauses for query: {query[:50]}...")
            
            # Step 1: Direct semantic search
            direct_results = await self.embedding_engine.search_similar(
                query, top_k=top_k * 2, threshold=self.similarity_threshold
            )
            
            # Step 2: Query expansion and refinement
            expanded_queries = self._expand_query(query)
            expanded_results = []
            
            for expanded_query in expanded_queries:
                exp_results = await self.embedding_engine.search_similar(
                    expanded_query, top_k=top_k, threshold=self.similarity_threshold * 0.8
                )
                expanded_results.extend(exp_results)
            
            # Step 3: Combine and deduplicate results
            all_results = direct_results + expanded_results
            unique_results = self._deduplicate_results(all_results)
            
            # Step 4: Re-rank based on multiple factors
            ranked_results = self._rerank_results(query, unique_results)
            
            # Step 5: Filter and return top results
            final_results = ranked_results[:top_k]
            
            logger.info(f"Found {len(final_results)} relevant clauses")
            return final_results
            
        except Exception as e:
            logger.error(f"Error finding relevant clauses: {str(e)}")
            return []
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand the query with synonyms and related terms for better matching
        """
        expanded_queries = []
        
        # Domain-specific expansions
        insurance_synonyms = {
            'cover': ['coverage', 'benefit', 'protection', 'include'],
            'surgery': ['operation', 'procedure', 'surgical treatment'],
            'waiting period': ['waiting time', 'waiting duration', 'wait period'],
            'premium': ['payment', 'cost', 'fee', 'charge'],
            'claim': ['reimbursement', 'compensation', 'payout'],
            'deductible': ['excess', 'co-payment', 'out-of-pocket'],
            'policy': ['insurance', 'plan', 'scheme', 'coverage'],
            'maternity': ['pregnancy', 'childbirth', 'delivery'],
            'pre-existing': ['existing condition', 'prior condition'],
            'exclude': ['exclusion', 'not covered', 'limitation'],
            'hospital': ['medical facility', 'healthcare facility'],
            'treatment': ['therapy', 'care', 'medical care']
        }
        
        legal_synonyms = {
            'contract': ['agreement', 'document', 'terms'],
            'clause': ['section', 'provision', 'term'],
            'liability': ['responsibility', 'obligation'],
            'breach': ['violation', 'non-compliance'],
            'terminate': ['end', 'cancel', 'discontinue']
        }
        
        hr_synonyms = {
            'employee': ['worker', 'staff member', 'personnel'],
            'leave': ['time off', 'absence', 'vacation'],
            'salary': ['wage', 'compensation', 'pay'],
            'benefit': ['perk', 'advantage', 'entitlement']
        }
        
        # Combine all synonyms
        all_synonyms = {**insurance_synonyms, **legal_synonyms, **hr_synonyms}
        
        # Create expanded queries
        query_lower = query.lower()
        
        for term, synonyms in all_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query != query_lower:
                        expanded_queries.append(expanded_query)
        
        # Add keyword-focused queries
        keywords = self._extract_keywords(query)
        if keywords:
            expanded_queries.append(' '.join(keywords))
        
        return expanded_queries[:3]  # Limit to top 3 expansions
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query"""
        # Remove common stop words
        stop_words = {
            'the', 'is', 'are', 'was', 'were', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'this', 'that', 'these', 'those', 'what', 'how', 'when', 'where', 'why',
            'does', 'do', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might'
        }
        
        words = query.lower().split()
        keywords = [word.strip('.,!?;:') for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on text similarity"""
        if not results:
            return []
        
        unique_results = []
        seen_texts = set()
        
        for result in results:
            text = result.get('text', '').strip()
            # Use first 100 characters as a key for deduplication
            text_key = text[:100].lower()
            
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)
        
        return unique_results
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-rank results based on multiple factors:
        1. Semantic similarity score
        2. Text length (prefer more substantial content)
        3. Source type (prefer certain sources)
        4. Keyword overlap
        """
        if not results:
            return []
        
        query_words = set(query.lower().split())
        
        for result in results:
            score = result.get('similarity_score', 0)
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            
            # Factor 1: Base similarity score (weight: 0.5)
            base_score = score * 0.5
            
            # Factor 2: Text length bonus (weight: 0.2)
            length_score = min(len(text) / 1000, 1.0) * 0.2
            
            # Factor 3: Source type bonus (weight: 0.2)
            source_score = 0
            source_type = metadata.get('source', '')
            if source_type == 'section':
                source_score = 0.2
            elif source_type == 'main_text':
                source_score = 0.15
            elif source_type == 'table':
                source_score = 0.1
            
            # Factor 4: Keyword overlap bonus (weight: 0.1)
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words) / len(query_words) if query_words else 0
            keyword_score = overlap * 0.1
            
            # Calculate final score
            final_score = base_score + length_score + source_score + keyword_score
            result['final_score'] = final_score
        
        # Sort by final score
        return sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
    
    async def find_specific_clause_types(self, 
                                       query: str, 
                                       clause_types: List[str],
                                       indexed_content: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find specific types of clauses (e.g., exclusions, conditions, definitions)
        
        Args:
            query: User query
            clause_types: Types of clauses to look for
            indexed_content: Indexed document content
            
        Returns:
            Dictionary mapping clause types to relevant clauses
        """
        try:
            results = {}
            
            for clause_type in clause_types:
                # Create targeted query for this clause type
                targeted_query = f"{query} {clause_type}"
                
                clause_results = await self.find_relevant_clauses(
                    targeted_query, indexed_content, top_k=3
                )
                
                # Filter results that actually relate to the clause type
                filtered_results = []
                for result in clause_results:
                    text = result.get('text', '').lower()
                    if any(keyword in text for keyword in self._get_clause_keywords(clause_type)):
                        filtered_results.append(result)
                
                results[clause_type] = filtered_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error finding specific clause types: {str(e)}")
            return {}
    
    def _get_clause_keywords(self, clause_type: str) -> List[str]:
        """Get keywords associated with specific clause types"""
        keywords_map = {
            'exclusions': ['exclude', 'excluded', 'exclusion', 'not covered', 'limitation', 'restriction'],
            'conditions': ['condition', 'requirement', 'must', 'shall', 'provided that'],
            'definitions': ['means', 'defined as', 'definition', 'shall mean', 'refers to'],
            'benefits': ['benefit', 'coverage', 'covered', 'entitled', 'reimbursement'],
            'waiting_periods': ['waiting period', 'wait', 'after', 'months', 'days', 'continuous'],
            'premiums': ['premium', 'payment', 'due', 'cost', 'fee', 'charge'],
            'claims': ['claim', 'reimbursement', 'submission', 'documentation', 'proof']
        }
        
        return keywords_map.get(clause_type, [])
    
    async def validate_clause_relevance(self, 
                                      query: str, 
                                      clause: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate how relevant a clause is to the query
        
        Args:
            query: User query
            clause: Clause to validate
            
        Returns:
            Validation results with confidence scores
        """
        try:
            clause_text = clause.get('text', '')
            similarity_score = clause.get('similarity_score', 0)
            
            # Check for direct keyword matches
            query_words = set(query.lower().split())
            clause_words = set(clause_text.lower().split())
            keyword_overlap = len(query_words & clause_words) / len(query_words) if query_words else 0
            
            # Check for negation patterns that might affect relevance
            negation_patterns = ['not', 'except', 'excluding', 'does not', 'will not', 'cannot']
            has_negation = any(pattern in clause_text.lower() for pattern in negation_patterns)
            
            # Calculate confidence score
            confidence = (similarity_score + keyword_overlap) / 2
            if has_negation:
                confidence *= 0.8  # Reduce confidence for potentially negative clauses
            
            return {
                'relevance_score': confidence,
                'keyword_overlap': keyword_overlap,
                'has_negation': has_negation,
                'confidence_level': 'high' if confidence > 0.7 else 'medium' if confidence > 0.4 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error validating clause relevance: {str(e)}")
            return {'relevance_score': 0, 'confidence_level': 'low'}
    
    def set_similarity_threshold(self, threshold: float):
        """Set the similarity threshold for clause matching"""
        self.similarity_threshold = max(0.1, min(1.0, threshold))
    
    def set_max_results(self, max_results: int):
        """Set the maximum number of results to return"""
        self.max_results = max(1, min(50, max_results))
