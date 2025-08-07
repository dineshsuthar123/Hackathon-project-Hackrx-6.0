"""
PROTOCOL 7.0: ReAct FRAMEWORK - MULTI-STEP REASONING ENGINE
Reason + Act framework for complex query decomposition and solution
"""

import logging
import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

class ReasoningStep(Enum):
    THOUGHT = "THOUGHT"
    ACTION = "ACTION"
    OBSERVATION = "OBSERVATION"
    FINAL_ANSWER = "FINAL_ANSWER"

@dataclass
class ReActStep:
    step_type: ReasoningStep
    content: str
    timestamp: float
    step_number: int

class PrecisionSearchTool:
    """Enhanced search tool for ReAct framework"""
    
    def __init__(self, document_content: str, logger):
        self.document_content = document_content
        self.logger = logger
        self.search_cache = {}
    
    async def search(self, query: str) -> str:
        """Precision search with semantic understanding"""
        
        # Cache check
        cache_key = query.lower().strip()
        if cache_key in self.search_cache:
            self.logger.info(f"🎯 CACHE HIT: {query}")
            return self.search_cache[cache_key]
        
        self.logger.info(f"🔍 PRECISION SEARCH: {query}")
        
        # Multi-strategy search
        results = []
        
        # Strategy 1: Direct keyword matching
        keywords = re.findall(r'\b\w+\b', query.lower())
        content_lower = self.document_content.lower()
        
        sentences = re.split(r'[.!?]+', self.document_content)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
                
            sentence_lower = sentence.lower()
            matches = sum(1 for keyword in keywords if keyword in sentence_lower)
            
            if matches >= 2:  # At least 2 keyword matches
                results.append((sentence, matches))
        
        # Strategy 2: Specific pattern matching for insurance queries
        patterns = {
            'eligibility': [
                r'eligible.*dependent.*daughter',
                r'daughter.*eligible',
                r'coverage.*child.*marriage',
                r'dependent.*marriage',
                r'female child.*married'
            ],
            'dental': [
                r'dental.*claim',
                r'dental.*treatment',
                r'dental.*coverage'
            ],
            'grievance': [
                r'grievance.*email',
                r'complaint.*email',
                r'csd@.*\.in',
                r'grievance.*redressal'
            ],
            'email': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'age_limit': [
                r'(\d+)\s*years.*age',
                r'age.*(\d+)\s*years',
                r'until.*(\d+)\s*years'
            ]
        }
        
        query_lower = query.lower()
        for category, pattern_list in patterns.items():
            if any(term in query_lower for term in [category]):
                for pattern in pattern_list:
                    matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                    for match in matches:
                        # Find the sentence containing this match
                        start = max(0, match.start() - 100)
                        end = min(len(self.document_content), match.end() + 100)
                        context = self.document_content[start:end]
                        results.append((context, 10))  # High priority for pattern matches
        
        # Sort by relevance and deduplicate
        results.sort(key=lambda x: x[1], reverse=True)
        
        if results:
            best_result = results[0][0]
            self.search_cache[cache_key] = best_result
            self.logger.info(f"✅ FOUND: {best_result[:100]}...")
            return best_result
        
        # Fallback search
        if keywords:
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    result = sentence.strip()
                    self.search_cache[cache_key] = result
                    return result
        
        no_result = f"No specific information found for: {query}"
        self.search_cache[cache_key] = no_result
        return no_result

class ReActReasoningEngine:
    """Multi-step reasoning engine using ReAct framework"""
    
    def __init__(self, groq_client, logger):
        self.groq_client = groq_client
        self.logger = logger
        self.max_steps = 8
        self.step_timeout = 10  # seconds per step
        
    async def reason_and_act(self, document_content: str, complex_query: str) -> str:
        """Main ReAct reasoning loop"""
        
        start_time = time.time()
        self.logger.info(f"🧠 REACT REASONING INITIATED")
        self.logger.info(f"❓ COMPLEX QUERY: {complex_query}")
        
        # Initialize tools
        search_tool = PrecisionSearchTool(document_content, self.logger)
        
        # Reasoning trace
        reasoning_steps = []
        step_number = 1
        
        try:
            # Step 1: Initial decomposition and planning
            thought = await self._generate_initial_thought(complex_query)
            reasoning_steps.append(ReActStep(
                ReasoningStep.THOUGHT, thought, time.time(), step_number
            ))
            self.logger.info(f"💭 STEP {step_number} THOUGHT: {thought}")
            step_number += 1
            
            # Reasoning loop
            current_context = ""
            final_answer = None
            
            while step_number <= self.max_steps and not final_answer:
                
                # Generate next action
                action_query = await self._plan_next_action(
                    complex_query, reasoning_steps, current_context
                )
                
                if action_query.startswith("FINAL_ANSWER:"):
                    # Ready for final answer
                    final_answer = action_query.replace("FINAL_ANSWER:", "").strip()
                    reasoning_steps.append(ReActStep(
                        ReasoningStep.FINAL_ANSWER, final_answer, time.time(), step_number
                    ))
                    break
                
                # Record action
                reasoning_steps.append(ReActStep(
                    ReasoningStep.ACTION, f"Search: {action_query}", time.time(), step_number
                ))
                self.logger.info(f"🎯 STEP {step_number} ACTION: Search for '{action_query}'")
                step_number += 1
                
                # Execute search
                search_result = await search_tool.search(action_query)
                current_context += f"\n{action_query}: {search_result}"
                
                # Record observation
                reasoning_steps.append(ReActStep(
                    ReasoningStep.OBSERVATION, search_result, time.time(), step_number
                ))
                self.logger.info(f"👁️ STEP {step_number} OBSERVATION: {search_result[:100]}...")
                step_number += 1
                
                # Generate next thought
                if step_number <= self.max_steps:
                    next_thought = await self._synthesize_next_thought(
                        complex_query, reasoning_steps, current_context
                    )
                    reasoning_steps.append(ReActStep(
                        ReasoningStep.THOUGHT, next_thought, time.time(), step_number
                    ))
                    self.logger.info(f"💭 STEP {step_number} THOUGHT: {next_thought}")
                    step_number += 1
            
            # Generate final answer if not already generated
            if not final_answer:
                final_answer = await self._generate_final_answer(
                    complex_query, reasoning_steps, current_context
                )
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(f"🎯 REACT REASONING COMPLETE: {execution_time:.1f}ms")
            self.logger.info(f"✅ FINAL ANSWER: {final_answer}")
            
            return final_answer
            
        except Exception as e:
            self.logger.error(f"❌ REACT REASONING FAILED: {e}")
            # Fallback to simple search
            search_tool = PrecisionSearchTool(document_content, self.logger)
            fallback_result = await search_tool.search(complex_query)
            return f"Based on the available information: {fallback_result}"
    
    async def _generate_initial_thought(self, query: str) -> str:
        """Generate initial decomposition thought"""
        
        if not self.groq_client:
            return self._simple_decomposition(query)
        
        try:
            response = await self.groq_client.chat.completions.create(
                model="llama3-8b-8192",  # Use fast model for reasoning
                messages=[
                    {
                        "role": "system",
                        "content": """You are a reasoning agent. Break down complex queries into logical steps.

Think step by step:
1. Identify the main components of the query
2. Determine the logical order to address them
3. Note any dependencies between components

Respond with a clear thought process starting with "This query has multiple parts..."
Keep it concise but comprehensive."""
                    },
                    {
                        "role": "user",
                        "content": f"Break down this complex query: {query}"
                    }
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Initial thought generation failed: {e}")
            return self._simple_decomposition(query)
    
    async def _plan_next_action(self, original_query: str, steps: List[ReActStep], context: str) -> str:
        """Plan the next search action based on reasoning so far"""
        
        if not self.groq_client:
            return self._simple_next_action(original_query, steps)
        
        try:
            # Build reasoning history
            history = ""
            for step in steps[-3:]:  # Last 3 steps for context
                history += f"{step.step_type.value}: {step.content}\n"
            
            response = await self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a reasoning agent planning the next search action.

Based on the reasoning history, determine what specific information to search for next.

If you have enough information to answer the original query, respond with "FINAL_ANSWER: [your complete answer]"

Otherwise, respond with a specific search query to find the missing information.
Examples:
- "eligibility criteria for dependent daughter"
- "dental claim process"
- "grievance email address"

Keep search queries focused and specific."""
                    },
                    {
                        "role": "user",
                        "content": f"""Original Query: {original_query}

Reasoning History:
{history}

Current Context:
{context}

What should I search for next?"""
                    }
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Next action planning failed: {e}")
            return self._simple_next_action(original_query, steps)
    
    async def _synthesize_next_thought(self, original_query: str, steps: List[ReActStep], context: str) -> str:
        """Synthesize next reasoning thought"""
        
        if not self.groq_client:
            return "Continuing analysis of the information found..."
        
        try:
            latest_observation = ""
            for step in reversed(steps):
                if step.step_type == ReasoningStep.OBSERVATION:
                    latest_observation = step.content
                    break
            
            response = await self.groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a reasoning agent analyzing search results.

Based on what you just found, think about:
1. What this information tells you
2. What questions it answers from the original query
3. What you still need to find
4. How this affects your next steps

Respond with a clear thought starting with "Based on this information..." or "This tells me..."
Keep it concise and focused."""
                    },
                    {
                        "role": "user",
                        "content": f"""Original Query: {original_query}

Latest Search Result: {latest_observation}

What do you think about this result?"""
                    }
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Thought synthesis failed: {e}")
            return "Analyzing the search results and planning next steps..."
    
    async def _generate_final_answer(self, original_query: str, steps: List[ReActStep], context: str) -> str:
        """Generate comprehensive final answer"""
        
        if not self.groq_client:
            return self._simple_final_answer(context)
        
        try:
            # Build complete reasoning trace
            reasoning_trace = ""
            for step in steps:
                reasoning_trace += f"{step.step_type.value}: {step.content}\n"
            
            response = await self.groq_client.chat.completions.create(
                model="llama3-70b-8192",  # Use powerful model for final answer
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert insurance policy analyst providing final comprehensive answers.

Based on the reasoning process and gathered information, provide a complete, accurate answer that addresses all parts of the original query.

Your answer should be:
1. Comprehensive - address all parts of the query
2. Accurate - based only on the information found
3. Clear - easy to understand
4. Professional - appropriate for insurance customers

If information is missing for any part, clearly state what cannot be determined."""
                    },
                    {
                        "role": "user",
                        "content": f"""Original Query: {original_query}

Complete Reasoning Process:
{reasoning_trace}

All Gathered Information:
{context}

Provide the final comprehensive answer:"""
                    }
                ],
                temperature=0.0,
                max_tokens=400
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.error(f"❌ Final answer generation failed: {e}")
            return self._simple_final_answer(context)
    
    def _simple_decomposition(self, query: str) -> str:
        """Simple fallback decomposition"""
        parts = []
        
        if 'dependent' in query.lower() or 'daughter' in query.lower():
            parts.append("eligibility of dependent")
        if 'dental' in query.lower() or 'claim' in query.lower():
            parts.append("dental claim process")
        if 'grievance' in query.lower() or 'email' in query.lower():
            parts.append("grievance contact information")
        if 'name' in query.lower() or 'update' in query.lower():
            parts.append("name update process")
        
        return f"This query has multiple parts: {', '.join(parts)}. I need to address each component systematically."
    
    def _simple_next_action(self, query: str, steps: List[ReActStep]) -> str:
        """Simple fallback for next action"""
        query_lower = query.lower()
        
        # Check what we've already searched
        searched_terms = set()
        for step in steps:
            if step.step_type == ReasoningStep.ACTION:
                searched_terms.add(step.content.lower())
        
        # Determine what to search next
        if 'eligibility' not in str(searched_terms) and 'dependent' in query_lower:
            return "eligibility criteria for dependent daughter"
        elif 'dental' not in str(searched_terms) and 'dental' in query_lower:
            return "dental claim process"
        elif 'grievance' not in str(searched_terms) and 'grievance' in query_lower:
            return "grievance redressal email"
        else:
            return "FINAL_ANSWER: Based on available information, I need more specific details to provide a complete answer."
    
    def _simple_final_answer(self, context: str) -> str:
        """Simple fallback final answer"""
        if context.strip():
            return f"Based on the available information: {context}"
        else:
            return "I was unable to find specific information to answer this query completely."
