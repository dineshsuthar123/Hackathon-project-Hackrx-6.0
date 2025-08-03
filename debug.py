"""
DEBUG SCRIPT - Core Retrieval Testing
This script isolates the retrieval mechanism to find the fundamental bug.
"""

import os
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def load_document():
    """Load the policy document content"""
    try:
        with open('Health_Insurance_Policy.pdf', 'r', encoding='utf-8') as file:
            content = file.read()
        print(f"✅ Document loaded: {len(content)} characters")
        return content
    except FileNotFoundError:
        # Fallback content for testing
        fallback_content = """
        4.1. Hospitalization 
        The Company shall indemnify Medical Expense incurred for Hospitalization of the Insured Person during the Policy Period:
        i. Room Rent, Boarding, Nursing Expenses all inclusive up to 2% of Sum Insured or actual whichever is less per day.
        ii. ICU Charges up to 5% of Sum Insured or actual whichever is less per day
        iii. Surgeon, Anesthetist, Medical Practitioner, Consultant, Specialist Fees
        iv. Anesthesia, blood, oxygen, operation theatre charges, surgical appliances, medicines and drugs, costs towards diagnostics
        v. Expenses incurred on road Ambulance subject to a maximum of Rs. 2,000/- per hospitalization

        5. CUMULATIVE BONUS (CB)
        Cumulative Bonus will be increased by 5% in respect of each claim free Policy Year (i.e. Policy Year during which no claim is made) subject to a maximum of 50% of the Sum Insured.

        7.5. Cataract
        Expenses related to the treatment of Cataract are payable only when treatment is taken after completion of 24 months of continuous coverage under the policy.

        3.22. Grace Period 
        Grace Period means the specified period of time, immediately following the premium due date during which a payment can be made to renew or continue a Policy in force without it being considered as a break in the Policy. The Grace Period is 30 days.

        3.15. Pre-existing Disease
        Pre-existing Disease means any condition, ailment, injury or disease that is diagnosed by a physician within 48 months prior to the effective date of the policy issued by the insurer.

        3.12. Dependent Child
        Dependent Child means named child of the Insured Person including adopted and step child who is financially dependent on the Insured Person and does not have independent source of income.
        """
        print(f"⚠️ Using fallback content: {len(fallback_content)} characters")
        return fallback_content

def create_chunks(content):
    """Create document chunks for indexing"""
    print("\n--- CREATING CHUNKS ---")
    
    # Simple paragraph-based chunking
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip() and len(p.strip()) > 50]
    
    print(f"Created {len(paragraphs)} chunks")
    for i, chunk in enumerate(paragraphs[:5]):  # Show first 5 chunks
        print(f"\n--- CHUNK {i+1} ---")
        print(f"Length: {len(chunk)} chars")
        print(f"Preview: {chunk[:200]}...")
    
    return paragraphs

def semantic_search(query, chunks, embedder, top_k=5):
    """Perform semantic search using sentence transformers"""
    print(f"\n--- SEMANTIC SEARCH ---")
    print(f"Query: {query}")
    
    # Encode query and chunks
    query_embedding = embedder.encode([query])
    chunk_embeddings = embedder.encode(chunks)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i, idx in enumerate(top_indices):
        score = similarities[idx]
        chunk = chunks[idx]
        results.append({
            'rank': i + 1,
            'score': score,
            'chunk': chunk,
            'index': idx
        })
        
    return results

def keyword_search(query, chunks, top_k=5):
    """Perform keyword search using TF-IDF"""
    print(f"\n--- KEYWORD SEARCH ---")
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    
    # Fit on all chunks + query
    all_texts = chunks + [query]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Query is the last item
    query_vector = tfidf_matrix[-1]
    chunk_vectors = tfidf_matrix[:-1]
    
    # Calculate similarities
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    
    # Get top results
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for i, idx in enumerate(top_indices):
        score = similarities[idx]
        chunk = chunks[idx]
        results.append({
            'rank': i + 1,
            'score': score,
            'chunk': chunk,
            'index': idx
        })
        
    return results

def pattern_search(query, chunks):
    """Search for specific patterns and keywords"""
    print(f"\n--- PATTERN SEARCH ---")
    
    # Define patterns based on query
    patterns = []
    
    if any(word in query.lower() for word in ['ambulance', 'transport']):
        patterns = [r'ambulance', r'transport', r'Rs\.?\s*2,?000', r'road ambulance']
    elif any(word in query.lower() for word in ['room', 'rent', 'boarding']):
        patterns = [r'room rent', r'boarding', r'2%', r'per day']
    elif any(word in query.lower() for word in ['bonus', 'cumulative']):
        patterns = [r'cumulative bonus', r'bonus', r'5%', r'claim free']
    elif any(word in query.lower() for word in ['cataract']):
        patterns = [r'cataract', r'24 months', r'continuous coverage']
    elif any(word in query.lower() for word in ['grace', 'premium']):
        patterns = [r'grace period', r'premium', r'30 days']
    
    results = []
    for i, chunk in enumerate(chunks):
        score = 0
        for pattern in patterns:
            matches = len(re.findall(pattern, chunk, re.IGNORECASE))
            score += matches
        
        if score > 0:
            results.append({
                'rank': len(results) + 1,
                'score': score,
                'chunk': chunk,
                'index': i
            })
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:5]

def debug_single_query():
    """Debug a single query end-to-end"""
    
    # Load models
    print("Loading models...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Embedder loaded")
    
    # Load document
    content = load_document()
    
    # Create chunks
    chunks = create_chunks(content)
    
    # Test query - something we know should have a clear answer
    query = "What is the maximum coverage for road ambulance expenses?"
    
    print(f"\n{'='*60}")
    print(f"DEBUGGING QUERY: {query}")
    print(f"{'='*60}")
    
    # Test semantic search
    semantic_results = semantic_search(query, chunks, embedder)
    print(f"\n--- SEMANTIC SEARCH RESULTS ---")
    for result in semantic_results:
        print(f"\nRank {result['rank']} (Score: {result['score']:.4f}):")
        print(f"Chunk {result['index']}: {result['chunk'][:300]}...")
    
    # Test keyword search
    keyword_results = keyword_search(query, chunks)
    print(f"\n--- KEYWORD SEARCH RESULTS ---")
    for result in keyword_results:
        print(f"\nRank {result['rank']} (Score: {result['score']:.4f}):")
        print(f"Chunk {result['index']}: {result['chunk'][:300]}...")
    
    # Test pattern search
    pattern_results = pattern_search(query, chunks)
    print(f"\n--- PATTERN SEARCH RESULTS ---")
    for result in pattern_results:
        print(f"\nRank {result['rank']} (Score: {result['score']}):")
        print(f"Chunk {result['index']}: {result['chunk'][:300]}...")
    
    # Look for the correct answer
    print(f"\n{'='*60}")
    print("MANUAL SEARCH FOR CORRECT ANSWER")
    print(f"{'='*60}")
    
    ambulance_keywords = ['ambulance', '2000', '2,000']
    for i, chunk in enumerate(chunks):
        if any(keyword.lower() in chunk.lower() for keyword in ambulance_keywords):
            print(f"\n--- CHUNK {i} CONTAINS AMBULANCE INFO ---")
            print(chunk)

def test_multiple_queries():
    """Test several known queries"""
    
    print("Loading models...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    content = load_document()
    chunks = create_chunks(content)
    
    test_queries = [
        "What is the maximum coverage for road ambulance expenses?",
        "What percentage is the room rent coverage?",
        "What is the cumulative bonus rate?",
        "What is the cataract waiting period?",
        "What is the grace period for premium payment?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"TESTING: {query}")
        print(f"{'='*80}")
        
        # Just show top 3 semantic results
        results = semantic_search(query, chunks, embedder, top_k=3)
        for result in results:
            print(f"\nScore: {result['score']:.4f}")
            print(f"Chunk: {result['chunk'][:200]}...")

if __name__ == "__main__":
    print("🔍 DEBUG SCRIPT - Core Retrieval Testing")
    print("This will isolate and test the fundamental retrieval mechanism")
    
    # First test a single query in detail
    debug_single_query()
    
    # Then test multiple queries
    print(f"\n\n{'='*100}")
    print("TESTING MULTIPLE QUERIES")
    print(f"{'='*100}")
    test_multiple_queries()
