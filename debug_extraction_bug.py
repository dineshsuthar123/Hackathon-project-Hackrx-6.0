"""
EXACT REPRODUCTION: Debug the extraction logic step by step
"""

import re

query = "What is the age range for dependent children coverage?"
context = """CHUNK 1: ERIOD There shall be a grace period of thirty days for payment of renewal premium. DEPENDENT CHILDREN AGE LIMIT Dependent children are covered from 3 months to 25 years of age. AYUSH HOSPITALS AYUSH hospitals must have minimum 5 in-patient beds and round the clock availability.

CHUNK 2: . ICU COVERAGE Intensive Care Unit (ICU/ICCU) expenses are covered up to 5% of sum insured per day. GRACE PERIOD There shall be a grace period of thirty days for payment of renewal premium. DEPENDENT CHILDREN AGE LIMIT Dependent children are covered from 3 months to 25 years of age. AYUSH HOSPITALS AYUSH hospitals must have minimum 5 in-patient beds and round the clock availability.

CHUNK 3: INSURANCE POLICY DOCUMENT WAITING PERIODS Pre-existing diseases are subject to a waiting period of 3 years from the date of first enrollment. Specific conditions waiting periods: - Cataract: 24 months - Joint replacement: 48 months - Gout and Rheumatism: 36 months - Hernia, Hydrocele, Congenital internal diseases: 24 months AMBULANCE COVERAGE Expenses incurred on road ambulance subject to maximum of Rs. 2,000/- per hospitalization are payable. ROOM RENT COVERAGE Room rent, boarding and nursing expenses are covered up to 2% of sum insured per day."""

query_lower = query.lower()
context_lower = context.lower()

print("=== STEP-BY-STEP EXTRACTION DEBUG ===")
print(f"Query: {query}")
print(f"Query lower: {query_lower}")
print()

# Test each condition in order (as they appear in the code)

print("1. Testing WAITING PERIOD condition:")
waiting_words = ['waiting', 'period']
has_waiting = any(word in query_lower for word in waiting_words)
print(f"   Words: {waiting_words}")
print(f"   Has waiting words: {has_waiting}")

if has_waiting:
    print("   -> SKIPPED (waiting condition)")
else:
    print("   -> PASSED (no waiting words)")

print()

print("2. Testing AMBULANCE condition:")
ambulance_words = ['ambulance', 'coverage', 'amount']
has_ambulance = any(word in query_lower for word in ambulance_words)
print(f"   Words: {ambulance_words}")
print(f"   Has ambulance words: {has_ambulance}")

if has_ambulance:
    print("   -> TESTING ambulance pattern...")
    match = re.search(r'ambulance.*?rs\.?\s*([0-9,]+)', context_lower)
    if match:
        amount = match.group(1)
        result = f"Road ambulance expenses are covered up to Rs. {amount} per hospitalization."
        print(f"   -> MATCHED! Result: {result}")
        print("   -> THIS IS THE BUG! Stopping here.")
    else:
        print("   -> No ambulance pattern match")
else:
    print("   -> PASSED (no ambulance words)")

print()

print("3. Testing GRACE condition (should not reach here):")
grace_words = ['grace', 'premium']
has_grace = any(word in query_lower for word in grace_words)
print(f"   Words: {grace_words}")
print(f"   Has grace words: {has_grace}")

print()

print("4. Testing AGE condition (should not reach here):")
age_words = ['age', 'dependent', 'children']
has_age = any(word in query_lower for word in age_words)
print(f"   Words: {age_words}")
print(f"   Has age words: {has_age}")

if has_age:
    print("   -> TESTING age patterns...")
    patterns = [
        r'dependent children.*?(\d+)\s*months.*?(\d+)\s*years',
        r'children.*?covered.*?(\d+)\s*months.*?(\d+)\s*years',
        r'dependent.*?(\d+)\s*months.*?(\d+)\s*years.*?age',
        r'from\s*(\d+)\s*months\s*to\s*(\d+)\s*years'
    ]
    
    for i, pattern in enumerate(patterns, 1):
        match = re.search(pattern, context_lower)
        if match:
            months = match.group(1)
            years = match.group(2)
            result = f"The age range for dependent children is {months} months to {years} years."
            print(f"   Pattern {i} MATCHED: {result}")
            break
    else:
        print("   -> No age pattern matches")

print()
print("=== DIAGNOSIS ===")
print(f"Query contains 'coverage': {'coverage' in query_lower}")
print("The bug is in the ambulance condition - it checks for 'coverage' which is in the age question!")
print("Need to make the ambulance condition more specific.")
