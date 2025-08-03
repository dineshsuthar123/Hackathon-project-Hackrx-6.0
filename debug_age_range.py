"""
DEBUG: Question 5 Issue - Age Range Extraction
"""

import re

# Test content
content = """
INSURANCE POLICY DOCUMENT

WAITING PERIODS
Pre-existing diseases are subject to a waiting period of 3 years from the date of first enrollment.
Specific conditions waiting periods:
- Cataract: 24 months
- Joint replacement: 48 months  
- Gout and Rheumatism: 36 months
- Hernia, Hydrocele, Congenital internal diseases: 24 months

AMBULANCE COVERAGE
Expenses incurred on road ambulance subject to maximum of Rs. 2,000/- per hospitalization are payable.

ROOM RENT COVERAGE  
Room rent, boarding and nursing expenses are covered up to 2% of sum insured per day.

ICU COVERAGE
Intensive Care Unit (ICU/ICCU) expenses are covered up to 5% of sum insured per day.

GRACE PERIOD
There shall be a grace period of thirty days for payment of renewal premium.

DEPENDENT CHILDREN AGE LIMIT
Dependent children are covered from 3 months to 25 years of age.

AYUSH HOSPITALS
AYUSH hospitals must have minimum 5 in-patient beds and round the clock availability.
"""

query = "What is the age range for dependent children coverage?"
query_lower = query.lower()
context_lower = content.lower()

print("=== DEBUGGING AGE RANGE EXTRACTION ===")
print(f"Query: {query}")
print(f"Query triggers: {[word for word in ['age', 'dependent', 'children'] if word in query_lower]}")

# Test the patterns
patterns = [
    r'dependent children.*?(\d+)\s*months.*?(\d+)\s*years',
    r'children.*?covered.*?(\d+)\s*months.*?(\d+)\s*years',
    r'dependent.*?(\d+)\s*months.*?(\d+)\s*years.*?age',
    r'from\s*(\d+)\s*months\s*to\s*(\d+)\s*years'
]

print("\n=== TESTING PATTERNS ===")
for i, pattern in enumerate(patterns, 1):
    print(f"Pattern {i}: {pattern}")
    match = re.search(pattern, context_lower)
    if match:
        print(f"  ✅ MATCH: {match.groups()}")
        months = match.group(1)
        years = match.group(2)
        print(f"  RESULT: The age range for dependent children is {months} months to {years} years.")
    else:
        print(f"  ❌ NO MATCH")

print("\n=== CONTENT ANALYSIS ===")
relevant_section = "dependent children are covered from 3 months to 25 years of age"
print(f"Relevant section: {relevant_section}")

# Test with the exact relevant section
for i, pattern in enumerate(patterns, 1):
    print(f"Pattern {i} on relevant section:")
    match = re.search(pattern, relevant_section)
    if match:
        print(f"  ✅ MATCH: {match.groups()}")
    else:
        print(f"  ❌ NO MATCH")
