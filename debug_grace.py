"""
Debug grace period issue
"""

import re

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

question = "What is the grace period for premium payment?"
question_lower = question.lower()
content_lower = content.lower()

print("=== GRACE PERIOD DEBUG ===")
print(f"Question: {question}")
print(f"Has grace words: {any(word in question_lower for word in ['grace', 'premium'])}")
print()

# Test patterns
patterns = [
    r'grace period.*?(\d+)\s*days',
    r'grace.*?(\d+)\s*days',
    r'thirty\s*days.*?grace',
    r'grace.*?thirty\s*days',
    r'grace period.*?thirty days'
]

print("Testing patterns:")
for i, pattern in enumerate(patterns, 1):
    match = re.search(pattern, content_lower)
    print(f"{i}. {pattern}")
    if match:
        print(f"   ✅ MATCH: {match.groups()}")
        if match.groups() and match.group(1):
            print(f"   → Would return: The grace period for premium payment is {match.group(1)} days.")
        else:
            print(f"   → No group 1")
    else:
        print(f"   ❌ NO MATCH")

print()
print("Special thirty check:")
has_thirty = 'thirty' in content_lower and 'grace' in content_lower
print(f"Has 'thirty' and 'grace': {has_thirty}")

if has_thirty:
    print("→ Would return: The grace period for premium payment is 30 days.")

# Find the exact text
print()
print("Relevant section:")
match = re.search(r'grace period.*?premium', content_lower)
if match:
    print(f"Found: '{match.group(0)}'")
else:
    print("No grace period section found")
