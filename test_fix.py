"""
FIXED PATTERN EXTRACTION - Testing with Simplified Logic
This fixes the overly strict validation that was causing wrong answers.
"""

import re
from typing import Optional

def fixed_pattern_extraction(query: str, context: str) -> Optional[str]:
    """Simplified, working pattern extraction"""
    context_lower = context.lower()
    query_lower = query.lower()
    
    # Ambulance coverage
    if any(word in query_lower for word in ['ambulance', 'transport']):
        # Look for ambulance amounts
        patterns = [
            r'ambulance.*?rs\.?\s*([0-9,]+)',
            r'road ambulance.*?rs\.?\s*([0-9,]+)',
            r'expenses.*?ambulance.*?rs\.?\s*([0-9,]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, context_lower)
            if match:
                amount = match.group(1)
                return f"Road ambulance expenses are covered up to Rs. {amount} per hospitalization."
    
    # Room rent coverage
    if any(word in query_lower for word in ['room', 'rent', 'boarding']):
        # Look for room rent percentage
        match = re.search(r'room rent.*?(\d+)%.*?sum insured', context_lower)
        if match:
            percentage = match.group(1)
            return f"Room rent is covered up to {percentage}% of sum insured per day."
    
    # Cumulative bonus
    if any(word in query_lower for word in ['cumulative', 'bonus']):
        # Look for bonus percentage
        match = re.search(r'cumulative bonus.*?(\d+)%.*?claim.*?free', context_lower)
        if match:
            percentage = match.group(1)
            return f"Cumulative bonus is {percentage}% per claim-free year."
    
    # Cataract waiting period
    if any(word in query_lower for word in ['cataract', 'waiting']):
        # Look for months
        match = re.search(r'cataract.*?(\d+)\s*months', context_lower)
        if match:
            months = match.group(1)
            return f"The waiting period for cataract treatment is {months} months."
    
    # Grace period
    if any(word in query_lower for word in ['grace', 'premium']):
        # Look for days
        match = re.search(r'grace period.*?(\d+)\s*days', context_lower)
        if match:
            days = match.group(1)
            return f"The grace period for premium payment is {days} days."
    
    return None

# Test with the content we know contains the right answers
test_content = """
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
"""

# Test queries
test_queries = [
    "What is the maximum coverage for road ambulance expenses?",
    "What percentage is the room rent coverage?", 
    "What is the cumulative bonus rate?",
    "What is the cataract waiting period?",
    "What is the grace period for premium payment?"
]

print("🔧 TESTING FIXED PATTERN EXTRACTION")
print("="*60)

for query in test_queries:
    result = fixed_pattern_extraction(query, test_content)
    print(f"\nQuery: {query}")
    print(f"Answer: {result}")
    print("-" * 40)
