import requests
import json

def test_server():
    questions = [
        'What is the ambulance coverage amount?',
        'What is the cumulative bonus percentage?',
        'What is the room rent limit?'
    ]

    headers = {'Authorization': 'Bearer a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36'}

    print("🧪 TESTING CURRENT SERVER")
    print("=" * 50)
    
    for i, q in enumerate(questions, 1):
        try:
            response = requests.get('http://localhost:8000/hackrx/run', 
                                  headers=headers, 
                                  params={'documents': 'test', 'questions': q})
            
            if response.status_code == 200:
                data = response.json()
                answer = data['answers'][0]
                print(f"✅ [{i}] {q}")
                print(f"    {answer[:100]}...")
                print()
            else:
                print(f"❌ [{i}] {q} - Status: {response.status_code}")
                print(f"    {response.text}")
                
        except Exception as e:
            print(f"❌ [{i}] {q} - Error: {e}")

if __name__ == "__main__":
    test_server()
