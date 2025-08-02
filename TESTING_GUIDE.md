# 🧪 Testing Guide - Real Documents

## 📋 Overview

This guide shows you how to test the LLM-Powered Intelligent Query-Retrieval System with actual documents including PDFs, DOCX files, and online documents.

## 🚀 Quick Test (5 minutes)

### **Step 1: Start the Server**

```bash
# Navigate to project directory
cd "d:\Majar Projects\hack-6.0-hackathon"

# Start the production server
python production_server.py
```

Server will start on: `http://localhost:8002`

### **Step 2: Run Real Document Tests**

```bash
# Run comprehensive test with real documents
python test_real_documents.py
```

This will automatically test with:
- ✅ **National Insurance Policy** (PDF) - 10 questions
- ✅ **Apple SEC Filing** (HTML) - 5 questions  
- ✅ **GPT-3 Research Paper** (PDF) - 5 questions

Expected output:
```
🧪 COMPREHENSIVE DOCUMENT TESTING SUITE
========================================
📈 Overall Statistics:
  • Total Documents Tested: 3
  • Successful Tests: 3
  • Success Rate: 100.0%
  • Average Response Time: 4.60s
  • Average Answer Quality: 4.30/5
```

## 📄 Document Types Supported

### **✅ PDF Documents**
- Insurance policies
- Legal contracts
- Research papers
- Financial reports
- Technical manuals

### **✅ DOCX Documents**
- Policy documents
- HR manuals
- Procedures
- Guidelines

### **✅ Online Documents**
- HTML pages
- Public PDFs
- SEC filings
- Government documents

## 🔬 Testing Methods

### **Method 1: Automated Testing (Recommended)**

Use our comprehensive test suite:

```bash
# Test all document types
python test_real_documents.py

# Test HackRx compliance
python test_hackrx_compliance.py

# Test specific features
python test_improved_answers.py
```

### **Method 2: Manual API Testing**

Test with curl commands:

```bash
# Basic test with insurance document
curl -X POST "http://localhost:8002/hackrx/run" \
  -H "Authorization: Bearer a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.nationalinsurance.nic.co.in/sites/default/files/2024-11/National%20Parivar%20Mediclaim%20Plus%20Policy%20Wording.pdf",
    "questions": [
      "What is the grace period for premium payment?",
      "Does this policy cover maternity expenses?",
      "What is the waiting period for pre-existing diseases?"
    ]
  }'
```

### **Method 3: Your Own Documents**

To test with your documents:

1. **Upload to cloud storage** (Google Drive, Dropbox, AWS S3)
2. **Make it publicly accessible**
3. **Copy the public URL**
4. **Use in your test**

Example with Google Drive:
1. Upload your PDF to Google Drive
2. Right-click → "Get link" → "Anyone with the link"
3. Modify URL format:
   - From: `https://drive.google.com/file/d/FILE_ID/view`
   - To: `https://drive.google.com/uc?export=download&id=FILE_ID`

## 📊 Sample Test Results

### **Insurance Policy Test**
```json
{
  "questions": [
    "What is the grace period for premium payment?",
    "Does this policy cover maternity expenses?",
    "What is the waiting period for pre-existing diseases?"
  ],
  "answers": [
    "Grace period of 30 days is provided for premium payment after the due date.",
    "Yes, the policy covers maternity expenses with a waiting period of 24 months.",
    "There is a waiting period of thirty-six (36) months for pre-existing diseases."
  ],
  "quality_scores": [4.5, 5.0, 5.0],
  "response_time": "3.2 seconds"
}
```

### **Research Paper Test**
```json
{
  "questions": [
    "What is the main contribution of this paper?",
    "How many parameters does GPT-3 have?"
  ],
  "answers": [
    "The paper introduces GPT-3, a language model with 175 billion parameters...",
    "GPT-3 has 175 billion parameters, making it the largest neural network..."
  ],
  "quality_scores": [4.0, 5.0],
  "response_time": "2.1 seconds"
}
```

## 🎯 Question Types That Work Best

### **✅ Specific Information Queries**
- "What is the premium amount?"
- "What are the eligibility criteria?"
- "How do I file a claim?"

### **✅ Yes/No Questions**
- "Does the policy cover dental treatment?"
- "Are pre-existing conditions covered?"
- "Is there a waiting period?"

### **✅ Procedural Questions**
- "How do I renew the policy?"
- "What documents are required?"
- "What is the claim process?"

### **✅ Comparison Questions**
- "What is the difference between Plan A and Plan B?"
- "Compare coverage limits"

### **❌ Avoid These Question Types**
- Very vague questions: "Tell me about this document"
- Questions requiring external knowledge
- Questions about information not in the document
- Complex mathematical calculations

## 🔧 Troubleshooting

### **Common Issues & Solutions**

1. **"Cannot download document"**
   - ✅ Check if URL is publicly accessible
   - ✅ Verify document format (PDF, DOCX)
   - ✅ Test URL in browser first

2. **"Authentication failed"**
   - ✅ Check API token in .env file
   - ✅ Verify Bearer token format
   - ✅ Ensure server is running

3. **"Poor answer quality"**
   - ✅ Use specific, clear questions
   - ✅ Ensure document contains relevant information
   - ✅ Try rephrasing questions

4. **"Slow response times"**
   - ✅ Check internet connection
   - ✅ Try smaller documents first
   - ✅ Monitor OpenAI API limits

### **Debug Mode**

Enable detailed logging:

```python
# Add to your test script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Benchmarks

### **Expected Performance**
- **Response Time**: 2-5 seconds per question
- **Accuracy**: 80%+ for domain-specific questions
- **Document Size**: Up to 50MB PDFs
- **Concurrent Users**: 10+ simultaneous requests

### **Optimization Tips**
- Use specific, well-formed questions
- Ensure documents are high-quality (not scanned)
- Batch related questions together
- Cache frequently accessed documents

## 🎉 Success Criteria

Your system is working correctly if:

✅ **Server starts without errors**  
✅ **Health check returns 200**  
✅ **Document downloads successfully**  
✅ **All questions get answers**  
✅ **Response time < 10 seconds**  
✅ **Answer quality > 3/5 average**  

## 📞 Need Help?

If you encounter issues:

1. **Check the logs** for error messages
2. **Verify your .env configuration**
3. **Test with the provided examples first**
4. **Ensure your OpenAI API key is valid**

## 🏆 Ready for Production

Once your tests pass:

1. **Deploy to Render/AWS/etc.**
2. **Configure production environment variables**
3. **Submit your webhook URL to HackRx**
4. **Monitor performance and logs**

---

**🎯 Your LLM-Powered System is ready to analyze any document and answer intelligent questions!**
