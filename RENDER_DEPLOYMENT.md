# RENDER DEPLOYMENT GUIDE - Groq ReAct Intelligence API

## 🚀 Render Deployment Configuration

### **Start Command:**
```bash
uvicorn app_groq_ultimate:app --host 0.0.0.0 --port $PORT
```

### **Alternative Start Command (using script):**
```bash
bash start.sh
```

### **Build Command:**
```bash
pip install -r requirements.txt
```

## 📋 Render Service Configuration

### **1. Service Settings:**
- **Name**: `groq-react-intelligence-api`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app_groq_ultimate:app --host 0.0.0.0 --port $PORT`

### **2. Environment Variables:**
Set these in your Render dashboard:

```bash
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=your_mongodb_connection_string
HACKRX_API_TOKEN=a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36
MONGODB_DATABASE=hackrx_groq_intelligence
MONGODB_COLLECTION=document_cache
```

### **3. Runtime Settings:**
- **Python Version**: `3.9` or higher
- **Memory**: `512MB` (sufficient for lightweight deployment)
- **Auto-Deploy**: `Yes` (deploy on git push)

## 🔧 Deployment Steps

### **Step 1: Create Render Service**
1. Go to [render.com](https://render.com)
2. Connect your GitHub repository
3. Select "Web Service"
4. Choose your repository: `Hackathon-project-Hackrx-6.0`

### **Step 2: Configure Service**
- **Name**: `groq-react-intelligence`
- **Branch**: `main`
- **Root Directory**: `.` (leave empty)
- **Environment**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app_groq_ultimate:app --host 0.0.0.0 --port $PORT`

### **Step 3: Set Environment Variables**
Add all the environment variables listed above in the Render dashboard.

### **Step 4: Deploy**
Click "Create Web Service" and Render will automatically deploy your application.

## 📱 Testing Your Deployment

Once deployed, your API will be available at:
- **Base URL**: `https://your-service-name.onrender.com`
- **Health Check**: `https://your-service-name.onrender.com/health`
- **API Endpoint**: `https://your-service-name.onrender.com/hackrx/run`

### **Test Commands:**
```bash
# Health check
curl https://your-service-name.onrender.com/health

# API test
curl -X POST https://your-service-name.onrender.com/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the waiting period for Gout and Rheumatism?"]
  }'
```

## 🔍 Troubleshooting

### **Common Issues:**

1. **Build Failures**: Check that all dependencies in `requirements.txt` are available
2. **Port Issues**: Render automatically sets the `$PORT` environment variable
3. **Memory Issues**: Current setup uses ~72MB, well under Render's limits
4. **Environment Variables**: Ensure all required variables are set in Render dashboard

### **Logs and Debugging:**
- View logs in Render dashboard under "Logs" tab
- Use the health endpoint to verify service status
- Check environment variables in the Render dashboard

## 📈 Performance Optimization

- **Cold Start**: ~3-5 seconds (typical for Render)
- **Memory Usage**: ~72MB (efficient)
- **Response Time**: <5 seconds for most queries
- **Auto-scaling**: Render handles scaling automatically

## 🔄 Continuous Deployment

Render will automatically redeploy when you push to the `main` branch. This enables:
- Seamless updates
- Zero-downtime deployments
- Automatic rollbacks on failures

Your Groq ReAct Intelligence API is optimized for Render deployment! 🚀
