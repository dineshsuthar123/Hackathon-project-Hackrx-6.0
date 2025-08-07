# Vercel Deployment for Groq Hyper-Intelligence API

## 🚀 Quick Deploy to Vercel

### Prerequisites
1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally with `npm i -g vercel`
3. **Environment Variables**: Set up your API keys

### 🔧 Environment Variables Setup

Before deploying, you need to set these environment variables in your Vercel dashboard:

```bash
# Required API Keys
GROQ_API_KEY=your_groq_api_key_here
MONGODB_URI=your_mongodb_connection_string
HACKRX_API_TOKEN=a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36

# Optional MongoDB Configuration
MONGODB_DATABASE=hackrx_groq_intelligence
MONGODB_COLLECTION=document_cache
MONGODB_MAX_POOL_SIZE=5
MONGODB_MIN_POOL_SIZE=1
```

### 📦 Deployment Steps

#### Method 1: Command Line (Recommended)
```bash
# 1. Login to Vercel
vercel login

# 2. Deploy from project directory
cd "d:\Majar Projects\hack-6.0-hackathon"
vercel

# 3. Follow the prompts:
#    - Set up and deploy? Y
#    - Which scope? (your account)
#    - Link to existing project? N
#    - Project name: hackrx-groq-intelligence
#    - Directory: ./
#    - Override settings? N

# 4. Set environment variables
vercel env add GROQ_API_KEY
vercel env add MONGODB_URI
vercel env add HACKRX_API_TOKEN

# 5. Redeploy with environment variables
vercel --prod
```

#### Method 2: GitHub Integration
1. Push code to GitHub repository
2. Import project in Vercel dashboard
3. Set environment variables in project settings
4. Deploy automatically

### 🔑 Environment Variables Configuration

In your Vercel dashboard (or via CLI):

1. **GROQ_API_KEY**: Your Groq API key from groq.com
   ```
   vercel env add GROQ_API_KEY production
   # Enter your actual Groq API key when prompted
   ```

2. **MONGODB_URI**: Your MongoDB connection string
   ```
   vercel env add MONGODB_URI production
   # Enter your MongoDB Atlas connection string
   ```

3. **HACKRX_API_TOKEN**: API authentication token
   ```
   vercel env add HACKRX_API_TOKEN production
   # Use: a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36
   ```

### 🎯 API Endpoints

Once deployed, your API will be available at:
- **Base URL**: `https://your-project-name.vercel.app`
- **Health Check**: `GET /health`
- **Main Endpoint**: `POST /hackrx/run`

### 📊 Testing Your Deployment

```bash
# Test health endpoint
curl https://your-project-name.vercel.app/health

# Test main API
curl -X POST https://your-project-name.vercel.app/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is the waiting period for Gout and Rheumatism?"]
  }'
```

### 🔧 Troubleshooting

#### Common Issues:

1. **Build Errors**: Check that all dependencies are in requirements.txt
2. **Timeout Issues**: Vercel functions have a 30-second timeout limit
3. **Memory Issues**: Current setup uses ~73MB, well under limits
4. **Environment Variables**: Ensure all required variables are set

#### Logs and Debugging:
```bash
# View function logs
vercel logs

# View build logs
vercel logs --build
```

### 🚀 Performance Optimizations

The deployment is optimized for:
- **Cold Start**: ~2-3 seconds
- **Memory Usage**: ~73MB
- **Response Time**: <5 seconds for most queries
- **Concurrent Requests**: Handles multiple simultaneous requests

### 📱 Monitoring

- **Vercel Dashboard**: Monitor function invocations, errors, and performance
- **Application Logs**: Built-in logging tracks cache hits, Groq calls, and performance
- **Health Endpoint**: Use `/health` for uptime monitoring

### 🔄 Continuous Deployment

Set up automatic deployments:
1. Connect GitHub repository to Vercel
2. Enable automatic deployments on push to main branch
3. Environment variables persist across deployments

Your Groq Hyper-Intelligence API is now ready for production! 🎉
