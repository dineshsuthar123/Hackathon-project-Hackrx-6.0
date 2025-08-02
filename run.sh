#!/bin/bash
# Unix/Linux/macOS script to run the LLM-Powered Query-Retrieval System

echo "🎯 LLM-Powered Intelligent Query-Retrieval System"
echo "🏆 Hack 6.0 Hackathon Submission"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "Creating .env file template..."
    cat > .env << EOF
OPENAI_API_KEY=your_openai_api_key_here
HACKRX_API_TOKEN=a3d1b4849a33b0269ac53fd27a8552eb1fbcc9cea01c70a1a85e11e330eb7c36
EOF
    echo ""
    echo "📝 Please edit .env file and add your OpenAI API key"
    echo "Then run this script again"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Run the system
echo "🚀 Starting the system..."
python3 start.py
