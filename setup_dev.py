#!/usr/bin/env python3
"""
Development Setup Script for Medical Voice Assistant
Optimized for Mac development environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    print("""
🏥 Medical Voice Assistant - Development Setup
===============================================
Ultra Human-Like Voice with Smart Selection
- Higgs Audio V2 (Best Quality)
- Chatterbox (Fast Backup) 
- Bark (Most Natural/Human-like)
- Kokoro (Real-time)
- Whisper STT + Gemini LLM
""")

def check_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False
    
    print(f"✅ Python {sys.version}")
    
    # Check OS
    system = platform.system()
    print(f"✅ Operating System: {system}")
    
    if system == "Darwin":  # macOS
        print("✅ Mac development environment detected")
    
    # Check if we're in the right directory
    if not Path("backend/main.py").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    return True

def setup_environment():
    """Setup Python environment"""
    print("\n🐍 Setting up Python environment...")
    
    try:
        # Create virtual environment
        if not Path("venv").exists():
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        else:
            print("✅ Virtual environment already exists")
        
        # Determine activation script path
        if platform.system() == "Windows":
            activate_script = "venv/Scripts/activate"
            pip_path = "venv/Scripts/pip"
        else:
            activate_script = "venv/bin/activate"
            pip_path = "venv/bin/pip"
        
        print(f"📝 To activate: source {activate_script}")
        
        # Install backend requirements
        print("Installing backend requirements...")
        subprocess.run([pip_path, "install", "-r", "backend/requirements.txt"], check=True)
        print("✅ Backend requirements installed")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Environment setup failed: {e}")
        return False

def setup_config():
    """Setup configuration files"""
    print("\n⚙️ Setting up configuration...")
    
    # Copy .env.example to .env if it doesn't exist
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Created .env from .env.example")
        else:
            print("⚠️ .env.example not found")
    else:
        print("✅ .env already exists")
    
    # Create necessary directories
    directories = [
        "logs",
        "data", 
        "models/whisper",
        "models/higgs",
        "models/chatterbox", 
        "models/bark",
        "models/kokoro",
        "voices"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API keys...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    env_content = env_file.read_text()
    
    # Check for Gemini API key
    if "GOOGLE_API_KEY=" in env_content and "your_google_gemini_key_here" not in env_content:
        print("✅ Google API key configured")
    else:
        print("⚠️ Google API key not configured")
        print("   Get your free key at: https://makersuite.google.com/app/apikey")
    
    # Check for Pinecone (optional)
    if "PINECONE_API_KEY=" in env_content and "pcsk_" in env_content:
        print("✅ Pinecone API key configured")
    else:
        print("ℹ️ Pinecone API key not configured (optional for RAG)")
    
    return True

def create_dev_scripts():
    """Create development scripts"""
    print("\n📝 Creating development scripts...")
    
    # Mac development script (CPU only)
    dev_script = """#!/bin/bash
echo "🏥 Starting Medical Voice Assistant (Mac Development Mode)"
echo "Using CPU-only mode for Mac compatibility"
echo ""

# Set environment variables for Mac development
export DEVICE=cpu
export DEBUG=true
export LOG_LEVEL=DEBUG
export ENVIRONMENT=development

# Start backend only (no GPU services for Mac)
echo "Starting backend server..."
cd backend
source ../venv/bin/activate
python main.py

echo ""
echo "🌐 Backend running at: http://localhost:8085"
echo "📊 Health check: http://localhost:8085/health"
echo "📖 API docs: http://localhost:8085/docs"
"""
    
    with open("run_dev_mac.sh", "w") as f:
        f.write(dev_script)
    
    os.chmod("run_dev_mac.sh", 0o755)
    print("✅ Created run_dev_mac.sh")
    
    # RunPod deployment script
    runpod_script = """#!/bin/bash
echo "🚀 Deploying to RunPod (GPU Mode)"
echo "Make sure you're connected to your RunPod instance"
echo ""

# Build and deploy with GPU support
docker-compose -f docker-compose.yml up -d

echo ""
echo "🌐 Services starting..."
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"  
echo "Whisper STT: http://localhost:8001"
echo "Higgs TTS: http://localhost:8002"
echo "Chatterbox TTS: http://localhost:8003"
echo "Kokoro TTS: http://localhost:8004"
echo "Bark TTS: http://localhost:8005"
"""
    
    with open("deploy_runpod.sh", "w") as f:
        f.write(runpod_script)
    
    os.chmod("deploy_runpod.sh", 0o755)
    print("✅ Created deploy_runpod.sh")

def print_next_steps():
    """Print next steps for the user"""
    print("""
🎉 Setup Complete!

Next Steps:
===========

1. 📝 Configure your API keys in .env:
   - Add your Gemini API key (required)
   - Add Pinecone key (optional, for RAG)

2. 🖥️ For Mac Development (CPU only):
   ./run_dev_mac.sh

3. 🚀 For RunPod Deployment (GPU):
   ./deploy_runpod.sh

4. 🧪 Test the voice models:
   curl http://localhost:8085/api/voice/models

5. 🎙️ Try the voice demo:
   curl -X POST http://localhost:8085/api/voice/demo \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Hello, I am your medical assistant", "voice_quality": "best"}'

Voice Quality Options:
=====================
- "best" → Higgs Audio V2 (most emotional)
- "natural" → Bark (most human-like)  
- "fast" → Chatterbox (balanced)
- "realtime" → Kokoro (ultra-fast)

🔗 Useful URLs:
- Backend API: http://localhost:8085
- API Documentation: http://localhost:8085/docs
- Health Check: http://localhost:8085/health
- Voice Models: http://localhost:8085/api/voice/models

Need help? Check the README.md for detailed instructions.
""")

def main():
    """Main setup function"""
    print_banner()
    
    if not check_requirements():
        sys.exit(1)
    
    if not setup_environment():
        sys.exit(1)
    
    setup_config()
    check_api_keys()
    create_dev_scripts()
    print_next_steps()

if __name__ == "__main__":
    main()
