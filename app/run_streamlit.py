"""
Runner script for Streamlit app with proper error handling.
"""

import sys
import os
import subprocess

def check_requirements():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama connected with {len(models)} models")
            return True
        else:
            print("❌ Ollama not responding correctly")
            return False
    except:
        print("❌ Cannot connect to Ollama")
        print("Please ensure Ollama is running: ollama serve")
        return False

def main():
    """Main runner function."""
    print("🚀 Starting World Bank RAG Streamlit App...")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check Ollama
    if not check_ollama():
        return
    
    # Change to app directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    
    print("📱 Launching Streamlit app...")
    print("🌐 App will open in your browser at: http://localhost:8501")
    print("⏹️ Press Ctrl+C to stop the app")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    main()
