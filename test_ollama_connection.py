"""
Simple test to verify Ollama connection from Python.
"""

import requests
import json

def test_ollama_api():
    """Test Ollama API directly."""
    try:
        # Test 1: Check if API is accessible
        print("🔍 Testing Ollama API connection...")
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        
        if response.status_code == 200:
            print("✅ Ollama API is accessible")
            models = response.json().get("models", [])
            print(f"📋 Available models: {[m['name'] for m in models]}")
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
        
        # Test 2: Try actual generation
        print("\n🧪 Testing text generation...")
        payload = {
            "model": "llama2:latest",
            "prompt": "Hello, please respond with just 'API test successful'",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "max_tokens": 50
            }
        }
        
        gen_response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if gen_response.status_code == 200:
            result = gen_response.json()
            answer = result.get("response", "")
            print(f"✅ Generation successful: {answer}")
            return True
        else:
            print(f"❌ Generation failed with status {gen_response.status_code}")
            print(f"Response: {gen_response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except requests.exceptions.Timeout as e:
        print(f"❌ Timeout error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Ollama Connection")
    print("=" * 50)
    
    success = test_ollama_api()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! Ollama is working correctly.")
        print("\n💡 If Streamlit still fails, the issue might be:")
        print("   1. Streamlit configuration")
        print("   2. Python environment variables")
        print("   3. Network/firewall settings")
        print("   4. Port conflicts")
    else:
        print("❌ Ollama connection tests failed.")
        print("\n🛠️ Troubleshooting steps:")
        print("   1. Ensure Ollama is running: ollama serve")
        print("   2. Check if port 11434 is accessible")
        print("   3. Try restarting Ollama")
        print("   4. Check firewall/antivirus settings")
