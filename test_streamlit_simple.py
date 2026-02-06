"""
Minimal Streamlit test to isolate connection issues.
"""

import streamlit as st
import requests
import sys

st.set_page_config(page_title="Ollama Connection Test")

def test_ollama_in_streamlit():
    """Test Ollama connection within Streamlit."""
    st.title("🔍 Ollama Connection Test")
    
    # Test basic connection
    st.subheader("Test 1: Basic Connection")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            st.success("✅ Ollama API is accessible")
            models = response.json().get("models", [])
            st.write(f"Available models: {[m['name'] for m in models]}")
        else:
            st.error(f"❌ API returned status {response.status_code}")
    except Exception as e:
        st.error(f"❌ Connection error: {e}")
    
    # Test generation
    st.subheader("Test 2: Text Generation")
    if st.button("🧪 Test Generation", type="primary"):
        try:
            with st.spinner("Testing generation..."):
                payload = {
                    "model": "llama2:latest",
                    "prompt": "Hello from Streamlit!",
                    "stream": False,
                    "options": {"max_tokens": 50}
                }
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("response", "")
                    st.success(f"✅ Generated: {answer}")
                else:
                    st.error(f"❌ Generation failed: {response.status_code}")
                    
        except Exception as e:
            st.error(f"❌ Generation error: {e}")
    
    # Debug info
    st.subheader("Debug Information")
    st.write(f"Python executable: {sys.executable}")
    st.write(f"Streamlit version: {st.__version__}")
    st.write(f"Requests can connect to: http://localhost:11434")

if __name__ == "__main__":
    test_ollama_in_streamlit()
