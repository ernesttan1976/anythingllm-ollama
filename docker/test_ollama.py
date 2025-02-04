import requests
import json
import sys
import time

def test_ollama_api(base_url="http://localhost:11434"):
    print(f"Testing Ollama API at {base_url}...")
    
    # Test 1: Check if API is responsive
    try:
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            print("✅ API is responsive")
            models = response.json()
            print("\nInstalled models:")
            for model in models['models']:
                print(f"- {model['name']}")
        else:
            print("❌ API returned unexpected status code:", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Ollama API")
        return False

    # Test 2: Test generation with Llama
    print("\nTesting Llama model...")
    try:
        payload = {
            "model": "llama3.1",
            "prompt": "What is 2+2? Reply with just the number.",
            "stream": False
        }
        response = requests.post(f"{base_url}/api/generate", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ Llama model responded successfully")
            print(f"Response: {result['response'].strip()}")
        else:
            print("❌ Llama model test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing Llama model: {str(e)}")
        return False

    # Test 3: Test embeddings with nomic-embed-text
    print("\nTesting nomic-embed-text model...")
    try:
        payload = {
            "model": "nomic-embed-text",
            "prompt": "Hello world"
        }
        response = requests.post(f"{base_url}/api/embeddings", json=payload)
        if response.status_code == 200:
            result = response.json()
            embedding_length = len(result['embedding'])
            print("✅ Embedding model responded successfully")
            print(f"Embedding dimension: {embedding_length}")
        else:
            print("❌ Embedding model test failed")
            return False
    except Exception as e:
        print(f"❌ Error testing embedding model: {str(e)}")
        return False

    print("\n✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    # Allow custom URL from command line argument
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:11434"
    test_ollama_api(base_url)