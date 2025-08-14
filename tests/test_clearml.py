# ClearML Installation Test Script
import requests
import json

def test_clearml_server():
    """Test ClearML server connectivity"""
    try:
        # Test API server
        api_response = requests.get("http://localhost:8008/debug.ping")
        print(f"✅ API Server Status: {api_response.status_code}")
        
        # Test Web interface
        web_response = requests.get("http://localhost:8080")
        print(f"✅ Web Interface Status: {web_response.status_code}")
        
        # Test File server
        files_response = requests.get("http://localhost:8081")
        print(f"✅ File Server Status: {files_response.status_code}")
        
        print("\n🎉 ClearML server is running successfully!")
        print("\n📋 Access Information:")
        print("   Web Interface: http://localhost:8080")
        print("   API Server:    http://localhost:8008")
        print("   File Server:   http://localhost:8081")
        
        return True
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_clearml_server()
