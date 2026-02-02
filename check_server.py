"""
Quick script to check if server is running
"""
import requests
import sys

def check_server():
    """Check if Flask server is running"""
    try:
        print("Checking if server is running...")
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is running!")
            print(f"   Status: {data.get('status')}")
            print(f"   Model loaded: {data.get('model_loaded')}")
            print(f"   Data loaded: {data.get('data_loaded')}")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Server is NOT running")
        print("\nTo start the server, run:")
        print("   python start_server.py")
        print("   OR")
        print("   python run.py")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False

if __name__ == "__main__":
    if check_server():
        sys.exit(0)
    else:
        sys.exit(1)

