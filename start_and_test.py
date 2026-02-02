"""
Automated script to start server and run tests
This script will start the server and then run the tests
"""
import subprocess
import sys
import time
import requests
import os

def check_port_available(port=5000):
    """Check if port is available"""
    try:
        response = requests.get(f'http://localhost:{port}/api/health', timeout=1)
        return False  # Port is in use
    except requests.exceptions.ConnectionError:
        return True  # Port is available
    except:
        return True

def start_server():
    """Start the Flask server in a subprocess"""
    print("=" * 60)
    print("Starting Flask Server")
    print("=" * 60)
    
    # Check if port is already in use
    if not check_port_available():
        print("⚠️  Port 5000 is already in use!")
        print("   Another server may be running.")
        print("   If not, kill the process using port 5000:")
        print("   netstat -ano | findstr :5000")
        return None
    
    try:
        # Start server as subprocess
        process = subprocess.Popen(
            [sys.executable, 'start_server.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        print("✅ Server process started")
        print("   Waiting for server to be ready...")
        
        # Wait for server to be ready (max 60 seconds)
        max_wait = 60
        waited = 0
        while waited < max_wait:
            try:
                response = requests.get('http://localhost:5000/api/health', timeout=2)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    return process
            except:
                pass
            
            time.sleep(2)
            waited += 2
            if waited % 10 == 0:
                print(f"   Still waiting... ({waited}/{max_wait} seconds)")
        
        print("❌ Server did not start in time")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return None

def run_tests():
    """Run the SARI improvements test"""
    print("\n" + "=" * 60)
    print("Running SARI Improvements Test")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            [sys.executable, 'test_sari_improvements.py'],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("SARI Improvements Test - Automated Runner")
    print("=" * 60)
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Start server
    server_process = start_server()
    
    if not server_process:
        print("\n❌ Failed to start server")
        print("\nPlease start the server manually:")
        print("   python start_server.py")
        print("\nThen run tests in another terminal:")
        print("   python test_sari_improvements.py")
        sys.exit(1)
    
    try:
        # Run tests
        success = run_tests()
        
        if success:
            print("\n✅ Tests completed successfully!")
        else:
            print("\n⚠️  Tests completed with some issues")
        
    finally:
        # Stop server
        print("\n" + "=" * 60)
        print("Stopping server...")
        print("=" * 60)
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
            print("✅ Server stopped")
        except:
            print("⚠️  Force killing server...")
            server_process.kill()
        
        print("\nDone!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)

