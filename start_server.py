"""
Simple script to start the Flask server
Use this if run.py doesn't work
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("=" * 60)
    print("Starting Sinhala Text Simplifier Server")
    print("=" * 60)
    
    # Check if sari_optimizer exists
    try:
        from sari_optimizer import select_best_candidate_with_sari
        print("✅ sari_optimizer module imported successfully")
    except ImportError as e:
        print(f"❌ Error importing sari_optimizer: {e}")
        print("Make sure sari_optimizer.py is in the same directory as app.py")
        sys.exit(1)
    
    # Import and start app
    try:
        from app import app
        print("✅ Flask app imported successfully")
        
        print("\n" + "=" * 60)
        print("Server starting on http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True,
            use_reloader=False  # Disable reloader to avoid issues
        )
        
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except KeyboardInterrupt:
    print("\n\nServer stopped by user")
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

