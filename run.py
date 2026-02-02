#!/usr/bin/env python3
"""
Development runner for Sinhala Text Simplifier
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """Check if required files exist"""
    required_files = [
        'sinhala_text_simplification_results.pkl',
        'final_sinhala_simplifier'
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print("Missing required files:")
        for file in missing:
            print(f"   - {file}")
        print("\nPlease copy your model files to the project directory:")
        print("1. sinhala_text_simplification_results.pkl")
        print("2. final_sinhala_simplifier/ directory")
        return False
    
    print("All required files found")
    return True

def main():
    """Main function"""
    print("Starting Sinhala Text Simplifier...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Import and run app
    try:
        from app import app
        print("Flask app imported successfully")
        print("Starting server at http://localhost:5000")
        print("The interface will be available in your browser")
        print("=" * 50)
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            threaded=True
        )
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()