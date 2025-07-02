#!/usr/bin/env python3
"""
Deployment script for Ball Tracking Web Application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'opencv-python', 'numpy', 'pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    else:
        print("All dependencies are installed!")

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'results', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")

def setup_environment():
    """Set up environment variables"""
    env_vars = {
        'FLASK_ENV': 'production',
        'FLASK_DEBUG': '0',
        'MAX_CONTENT_LENGTH': '104857600',  # 100MB
    }
    
    print("Environment variables to set:")
    for key, value in env_vars.items():
        print(f"export {key}={value}")

def main():
    print("🚀 Setting up Ball Tracking Web Application...")
    
    # Check and install dependencies
    check_dependencies()
    
    # Create directories
    create_directories()
    
    # Setup environment
    setup_environment()
    
    print("\n✅ Setup complete!")
    print("\nTo run the application:")
    print("1. Set environment variables (see above)")
    print("2. Run: python -m ball_tracking.web")
    print("3. Open http://localhost:5000 in your browser")
    
    print("\nFor production deployment:")
    print("1. Use a WSGI server like Gunicorn:")
    print("   pip install gunicorn")
    print("   gunicorn -w 4 -b 0.0.0.0:5000 ball_tracking.web:app")
    print("2. Set up a reverse proxy (nginx/apache)")
    print("3. Use environment variables for configuration")

if __name__ == '__main__':
    main() 