#!/usr/bin/env python3
"""
Health check script for the Bean Classification API
"""

import sys
import requests
import json
from typing import Dict, Any

def check_health() -> Dict[str, Any]:
    """Check the health of the API service"""
    try:
        # Check basic health endpoint
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code != 200:
            return {
                "status": "unhealthy",
                "error": f"Health endpoint returned {response.status_code}",
                "details": response.text
            }
        
        # Check model info endpoint
        model_response = requests.get("http://localhost:8000/model/info", timeout=10)
        if model_response.status_code != 200:
            return {
                "status": "unhealthy",
                "error": f"Model info endpoint returned {model_response.status_code}",
                "details": model_response.text
            }
        
        model_info = model_response.json()
        
        return {
            "status": "healthy",
            "api_version": response.json().get("version", "unknown"),
            "model_loaded": model_info.get("model_loaded", False),
            "model_format": model_info.get("model_format", "unknown"),
            "timestamp": model_info.get("timestamp")
        }
        
    except requests.exceptions.ConnectionError:
        return {
            "status": "unhealthy",
            "error": "Cannot connect to API service"
        }
    except requests.exceptions.Timeout:
        return {
            "status": "unhealthy",
            "error": "API service timeout"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": f"Unexpected error: {str(e)}"
        }

def main():
    """Main health check function"""
    result = check_health()
    
    print(json.dumps(result, indent=2))
    
    if result["status"] == "healthy":
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()