import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        print("CONFIG_ERROR: Missing QDRANT_URL or QDRANT_API_KEY in environment.")
        return

    try:
        # Initialize client for Qdrant Cloud
        client = QdrantClient(
            url=url, 
            api_key=api_key,
        )
        
        # Perform a read-only operation: get collections
        collections_response = client.get_collections()
        collections = collections_response.collections
        
        print(f"CONNECTION_SUCCESS")
        print(f"COLLECTIONS_COUNT: {len(collections)}")
        for col in collections:
            print(f"COLLECTION_NAME: {col.name}")
            
    except Exception as e:
        # Sanitize error reporting
        err_msg = str(e).lower()
        if "401" in err_msg or "unauthorized" in err_msg:
            print("AUTH_FAILURE: Authentication failed. Check API key.")
        elif "dns" in err_msg or "getaddrinfo" in err_msg:
            print("NETWORK_FAILURE: DNS or Network resolution failed.")
        elif "ssl" in err_msg or "tls" in err_msg:
            print("TLS_ERROR: SSL/TLS connection error.")
        elif "timeout" in err_msg:
            print("TIMEOUT_ERROR: Connection timed out.")
        else:
            print(f"CONNECTION_ERROR: {type(e).__name__}")

if __name__ == "__main__":
    main()
