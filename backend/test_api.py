import urllib.request
import json
import sys

def test_root():
    print("Testing Root...")
    try:
        with urllib.request.urlopen("http://localhost:8000/") as response:
            status = response.getcode()
            body = response.read().decode('utf-8')
            print(f"Status: {status}")
            print(f"Response: {body}")
    except Exception as e:
        print(f"Error: {e}")

def test_predict():
    print("\nTesting Predict (ML Models)...")
    try:
        url = "http://localhost:8000/api/predict"
        payload = json.dumps({"customer_id": "101"}).encode('utf-8')
        req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            body = response.read().decode('utf-8')
            print(f"Status: {status}")
            print(f"Response: {json.dumps(json.loads(body), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_streams():
    print("\nTesting Streams Status...")
    try:
        with urllib.request.urlopen("http://localhost:8000/api/streams/status") as response:
            status = response.getcode()
            body = response.read().decode('utf-8')
            print(f"Status: {status}")
            print(f"Response: {json.dumps(json.loads(body), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_dashboard():
    print("\nTesting Dashboard Overview (Dynamic)...")
    try:
        with urllib.request.urlopen("http://localhost:8000/api/dashboard/overview") as response:
            status = response.getcode()
            body = response.read().decode('utf-8')
            print(f"Status: {status}")
            print(f"Response: {json.dumps(json.loads(body), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_alert_details():
    print("\nTesting Alert Details...")
    try:
        with urllib.request.urlopen("http://localhost:8000/api/details/alert/101") as response:
            status = response.getcode()
            body = response.read().decode('utf-8')
            print(f"Status: {status}")
            print(f"Response: {json.dumps(json.loads(body), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_product_details():
    print("\nTesting Product Details...")
    try:
        # Use quote for URL encoding spaces
        url = "http://localhost:8000/api/details/product/Personal%20Loan"
        with urllib.request.urlopen(url) as response:
            status = response.getcode()
            body = response.read().decode('utf-8')
            print(f"Status: {status}")
            print(f"Response: {json.dumps(json.loads(body), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_root()
    test_predict()
    test_dashboard()
    test_alert_details()
    test_product_details()
    test_streams()
