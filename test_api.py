import os
import requests
import base64
import json

# Configuration
API_URL = "http://localhost:5000/api/filter"
REF_IMAGE_PATH = "reference_face.jpg"
# A small public Google Drive folder containing some sample images of faces
# (using a folder ID that we know exists or can test)
TEST_FOLDER_URL = "test_gallery"

def test_api():
    print("=== Pixify Backend API Integration Test ===")
    
    # 1. Check if reference_face.jpg exists
    if not os.path.exists(REF_IMAGE_PATH):
        print(f"[ERROR] Reference image not found at {REF_IMAGE_PATH}")
        print("Creating a mock reference image for testing...")
        # Create a small dummy image for test compilation if it's missing
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save(REF_IMAGE_PATH)
        print(f"Created dummy image {REF_IMAGE_PATH}")
        
    # 2. Encode reference image to base64
    print("Encoding reference image to base64...")
    with open(REF_IMAGE_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    base64_payload = f"data:image/jpeg;base64,{encoded_string}"
    
    # 3. Construct payload
    payload = {
        "google_drive_link": TEST_FOLDER_URL,
        "reference_image": base64_payload,
        "similarity_threshold": 0.35, # Use loose threshold for testing
        "max_results": 5
    }
    
    # 4. Send request
    print(f"Sending POST request to {API_URL}...")
    try:
        response = requests.post(
            API_URL, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60 # Google Drive download might take some time
        )
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("API response received successfully!")
            print(f"Status: {result.get('status')}")
            results_list = result.get('results', [])
            print(f"Total matching images: {len(results_list)}")
            
            for idx, item in enumerate(results_list, 1):
                print(f"Match #{idx}: {item.get('filename')} - Similarity: {item.get('similarity')} - Size: {item.get('size')} - Res: {item.get('dimensions')}")
                # Print first few chars of base64 mock_thumbnail to verify
                thumb = item.get('mock_thumbnail', '')
                print(f"   Thumbnail: {thumb[:50]}...")
        else:
            print(f"[FAILED] Error response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to Flask server. Is it running on http://localhost:5000?")
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")

if __name__ == "__main__":
    test_api()
