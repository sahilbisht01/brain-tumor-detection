import requests
import os

# Test backend API with a sample image
def test_backend_api():
    print("="*70)
    print("BACKEND API TEST")
    print("="*70)
    
    # Backend URL
    url = "http://localhost:5000/predict"
    
    # Test image path
    test_image = "dataset/Testing/glioma/Te-gl_0010.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"\n📤 Uploading test image: {test_image}")
    print(f"   Expected class: glioma")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Backend responded successfully!")
            print(f"\n📊 Prediction Results:")
            print(f"   Label:      {result.get('label', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A')}")
            print(f"   Class:      {result.get('class', 'N/A')}")
            
            if 'all_probabilities' in result:
                print(f"\n📈 All Class Probabilities:")
                for class_name, prob in result['all_probabilities'].items():
                    print(f"   {class_name:12s}: {prob}")
            
            # Check if prediction is correct
            if result.get('class') == 'glioma':
                print(f"\n✅ CORRECT PREDICTION! Expected: glioma, Got: {result.get('class')}")
            else:
                print(f"\n❌ INCORRECT PREDICTION! Expected: glioma, Got: {result.get('class')}")
        else:
            print(f"\n❌ Backend error: Status {response.status_code}")
            print(f"   Response: {response.text}")
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Could not connect to backend at {url}")
        print(f"   Make sure the backend server is running!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    test_backend_api()
