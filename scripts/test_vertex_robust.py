import os
import vertexai
from vertexai.generative_models import GenerativeModel
import json

def test_vertex_robust():
    key_path = "/Users/wmh/Downloads/gen-lang-client-0752774059-73ec68aeb54b.json"
    project_id = "gen-lang-client-0752774059"
    
    print(f"🚀 Final Robust Test for Project: {project_id}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    
    # Try multiple regions and model name variations
    configs = [
        {"region": "us-central1", "model": "gemini-2.0-flash-001"},
        {"region": "us-central1", "model": "gemini-1.5-flash-001"},
        {"region": "us-central1", "model": "gemini-1.5-flash-002"},
        {"region": "us-central1", "model": "gemini-1.5-flash"},
        {"region": "asia-east1", "model": "gemini-1.5-flash-001"}
    ]
    
    for config in configs:
        print(f"\nTrying {config['model']} in {config['region']}...")
        try:
            vertexai.init(project=project_id, location=config['region'])
            model = GenerativeModel(config['model'])
            response = model.generate_content("Ping")
            print(f"✅ SUCCESS! Region: {config['region']}, Model: {config['model']}")
            print(f"Response: {response.text.strip()}")
            return True
        except Exception as e:
            print(f"❌ Failed {config['model']} in {config['region']}: {str(e)}")
            
    return False

if __name__ == "__main__":
    test_vertex_robust()
