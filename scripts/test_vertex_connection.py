import os
import vertexai
from vertexai.generative_models import GenerativeModel

def test_vertex():
    key_path = "/Users/wmh/Downloads/gen-lang-client-0752774059-73ec68aeb54b.json"
    project_id = "gen-lang-client-0752774059"
    
    print(f"Initializing Vertex AI with project: {project_id}...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    
    try:
        vertexai.init(project=project_id, location="us-central1")
        # Try a specific version if the base name fails
        model = GenerativeModel("gemini-1.5-flash-002")
        
        print("Sending test request...")
        response = model.generate_content("Hello, this is a test from the prompt compressor project. Are you there?")
        
        print("\n✅ Success! Vertex AI Response:")
        print(response.text)
        return True
    except Exception as e:
        print(f"\n❌ Vertex AI Test Failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_vertex()
