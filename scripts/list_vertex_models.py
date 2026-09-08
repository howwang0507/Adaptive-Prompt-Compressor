import os
from google.cloud import aiplatform

def list_models():
    key_path = "/Users/wmh/Downloads/gen-lang-client-0752774059-73ec68aeb54b.json"
    project_id = "gen-lang-client-0752774059"
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    
    regions = ["us-central1", "asia-east1", "us-east1"]
    
    for region in regions:
        print(f"\n--- Checking Region: {region} ---")
        try:
            aiplatform.init(project=project_id, location=region)
            models = aiplatform.Model.list()
            if not models:
                print(f"No models found in {region} (or no permission to list).")
            else:
                for model in models:
                    print(f"Found Model: {model.display_name} ({model.resource_name})")
        except Exception as e:
            print(f"Error listing models in {region}: {str(e)}")

if __name__ == "__main__":
    list_models()
