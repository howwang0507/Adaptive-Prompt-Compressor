import os
import json
import urllib.request
import urllib.error

def get_access_token(service_account_file):
    # Use gcloud to get the token for the service account
    import subprocess
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file
    result = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"], 
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Failed to get token:", result.stderr)
        return None
    return result.stdout.strip()

def test_vertex_rest():
    key_path = "/Users/wmh/Downloads/gen-lang-client-0752774059-73ec68aeb54b.json"
    project_id = "gen-lang-client-0752774059"
    location = "us-central1"
    
    # Let's try 1.5-flash and 2.0-flash-001
    models = ["gemini-1.5-flash", "gemini-2.0-flash-001"]
    
    token = get_access_token(key_path)
    if not token:
        return
        
    print(f"🚀 Testing Vertex AI directly via REST API for project: {project_id}")
    
    for model_id in models:
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{"role": "user", "parts": [{"text": "Hello, are you there?"}]}]
        }
        
        req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
        
        print(f"\n--- Trying Model: {model_id} ---")
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                print("✅ SUCCESS! Response:")
                try:
                    print(result['candidates'][0]['content']['parts'][0]['text'])
                    # If we succeed, we can break and use this!
                    return
                except KeyError:
                    print(result)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            print(f"❌ HTTP {e.code}: {error_body}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_vertex_rest()
