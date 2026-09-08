import json
import urllib.request
import urllib.error
import subprocess

def get_gcloud_access_token():
    """Generates an access token using the user's current gcloud login."""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-access-token"], 
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"❌ Failed to get gcloud token: {e}")
        return None

def test_enterprise_oauth():
    # From your documentation: 
    # Base URL: {api_endpoint}/v1/projects/{project_id}/locations/{location_id}/publishers/google/models/{model_id}:generateContent
    project_id = "gen-lang-client-0752774059"
    location = "us-central1"
    model_id = "gemini-1.5-flash" # We can also try gemini-2.0-flash-001
    
    token = get_gcloud_access_token()
    if not token:
        print("Please run 'gcloud auth login' in your terminal first.")
        return

    url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent"
    
    print("🚀 Testing Enterprise API via OAuth 2.0 Bearer Token...")
    print(f"Project: {project_id} | Model: {model_id}")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Testing connectivity for NeurIPS paper."}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 50}
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            print("✅ SUCCESS! Response received:")
            print(result['candidates'][0]['content']['parts'][0]['text'])
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"❌ HTTP {e.code} Error:")
        print(error_body)
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_enterprise_oauth()
