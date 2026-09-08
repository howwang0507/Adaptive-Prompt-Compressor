import json
import urllib.request
import urllib.error

def test_vertex_api_key(api_key):
    projects = ["gen-lang-client-0752774059", "519205263973"]
    location = "us-central1"
    models = ["gemini-1.5-flash", "gemini-2.0-flash-001"]
    
    print("🚀 Testing Vertex AI / Agent Platform via REST with API Key...")
    
    for project_id in projects:
        for model_id in models:
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent?key={api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [{"role": "user", "parts": [{"text": "Hello, are you there?"}]}]
            }
            
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers, method="POST")
            
            print(f"\n--- Trying Project: {project_id} | Model: {model_id} ---")
            try:
                with urllib.request.urlopen(req) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    print("✅ SUCCESS! Response:")
                    try:
                        print(result['candidates'][0]['content']['parts'][0]['text'])
                        return
                    except KeyError:
                        print(result)
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8")
                print(f"❌ HTTP {e.code}: {error_body}")
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    
    test_vertex_api_key(args.key)
