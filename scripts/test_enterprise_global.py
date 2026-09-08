import os
import json
import urllib.request
import urllib.error

def test_enterprise_global(api_key):
    project_id = "gen-lang-client-0752774059"
    # CRITICAL FINDING: Enterprise Agent Platform uses 'global' location
    location = "global"
    
    # Models explicitly mentioned in the Enterprise docs
    models = ["gemini-1.5-flash", "gemini-2.0-flash-001", "gemini-2.5-flash"]
    
    print(f"🚀 Precision Test: Enterprise Global Endpoint")
    print(f"Project: {project_id} | Location: {location}\n")
    
    for model_id in models:
        # Note the endpoint is aiplatform.googleapis.com (no region prefix)
        url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent?key={api_key}"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "Connectivity check for NeurIPS paper."}]}],
            "generationConfig": {"maxOutputTokens": 20}
        }
        
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        
        print(f"Trying: {model_id}...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                print("✅ SUCCESS!")
                print(f"   Response: {result['candidates'][0]['content']['parts'][0]['text'].strip()}")
                return # If one works, we found the path!
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            print(f"❌ Error {e.code}: {err_body}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    test_enterprise_global(args.key)
