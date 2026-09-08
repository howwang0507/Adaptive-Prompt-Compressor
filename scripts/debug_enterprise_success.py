import os
import json
import urllib.request
import urllib.error

def debug_enterprise_success(api_key):
    project_id = "gen-lang-client-0752774059"
    location = "global"
    model_id = "gemini-2.5-flash"
    
    url = f"https://aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": "Hello! Please reply with 'connected'."}]}],
        "generationConfig": {"maxOutputTokens": 20}
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    
    print(f"🚀 Final Debugging for: {model_id}")
    try:
        with urllib.request.urlopen(req) as response:
            raw_res = response.read().decode("utf-8")
            result = json.loads(raw_res)
            print("✅ RAW SUCCESS JSON RECEIVED:")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Error during debug: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    debug_enterprise_success(args.key)
