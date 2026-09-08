import os
import json
import urllib.request
import urllib.error

def test_multiple_models_enterprise(api_key):
    project_id = "gen-lang-client-0752774059"
    location = "us-central1"
    
    # Exhaustive list of potential model IDs in Vertex/Enterprise
    model_list = [
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-001",
        "gemini-1.0-pro-001",
        "gemini-2.0-flash-exp",
        "gemini-2.0-flash-001",
        "text-bison@002",
        "chat-bison@002"
    ]
    
    print(f"🚀 Exhaustive Model Sweep for Project: {project_id}")
    print("Testing across Gemini and PaLM families via Enterprise REST API...\n")
    
    success_found = False

    for model_id in model_list:
        url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:generateContent?key={api_key}"
        
        # Bison models use a slightly different endpoint structure in some versions, 
        # but let's try the generativeContent standard first.
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
            "generationConfig": {"maxOutputTokens": 10}
        }
        
        # Special payload for older Bison models if needed
        if "bison" in model_id:
            payload = {
                "instances": [{"content": "ping"}],
                "parameters": {"maxOutputTokens": 10}
            }
            url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models/{model_id}:predict?key={api_key}"

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        
        print(f"Trying: {model_id}...", end=" ", flush=True)
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode("utf-8"))
                print("✅ SUCCESS!")
                print(f"   Response: {result}")
                success_found = True
                # Don't break, let's see what else works
        except urllib.error.HTTPError as e:
            print(f"❌ Error {e.code}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

    if not success_found:
        print("\n🏁 Sweep Complete. No models found in this region for this project.")
    else:
        print("\n🎉 Sweep Complete. Some models are accessible!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    test_multiple_models_enterprise(args.key)
