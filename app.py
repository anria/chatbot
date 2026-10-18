# modal deploy app.py
# https://<your-username>--ollama-gateway-api.modal.run"
# 

import os
import subprocess
import time
from modal import App, Image, Volume, asgi_app, fastapi_endpoint

# Define the model requested
MODEL_NAME = "granite3.1-moe:1b"
# HOST = "https://anria-ai-bots-ollama-gateway-api.us-east.modal.run"
HOST = "http://localhost:11434"


# 1. Define a build-time function to pull the model files
def download_granite_model():
    """Starts Ollama in the background and bakes the model weights into the container image layer."""
    # Start the server in the background
    server_process = subprocess.Popen(["ollama", "serve"])
    
    # Wait for the Ollama server to become available
    print("Waiting for Ollama service to start up...")
    for _ in range(30):
        try:
            import httpx
            response = httpx.get("http://127.0.0.1:11434")
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        server_process.terminate()
        raise TimeoutError("Ollama service failed to start during the build stage.")

    # Pull the required Granite 1B model into the image cache
    print("Pulling {MODEL_NAME}...")
    subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
    
    # Clean up the background server process cleanly
    server_process.terminate()
    server_process.wait()

# Create a persistent volume to cache downloaded Ollama models
# This prevents downloading the model from scratch on every cold start
ollama_volume = Volume.from_name("ollama-models-cache", create_if_missing=True)

# 2. Configure the container image environment
ollama_image = (
    Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.11")
    #Image.debian_slim()
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh",)
    .pip_install("fastapi[standard]", "ollama", "httpx")
    .run_function(download_granite_model)
)

app = App(name="ollama-gateway", image=ollama_image)

@app.function()
def generate_text(prompt: str):
    import ollama
    
    # Ensure the background server daemon is up for runtime execution
    server_process = subprocess.Popen(["ollama", "serve"])
    time.sleep(2)  # Short pause to let it bind to the local port
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        return response["response"]
    finally:
        server_process.terminate()
        server_process.wait()

@fastapi_endpoint(docs=True)
def api():
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import httpx

    fastapi_app = FastAPI(title="Ollama API Gateway")

    # Start the Ollama background service when the container launches
    print("Starting Ollama background service...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for the Ollama server to become responsive
    for _ in range(30):
        try:
            with httpx.Client() as client:
                if client.get("http://localhost:11434").status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.5)

    # Automatically pull the requested granite model if it isn't already cached
    print(f"Checking/Pulling model: {MODEL_NAME}...")
    subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
    print(f"Model {MODEL_NAME} is ready!")

    # Reverse proxy middleware to forward incoming API traffic directly to local Ollama
    @fastapi_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def proxy_ollama(path: Request, path_name: str = ""):
        req_path = path.url.path
        if path.url.query:
            req_path += f"?{path.url.query}"

        url = f"http://localhost:11434{req_path}"
        
        async def stream_response():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    method=path.method,
                    url=url,
                    headers=dict(path.headers),
                    content=await path.body()
                ) as response:
                    async for chunk in response.aiter_raw():
                        yield chunk

        return StreamingResponse(stream_response())

    return fastapi_app

@asgi_app()
def web_entrypoint():
    import httpx
    
    # 1. Boot up the Ollama background daemon on CPU
    print("Starting background Ollama service...")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Wait for Ollama loopback to become healthy
    for _ in range(30):
        try:
            with httpx.Client() as client:
                if client.get("http://localhost:11434").status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.5)

    # 3. Securely pull/verify your local model weights inside the container volume
    print(f"Ensuring model availability: {MODEL_NAME}...")
    subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
    print(f"Model {MODEL_NAME} is active!")

    # 4. Dynamically import and expose the app object declared inside your main.py
    from main import app as user_web_app
    return user_web_app    