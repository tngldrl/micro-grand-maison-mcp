from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import tempfile
import subprocess
import shutil
import httpx
from dotenv import load_dotenv

from scanner import scan_code_chunks, scan_metadata
from analyzer import extract_skeleton, extract_partial_graph, synthesize_architecture, chat_with_character
from generator import generate_avatar

load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

app = FastAPI(title="Architecture World MCP Server")

# Serve avatars statically
output_dir = "output/avatars"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
app.mount("/static/avatars", StaticFiles(directory=output_dir), name="avatars")

class AnalyzeRequest(BaseModel):
    repo_urls: list[str]
    project_id: str
    callback_url: str

def run_async_analysis(repo_urls: list[str], callback_url: str, project_id: str):
    cloned_dirs = []
    repo_configs = []
    
    try:
        # Clone each repository dynamically
        for url in repo_urls:
            url_str = url.strip()
            if not url_str:
                continue
                
            temp_dir = tempfile.mkdtemp(prefix="repo-")
            cloned_dirs.append(temp_dir)
            
            # Run git clone --depth 1
            print(f"Cloning {url_str} to {temp_dir}...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", url_str, temp_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"Failed to clone repository {url_str}: {result.stderr}")
                
            repo_configs.append((temp_dir, url_str))
            
        if not repo_configs:
            raise Exception("No valid repositories to analyze.")

        # PHASE 1
        metadata_context = scan_metadata(repo_configs)
        skeleton_json = extract_skeleton(metadata_context, GCP_PROJECT_ID)
        
        # PHASE 2
        code_chunks = scan_code_chunks(repo_configs)
        partial_graphs = []
        for i, chunk_context in enumerate(code_chunks):
            try:
                partial_graph = extract_partial_graph(chunk_context, skeleton_json, GCP_PROJECT_ID)
                partial_graphs.append(partial_graph)
            except Exception as e:
                print(f"Failed Chunk {i+1}: {e}")
                
        if not partial_graphs:
            raise Exception("No chunks were successfully analyzed.")
            
        # PHASE 3
        final_architecture_json = synthesize_architecture(partial_graphs, GCP_PROJECT_ID)
        data = json.loads(final_architecture_json)
        
        # PHASE 4: Avatars
        microservices = data.get("microservices", [])
        for ms in microservices:
            name = ms.get("name", "unknown")
            prompt = ms.get("avatar_prompt", "")
            
            if not prompt:
                continue
                
            safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(" ", "_").lower()
            image_filename = f"{safe_name}.png"
            image_path = os.path.join(output_dir, image_filename)
            
            # Generate Avatar if not exists
            if not os.path.exists(image_path):
                generate_avatar(prompt, image_path, GCP_PROJECT_ID)
            
            # Update the avatar URL in the response
            # Note: Hardcoded to localhost:8001 for MVP
            ms["avatar_image_url"] = f"http://localhost:8001/static/avatars/{image_filename}"
            
        # Send callback with success status
        callback_payload = {
            "project_id": project_id,
            "status": "success",
            "data": data
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(callback_url, json=callback_payload)
            resp.raise_for_status()
            print(f"Callback sent successfully to {callback_url}")

    except Exception as e:
        print(f"Analysis failed for project {project_id}: {e}")
        # Send callback with error status
        callback_payload = {
            "project_id": project_id,
            "status": "error",
            "error": str(e)
        }
        try:
            with httpx.Client(timeout=60.0) as client:
                client.post(callback_url, json=callback_payload)
        except Exception as callback_err:
            print(f"Failed to send error callback to {callback_url}: {callback_err}")
    finally:
        # Cleanup cloned directories
        for temp_dir in cloned_dirs:
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as cleanup_err:
                print(f"Error cleaning up temporary directory {temp_dir}: {cleanup_err}")

@app.post("/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    if not GCP_PROJECT_ID:
        raise HTTPException(status_code=500, detail="GCP_PROJECT_ID is not configured.")
        
    background_tasks.add_task(
        run_async_analysis,
        req.repo_urls,
        req.callback_url,
        req.project_id
    )
    return {"status": "queued"}

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    system_prompt: str
    history: list[Message]
    new_message: str

@app.post("/chat")
def chat(req: ChatRequest):
    if not GCP_PROJECT_ID:
        raise HTTPException(status_code=500, detail="GCP_PROJECT_ID is not configured.")
    try:
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in req.history]
        response_text = chat_with_character(
            system_prompt=req.system_prompt,
            history=history_dicts,
            new_message=req.new_message,
            project_id=GCP_PROJECT_ID
        )
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
