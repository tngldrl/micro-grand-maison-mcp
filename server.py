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
import math
from PIL import Image

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

def calculate_layout_coordinates(data: dict) -> dict:
    """
    Given the integrated architecture data dictionary containing "layout_pattern"
    and a list of "microservices" (each having a "layout_metadata" dict),
    calculate the 2D pixel coordinates (x, y) for each microservice and store it
    in the microservice's "position" property as {"x": x, "y": y}.
    """
    layout_pattern = data.get("layout_pattern", "mesh")
    microservices = data.get("microservices", [])
    if not microservices:
        return data

    # Safe fallback layout if metadata is missing or calculations error out
    def apply_fallback_layout():
        for i, ms in enumerate(microservices):
            ms["position"] = {
                "x": float(100 + (i * 250)),
                "y": float(100 + ((i % 2) * 200))
            }

    try:
        if layout_pattern == "hierarchical":
            # Group by rank
            ranks = {}
            for ms in microservices:
                meta = ms.get("layout_metadata") or {}
                rank = meta.get("rank")
                if rank is None:
                    rank = 0
                ranks.setdefault(rank, []).append(ms)

            for rank, ms_list in ranks.items():
                ms_list.sort(key=lambda x: (x.get("layout_metadata") or {}).get("index_in_rank", 0) or 0)
                count = len(ms_list)
                for idx, ms in enumerate(ms_list):
                    ms["position"] = {
                        "x": float(400 + (idx - (count - 1) / 2.0) * 350),
                        "y": float(100 + rank * 300)
                    }

        elif layout_pattern == "radial":
            # Find hub
            hubs = [ms for ms in microservices if (ms.get("layout_metadata") or {}).get("is_hub") is True]
            hub = hubs[0] if hubs else None
            
            # Spokes
            spokes = [ms for ms in microservices if ms != hub]
            spokes.sort(key=lambda x: (x.get("layout_metadata") or {}).get("spoke_index", 0) or 0)
            
            if hub:
                hub["position"] = {"x": 500.0, "y": 500.0}
            
            N = len(spokes)
            for i, ms in enumerate(spokes):
                angle = (2.0 * math.pi * i) / N if N > 0 else 0
                ms["position"] = {
                    "x": float(500.0 + 400.0 * math.cos(angle)),
                    "y": float(500.0 + 400.0 * math.sin(angle))
                }
                
        elif layout_pattern == "clustering":
            # Group by cluster_id
            clusters = {}
            for ms in microservices:
                meta = ms.get("layout_metadata") or {}
                c_id = meta.get("cluster_id", 0) or 0
                clusters.setdefault(c_id, []).append(ms)

            # Map cluster_id to a center coordinate
            cluster_centers = {
                0: (300.0, 300.0),
                1: (1000.0, 300.0),
                2: (600.0, 950.0),
                3: (1300.0, 950.0)
            }
            
            for c_id, ms_list in clusters.items():
                center_x, center_y = cluster_centers.get(c_id, (300.0 + c_id * 500.0, 300.0))
                ms_list.sort(key=lambda x: (x.get("layout_metadata") or {}).get("index_in_cluster", 0) or 0)
                M = len(ms_list)
                for j, ms in enumerate(ms_list):
                    angle = (2.0 * math.pi * j) / M if M > 0 else 0
                    ms["position"] = {
                        "x": float(center_x + 180.0 * math.cos(angle)),
                        "y": float(center_y + 180.0 * math.sin(angle))
                    }

        elif layout_pattern == "boundary":
            # Group by boundary_id
            boundaries = {}
            for ms in microservices:
                meta = ms.get("layout_metadata") or {}
                b_id = meta.get("boundary_id", 0) or 0
                boundaries.setdefault(b_id, []).append(ms)

            # Map boundary_id to start coordinates side-by-side
            for b_id, ms_list in boundaries.items():
                start_x = 100.0 + b_id * 700.0
                start_y = 100.0
                ms_list.sort(key=lambda x: (x.get("layout_metadata") or {}).get("index_in_boundary", 0) or 0)
                for idx, ms in enumerate(ms_list):
                    row = idx // 2
                    col = idx % 2
                    ms["position"] = {
                        "x": float(start_x + col * 300.0),
                        "y": float(start_y + row * 300.0)
                    }

        elif layout_pattern == "mesh":
            # Circular layout
            microservices_sorted = sorted(microservices, key=lambda x: (x.get("layout_metadata") or {}).get("index", 0) or 0)
            N = len(microservices_sorted)
            for i, ms in enumerate(microservices_sorted):
                angle = (2.0 * math.pi * i) / N if N > 0 else 0
                ms["position"] = {
                    "x": float(500.0 + 400.0 * math.cos(angle)),
                    "y": float(500.0 + 400.0 * math.sin(angle))
                }

        elif layout_pattern == "matrix":
            for ms in microservices:
                meta = ms.get("layout_metadata") or {}
                row = meta.get("row", 0) or 0
                col = meta.get("col", 0) or 0
                ms["position"] = {
                    "x": float(100.0 + col * 400.0),
                    "y": float(100.0 + row * 300.0)
                }
        else:
            apply_fallback_layout()

    except Exception as err:
        print(f"Error calculating coordinates for layout {layout_pattern}: {err}. Applying fallback.")
        apply_fallback_layout()

    return data

def make_background_transparent(image_path: str):
    """
    Load an image, dynamically detect if the background is green-screen, black, or white,
    and convert the background area to alpha transparency.
    For green screens, global color keying is used to clear enclosed spaces (holes),
    and boundary BFS is used for smooth edges.
    For black/white screens, BFS is used to protect internal parts.
    """
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    pixels = list(img.getdata())
    
    # 四隅のサンプリング
    corners = [
        pixels[0],
        pixels[width - 1],
        pixels[(height - 1) * width],
        pixels[height * width - 1]
    ]
    
    # 相対的色相による頑強な緑判定
    green_corners = sum(1 for r, g, b, a in corners if g > r + 30 and g > b + 30 and g > 80)
    dark_corners = sum(1 for r, g, b, a in corners if r < 60 and g < 60 and b < 60)
    
    new_pixels = []
    
    if green_corners >= 2:
        background_pixels = set()
        
        # 1. 内側の孤立領域 (ホール) を消去するためのグローバル透過 (厳格判定)
        for idx, (r, g, b, a) in enumerate(pixels):
            if g > r + 40 and g > b + 40 and g > 90:
                background_pixels.add((idx % width, idx // width))
                
        # 2. キャラクター外周のエッジやグラデーションを消去するための境界BFS (寛容判定)
        visited = set()
        queue = []
        for x in range(width):
            queue.append((x, 0))
            queue.append((x, height - 1))
        for y in range(1, height - 1):
            queue.append((0, y))
            queue.append((width - 1, y))
            
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            idx = cy * width + cx
            r, g, b, a = pixels[idx]
            
            # 少し緩めの緑色優位度チェック
            if g > r + 20 and g > b + 20 and g > 50:
                background_pixels.add((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if (nx, ny) not in visited:
                            queue.append((nx, ny))
                            
        # ピクセル書き換え
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                r, g, b, a = pixels[idx]
                if (x, y) in background_pixels:
                    new_pixels.append((0, 0, 0, 0))
                else:
                    new_pixels.append((r, g, b, a))
                    
    else:
        # 黒/白背景向け (下位互換性維持用のBFS境界探索)
        is_dark_bg = dark_corners >= 2
        visited = set()
        queue = []
        for x in range(width):
            queue.append((x, 0))
            queue.append((x, height - 1))
        for y in range(1, height - 1):
            queue.append((0, y))
            queue.append((width - 1, y))
            
        background_pixels = set()
        while queue:
            cx, cy = queue.pop(0)
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            
            idx = cy * width + cx
            r, g, b, a = pixels[idx]
            
            should_remove = False
            if is_dark_bg:
                if r < 35 and g < 35 and b < 35:
                    should_remove = True
            else:
                if r > 220 and g > 220 and b > 220:
                    should_remove = True
                    
            if should_remove:
                background_pixels.add((cx, cy))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        if (nx, ny) not in visited:
                            queue.append((nx, ny))
                            
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                r, g, b, a = pixels[idx]
                if (x, y) in background_pixels:
                    new_pixels.append((0, 0, 0, 0))
                else:
                    new_pixels.append((r, g, b, a))
                    
    img.putdata(new_pixels)
    img.save(image_path, "PNG")

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
        data = calculate_layout_coordinates(data)
        
        # PHASE 4: Avatars
        microservices = data.get("microservices", [])

        # Clean up stale avatar files from previous analyses of the same project.
        # Only files prefixed with this project's safe_project_id are removed;
        # other projects' files are left untouched.
        safe_project_id = "".join([c for c in project_id if c.isalpha() or c.isdigit() or c=='-']).lower()
        prefix = f"{safe_project_id}_"
        for existing_file in os.listdir(output_dir):
            if existing_file.startswith(prefix) and existing_file.endswith(".png"):
                try:
                    os.remove(os.path.join(output_dir, existing_file))
                    print(f"Removed stale avatar: {existing_file}")
                except Exception as rm_err:
                    print(f"Could not remove stale avatar {existing_file}: {rm_err}")

        for ms in microservices:
            name = ms.get("name", "unknown")
            prompt = ms.get("avatar_prompt", "")
            
            if not prompt:
                continue
                
            safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(" ", "_").lower()
            # Include project_id in the filename to avoid cross-project collisions
            image_filename = f"{safe_project_id}_{safe_name}.png"
            image_path = os.path.join(output_dir, image_filename)
            
            # Always regenerate to reflect the latest prompt (no skip-if-exists)
            if generate_avatar(prompt, image_path, GCP_PROJECT_ID):
                try:
                    make_background_transparent(image_path)
                    print(f"Successfully removed background from avatar for {name}")
                except Exception as bg_err:
                    print(f"Failed to remove background from avatar for {name}: {bg_err}")
            
            mcp_service_url = os.getenv("MCP_SERVICE_URL", "http://localhost:8001")
            ms["avatar_image_url"] = f"{mcp_service_url}/static/avatars/{image_filename}"
            
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
