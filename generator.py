import os
import time
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def generate_avatar(prompt: str, output_path: str, project_id: str, locations: list[str] = None) -> bool:
    """
    Generate an avatar image using Vertex AI Imagen 3 and save it to the local disk.
    Supports a list of target locations with automatic fallback in case of transient quota errors.
    """
    if locations is None:
        locations = ["us-central1", "us-east4", "europe-west9", "asia-northeast1"]
        
    max_retries = 3
    base_delay = 5
    
    for loc_idx, location in enumerate(locations):
        print(f"Initializing Vertex AI at location: {location} (Attempting {loc_idx + 1}/{len(locations)})")
        try:
            vertexai.init(project=project_id, location=location)
        except Exception as init_err:
            print(f"Failed to initialize Vertex AI at location {location}: {init_err}")
            if loc_idx < len(locations) - 1:
                print("Attempting fallback to next location...")
                continue
            return False
            
        for attempt in range(max_retries):
            try:
                # Use the latest Imagen 3 model
                model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
                
                print(f"Generating image for prompt in {location}: {prompt}")
                
                response = model.generate_images(
                    prompt=prompt,
                    number_of_images=1,
                    aspect_ratio="1:1",
                    # We want high quality, clear background
                    negative_prompt="photorealistic, photography, realistic, 3d render, complex background, scenery, environment, landscape, room, messy, low resolution, ugly, disfigured, text, words",
                    guidance_scale=7.5
                )
                
                if response.images:
                    image = response.images[0]
                    image.save(output_path)
                    print(f"Successfully saved image to {output_path} using location {location}")
                    return True
                else:
                    print(f"No image was returned from the model at location {location}.")
                    return False
                    
            except Exception as e:
                err_msg = str(e)
                # Check for transient quota/resource errors
                is_transient = (
                    "429" in err_msg or 
                    "Quota" in err_msg or 
                    "ResourceExhausted" in err_msg or 
                    "Resource exhausted" in err_msg
                )
                
                if is_transient:
                    if attempt < max_retries - 1:
                        sleep_time = base_delay * (2 ** attempt)
                        print(f"[{location} - Quota Exceeded] Retrying in {sleep_time} seconds... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(sleep_time)
                        continue
                    else:
                        print(f"[{location} - Quota Exceeded] All retries exhausted at location {location}.")
                        break
                else:
                    # For other errors (permission, invalid prompt, regional support, model not visible, etc.),
                    # print the error but fall back to the next location instead of aborting the entire function.
                    print(f"Error generating image at location {location}: {e}")
                    break
                    
    return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python generator.py <gcp_project_id> <output_file_path> <prompt>")
        sys.exit(1)
        
    project_id = sys.argv[1]
    output_file = sys.argv[2]
    user_prompt = " ".join(sys.argv[3:])
    
    generate_avatar(user_prompt, output_file, project_id)
