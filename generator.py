import os
import time
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def generate_avatar(prompt: str, output_path: str, project_id: str, location: str = "us-central1") -> bool:
    """
    Generate an avatar image using Vertex AI Imagen 3 and save it to the local disk.
    """
    vertexai.init(project=project_id, location=location)
    
    max_retries = 5
    base_delay = 20
    
    for attempt in range(max_retries):
        try:
            # Use the latest Imagen 3 model (adjust version string as available in the GCP project)
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
            
            print(f"Generating image for prompt: {prompt}")
            
            # In Phase 1, we save locally. In Phase 3, this will upload to Google Cloud Storage.
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
                print(f"Successfully saved image to {output_path}")
                return True
            else:
                print("No image was returned from the model.")
                return False
                
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt)
                    print(f"[429 Quota Exceeded] Retrying in {sleep_time} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(sleep_time)
                    continue
            print(f"Error generating image: {e}")
            return False
            
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
