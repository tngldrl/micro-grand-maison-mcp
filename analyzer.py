import os
import json
import vertexai
from vertexai.generative_models import GenerativeModel

WORLD_SETTING = """
WORLD SETTING FOR AVATARS:
The ecosystem must be represented as "A restaurant where various land, sea, and air animals work."
You must assign one of the following restaurant staff roles to each microservice based on its technical function, and describe the character as an animal performing that role.

Roles:
1. Service Staff (Front of House)
- Manager: Overall responsibility, sales, shift management, customer relations.
- Captain (Head Waiter): Floor leader, guiding guests, taking orders, directing timing.
- Commis (Waiter/Server): Main floor staff, serving dishes/drinks, clearing tables.
- Sommelier: Beverage professional, wine pairing, quality control.
- Host/Hostess (Receptionist): The "face" of the shop, greeting, reservation management.
- Bartender: Preparing drinks at the counter.
- Basser (Runner): Clearing tables, water refills, transporting food from kitchen to floor.

2. Kitchen Staff (Back of House)
- Chef de Cuisine (Head Chef): Kitchen boss, menu dev, ingredient sourcing, quality check.
- Sous Chef (Deputy Chef): Right hand to chef, directing cooking, guiding staff.
- Chef de Partie: Saucier (meat/sauce), Poissonnier (fish), Entremetier (soup/veg/egg), Garde-manger (cold dishes/inventory).
- Commis (Prep Cook): Prep work, simple cooking, plating.
- Patissier: Desserts, baked goods, bread.
- Dishwasher (Steward): Washing dishes, cleaning, managing equipment.

For each microservice, the `role_type` should be one of the above titles, and the `avatar_prompt` MUST describe an animal character performing this exact role in a bustling restaurant.

CRITICAL VISUAL CONSTRAINTS for `avatar_prompt`:
1. Scale/Complexity -> Animal Size/Type: If the service is small/low complexity, choose a small agile animal (e.g., squirrel, bird). If it is large/complex, choose a large imposing animal (e.g., elephant, bear).
2. Importance/Centrality -> Outfit/Aura: If the service is highly important/central, give them a highly decorated, luxurious uniform or a commanding aura. If peripheral/low importance, give them a simple basic uniform or apron.
3. The character MUST be drawn in full-body. Add "Full-body shot of the character." to the prompt.
4. The background MUST be completely empty/solid. Add "Solid black background, absolutely no background elements or scenery." to the prompt.
5. The art style MUST be a friendly caricature. Add "Art style: Friendly and approachable animal caricature, non-photorealistic, stylized 2D illustration." to the prompt.
"""

def extract_skeleton(metadata_context: str, project_id: str, location: str = "us-central1") -> str:
    """Phase 1: Extract the skeleton from metadata files."""
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-2.5-flash")
    
    prompt = f"""You are an expert software architect.
Analyze the following metadata and configuration files from multiple repositories.
Identify the microservices, their primary technologies, and any explicit network or infrastructural dependencies.
Output a JSON array of basic service objects containing 'name', 'technology_stack', and 'basic_role'.
Do NOT extract detailed API logic, just the skeleton.

Context:
{metadata_context}
"""
    schema = {
        "type": "OBJECT",
        "properties": {
            "skeleton": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "technology_stack": {"type": "STRING"},
                        "basic_role": {"type": "STRING"}
                    },
                    "required": ["name", "technology_stack", "basic_role"]
                }
            }
        },
        "required": ["skeleton"]
    }
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json", "response_schema": schema}
    )
    return response.text

def extract_partial_graph(chunk_context: str, skeleton_json: str, project_id: str, location: str = "us-central1") -> str:
    """Phase 2: Extract partial architectural graph from an arbitrary chunk of code."""
    vertexai.init(project=project_id, location=location)
    
    system_instruction = (
        "You are an expert software architect and creative persona designer. "
        "Your task is to analyze an ARBITRARY CHUNK of source code, using the provided overall ecosystem skeleton as background knowledge. "
        "This chunk may contain code for MULTIPLE distinct microservices, or just a fragment of one service. "
        "Identify WHICH logical service(s) from the skeleton this code belongs to. "
        "Identify the source repository URL from the bracketed file prefixes (e.g. [https://github.com/...] in the file paths) for each service. "
        "Extract the roles, scales, repository_url, and dependencies for ALL services found in this chunk. "
        "Design a character prompt representing each service found."
    )
    model = GenerativeModel("gemini-2.5-flash", system_instruction=[system_instruction])
    
    prompt = f"""Overall Ecosystem Skeleton:
{skeleton_json}

Now, analyze the following arbitrary chunk of source code from the ecosystem.
Extract the partial microservice profiles for EVERY service you identify in this chunk.
Ensure you set the 'repository_url' for each microservice matching the repository URL found in the file path brackets (e.g., [https://github.com/owner/repo.git]).

{WORLD_SETTING}

Code Chunk Context:
{chunk_context}
"""

    vertex_schema = {
        "type": "OBJECT",
        "properties": {
            "microservices": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "description": {"type": "STRING"},
                        "scale_and_complexity": {"type": "STRING"},
                        "importance_and_centrality": {"type": "STRING"},
                        "role_type": {"type": "STRING"},
                        "repository_url": {"type": "STRING"},
                        "dependencies": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "service_name": {"type": "STRING"},
                                    "description": {"type": "STRING"}
                                },
                                "required": ["service_name", "description"]
                            }
                        },
                        "avatar_prompt": {"type": "STRING"}
                    },
                    "required": ["name", "description", "scale_and_complexity", "importance_and_centrality", "role_type", "repository_url", "dependencies", "avatar_prompt"]
                }
            }
        },
        "required": ["microservices"]
    }
    
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "response_mime_type": "application/json", "response_schema": vertex_schema}
    )
    return response.text

def synthesize_architecture(partial_graphs: list, project_id: str, location: str = "us-central1") -> str:
    """Phase 3: Combine all partial graphs into a final, consistent architecture."""
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-2.5-flash")
    
    combined_json_str = "\n---\n".join(partial_graphs)
    
    prompt = f"""You are a master system integrator.
I have analyzed the codebase in arbitrary chunks and generated partial JSON profiles representing fragments of the ecosystem.
Your task is to merge these PARTIAL GRAPHS into a single, unified Logical Architecture JSON.
Crucially: DEDUPLICATE components. If 'cartservice' appears in 5 different chunks, merge its descriptions and dependencies into ONE single 'cartservice' object.
Resolve any conflicting names, ensure dependencies refer to existing logical services, and output the final validated array.
Ensure you retain and resolve the correct 'repository_url' for each merged microservice.

{WORLD_SETTING}

Partial Analyses (from various chunks):
{combined_json_str}
"""
    vertex_schema = {
        "type": "OBJECT",
        "properties": {
            "microservices": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "description": {"type": "STRING"},
                        "scale_and_complexity": {"type": "STRING"},
                        "importance_and_centrality": {"type": "STRING"},
                        "role_type": {"type": "STRING"},
                        "repository_url": {"type": "STRING"},
                        "dependencies": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "service_name": {"type": "STRING"},
                                    "description": {"type": "STRING"}
                                },
                                "required": ["service_name", "description"]
                            }
                        },
                        "avatar_prompt": {"type": "STRING"}
                    },
                    "required": ["name", "description", "scale_and_complexity", "importance_and_centrality", "role_type", "repository_url", "dependencies", "avatar_prompt"]
                }
            }
        },
        "required": ["microservices"]
    }

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json", "response_schema": vertex_schema}
    )
    return response.text

from vertexai.generative_models import Content, Part

def chat_with_character(system_prompt: str, history: list, new_message: str, project_id: str, location: str = "us-central1") -> str:
    """Phase 3: Chat with a specific character agent."""
    vertexai.init(project=project_id, location=location)
    
    # Configure the persona
    model = GenerativeModel("gemini-2.5-flash", system_instruction=[system_prompt])
    
    # Reconstruct history
    chat_history = []
    for msg in history:
        role = msg.get("role")
        # vertexai expects 'user' or 'model'
        if role not in ["user", "model"]:
            role = "user"
        content = msg.get("content", "")
        chat_history.append(Content(role=role, parts=[Part.from_text(content)]))
        
    chat_session = model.start_chat(history=chat_history)
    response = chat_session.send_message(new_message)
    return response.text
