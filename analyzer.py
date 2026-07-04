import os
import json
import time
import vertexai
from vertexai.generative_models import GenerativeModel
import google.api_core.exceptions

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def call_with_retry(func, *args, max_retries=5, initial_backoff=2, **kwargs):
    """Call a Vertex AI function with exponential backoff on transient and 429 errors."""
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.ServiceUnavailable, google.api_core.exceptions.GoogleAPICallError) as e:
            print(f"[Attempt {attempt+1}/{max_retries}] Vertex AI transient error: {e}", flush=True)
            if attempt == max_retries - 1:
                raise e
            print(f"Retrying in {backoff} seconds...", flush=True)
            time.sleep(backoff)
            backoff *= 2
        except Exception as e:
            err_str = str(e).lower()
            if any(term in err_str for term in ["429", "quota", "exhausted", "rate limit", "overloaded"]):
                print(f"[Attempt {attempt+1}/{max_retries}] Rate limit or resource exhausted: {e}", flush=True)
                if attempt == max_retries - 1:
                    raise e
                print(f"Retrying in {backoff} seconds...", flush=True)
                time.sleep(backoff)
                backoff *= 2
            else:
                raise e

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

For each microservice, the `role_type` should be one of the above titles, and the `avatar_prompt` MUST describe only the animal character itself performing this role (e.g., wearing the appropriate uniform, holding utensils/tools, or performing actions). The prompt MUST NOT describe any restaurant environment, table, kitchen, floor, background scenery, room, or setting.

You must also generate a second prompt `avatar_chat_prompt` for a chat-specific avatar. The `avatar_chat_prompt` MUST describe the exact same animal character (same animal type, colors, outfit, uniform, and features) as in `avatar_prompt` to maintain complete visual consistency. However, instead of the action/pose in `avatar_prompt`, the character in `avatar_chat_prompt` MUST be facing forward (looking directly at the user) and speaking cheerfully with a friendly smile, as if introducing or explaining themselves to the user. Like `avatar_prompt`, it MUST NOT describe any background.

CRITICAL VISUAL CONSTRAINTS for `avatar_prompt` (Integrate these strictly in English):
1. Scale/Complexity -> Animal Size/Type: If the service is small/low complexity, choose a small agile animal (e.g., squirrel, bird). If it is large/complex, choose a large imposing animal (e.g., elephant, bear).
2. Importance/Centrality -> Outfit/Aura: If the service is highly important/central, give them a highly decorated, luxurious uniform or a commanding aura. If peripheral/low importance, give them a simple basic uniform or apron.
3. The prompt MUST start with: "Full-body shot of the character, isolated, "
4. The background MUST be completely flat, solid, and uniform chroma-key green. The prompt MUST end with: "The entire background is a single flat solid pure chroma-key green color (RGB: 0, 255, 0) with absolutely no details, no shadows, no flooring, no lighting effects on the background, and no scenery. The character is completely isolated against this flat green background."
5. The art style MUST follow a Pokemon-style creature design. Add the following to the prompt: "Art style: Pokemon-style creature art, anime monster mascot design, extremely oversized sparkling eyes with large white catchlight highlights (eyes are the most prominent facial feature), super-deformed chibi proportions (large head, tiny body), thick bold black outlines, bright cel-shading, vivid saturated colors."
6. NO Background Contradiction: Do NOT include any descriptions of environments, locations, floor textures, ground shadows, or environment lighting that could contradict the flat green background instruction.

CRITICAL VISUAL CONSTRAINTS for `avatar_chat_prompt` (Integrate these strictly in English):
1. Copy the exact same animal, colors, clothing, and details from `avatar_prompt` for visual identity.
2. The prompt MUST start with: "Full-body shot of the character, isolated, facing forward, speaking cheerfully with a friendly smile, "
3. The background MUST be completely flat, solid, and uniform chroma-key green. The prompt MUST end with: "The entire background is a single flat solid pure chroma-key green color (RGB: 0, 255, 0) with absolutely no details, no shadows, no flooring, no lighting effects on the background, and no scenery. The character is completely isolated against this flat green background."
4. The art style MUST follow the exact same Pokemon-style creature design. Add the following to the prompt: "Art style: Pokemon-style creature art, anime monster mascot design, extremely oversized sparkling eyes with large white catchlight highlights (eyes are the most prominent facial feature), super-deformed chibi proportions (large head, tiny body), thick bold black outlines, bright cel-shading, vivid saturated colors."
5. NO Background Contradiction: Do NOT include any descriptions of environments, locations, floor textures, ground shadows, or environment lighting that could contradict the flat green background instruction.
"""

def extract_skeleton(metadata_context: str, project_id: str, location: str = "us-central1") -> str:
    """Phase 1: Extract the skeleton from metadata files."""
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(GEMINI_MODEL)
    
    prompt = f"""You are an expert software architect.
Analyze the following metadata and configuration files from multiple repositories.
Identify the internal, custom microservices and applications.

CRITICAL INSTRUCTIONS FOR MICROSERVICE IDENTIFICATION:
1. Do NOT extract generic, off-the-shelf public technologies, databases, external APIs, or managed services (such as PostgreSQL, Firebase, GitHub API, Redis, AWS S3, etc.) as standalone microservice components under their generic names.
2. If a repository is dedicated to deploying, wrapping, or managing a database or infrastructure component, deduce an architecturally meaningful, custom logical name for it (e.g., "Restaurant Inventory Database" instead of "PostgreSQL") using the repository name, path, README, or structure.
3. If a generic technology is merely utilized/accessed by a custom microservice, it should be listed in the 'technology_stack' of the utilizing microservice, rather than being drawn as a separate component node.

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
    
    response = call_with_retry(
        model.generate_content,
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
        "Extract the roles, scales, scale_tier (integer rating from 1 to 5 based on code volume and responsibility), repository_url, and dependencies for ALL services found in this chunk. "
        "Design a character prompt representing each service found. "
        "Additionally, identify the most important source files for each service to serve as exploration anchors during future chat sessions. "
        "Specifically, identify the representative technologies used by each service."
    )
    model = GenerativeModel(GEMINI_MODEL, system_instruction=[system_instruction])
    
    prompt = f"""Overall Ecosystem Skeleton:
{skeleton_json}

Now, analyze the following arbitrary chunk of source code from the ecosystem.
Extract the partial microservice profiles for EVERY service you identify in this chunk.
Ensure you set the 'repository_url' for each microservice matching the repository URL found in the file path brackets (e.g., [https://github.com/owner/repo.git]).

CRITICAL INSTRUCTIONS FOR MICROSERVICE IDENTIFICATION:
1. Do NOT extract generic, off-the-shelf public technologies, databases, external APIs, or managed services (such as PostgreSQL, Firebase, GitHub API, Redis, AWS S3, etc.) as standalone microservice components under their generic names.
2. If a repository is dedicated to deploying, wrapping, or managing a database or infrastructure component, deduce an architecturally meaningful, custom logical name for it (e.g., "Restaurant Inventory Database" instead of "PostgreSQL") using the repository name, path, README, or structure.
3. If a generic technology is merely utilized/accessed by a custom microservice, it should be listed in the 'technologies' array of the utilizing microservice, rather than being drawn as a separate component node.

{WORLD_SETTING}

For the 'key_files' field, identify up to 10 source files that are the best entry points
for understanding this service during a chat session. Rules:
1. ALWAYS include files for these two perspectives (if present in the code chunk):
   - 'core_business_logic': The primary file(s) implementing this service's core functionality
   - 'configuration': Environment variables, config files, or settings (e.g., .env.example, config.py, application.yml)
2. In addition, based on the SPECIFIC CHARACTERISTICS of this service, determine up to 8 more
   perspectives that are uniquely relevant. Examples (use only what applies):
   - 'api_contracts' for services with REST/gRPC interfaces
   - 'data_schema' for services with DB models or migrations
   - 'routing_rules' for gateways or routers
   - 'event_handlers' for async/event-driven services
   - 'client_definitions' for services that call external APIs
   - 'authentication_strategy' for auth services
   - 'migration_files' for database services
   - 'test_coverage' for services with test suites that reveal expected behavior
3. CRITICAL: Only use file paths that ACTUALLY APPEAR in the provided code chunk.
   Do NOT guess or invent file paths.
4. Use relative paths from the repository root (strip the bracketed repo URL prefix).

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
                        "scale_tier": {
                            "type": "INTEGER",
                            "description": "Complexity and scale rating from 1 (very small/simple helper service) to 5 (critical high-scale core/gateway monolith)."
                        },
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
                        "avatar_prompt": {"type": "STRING"},
                        "avatar_chat_prompt": {"type": "STRING"},
                        "technologies": {
                            "type": "ARRAY",
                            "items": {"type": "STRING"},
                            "description": "List of representative technologies, databases, frameworks, languages, or external APIs utilized by this microservice (e.g. PostgreSQL, FastAPI, Python, Firebase, GitHub API). Max 10 entries."
                        },
                        "key_files": {
                            "type": "ARRAY",
                            "description": "Key source files for this service, used as exploration anchors during chat. Max 10 entries.",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "path": {
                                        "type": "STRING",
                                        "description": "Relative file path from repository root. Must be a path that actually exists in the provided code chunk."
                                    },
                                    "perspective": {
                                        "type": "STRING",
                                        "description": "The aspect of the service this file represents (e.g., 'core_business_logic', 'configuration', 'api_contracts', 'data_schema', 'routing_rules', etc.)"
                                    },
                                    "reason": {
                                        "type": "STRING",
                                        "description": "Brief explanation of why this file is important for understanding this service."
                                    }
                                },
                                "required": ["path", "perspective", "reason"]
                            }
                        }
                    },
                    "required": ["name", "description", "scale_and_complexity", "scale_tier", "importance_and_centrality", "role_type", "repository_url", "dependencies", "avatar_prompt", "avatar_chat_prompt", "key_files", "technologies"]
                }
            }
        },
        "required": ["microservices"]
    }
    
    response = call_with_retry(
        model.generate_content,
        prompt,
        generation_config={"temperature": 0.2, "response_mime_type": "application/json", "response_schema": vertex_schema}
    )
    return response.text

def synthesize_architecture(partial_graphs: list, project_id: str, location: str = "us-central1") -> str:
    """Phase 3: Combine all partial graphs into a final, consistent architecture."""
    # -----------------------------------------------------------------------
    # Deterministic Data Backup (Post-processing Guardrails)
    # Back up the original, detailed avatar_prompt, role_type and unique key_files
    # from the partial graphs to prevent LLM from losing or shortening them.
    # -----------------------------------------------------------------------
    backup_prompts = {}
    backup_key_files = {}
    backup_technologies = {}
    
    for pg_str in partial_graphs:
        try:
            pg_data = json.loads(pg_str)
            for ms in pg_data.get("microservices", []):
                ms_name = ms.get("name")
                if not ms_name:
                    continue
                
                # Keep the longest (most detailed) avatar prompt
                role_type = ms.get("role_type")
                avatar_prompt = ms.get("avatar_prompt")
                avatar_chat_prompt = ms.get("avatar_chat_prompt")
                if avatar_prompt:
                    current_best = backup_prompts.get(ms_name)
                    if not current_best or len(avatar_prompt) > len(current_best[1]):
                        backup_prompts[ms_name] = (role_type, avatar_prompt, avatar_chat_prompt)
                
                # Gather unique key_files
                kfs = ms.get("key_files", [])
                if kfs:
                    existing_kfs = backup_key_files.setdefault(ms_name, {})
                    for kf in kfs:
                        path = kf.get("path")
                        if path and path not in existing_kfs:
                            existing_kfs[path] = kf

                # Gather unique technologies
                techs = ms.get("technologies", [])
                if techs:
                    existing_techs = backup_technologies.setdefault(ms_name, set())
                    for t in techs:
                        if t:
                            existing_techs.add(t)
        except Exception as parse_err:
            print(f"Warning: Failed to parse partial graph JSON for backup: {parse_err}")

    vertexai.init(project=project_id, location=location)
    model = GenerativeModel(GEMINI_MODEL)
    
    combined_json_str = "\n---\n".join(partial_graphs)
    
    prompt = f"""You are a master system integrator.
I have analyzed the codebase in arbitrary chunks and generated partial JSON profiles representing fragments of the ecosystem.
Your task is to merge these PARTIAL GRAPHS into a single, unified Logical Architecture JSON.
Crucially: DEDUPLICATE components. If 'cartservice' appears in 5 different chunks, merge its descriptions, scale_tier (retaining the max or most representative tier), key_files (combining unique paths across chunks, preserving perspectives and reasons, up to 10 total items), technologies (combining all unique technologies across chunks, preserving original technology names, up to 10 items total), and dependencies into ONE single 'cartservice' object.
Resolve any conflicting names, ensure dependencies refer to existing logical services, and output the final validated array.
Ensure you retain and resolve the correct 'repository_url', 'scale_tier', 'key_files', 'role_type', 'technologies', and 'avatar_prompt' (ensuring the detailed creative prompt is preserved or synthesized) for each merged microservice.

Also, you must select the most appropriate overall visual layout pattern for this microservices graph and output it in `layout_pattern`. You must select exactly ONE of the following 6 patterns, and for each microservice, fill the abstract logical attributes in `layout_metadata`.

Available Layout Patterns:
1. "hierarchical":
   - Criteria: Use if there is a clear vertical or horizontal data flow/layers (e.g. gateway -> business logic -> database) and dependencies mostly flow in one direction.
   - Nodes metadata: Set `rank` (integer hierarchical level, e.g. 0 for gateway/frontend, 1 for business logic, 2 for database) and `index_in_rank` (sequential order of nodes in the same rank layer, starting from 0).
2. "radial":
   - Criteria: Use if there is one centralized hub service (e.g. API Gateway, Message Broker, Event Bus) that mediates communications for all other peripheral services (spokes).
   - Nodes metadata: For the central hub node, set `is_hub` to true. For all other spoke nodes, set `is_hub` to false and assign a unique sequential `spoke_index` (from 0 to N-1).
3. "clustering":
   - Criteria: Use if the graph naturally splits into multiple isolated or weakly connected sub-graphs/islands (e.g. distinct DDD bounded contexts or isolated domains).
   - Nodes metadata: Set `cluster_id` (0-indexed integer identifying the cluster group) and `index_in_cluster` (unique sequential index within that cluster).
4. "boundary":
   - Criteria: Use if there are clear nested containers/groups represented in directory names, namespaces, or distinct structural boundaries (e.g., Kubernetes namespaces or VPC subnets).
   - Nodes metadata: Set `boundary_id` (0-indexed integer identifying the boundary container) and `index_in_boundary` (unique sequential index inside the boundary).
5. "mesh":
   - Criteria: Use if the services are peer-to-peer or form a highly dense, mutually connected network without a clear single hub or simple rank layers.
   - Nodes metadata: Set `index` (0-indexed unique sequential identifier for the nodes to lay them out in a large circle).
6. "matrix":
   - Criteria: Use if there are distinct columns and rows of service variations (e.g., Command services vs Query services, or Multi-Region service duplicates).
   - Nodes metadata: Set `row` (0-indexed row number) and `col` (0-indexed column number) to line up in a grid.

Make sure you output ALL the required properties for each node. If layout_metadata does not apply to a specific node, fill the relevant fields for the chosen layout_pattern anyway (leave the other fields blank/unset).

{WORLD_SETTING}

Partial Analyses (from various chunks):
{combined_json_str}
"""
    vertex_schema = {
        "type": "OBJECT",
        "properties": {
            "layout_pattern": {
                "type": "STRING",
                "description": "Selected visual layout pattern for the system. Must be one of: 'hierarchical', 'radial', 'clustering', 'boundary', 'mesh', 'matrix'."
            },
            "microservices": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "name": {"type": "STRING"},
                        "description": {"type": "STRING"},
                        "scale_and_complexity": {"type": "STRING"},
                        "scale_tier": {
                            "type": "INTEGER",
                            "description": "Complexity and scale rating from 1 (very small/simple helper service) to 5 (critical high-scale core/gateway monolith)."
                        },
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
                        "avatar_prompt": {"type": "STRING"},
                        "avatar_chat_prompt": {"type": "STRING"},
                        "key_files": {
                            "type": "ARRAY",
                            "description": "Key source files for this service, used as exploration anchors during chat. Max 10 entries.",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "path": {
                                        "type": "STRING",
                                        "description": "Relative file path from repository root."
                                    },
                                    "perspective": {
                                        "type": "STRING",
                                        "description": "The aspect of the service this file represents (e.g., 'core_business_logic', 'configuration', 'api_contracts', 'data_schema', etc.)"
                                    },
                                    "reason": {
                                        "type": "STRING",
                                        "description": "Brief explanation of why this file is important."
                                    }
                                },
                                "required": ["path", "perspective", "reason"]
                            }
                        },
                        "technologies": {
                            "type": "ARRAY",
                            "description": "List of representative technologies, databases, frameworks, languages, or external APIs utilized by this microservice. Max 10 entries.",
                            "items": {"type": "STRING"}
                        },
                        "layout_metadata": {
                            "type": "OBJECT",
                            "properties": {
                                "rank": {"type": "INTEGER", "description": "Hierarchical level, 0 for topmost (e.g. gateway/frontend). Required if layout_pattern is 'hierarchical'."},
                                "index_in_rank": {"type": "INTEGER", "description": "Index within the same rank layer (0-indexed). Required if layout_pattern is 'hierarchical'."},
                                "is_hub": {"type": "BOOLEAN", "description": "True if this service acts as a centralized broker or gateway. Required if layout_pattern is 'radial'."},
                                "spoke_index": {"type": "INTEGER", "description": "Angle/spoke index for radial layout. Required if layout_pattern is 'radial'."},
                                "cluster_id": {"type": "INTEGER", "description": "ID of the isolated island cluster. Required if layout_pattern is 'clustering'."},
                                "index_in_cluster": {"type": "INTEGER", "description": "Index of the node inside its cluster. Required if layout_pattern is 'clustering'."},
                                "boundary_id": {"type": "INTEGER", "description": "Boundary container ID (e.g. VPC, namespace). Required if layout_pattern is 'boundary'."},
                                "index_in_boundary": {"type": "INTEGER", "description": "Index of the node inside its boundary. Required if layout_pattern is 'boundary'."},
                                "index": {"type": "INTEGER", "description": "Simple sequential index for mesh/circular layouts. Required if layout_pattern is 'mesh'."},
                                "row": {"type": "INTEGER", "description": "Row index for grid/matrix layouts. Required if layout_pattern is 'matrix'."},
                                "col": {"type": "INTEGER", "description": "Column index for grid/matrix layouts. Required if layout_pattern is 'matrix'."}
                            },
                            "description": "Logical placement attributes for calculating 2D coordinates."
                        }
                    },
                    "required": ["name", "description", "scale_and_complexity", "scale_tier", "importance_and_centrality", "role_type", "repository_url", "dependencies", "avatar_prompt", "avatar_chat_prompt", "layout_metadata", "key_files", "technologies"]
                }
            }
        },
        "required": ["layout_pattern", "microservices"]
    }
    
    response = call_with_retry(
        model.generate_content,
        prompt,
        generation_config={"temperature": 0.1, "response_mime_type": "application/json", "response_schema": vertex_schema}
    )
    
    # -----------------------------------------------------------------------
    # Deterministic Data Restoration (Post-processing Guardrails)
    # Recover any avatar_prompt, role_type, or key_files that were dropped or
    # severely shortened during the LLM synthesis phase.
    # -----------------------------------------------------------------------
    try:
        final_data = json.loads(response.text)
        modified = False
        
        for ms in final_data.get("microservices", []):
            ms_name = ms.get("name")
            if not ms_name:
                continue
                
            # Restore avatar prompt if it's missing or significantly shortened (< 70% of original)
            backup = backup_prompts.get(ms_name)
            if backup:
                role_type, avatar_prompt, avatar_chat_prompt = backup
                current_prompt = ms.get("avatar_prompt", "")
                if not current_prompt or len(current_prompt) < len(avatar_prompt) * 0.7:
                    ms["avatar_prompt"] = avatar_prompt
                    if role_type and not ms.get("role_type"):
                        ms["role_type"] = role_type
                    modified = True
                    print(f"Deterministic recovery: Restored original avatar_prompt for microservice '{ms_name}'")
                
                # Restore avatar_chat_prompt if it's missing or significantly shortened (< 70% of original)
                current_chat_prompt = ms.get("avatar_chat_prompt", "")
                if avatar_chat_prompt and (not current_chat_prompt or len(current_chat_prompt) < len(avatar_chat_prompt) * 0.7):
                    ms["avatar_chat_prompt"] = avatar_chat_prompt
                    modified = True
                    print(f"Deterministic recovery: Restored original avatar_chat_prompt for microservice '{ms_name}'")
            
            # Deterministically merge missing key_files back (up to 10 max)
            backup_kfs = backup_key_files.get(ms_name)
            if backup_kfs:
                current_kfs = ms.get("key_files", [])
                current_paths = {kf.get("path") for kf in current_kfs if kf.get("path")}
                
                added = False
                for path, kf_obj in backup_kfs.items():
                    if len(current_kfs) >= 10:
                        break
                    if path not in current_paths:
                        current_kfs.append(kf_obj)
                        current_paths.add(path)
                        added = True
                        
                if added:
                    ms["key_files"] = current_kfs
                    modified = True
                    print(f"Deterministic recovery: Merged missing key_files for '{ms_name}'")

            # Deterministically merge technologies back (up to 10 max)
            backup_techs = backup_technologies.get(ms_name)
            if backup_techs:
                current_techs = ms.get("technologies", [])
                current_tech_lower = {t.lower() for t in current_techs}
                
                added_tech = False
                for t in backup_techs:
                    if len(current_techs) >= 10:
                        break
                    if t.lower() not in current_tech_lower:
                        current_techs.append(t)
                        current_tech_lower.add(t.lower())
                        added_tech = True
                
                if added_tech:
                    ms["technologies"] = current_techs
                    modified = True
                    print(f"Deterministic recovery: Merged technologies for '{ms_name}'")
                    
        if modified:
            return json.dumps(final_data)
            
    except Exception as recovery_err:
        print(f"Warning: Failed during deterministic recovery post-processing: {recovery_err}")

    return response.text

from vertexai.generative_models import Content, Part

def chat_with_character(system_prompt: str, history: list, new_message: str, project_id: str, location: str = "us-central1") -> str:
    """Phase 3: Chat with a specific character agent."""
    vertexai.init(project=project_id, location=location)
    
    # Configure the persona
    model = GenerativeModel(GEMINI_MODEL, system_instruction=[system_prompt])
    
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
    response = call_with_retry(chat_session.send_message, new_message)
    return response.text
