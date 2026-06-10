import os
import pathspec

def load_gitignore(repo_path: str):
    """Load .gitignore if it exists and return a pathspec object."""
    gitignore_path = os.path.join(repo_path, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, lines)
    return None

def scan_metadata(repo_configs: list) -> str:
    """
    Scan only metadata files (configs, package definitions, root README, and docs/spec folders).
    This provides a lightweight high-level view of the ecosystem.
    repo_configs: list of (repo_path, original_url) tuples
    """
    metadata_files = {'docker-compose.yml', 'pom.xml', 'package.json', 'go.mod', 'Dockerfile'}
    allowed_exts = ('.yaml', '.yml')
    doc_dirs = {'spec', 'docs'}
    
    all_output = []
    for repo_path, original_url in repo_configs:
        all_output.append("=" * 80)
        all_output.append(f"=== REPOSITORY METADATA: {original_url} ===")
        all_output.append("=" * 80 + "\n")
        
        for root, dirs, files in os.walk(repo_path):
            rel_root = os.path.relpath(root, repo_path)
            
            # Fast prune: skip hidden dirs or build/vendor
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules', 'venv', 'vendor', 'dist', 'build')]
            
            is_root = (rel_root == '.')
            is_doc_dir = any(part in doc_dirs for part in rel_root.split(os.sep))
            
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                # Check if it's a target metadata file
                include = False
                if is_root and (file in metadata_files or file.endswith(allowed_exts) or file.lower() == 'readme.md'):
                    include = True
                elif is_doc_dir and file.endswith(('.md', '.txt')):
                    include = True
                    
                if include:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            all_output.append(f"--- File: [{original_url}] {rel_path} ---\n{f.read()}\n")
                    except Exception:
                        pass
                        
    return "\n".join(all_output)

def scan_code_chunks(repo_configs: list, max_chars: int = 300000) -> list:
    """
    Scan multiple directories and pack files into chunks of up to max_chars.
    This entirely ignores repository boundaries and groups code purely by token limits.
    repo_configs: list of (repo_path, original_url) tuples
    """
    default_ignores = [
        ".git", "node_modules", "venv", "__pycache__", ".next", "dist", "build", ".venv",
        "vendor", "test", "tests", "public", "static", "assets", "mock", "mocks", "fixtures"
    ]
    ignore_files = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "go.sum", "Cargo.lock", "Gemfile.lock"
    }
    ignore_file_suffixes = (
        '_test.go', '.spec.js', '.test.js', '.spec.ts', '.test.ts', '.min.js', '.min.css'
    )
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for repo_path, original_url in repo_configs:
        ignore_spec = load_gitignore(repo_path)
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in default_ignores]
            if ignore_spec:
                rel_root = os.path.relpath(root, repo_path)
                dirs[:] = [d for d in dirs if not ignore_spec.match_file(os.path.join(rel_root, d))]
    
            for file in files:
                if file.startswith('.') or file in ignore_files or file.endswith(ignore_file_suffixes):
                    continue
                if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar', '.gz')):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                if ignore_spec and ignore_spec.match_file(rel_path):
                    continue
                    
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        file_block = f"--- File: [{original_url}] {rel_path} ---\n{content}\n{'-'*40}\n"
                        block_len = len(file_block)
                        
                        if current_length + block_len > max_chars and current_length > 0:
                            chunks.append("".join(current_chunk))
                            current_chunk = []
                            current_length = 0
                            
                        current_chunk.append(file_block)
                        current_length += block_len
                except Exception:
                    pass
                    
    if current_chunk:
        chunks.append("".join(current_chunk))
        
    return chunks

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
        configs = [(p, p) for p in paths]
        chunks = scan_code_chunks(configs)
        print(f"Generated {len(chunks)} chunks.")
    else:
        print("Usage: python scanner.py <path_to_directory1> [path_to_directory2 ...]")
