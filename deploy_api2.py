import urllib.request, urllib.error, json, os, base64

TOKEN = open(os.path.join(os.path.dirname(__file__), ".ghtoken")).read().strip()
OWNER = "lirafael-fun"
REPO = "language-learning-app"
API = "https://api.github.com"

def api(method, path, data=None):
    url = f"{API}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"token {TOKEN}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "hermes-deploy")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = json.loads(e.read())
        return err

project_dir = os.path.dirname(__file__)
files_to_push = []
for root, dirs, files in os.walk(project_dir):
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', 'templates']]
    for f in files:
        if f in ['.ghtoken', 'deploy.py', 'deploy_api.py', 'test_summary.py']:
            continue
        full = os.path.join(root, f)
        rel = os.path.relpath(full, project_dir).replace('\\', '/')
        with open(full, 'rb') as fh:
            content = fh.read()
        files_to_push.append((rel, content))

# Also add templates/index.html
tpl_path = os.path.join(project_dir, 'templates', 'index.html')
if os.path.exists(tpl_path):
    with open(tpl_path, 'rb') as fh:
        files_to_push.append(('templates/index.html', fh.read()))

print(f"Files to push: {[f[0] for f in files_to_push]}")

for path, content in files_to_push:
    print(f"  Uploading {path} ({len(content)} bytes)...")
    result = api("PUT", f"/repos/{OWNER}/{REPO}/contents/{path}", {
        "message": f"Add {path}",
        "content": base64.b64encode(content).decode(),
        "branch": "main"
    })
    if "content" in result:
        print(f"    OK: {result['content']['sha'][:7]}")
    else:
        print(f"    ERROR: {result.get('message', result)}")

print(f"\nDone! https://github.com/{OWNER}/{REPO}")
