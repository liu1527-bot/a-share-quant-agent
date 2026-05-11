"""
通过 PUT /repos/{owner}/{repo}/contents/{path} 单文件更新。
比 push_via_api.py 快得多(适合改 1-2 个文件)。
"""
import os, sys, base64, json, urllib.request, urllib.error

TOKEN = os.environ['TOKEN']
OWNER = os.environ['OWNER']
REPO = os.environ['REPO']
PATH = os.environ['FILEPATH']  # 仓库内路径
LOCAL = os.environ.get('LOCAL', PATH)
MSG = os.environ.get('MSG', f'update {PATH}')
BRANCH = os.environ.get('BRANCH', 'main')

API = f'https://api.github.com/repos/{OWNER}/{REPO}/contents/{PATH}'
HEADERS = {
    'Authorization': f'Bearer {TOKEN}',
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}


def req(method, url, data=None):
    body = json.dumps(data).encode() if data else None
    r = urllib.request.Request(url, data=body, method=method,
                                headers={**HEADERS, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# 1. 拿当前 sha (如果文件已存在)
code, info = req('GET', f'{API}?ref={BRANCH}')
sha = info['sha'] if code == 200 else None

# 2. PUT
content_b64 = base64.b64encode(open(LOCAL, 'rb').read()).decode()
payload = {'message': MSG, 'content': content_b64, 'branch': BRANCH}
if sha:
    payload['sha'] = sha
code, data = req('PUT', API, payload)
if code in (200, 201):
    print(f"[OK] commit {data['commit']['sha'][:7]} -> {PATH}")
else:
    print(f"[FAIL] {data.get('message')}")
    sys.exit(1)
