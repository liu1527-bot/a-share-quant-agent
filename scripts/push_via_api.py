#!/usr/bin/env python
"""
绕过 git push,直接用 GitHub Contents API 把仓库内容传上去。
适合: 国内网络 git push 被重置但 api.github.com 可达的情况。

用法: TOKEN=ghp_xxx OWNER=liu1527-bot REPO=a-share-quant-agent python scripts/push_via_api.py
"""
import os
import sys
import base64
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

TOKEN = os.environ['TOKEN']
OWNER = os.environ['OWNER']
REPO = os.environ['REPO']
BRANCH = os.environ.get('BRANCH', 'main')
COMMIT_MSG = os.environ.get('MSG', 'Initial commit via API')

API = f'https://api.github.com/repos/{OWNER}/{REPO}'
HEADERS = {
    'Authorization': f'Bearer {TOKEN}',
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}


def api_request(method, url, data=None):
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method, headers={
        **HEADERS, 'Content-Type': 'application/json'
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def list_git_files():
    """获取 git 已 add 的所有文件(尊重 .gitignore)"""
    out = subprocess.check_output(['git', 'ls-files'], text=True, encoding='utf-8')
    return [f.strip() for f in out.splitlines() if f.strip()]


def get_branch_sha():
    code, data = api_request('GET', f'{API}/branches/{BRANCH}')
    if code == 200:
        return data['commit']['sha'], data['commit']['commit']['tree']['sha']
    return None, None


def bootstrap_empty_repo():
    """空仓库需要先用 PUT /contents 创建第一个文件,触发分支创建"""
    print('[INFO] 用 PUT /contents 创建初始 README 触发分支...')
    init_content = base64.b64encode(b'# Initializing...\n').decode()
    code, data = api_request('PUT', f'{API}/contents/.gh-init', {
        'message': 'bootstrap empty repo',
        'content': init_content,
        'branch': BRANCH,
    })
    if code not in (200, 201):
        print(f'[FAIL] bootstrap: {data.get("message")}')
        return False
    print(f'[OK] bootstrap commit {data["commit"]["sha"][:7]}')
    return True


def main():
    files = list_git_files()
    print(f'[INFO] 要上传 {len(files)} 个文件')

    base_sha, base_tree_sha = get_branch_sha()
    if not base_sha:
        print(f'[INFO] 远程 {BRANCH} 是空仓,先 bootstrap')
        if not bootstrap_empty_repo():
            return 1
        base_sha, base_tree_sha = get_branch_sha()
    print(f'[INFO] 基于 base_sha={base_sha[:7]}')

    # 1. 把每个文件作为 blob 上传
    blobs = []
    for i, fp in enumerate(files, 1):
        path = Path(fp)
        if not path.exists():
            print(f'[SKIP] {fp} 不存在')
            continue
        content = path.read_bytes()
        b64 = base64.b64encode(content).decode()
        code, blob = api_request('POST', f'{API}/git/blobs', {
            'content': b64, 'encoding': 'base64'
        })
        if code != 201:
            print(f'[FAIL] blob {fp}: {blob.get("message")}')
            return 1
        blobs.append({'path': fp.replace('\\', '/'), 'mode': '100644', 'type': 'blob', 'sha': blob['sha']})
        print(f'  [{i:>2}/{len(files)}] {fp}  -> {blob["sha"][:7]}')

    # 2. 创建 tree (不基于 base_tree,避免 bootstrap 占位文件残留)
    tree_payload = {'tree': blobs}
    code, tree = api_request('POST', f'{API}/git/trees', tree_payload)
    if code != 201:
        print(f'[FAIL] create tree: {tree.get("message")}')
        return 1
    print(f'[OK] tree {tree["sha"][:7]}')

    # 3. 创建 commit (不带 parent,做成 root commit,然后 force push 覆盖 bootstrap)
    commit_payload = {'message': COMMIT_MSG, 'tree': tree['sha'], 'parents': []}
    code, commit = api_request('POST', f'{API}/git/commits', commit_payload)
    if code != 201:
        print(f'[FAIL] create commit: {commit.get("message")}')
        return 1
    print(f'[OK] commit {commit["sha"][:7]}')

    # 4. 强制更新分支引用 (force=True 覆盖 bootstrap commit)
    code, ref = api_request('PATCH', f'{API}/git/refs/heads/{BRANCH}', {
        'sha': commit['sha'], 'force': True
    })
    if code not in (200, 201):
        print(f'[FAIL] update ref: {ref.get("message")}')
        return 1
    print(f'[OK] {BRANCH} -> {commit["sha"][:7]}')
    print(f'\n[DONE] https://github.com/{OWNER}/{REPO}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
