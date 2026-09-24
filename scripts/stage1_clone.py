import subprocess
import os
import shutil

REPOS = [
    {
        "name": "Qwen3-Omni",
        "url": "https://github.com/QwenLM/Qwen3-Omni.git",
        "commit": "e4235853125589c789f06a2dd83e9f4126df5e9d",
        "path": "src/Qwen3-Omni"
    },
    {
        "name": "DuplexOmni",
        "url": "https://github.com/MuyeHuang/DuplexOmni.git",
        "commit": "33bfba1a821b09c5aa66790944f9098584979d34",
        "path": "src/DuplexOmni"
    },
    {
        "name": "Qwen-Omni-Training-Talker",
        "url": "https://github.com/Hert4/Qwen-Omni-Training-Talker.git",
        "commit": "d9ad37e150a8e93d16169449d33b48a3141372bf",
        "path": "src/Qwen-Omni-Training-Talker"
    },
    {
        "name": "Qwen3-TTS",
        "url": "https://github.com/QwenLM/Qwen3-TTS.git",
        "commit": "022e286b98fbec7e1e916cb940cdf532cd9f488e",
        "path": "src/Qwen3-TTS"
    }
]

git_bin = shutil.which("git")
if not git_bin:
    raise RuntimeError("Git was not found on PATH")

def run_git(cmd, cwd=None):
    res = subprocess.run([git_bin] + cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Git error ({' '.join(cmd)}) in {cwd}: {res.stderr.strip()}")
        raise RuntimeError(res.stderr.strip())
    return res.stdout.strip()

for repo in REPOS:
    print(f"\n=== Processing {repo['name']} ===")
    git_dir = os.path.join(repo['path'], ".git")
    if not os.path.exists(git_dir):
        print(f"Cloning {repo['url']} into {repo['path']}...")
        run_git(["clone", repo['url'], repo['path']])
    else:
        print(f"Repo already exists at {repo['path']}")
    
    # checkout commit
    try:
        run_git(["checkout", repo['commit']], cwd=repo['path'])
        actual_sha = run_git(["rev-parse", "HEAD"], cwd=repo['path'])
        print(f"Checked out commit: {actual_sha}")
    except Exception as e:
        actual_sha = run_git(["rev-parse", "HEAD"], cwd=repo['path'])
        print(f"Current HEAD: {actual_sha} (target was {repo['commit']})")
