import subprocess, json, urllib.request, sys, time
from pathlib import Path

cred = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                      capture_output=True, text=True).stdout
token = [l.split("=",1)[1] for l in cred.split("\n") if l.startswith("password=")][0]

# 获取release的upload_url
req = urllib.request.Request("https://api.github.com/repos/94346hddjna/muzero-adaptive-sonar/releases/latest",
                             headers={"Authorization": f"token {token}", "User-Agent": "uploader"})
with urllib.request.urlopen(req) as r:
    rel = json.load(r)
upload_url = rel["upload_url"].replace("{?name,label}", "")
print("上传到 release:", rel["tag_name"])

for f in sorted(Path("release_assets").glob("*.zip")):
    size_mb = f.stat().st_size / 1e6
    print(f"上传 {f.name} ({size_mb:.0f} MB)...", flush=True)
    t0 = time.time()
    url = f"{upload_url}?name={f.name}"
    with open(f, "rb") as fh:
        data = fh.read()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Authorization": f"token {token}",
                                          "Content-Type": "application/zip",
                                          "User-Agent": "uploader"})
    try:
        with urllib.request.urlopen(req) as r:
            json.load(r)
        print(f"  ✅ 完成 ({size_mb:.0f} MB, {(time.time()-t0)/60:.1f} 分钟)")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
print("全部上传完成")
