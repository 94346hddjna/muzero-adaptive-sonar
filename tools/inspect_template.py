# -*- coding: utf-8 -*-
"""分析期刊模板包结构。"""
import zipfile, os, re

z = zipfile.ZipFile(os.environ["TEMP"] + "/template_pkg.zip")
tex = z.read("sn-article-template/sn-article.tex").decode("utf-8", errors="replace")

sections = re.findall(r"\\section\*?\{([^}]+)\}", tex)
print("章节:", sections)

m = re.search(r"\\abstract\{(.*?)\}", tex, re.S)
if m:
    print("摘要:", m.group(1)[:400])
m = re.search(r"\\keywords\{(.*?)\}", tex, re.S)
if m:
    print("关键词:", m.group(1)[:200])

print("公式数:", len(re.findall(r"\\begin\{equation\}", tex)))
print("figure数:", len(re.findall(r"\\begin\{figure", tex)))
print("table数:", len(re.findall(r"\\begin\{table", tex)))
m = re.search(r"\\bibliographystyle\{([^}]+)\}", tex)
if m:
    print("bib样式:", m.group(1))

# 看一个完整的figure和table示例
fig = re.search(r"(\\begin\{figure.*?\\end\{figure.*?\})", tex, re.S)
if fig:
    print("\n--- figure示例 ---")
    print(fig.group(1)[:600])

# 摘要上下结构
idx = tex.find("abstract")
print("\n--- 摘要上下文 ---")
print(tex[idx-50:idx+600])
