# -*- coding: utf-8 -*-
"""完整提取开题报告 docx：段落(含样式) + 表格 + OMML数学公式(转线性文本)。
输出 markdown 到 kaiti_extracted.md，公式逐个编号以便核对。"""
import sys
import io
import zipfile
import defusedxml.ElementTree as ET

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DOCX = r"C:\Users\renao\Desktop\Muzero\开题报告(1).docx"
OUT = r"C:\Users\renao\Desktop\Muzero\kaiti_extracted.md"

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def tag(el):
    return el.tag.split("}")[-1]


def omml_to_linear(el):
    """递归把 OMML 元素转成线性可读文本。"""
    t = tag(el)
    if t == "t":  # m:t
        return el.text or ""
    if t == "r":  # m:r run
        return "".join(omml_to_linear(c) for c in el)
    if t == "f":  # m:f fraction
        num = el.find(f"{{{M}}}num")
        den = el.find(f"{{{M}}}den")
        return f"({omml_to_linear(num) if num is not None else ''})/({omml_to_linear(den) if den is not None else ''})"
    if t == "sSup":
        e = el.find(f"{{{M}}}e")
        sup = el.find(f"{{{M}}}sup")
        return f"{omml_to_linear(e) if e is not None else ''}^({omml_to_linear(sup) if sup is not None else ''})"
    if t == "sSub":
        e = el.find(f"{{{M}}}e")
        sub = el.find(f"{{{M}}}sub")
        return f"{omml_to_linear(e) if e is not None else ''}_({omml_to_linear(sub) if sub is not None else ''})"
    if t == "sSubSup":
        e = el.find(f"{{{M}}}e")
        sub = el.find(f"{{{M}}}sub")
        sup = el.find(f"{{{M}}}sup")
        return (f"{omml_to_linear(e) if e is not None else ''}"
                f"_({omml_to_linear(sub) if sub is not None else ''})"
                f"^({omml_to_linear(sup) if sup is not None else ''})")
    if t == "rad":  # radical
        e = el.find(f"{{{M}}}e")
        deg = el.find(f"{{{M}}}deg")
        return f"root[{omml_to_linear(deg) if deg is not None else ''}]({omml_to_linear(e) if e is not None else ''})"
    if t == "d":  # delimiter (parentheses)
        inner = "".join(omml_to_linear(c) for c in el.findall(f"{{{M}}}e"))
        return f"({inner})"
    if t == "nary":  # n-ary operator (sum, integral, product)
        naryPr = el.find(f"{{{M}}}naryPr")
        chr_el = naryPr.find(f"{{{M}}}chr") if naryPr is not None else None
        op = chr_el.get(f"{{{M}}}val") if chr_el is not None else "∫"
        sub = el.find(f"{{{M}}}sub")
        sup = el.find(f"{{{M}}}sup")
        e = el.find(f"{{{M}}}e")
        return (f"{op}_({omml_to_linear(sub) if sub is not None else ''})"
                f"^({omml_to_linear(sup) if sup is not None else ''})"
                f"[{omml_to_linear(e) if e is not None else ''}]")
    if t == "func":
        fname = el.find(f"{{{M}}}fName")
        e = el.find(f"{{{M}}}e")
        return f"{omml_to_linear(fname) if fname is not None else ''}({omml_to_linear(e) if e is not None else ''})"
    if t in ("e", "num", "den", "sup", "sub", "deg", "fName", "oMath", "oMathPara"):
        return "".join(omml_to_linear(c) for c in el)
    if t == "limLow":
        e = el.find(f"{{{M}}}e")
        lim = el.find(f"{{{M}}}lim")
        return f"{omml_to_linear(e) if e is not None else ''}_lim({omml_to_linear(lim) if lim is not None else ''})"
    if t == "lim":
        return "".join(omml_to_linear(c) for c in el)
    if t == "acc":
        e = el.find(f"{{{M}}}e")
        return omml_to_linear(e) if e is not None else ""
    if t == "bar":
        e = el.find(f"{{{M}}}e")
        return f"bar({omml_to_linear(e) if e is not None else ''})"
    if t == "m":  # matrix
        rows = el.findall(f".//{{{M}}}mr")
        rtxt = [" & ".join(omml_to_linear(e) for e in row.findall(f"{{{M}}}e")) for row in rows]
        return "[MATRIX: " + " ; ".join(rtxt) + "]"
    if t == "mr":
        return " ".join(omml_to_linear(c) for c in el)
    # 兜底：直接拼接子节点
    return "".join(omml_to_linear(c) for c in el)


def para_has_math(p):
    return p.find(f".//{{{M}}}oMath") is not None or p.find(f".//{{{M}}}oMathPara") is not None


def extract_para_text(p):
    """按顺序提取段落中的普通文本(w:t)与公式(oMath)，公式以 $...$ 标记。"""
    parts = []
    for child in p.iter():
        ct = child.tag
        if ct == f"{{{W}}}t" and child.text:
            # 确认不在 math 内部(omml的t是 m:t, w:t 不会在 math 内)
            parts.append(child.text)
        elif ct == f"{{{M}}}oMath":
            parts.append(" $[" + omml_to_linear(child) + "]$ ")
        elif ct == f"{{{M}}}oMathPara":
            for om in child.findall(f"{{{M}}}oMath"):
                parts.append(" $[" + omml_to_linear(om) + "]$ ")
    return "".join(parts)


def get_style(p, styles_map):
    pPr = p.find(f"{{{W}}}pPr")
    if pPr is not None:
        pStyle = pPr.find(f"{{{W}}}pStyle")
        if pStyle is not None:
            sid = pStyle.get(f"{{{W}}}val")
            return styles_map.get(sid, sid)
    return "Normal"


def main():
    with zipfile.ZipFile(DOCX) as z:
        names = z.namelist()
        doc_xml = z.read("word/document.xml")
        styles_xml = z.read("word/styles.xml")

    # 样式映射
    sroot = ET.fromstring(styles_xml)
    styles_map = {}
    for st in sroot.findall(f"{{{W}}}style"):
        sid = st.get(f"{{{W}}}styleId")
        name_el = st.find(f"{{{W}}}name")
        if sid and name_el is not None:
            styles_map[sid] = name_el.get(f"{{{W}}}val")

    root = ET.fromstring(doc_xml)
    body = root.find(f"{{{W}}}body")

    lines = []
    formula_count = 0
    table_count = 0

    def walk(element):
        nonlocal formula_count, table_count
        for child in element:
            ct = tag(child)
            if ct == "p":
                txt = extract_para_text(child).strip()
                style = get_style(child, styles_map)
                if "$[" in txt:
                    formula_count += txt.count("$[")
                if txt:
                    if "Heading" in style or "标题" in style:
                        level = "".join(c for c in style if c.isdigit()) or "1"
                        lines.append("#" * min(int(level), 6) + " " + txt)
                    else:
                        lines.append(txt)
                    lines.append("")
            elif ct == "tbl":
                table_count += 1
                lines.append(f"<!-- TABLE {table_count} START -->")
                rows_data = []
                for tr in child.findall(f"{{{W}}}tr"):
                    cells = []
                    for tc in tr.findall(f"{{{W}}}tc"):
                        cell_txt = " ".join(
                            extract_para_text(p).strip() for p in tc.findall(f".//{{{W}}}p")
                        ).strip()
                        cells.append(cell_txt.replace("|", "/"))
                    rows_data.append(cells)
                if rows_data:
                    ncols = max(len(r) for r in rows_data)
                    for r in rows_data:
                        r = r + [""] * (ncols - len(r))
                    lines.append("| " + " | ".join(rows_data[0]) + " |")
                    lines.append("|" + "---|" * ncols)
                    for r in rows_data[1:]:
                        lines.append("| " + " | ".join(r) + " |")
                lines.append(f"<!-- TABLE {table_count} END -->")
                lines.append("")

    walk(body)

    text = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Extraction complete: {len(lines)} lines, {table_count} tables, ~{formula_count} inline formulas")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
