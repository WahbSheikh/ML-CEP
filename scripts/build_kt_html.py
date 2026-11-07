"""Render the dated KT handbook and exact schema into a portable offline HTML.

This only updates KT documentation. It never fits models or edits results.
After a new experiment, review the narrative before changing the snapshot guard.
"""

from pathlib import Path
import base64
import html
import json
import re

import mistune

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/KT_BEGINNER_GUIDE.md"


def describe_feature(name):
    """Connect every exact column name to the feature-family teaching text."""
    if name == "pitch_voiced_fraction":
        return "Nonzero selected pitch frames ka fraction; voiced proxy, not verified voicing"
    if name.startswith("bicoherence_"):
        parts = name.split("_", 2)
        return f"Bicoherence {parts[1]} ka frequency-pair {parts[2]}"
    match = re.fullmatch(r"fourier_(\d+)_(\d+)_(\d+)_(mean|std)", name)
    if match:
        low, high, component, statistic = match.groups()
        return f"[{low}, {high}) Hz relative band power ka frame {statistic}"
    match = re.fullmatch(r"(.+)_(\d+)_(mean|std)", name)
    if not match:
        raise ValueError(f"Unrecognized schema field: {name}")
    family, component, statistic = match.groups()
    labels = {
        "mfcc": "MFCC coefficient",
        "mfcc_delta": "First-order MFCC delta",
        "mfcc_delta2": "Second-order MFCC delta",
        "mel": "Log-Mel band",
        "chroma": "Chroma pitch-class bin",
        "spectral_centroid": "Spectral centroid",
        "spectral_bandwidth": "Spectral bandwidth",
        "spectral_contrast": "Spectral contrast row",
        "spectral_rolloff": "Spectral rolloff",
        "spectral_flatness": "Spectral flatness",
        "rms": "Frame RMS",
        "zcr": "Zero-crossing rate",
        "pitch": "Nonzero pitch proxy",
    }
    if family not in labels:
        raise ValueError(f"Unrecognized feature family: {family}")
    axis = "nonzero-frame" if family == "pitch" else "time-frame"
    return f"{labels[family]} {int(component)} ka {axis} {statistic}"


class HandbookRenderer(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__(escape=False)
        self.toc = []
        self.ids = set()
        self.embedded_images = 0

    def heading(self, text, level, **attrs):
        plain = html.unescape(re.sub(r"<[^>]+>", "", text))
        identifier = re.sub(r"[^\w\s-]", "", plain.lower())
        identifier = re.sub(r"\s+", "-", identifier).strip("-")
        original = identifier
        counter = 1
        while identifier in self.ids:
            identifier = f"{original}-{counter}"
            counter += 1
        self.ids.add(identifier)
        if level == 2 and re.match(r"^\d{2} ", plain):
            self.toc.append((identifier, plain))
        return f'<h{level} id="{identifier}">{text}</h{level}>\n'

    def image(self, text, url, title=None):
        if url.startswith(("http:", "https:", "data:")):
            raise ValueError("KT export expects local project figures only")
        path = (DOC.parent / url).resolve()
        if not path.is_relative_to(ROOT) or path.suffix.lower() != ".png":
            raise ValueError(f"Unexpected figure path: {url}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        self.embedded_images += 1
        alt = html.escape(text, quote=True)
        return f'<figure><img src="data:image/png;base64,{encoded}" alt="{alt}" loading="lazy"><figcaption>{html.escape(text)}</figcaption></figure>'


CSS = """
:root { color-scheme:light; --ink:#192b3a; --accent:#087f8c; --paper:#fff; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:1.5rem; }
body { margin:0; color:var(--ink); background:#f1f5f7; font-family:Segoe UI,Arial,sans-serif; font-size:17px; line-height:1.75; }
.layout { display:grid; grid-template-columns:290px minmax(0,1fr); max-width:1550px; margin:auto; }
nav { position:sticky; top:0; height:100vh; overflow:auto; padding:26px 20px; font-size:14px; background:#122a38; color:#eff9fc; }
nav strong { display:block; font-size:21px; margin-bottom:8px; }
nav p { color:#bdd0d9; line-height:1.6; }
nav ol { padding-left:0; list-style:none; }
nav li { margin:0 0 9px; }
nav a { color:#d5edf2; text-decoration:none; display:block; border-left:2px solid #315262; padding-left:10px; }
nav a:hover { color:#fff; border-color:#65d9d7; }
main { min-width:0; padding:45px 55px 80px; background:var(--paper); }
.eyebrow { color:var(--accent); font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:13px; }
.reading-note { border-left:4px solid var(--accent); background:#edf8f8; padding:15px 20px; margin:20px 0 35px; }
h1 { line-height:1.2; font-size:39px; color:#102f43; margin:14px 0 22px; }
h2 { font-size:29px; line-height:1.3; margin-top:65px; padding-top:18px; border-top:2px solid #d9e6eb; color:#12435b; }
h3 { font-size:21px; line-height:1.45; margin-top:34px; color:#14576a; }
p { margin:13px 0; }
a { color:#086a85; text-underline-offset:3px; }
strong { color:#123a50; }
code { background:#edf2f5; border-radius:4px; padding:2px 5px; font-size:.88em; font-family:Consolas,monospace; overflow-wrap:anywhere; }
pre { background:#142c3a; color:#ecf5fa; border-radius:8px; padding:20px; overflow:auto; line-height:1.6; font-size:14px; }
pre code { background:transparent; color:inherit; padding:0; font-size:inherit; overflow-wrap:normal; }
.table-wrap { overflow-x:auto; margin:22px 0; }
table { border-collapse:collapse; width:100%; font-size:14px; line-height:1.6; }
th,td { text-align:left; vertical-align:top; padding:11px 12px; border:1px solid #d6e1e6; overflow-wrap:anywhere; }
th { background:#e9f1f5; color:#173f54; }
tbody tr:nth-child(even) { background:#f8fafb; }
figure { margin:28px 0; }
figure img { display:block; width:100%; height:auto; border:1px solid #dde6eb; }
figcaption { color:#506777; font-size:13px; margin-top:8px; }
blockquote { margin:24px 0; padding:8px 20px; border-left:4px solid #a3c3cc; background:#f6f9fa; }
footer { border-top:1px solid #d6e1e6; margin-top:60px; padding-top:20px; color:#526b78; font-size:14px; }
@media(max-width:1000px) { .layout{display:block;} nav{position:relative;height:auto;max-height:340px;} main{padding:28px 22px;} h1{font-size:31px;} h2{font-size:25px;} }
@media print { @page{size:A4;margin:16mm;} body{background:white;font-size:10.5pt;line-height:1.5;} .layout{display:block;} nav,.reading-note{display:none;} main{padding:0;} h1{font-size:25pt;} h2{font-size:19pt;break-after:avoid;} h3{font-size:14pt;break-after:avoid;} table{font-size:8pt;} thead{display:table-header-group;} tr{break-inside:avoid;} .table-wrap{overflow:visible;} pre{background:#f4f6f8;color:#142c3a;white-space:pre-wrap;overflow-wrap:anywhere;font-size:8.5pt;} figure{break-inside:avoid;} a{color:#14516b;} }
"""


def main():
    names = json.loads((ROOT / "artifacts/feature_schema.json").read_text())
    frozen = json.loads((ROOT / "artifacts/frozen_selection.json").read_text())
    expected = json.loads(frozen["models"]["SVM"]["parameters"])
    if (
        len(names) != 273
        or frozen["winner"] != "SVM"
        or expected != {"C": 100, "gamma": 0.001}
    ):
        raise ValueError(
            "KT is a dated snapshot: review the narrative before rebuilding for new results"
        )
    source = DOC.read_text(encoding="utf-8")
    rows = ["| Zero-based column | Exact name | Meaning |", "|---:|---|---|"]
    rows += [
        f"| {index} | `{name}` | {describe_feature(name)} |"
        for index, name in enumerate(names)
    ]
    content = "\n\n" + "\n".join(rows) + "\n\n"
    pattern = r"(<!-- FEATURE_DICTIONARY -->).*?(<!-- /FEATURE_DICTIONARY -->)"
    source, count = re.subn(
        pattern, lambda match: match[1] + content + match[2], source, flags=re.S
    )
    if count != 1:
        raise ValueError("Expected exactly one feature dictionary region")
    DOC.write_text(source, encoding="utf-8")
    renderer = HandbookRenderer()
    markdown = mistune.create_markdown(
        renderer=renderer, plugins=["table", "strikethrough"]
    )
    body = markdown(source)
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace(
        "</table>", "</table></div>"
    )
    assert len(renderer.toc) == 30 and renderer.embedded_images == 6
    toc = "".join(
        f'<li><a href="#{identifier}">{html.escape(label)}</a></li>'
        for identifier, label in renderer.toc
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ML-CEP Complete KT - Roman Urdu Beginner Handbook</title><style>{CSS}</style></head>
<body><div class="layout"><nav aria-label="Chapters"><strong>ML-CEP Complete KT</strong>
<p>Roman Urdu beginner handbook<br>30 chapters · 273 exact features<br>Snapshot: 7 November 2026</p><ol>{toc}</ol></nav>
<main><div class="eyebrow">Learn · Explain · Reproduce · Transfer</div>
<div class="reading-note">Yeh reading copy offline kaam karti hai: saare six charts embedded hain. Browser mein Ctrl+F se search, Ctrl+P se print/save kar sakte hain. Project source links ke liye complete folder chahiye. Pehli reading chapter1 ki learning route follow karein.</div>
{body}<footer>Generated from KT_BEGINNER_GUIDE.md by scripts/build_kt_html.py. Narrative is a dated learning snapshot; no model training or result modification is performed by this export.</footer></main></div></body></html>"""
    output = DOC.with_suffix(".html")
    output.write_text(page, encoding="utf-8")
    print(
        f"KT: {len(source.split()):,} whitespace-separated words; 30 chapters; {len(names)} schema entries; {renderer.embedded_images} embedded charts"
    )
    print(f"HTML: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
