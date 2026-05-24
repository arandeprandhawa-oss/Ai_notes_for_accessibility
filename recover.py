import os
import sys
import pathlib
import traceback
import re
import base64
import html as html_lib

if getattr(sys, 'frozen', False):
    BASE_DIR = pathlib.Path(sys.executable).parent
else:
    BASE_DIR = pathlib.Path(__file__).parent

SESSIONS_DIR = BASE_DIR / "sessions"

print("=" * 52)
print("  LectureLens — Recovery Tool")
print("  Updated for Sonnet + Local Pre-filter builds")
print("=" * 52)

if not SESSIONS_DIR.exists():
    print("\n  No sessions folder found on this USB.")
    input("\n  Press Enter to exit...")
    sys.exit(1)

sessions = sorted([d for d in SESSIONS_DIR.iterdir() if d.is_dir()], reverse=True)

if not sessions:
    print("\n  No sessions found.")
    input("\n  Press Enter to exit...")
    sys.exit(1)

print("\n  Available sessions:\n")
for i, s in enumerate(sessions):
    backup_dir = s / "backups"
    txt_files = list(backup_dir.glob("*.txt")) if backup_dir.exists() else []
    duplicate_dir = backup_dir / "duplicate_skipped_captures" if backup_dir.exists() else None
    dup_count = len(list(duplicate_dir.glob("*"))) if duplicate_dir and duplicate_dir.exists() else 0
    has_html = any(s.glob("*.html"))
    status = "HTML files exist" if has_html else f"{len(txt_files)} backup files"
    if dup_count:
        status += f", {dup_count} skipped duplicate image(s)"
    print(f"  [{i+1}] {s.name}  ({status})")

print()
choice = input("  Enter session number to recover (or press Enter for most recent): ").strip()

if choice == "":
    SESSION_DIR = sessions[0]
elif choice.isdigit() and 1 <= int(choice) <= len(sessions):
    SESSION_DIR = sessions[int(choice) - 1]
else:
    print("  Invalid choice.")
    input("  Press Enter to exit...")
    sys.exit(1)

BACKUP_DIR = SESSION_DIR / "backups"
IMAGES_DIR = SESSION_DIR / "images"
REPLACED_DIR = BACKUP_DIR / "replaced_captures"
DUPLICATE_DIR = BACKUP_DIR / "duplicate_skipped_captures"

print(f"\n  Recovering: {SESSION_DIR.name}")

if not BACKUP_DIR.exists():
    print("\n  No backups folder found.")
    input("\n  Press Enter to exit...")
    sys.exit(1)

# ── Helpers ────────────────────────────────────────────────────────────────
def parse_ts_from_header_or_name(txt_path, raw, kind="capture"):
    """Try to recover HH-MM-SS timestamp from a backup file."""
    lines = raw.split("\n")
    header = lines[0] if lines else ""

    # Preferred: timestamps in the file name.
    m = re.search(r"(\d{2}-\d{2}-\d{2})", txt_path.name)
    if m:
        return m.group(1)

    if kind == "capture":
        if "—" in header:
            return header.split("—")[-1].strip()
        if "--" in header:
            return header.split("--")[-1].strip()
    else:
        if " at " in header:
            return header.split(" at ")[-1].strip()

    return txt_path.stem


def find_image_for(prefix, ts):
    if not IMAGES_DIR.exists():
        return ""
    for ext in [".jpg", ".jpeg", ".png"]:
        p = IMAGES_DIR / f"{prefix}_{ts}{ext}"
        if p.exists():
            return str(p)
    for img in IMAGES_DIR.glob(f"{prefix}_{ts}*"):
        return str(img)
    return ""


def find_replaced_image_for(ts):
    if not REPLACED_DIR.exists():
        return ""
    for img in REPLACED_DIR.glob(f"capture_{ts}*"):
        return str(img)
    return ""


def read_backup_notes(txt_path):
    r"""Return (raw, notes) from a backup file.

    Backup files are not all shaped the same way:
      - Normal capture files:  header / "===" / blank / notes...      (notes at line 3)
      - LOOKBACK_CLEANED files: header / "Smart Lookback checked..." /
        optional "This capture replaced..." / "Reason: ..." / "===" / blank / notes
      - Screenshot files:      header / "===" / blank / [optional Instructor note] / notes

    So the notes always begin AFTER the "=====" separator line, but that line is at
    a variable position. Splitting on a hardcoded line index (the old lines[3:])
    prepended several junk metadata lines onto every Smart-Lookback-cleaned note.
    We find the separator instead.
    """
    raw = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = raw.split("\n")

    sep_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        # Real LectureLens separators are a run of '=' (the app writes 50 of them).
        # Accept any run of 3+ to be tolerant of truncated/edited backups.
        if s and set(s) == {"="} and len(s) >= 3:
            sep_idx = i
            break

    if sep_idx is not None:
        body = lines[sep_idx + 1:]
    else:
        # No separator found (unusual/legacy file). The normal capture format puts
        # notes at line 3, but LOOKBACK_CLEANED files have extra metadata lines
        # first. Skip any leading known-metadata lines so we never prepend junk.
        start = 1  # always skip the header line
        meta_prefixes = (
            "Smart Lookback checked", "This capture replaced",
            "Reason:", "Status:",
        )
        while start < len(lines) and (
            lines[start].strip() == "" or lines[start].startswith(meta_prefixes)
        ):
            start += 1
        body = lines[start:]

    return raw, "\n".join(body).strip()

# ── Detect smart-lookback replacements and cleaned files ───────────────────
replaced_old_ts = set()
replacement_map = {}  # new_ts -> list old_ts
replaced_reasons = {}

for txt in BACKUP_DIR.glob("notes_*_REPLACED_BY_CAPTURE_*.txt"):
    try:
        nums = re.findall(r"(\d{2}-\d{2}-\d{2})", txt.name)
        if len(nums) >= 2:
            old_ts, new_ts = nums[0], nums[-1]
            replaced_old_ts.add(old_ts)
            replacement_map.setdefault(new_ts, []).append(old_ts)
            raw = txt.read_text(encoding="utf-8", errors="replace")
            reason = "newer capture was clearer"
            for line in raw.splitlines():
                if line.lower().startswith("reason:"):
                    reason = line.split(":", 1)[1].strip()
                    break
            replaced_reasons[new_ts] = reason
    except Exception:
        pass

# Prefer LOOKBACK_CLEANED backups over original notes for the same timestamp.
cleaned_note_files = {}
for txt in BACKUP_DIR.glob("notes_*_LOOKBACK_CLEANED.txt"):
    m = re.search(r"(\d{2}-\d{2}-\d{2})", txt.name)
    if m:
        cleaned_note_files[m.group(1)] = txt

all_entries = []
seen_capture_ts = set()

# ── Capture notes ──────────────────────────────────────────────────────────
for txt in sorted(BACKUP_DIR.glob("notes_*.txt")):
    try:
        name = txt.name
        if "_screenshot" in name:
            continue
        if "_REPLACED_BY_CAPTURE_" in name:
            # Metadata file; not an actual note entry.
            continue
        if "_LOOKBACK_CLEANED" in name:
            # Added via cleaned_note_files below.
            continue

        raw, notes = read_backup_notes(txt)
        ts = parse_ts_from_header_or_name(txt, raw, kind="capture")

        if ts in seen_capture_ts:
            continue

        if ts in cleaned_note_files:
            raw2, notes2 = read_backup_notes(cleaned_note_files[ts])
            notes = notes2 or notes

        skip_final = ts in replaced_old_ts
        img_path = find_image_for("capture", ts)
        if not img_path and skip_final:
            img_path = find_replaced_image_for(ts)

        entry = {
            "type": "capture",
            "timestamp": ts,
            "notes": notes,
            "path": img_path or "",
            "skip_in_final_outputs": skip_final,
        }

        if ts in replacement_map:
            entry["replaced_previous_timestamps"] = replacement_map.get(ts, [])
            entry["replaced_previous_reason"] = replaced_reasons.get(ts, "newer capture was clearer")

        all_entries.append(entry)
        seen_capture_ts.add(ts)
    except Exception as e:
        print(f"  [!] Skipped {txt.name}: {e}")

# Handle cleaned files that have no normal original backup.
for ts, txt in cleaned_note_files.items():
    if ts in seen_capture_ts:
        continue
    try:
        raw, notes = read_backup_notes(txt)
        entry = {
            "type": "capture",
            "timestamp": ts,
            "notes": notes,
            "path": find_image_for("capture", ts),
            "skip_in_final_outputs": ts in replaced_old_ts,
        }
        if ts in replacement_map:
            entry["replaced_previous_timestamps"] = replacement_map.get(ts, [])
            entry["replaced_previous_reason"] = replaced_reasons.get(ts, "newer capture was clearer")
        all_entries.append(entry)
        seen_capture_ts.add(ts)
    except Exception as e:
        print(f"  [!] Skipped {txt.name}: {e}")

# ── Screenshot notes ───────────────────────────────────────────────────────
for txt in sorted(BACKUP_DIR.glob("*_screenshot.txt")):
    try:
        raw = txt.read_text(encoding="utf-8", errors="replace")
        lines = raw.split("\n")
        ts = parse_ts_from_header_or_name(txt, raw, kind="screenshot")

        # Notes begin after the "=====" separator. Anything before it is header.
        sep_idx = None
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s and set(s) == {"="} and len(s) >= 3:
                sep_idx = i
                break
        body = lines[sep_idx + 1:] if sep_idx is not None else lines[2:]

        instructor_note = ""
        notes_lines = []
        for line in body:
            if line.startswith("Instructor note:"):
                instructor_note = line.replace("Instructor note:", "").strip()
            else:
                notes_lines.append(line)
        img_path = find_image_for("screenshot", ts)
        all_entries.append({
            "type": "screenshot",
            "timestamp": ts,
            "notes": "\n".join(notes_lines).strip(),
            "path": img_path or "",
            "instructor_note": instructor_note,
        })
    except Exception as e:
        print(f"  [!] Skipped {txt.name}: {e}")

def parse_qa_note(raw):
    """Parse a text/photo note backup into (question, answer).

    File shape: header / "=====" / blank / "QUESTION/NOTE:" / text... /
                "CLAUDE'S ANSWER:" / answer...
    The marker lines flip the section and are not themselves included.
    """
    question_lines = []
    answer_lines = []
    section = None
    for line in raw.split("\n"):
        if line.strip() and set(line.strip()) == {"="} and len(line.strip()) >= 3:
            continue  # separator line — never content
        if "QUESTION/NOTE:" in line:
            section = "q"
        elif "CLAUDE" in line and "ANSWER" in line:
            section = "a"
        elif section == "q":
            question_lines.append(line)
        elif section == "a":
            answer_lines.append(line)
    return "\n".join(question_lines).strip(), "\n".join(answer_lines).strip()


# ── Text notes ─────────────────────────────────────────────────────────────
for txt in sorted(BACKUP_DIR.glob("note_*.txt")):
    try:
        if "_photo" in txt.name:
            continue
        raw = txt.read_text(encoding="utf-8", errors="replace")
        ts = parse_ts_from_header_or_name(txt, raw, kind="note")
        question, answer = parse_qa_note(raw)
        all_entries.append({"type": "text_note", "timestamp": ts, "note": question, "answer": answer})
    except Exception as e:
        print(f"  [!] Skipped {txt.name}: {e}")

# ── Photo notes ────────────────────────────────────────────────────────────
for txt in sorted(BACKUP_DIR.glob("*_photo.txt")):
    try:
        raw = txt.read_text(encoding="utf-8", errors="replace")
        ts = parse_ts_from_header_or_name(txt, raw, kind="photo_note")
        question, answer = parse_qa_note(raw)
        all_entries.append({
            "type": "photo_note",
            "timestamp": ts,
            "note": question,
            "answer": answer,
            "photo_path": find_image_for("note_photo", ts),
        })
    except Exception as e:
        print(f"  [!] Skipped {txt.name}: {e}")

all_entries.sort(key=lambda x: x.get("timestamp", ""))
active_entries = [e for e in all_entries if not e.get("skip_in_final_outputs")]
replaced_count = len([e for e in all_entries if e.get("skip_in_final_outputs")])
duplicate_count = len(list(DUPLICATE_DIR.glob("*"))) if DUPLICATE_DIR.exists() else 0

print(f"  Found {len(all_entries)} total entries.")
if replaced_count:
    print(f"  Smart Lookback: hiding {replaced_count} replaced old capture(s) from recovered final HTML.")
if duplicate_count:
    print(f"  Local pre-filter: found {duplicate_count} skipped duplicate image(s) in backup.")

if not active_entries:
    print("\n  No usable active data found.")
    input("\n  Press Enter to exit...")
    sys.exit(1)

# ── HTML rendering helpers ─────────────────────────────────────────────────
MATHJAX = r'''
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  }
};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
'''

CSS = """<style>
body{font-family:Georgia,serif;max-width:900px;margin:0 auto;padding:24px;color:#1a1a1a;background:#fafafa}
h1{color:#1e5aa0;border-bottom:2px solid #1e5aa0;padding-bottom:8px}
h2{color:#1e5aa0;font-size:1.15em;margin:20px 0 8px}
h3{color:#444;font-size:1em;margin:14px 0 6px}
.cap-hdr{background:#1e5aa0;color:white;padding:10px 16px;border-radius:4px;margin:24px 0 12px;font-family:Arial,sans-serif;font-size:13px;display:flex;justify-content:space-between}
.ss-hdr{background:#2d6a4f;color:white;padding:10px 16px;border-radius:4px;margin:24px 0 12px;font-family:Arial,sans-serif;font-size:13px}
.note-hdr{background:#a05020;color:white;padding:10px 16px;border-radius:4px;margin:24px 0 12px;font-family:Arial,sans-serif;font-size:13px}
.img-type{background:#fff8e6;border:1px solid #e8c840;border-left:3px solid #c8a000;padding:6px 12px;border-radius:0 4px 4px 0;font-size:12px;color:#7a5800;margin-bottom:12px;font-family:Arial,sans-serif}
.smart-replace{background:#eef8ff;border:1px solid #8bbce8;border-left:4px solid #1e5aa0;padding:8px 12px;border-radius:0 4px 4px 0;font-size:12px;color:#24445f;margin:8px 0 12px;font-family:Arial,sans-serif}
.prefilter{background:#f6f6f6;border:1px solid #ccc;border-left:4px solid #666;padding:8px 12px;border-radius:0 4px 4px 0;font-size:12px;color:#333;margin:8px 0 12px;font-family:Arial,sans-serif}
img.cimg{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:8px 0 16px}
.ans-lbl{color:#2d6a4f;font-weight:bold;margin-top:12px;font-family:Arial,sans-serif;font-size:13px}
.q-txt{font-style:italic;color:#555;margin:8px 0}
ul,ol{padding-left:1.5em}li{margin-bottom:4px}
p{line-height:1.7;margin:8px 0}
.footer{margin-top:40px;padding-top:12px;border-top:1px solid #ddd;font-size:11px;color:#aaa;font-family:Arial,sans-serif;text-align:center}
.toc{background:#f0f5ff;border:1px solid #c0d0f0;border-radius:4px;padding:16px 20px;margin:16px 0}
.toc a{color:#1e5aa0;text-decoration:none;display:block;margin:4px 0;font-family:Arial,sans-serif;font-size:13px}
.recovered{background:#f0f8f5;border:1px solid #9fd4b8;border-radius:4px;padding:10px 16px;margin:12px 0;font-family:Arial,sans-serif;font-size:12px;color:#2d6a4f}
.math-block{margin:12px 0; overflow-x:auto; text-align:center}
</style>"""


def md_to_html(text):
    r"""Convert simple markdown to HTML while preserving LaTeX for MathJax."""
    text = text or ""
    lines = text.split("\n")
    out = []
    in_ul = False
    in_math = False
    math_end = None
    math_lines = []

    latex_envs = (
        "cases", "aligned", "align", "align*", "array",
        "matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix",
        "smallmatrix", "split", "gather", "gathered",
    )

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def escape_keep_inline_math(s):
        parts = re.split(r'(\$[^$]+\$)', s)
        fixed = []
        for part in parts:
            is_dollar_span = (
                part.startswith('$') and part.endswith('$') and len(part) > 2
            )
            # A $...$ span is only real inline math if the inside actually looks
            # mathematical. This stops prose like "it costs $5 and $30" from being
            # rendered as math (the text between the two $ signs would otherwise
            # be swallowed). Math-ish = contains a LaTeX command, super/subscript,
            # braces, or an operator/relation — not just digits and words.
            if is_dollar_span:
                inner = part[1:-1]
                looks_mathy = bool(re.search(r'[\\^_{}=+]|\\[a-zA-Z]', inner)) or \
                              bool(re.search(r'[a-zA-Z]\d|\d[a-zA-Z]', inner))
                if not looks_mathy:
                    is_dollar_span = False
            if is_dollar_span:
                fixed.append(part)
            else:
                esc = html_lib.escape(part)
                esc = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", esc)
                fixed.append(esc)
        return "".join(fixed)

    def begin_math_block(line):
        stripped = line.strip()
        if stripped.startswith("$$"):
            if stripped.endswith("$$") and stripped != "$$":
                return ("single", "$$")
            return ("start", "$$")
        if stripped.startswith(r"\["):
            if stripped.endswith(r"\]") and stripped != r"\[":
                return ("single", r"\]")
            return ("start", r"\]")
        for env in latex_envs:
            if rf"\begin{{{env}}}" in line:
                if rf"\end{{{env}}}" in line:
                    return ("single_env", env)
                return ("start_env", env)
        return None

    def emit_math_block(block_lines, wrap_raw_env=True):
        block = "\n".join(block_lines).strip()
        if wrap_raw_env and block.startswith(r"\begin"):
            block = "$$\n" + block + "\n$$"
        out.append('<div class="math-block">\n' + block + '\n</div>')

    for line in lines:
        stripped = line.strip()

        if in_math:
            math_lines.append(line)
            if math_end == "$$" and stripped.endswith("$$"):
                emit_math_block(math_lines, wrap_raw_env=False)
                in_math = False; math_lines = []; math_end = None
            elif math_end == r"\]" and stripped.endswith(r"\]"):
                emit_math_block(math_lines, wrap_raw_env=False)
                in_math = False; math_lines = []; math_end = None
            elif math_end and math_end not in ("$$", r"\]") and rf"\end{{{math_end}}}" in line:
                emit_math_block(math_lines, wrap_raw_env=True)
                in_math = False; math_lines = []; math_end = None
            continue

        math_info = begin_math_block(line)
        if math_info:
            close_ul()
            kind, ending = math_info
            if kind in ("single", "single_env"):
                emit_math_block([line], wrap_raw_env=(kind == "single_env"))
            else:
                in_math = True
                math_end = ending
                math_lines = [line]
            continue

        if stripped == "":
            close_ul()
            continue

        if line.startswith("### "):
            close_ul(); out.append(f"<h3>{escape_keep_inline_math(line[4:])}</h3>")
        elif line.startswith("## "):
            close_ul(); out.append(f"<h2>{escape_keep_inline_math(line[3:])}</h2>")
        elif line.startswith("# "):
            close_ul(); out.append(f"<h2>{escape_keep_inline_math(line[2:])}</h2>")
        elif line.startswith("- ") or line.startswith("* "):
            txt = escape_keep_inline_math(line[2:].strip())
            if txt:
                if not in_ul:
                    out.append("<ul>"); in_ul = True
                out.append(f"<li>{txt}</li>")
        else:
            close_ul()
            out.append(f"<p>{escape_keep_inline_math(line)}</p>")

    close_ul()
    if in_math and math_lines:
        emit_math_block(math_lines, wrap_raw_env=True)
    return "\n".join(out)


def img_tag(path, style=""):
    try:
        p = pathlib.Path(path)
        if not p.exists():
            return '<p style="color:#999">[Image not available]</p>'
        with open(p, 'rb') as f:
            b = base64.standard_b64encode(f.read()).decode()
        ext = p.suffix.lower().replace('.', '')
        mime = 'jpeg' if ext in ('jpg', 'jpeg') else ext
        return f'<img class="cimg" src="data:image/{mime};base64,{b}" {style}>'
    except Exception:
        return ''

print("\n  Generating HTML files...")
session_name = SESSION_DIR.name

# ── 01_photos.html ─────────────────────────────────────────────────────────
try:
    photo_entries = [e for e in active_entries if e["type"] == "capture"]
    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Photos | {session_name}</title>{CSS}</head><body>
<h1>Lecture Photos (Recovered)</h1>
<p style="font-family:Arial,sans-serif;font-size:13px;color:#666">Session: {session_name} | {len(photo_entries)} active captures | {replaced_count} replaced | {duplicate_count} skipped duplicates | LectureLens Recovery</p>\n'''
    if duplicate_count:
        html += f'<div class="prefilter"><strong>Local pre-filter:</strong> {duplicate_count} near-duplicate image(s) were skipped by the main app and kept in backups/duplicate_skipped_captures.</div>\n'
    cn = 0
    for e in active_entries:
        if e["type"] != "capture":
            continue
        cn += 1
        html += f'<div class="cap-hdr"><span>Capture #{cn} | {e["timestamp"]}</span></div>\n'
        if e.get("replaced_previous_timestamps"):
            repl = ", ".join(e.get("replaced_previous_timestamps", []))
            reason = e.get("replaced_previous_reason", "newer capture was clearer")
            html += f'<div class="smart-replace"><strong>Smart Lookback:</strong> this clearer capture replaced earlier capture(s): {html_lib.escape(repl)}. Reason: {html_lib.escape(reason)}</div>\n'
        html += img_tag(e.get("path", "")) if e.get("path") else '<p style="color:#999">[Image not available]</p>'
    html += f'<div class="footer">LectureLens Recovery &mdash; {session_name}</div></body></html>'
    (SESSION_DIR / "01_photos.html").write_text(html, encoding='utf-8')
    print("  + 01_photos.html")
except Exception as e:
    print(f"  [!] Photos error: {e}")

# ── 02_ai_notes.html ───────────────────────────────────────────────────────
try:
    toc = ""
    cn = sn = nn = 0
    for e in active_entries:
        ts = e.get("timestamp", "")
        if e["type"] == "capture":
            cn += 1; toc += f'<a href="#c{ts}">Capture #{cn} — {ts}</a>\n'
        elif e["type"] == "screenshot":
            sn += 1; toc += f'<a href="#s{ts}">Screenshot #{sn} — {ts}</a>\n'
        elif e["type"] in ("text_note", "photo_note"):
            nn += 1; toc += f'<a href="#n{ts}">Note #{nn} — {ts}</a>\n'

    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>AI Notes | {session_name}</title>{CSS}{MATHJAX}</head><body>
<h1>AI Lecture Notes (Recovered)</h1>
<p style="font-family:Arial,sans-serif;font-size:13px;color:#666">Session: {session_name} | {len(active_entries)} active entries | {replaced_count} replaced | {duplicate_count} skipped duplicates | LectureLens Recovery</p>
<div class="recovered">These notes were recovered from backup files. MathJax is configured for $...$, \\(...\\), $$...$$, and \\[...\\] LaTeX rendering.</div>\n'''
    if duplicate_count:
        html += f'<div class="prefilter"><strong>Local pre-filter:</strong> {duplicate_count} duplicate image(s) were intentionally skipped before Claude processing to reduce cost.</div>\n'
    html += f'<div class="toc"><strong style="font-family:Arial,sans-serif;font-size:13px">Contents</strong>\n{toc}</div>\n'

    cn = sn = nn = 0
    for e in active_entries:
        ts = e.get("timestamp", "")
        if e["type"] == "capture":
            cn += 1
            html += f'<div class="cap-hdr" id="c{ts}"><span>Capture #{cn} | {ts} | Recovered</span></div>\n'
            if e.get("replaced_previous_timestamps"):
                repl = ", ".join(e.get("replaced_previous_timestamps", []))
                reason = e.get("replaced_previous_reason", "newer capture was clearer")
                html += f'<div class="smart-replace"><strong>Smart Lookback:</strong> this clearer capture replaced earlier capture(s): {html_lib.escape(repl)}. Reason: {html_lib.escape(reason)}</div>\n'
            notes = e.get("notes", "")
            if 'Image type:' in notes or 'Screenshot type:' in notes:
                first = notes.split('\n')[0]
                html += f'<div class="img-type">{html_lib.escape(first)}</div>\n'
                notes = '\n'.join(notes.split('\n')[1:])
            html += md_to_html(notes) + '\n'
        elif e["type"] == "screenshot":
            sn += 1
            html += f'<div class="ss-hdr" id="s{ts}">Screenshot #{sn} | {ts} | Recovered</div>\n'
            if e.get("path"):
                html += img_tag(e["path"], 'style="max-height:300px"')
            if e.get("instructor_note"):
                html += f'<p><strong>Teacher note:</strong> {html_lib.escape(e["instructor_note"])}</p>\n'
            html += md_to_html(e.get("notes", "")) + '\n'
        elif e["type"] in ("text_note", "photo_note"):
            nn += 1
            is_photo = e["type"] == "photo_note"
            label = "Photo Note" if is_photo else "Text Note"
            html += f'<div class="note-hdr" id="n{ts}">{label} #{nn} | {ts}</div>\n'
            if is_photo and e.get("photo_path"):
                html += img_tag(e["photo_path"], 'style="max-height:200px"')
            html += f'<p class="q-txt"><strong>Question:</strong> {html_lib.escape(e.get("note", ""))}</p>\n'
            if e.get("answer"):
                html += '<p class="ans-lbl">Claude Answer:</p>\n'
                html += md_to_html(e["answer"]) + '\n'

    html += f'<div class="footer">LectureLens Recovery &mdash; {session_name}</div></body></html>'
    (SESSION_DIR / "02_ai_notes.html").write_text(html, encoding='utf-8')
    print("  + 02_ai_notes.html")
except Exception as e:
    print(f"  [!] Notes error: {e}"); traceback.print_exc()

# ── 03_summary.html ────────────────────────────────────────────────────────
try:
    summary_parts = []
    cn = sn = nn = 0
    for e in active_entries:
        if e["type"] == "capture":
            cn += 1; summary_parts.append(f"[Capture #{cn} at {e['timestamp']}]\n{e.get('notes', '')}")
        elif e["type"] == "screenshot":
            sn += 1; summary_parts.append(f"[Screenshot #{sn} at {e['timestamp']}]\n{e.get('notes', '')}")
        elif e["type"] in ("text_note", "photo_note"):
            nn += 1; summary_parts.append(f"[Q&A #{nn} at {e['timestamp']}]\nQ: {e.get('note', '')}\nA: {e.get('answer', '')}")

    summary_md = "\n\n---\n\n".join(summary_parts)

    html = f'''<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>Summary | {session_name}</title>{CSS}{MATHJAX}</head><body>
<h1>Lecture Summary (Recovered)</h1>
<p style="font-family:Arial,sans-serif;font-size:13px;color:#666">Session: {session_name} | {len(active_entries)} active entries | {replaced_count} replaced | {duplicate_count} skipped duplicates | LectureLens Recovery</p>
<div class="recovered">Compiled from backup files. Replaced captures are hidden, skipped duplicates remain in backup, and LaTeX rendering uses the updated MathJax configuration.</div>\n'''
    html += md_to_html(summary_md)
    html += f'<div class="footer">LectureLens Recovery &mdash; {session_name}</div></body></html>'
    (SESSION_DIR / "03_summary.html").write_text(html, encoding='utf-8')
    print("  + 03_summary.html")
except Exception as e:
    print(f"  [!] Summary error: {e}")

print(f"\n{'='*52}")
print(f"  HTML files saved to:\n  {SESSION_DIR}")
print(f"  Open .html files in any browser for MathJax LaTeX rendering.")
print(f"{'='*52}")
input("\n  Press Enter to exit...")
