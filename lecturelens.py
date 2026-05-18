import cv2
import anthropic
import time
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.mathtext as mathtext
from io import BytesIO
try:
    from PIL import Image
except ImportError:
    Image = None
import subprocess
import ctypes
import base64
import datetime
import os
import pathlib
import threading
import queue
import sys
import traceback
from fpdf import FPDF
from pynput import keyboard
import numpy as np

# ── Base dir ───────────────────────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = pathlib.Path(sys.executable).parent
else:
    BASE_DIR = pathlib.Path(__file__).parent

# ── Load API key — tries config.txt first, falls back to hardcoded ─────────
def load_api_key():
    config_path = BASE_DIR / "config.txt"
    if config_path.exists():
        try:
            raw = config_path.read_bytes()
            # Strip BOM if present
            if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
                raw = raw[2:]
            if raw.startswith(b'\xef\xbb\xbf'):
                raw = raw[3:]
            text = raw.decode("utf-8", errors="ignore")
            # Remove ALL whitespace including hidden line breaks
            text = "".join(text.split())
            if "ANTHROPIC_API_KEY=" in text:
                return text.split("ANTHROPIC_API_KEY=")[1].split(";")[0].strip()
            return text.strip()
        except Exception as e:
            print(f"  Warning: could not read config.txt: {e}")

    # Hardcoded fallback option.
    # Paste your key locally between the quotes below if you really want a hardcoded fallback.
    HARDCODED_ANTHROPIC_API_KEY = "PASTE_YOUR_ANTHROPIC_API_KEY_HERE"
    if HARDCODED_ANTHROPIC_API_KEY and HARDCODED_ANTHROPIC_API_KEY != "PASTE_YOUR_ANTHROPIC_API_KEY_HERE":
        return HARDCODED_ANTHROPIC_API_KEY.strip()

    env_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key

    raise RuntimeError("Missing Anthropic API key. Put ANTHROPIC_API_KEY=your_key_here in config.txt, set ANTHROPIC_API_KEY as an environment variable, or paste it into HARDCODED_ANTHROPIC_API_KEY locally.")

os.environ["ANTHROPIC_API_KEY"] = load_api_key()

# ── Claude client ──────────────────────────────────────────────────────────
client = anthropic.Anthropic()

# ── Ask Claude to name this session ────────────────────────────────────────
session_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
now = datetime.datetime.now()
time_str = now.strftime("%I:%M %p")  # e.g. "02:42 PM"
day_str  = now.strftime("%A")        # e.g. "Friday"

print("  Naming session with Claude...")
try:
    name_resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        messages=[{
            "role": "user",
            "content": f"Give me a short 2-4 word folder name for a lecture session on {day_str} at {time_str}. "
                       f"Use lowercase words separated by hyphens. Return only the folder name, nothing else. "
                       f"Example: morning-biology-class"
        }]
    )
    raw_name = name_resp.content[0].text.strip().lower()
    # Sanitize — keep only letters, numbers, hyphens
    import re
    safe_name = re.sub(r'[^a-z0-9\-]', '', raw_name)[:40]
    if not safe_name:
        safe_name = "lecture-session"
    session_name = f"{session_time}_{safe_name}"
except Exception:
    session_name = session_time  # fallback to timestamp only

# ── Session folder (dated + named) inside sessions/ on the USB ─────────────
SESSION_DIR = BASE_DIR / "sessions" / session_name
SESSION_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR = SESSION_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
REPLACED_DIR = BACKUP_DIR / "replaced_captures"
REPLACED_DIR.mkdir(parents=True, exist_ok=True)
DUPLICATE_SKIPPED_DIR = BACKUP_DIR / "duplicate_skipped_captures"
DUPLICATE_SKIPPED_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = SESSION_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
captures = []          # list of {path, notes, timestamp}
active_threads = 0     # tracks screenshot/note threads still running
captures_lock = threading.Lock()  # for safely appending to captures list
instructor_notes = []  # list of {timestamp, note}
notes_lock = threading.Lock()
stopped = False
flash_frame = False    # signals preview to flash green border
note_flash = False     # signals preview to flash blue border (note added)
last_capture_time = 0  # cooldown prevents double/auto captures
SMART_LOOKBACK_REPAIR = True  # lets newer captures fix unclear/wrong older capture notes
SMART_DELETE_WORSE_CAPTURES = True  # hides/moves older captures when a newer capture is clearly better
LOCAL_DUPLICATE_PREFILTER = True  # free local check before paying Claude for near-identical board photos
DUPLICATE_DIFF_THRESHOLD = 0.010  # conservative: lower = skip fewer photos
DUPLICATE_EDGE_DIFF_THRESHOLD = 0.020
SHARPNESS_KEEP_MULTIPLIER = 1.18  # if new duplicate is much sharper, still send it to Claude
duplicate_skipped_count = 0
camera_index = 0       # which camera is active
capture_queue = queue.Queue()  # queue of raw frames waiting to be processed
note_input_active = False  # True while user is typing a note in terminal
screenshot_threads = []  # non-daemon screenshot workers that must finish before exit

print("=" * 52)
print("  LectureLens — AI Lecture Notes")
print("=" * 52)
print(f"  Session: {session_name}")
print(f"  Saving to: {SESSION_DIR}")
print("=" * 52)
print("  SPACE / remote button = capture")
print("  N                     = add instructor note (with optional AI reply)")
print("  ESC                   = stop & generate PDFs")
print("=" * 52)

# ── Camera detection ──────────────────────────────────────────────────────
def detect_cameras():
    found = []
    for i in range(6):
        test = cv2.VideoCapture(i)
        if test.isOpened():
            found.append(i)
            test.release()
    return found

available_cameras = detect_cameras()
if not available_cameras:
    print("\nERROR: No cameras detected.")
    input("Press Enter to exit...")
    sys.exit(1)

camera_index = 0  # index into available_cameras list
cam = cv2.VideoCapture(available_cameras[camera_index])
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

camera_labels = []
for i, idx in enumerate(available_cameras):
    if i == 0:
        camera_labels.append(f"Camera {idx} (ceiling/main)")
    elif i == 1:
        camera_labels.append(f"Camera {idx} (doc cam)")
    else:
        camera_labels.append(f"Camera {idx}")

print(f"\n  Found {len(available_cameras)} camera(s):")
for label in camera_labels:
    print(f"    • {label}")
if len(available_cameras) > 1:
    print("  TAB = toggle between cameras")
print("  Camera ready. Waiting for input...\n")

# ── Local pre-filter: skip near-duplicate board photos before paying Claude ──
def _image_focus_score(img_bgr):
    """Return a simple sharpness score. Higher usually means clearer text/board writing."""
    try:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _compare_board_images(path_a, path_b):
    """Return similarity metrics for two saved images.

    This is intentionally conservative. It only calls something a duplicate when
    both the full-image difference and edge difference are tiny. That way a small
    new equation or extra row on the board is less likely to be skipped.
    """
    try:
        a = cv2.imread(str(path_a))
        b = cv2.imread(str(path_b))
        if a is None or b is None:
            return None

        # Resize to make comparison fast and independent of camera resolution.
        size = (320, 180)
        ag = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), size)
        bg = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), size)

        # Smooth lighting/noise a bit, but keep real board changes visible.
        ag_blur = cv2.GaussianBlur(ag, (5, 5), 0)
        bg_blur = cv2.GaussianBlur(bg, (5, 5), 0)
        diff_ratio = float(cv2.mean(cv2.absdiff(ag_blur, bg_blur))[0] / 255.0)

        # Edge difference helps catch new writing even when lighting is similar.
        ae = cv2.Canny(ag, 60, 160)
        be = cv2.Canny(bg, 60, 160)
        edge_diff_ratio = float(cv2.mean(cv2.absdiff(ae, be))[0] / 255.0)

        old_focus = _image_focus_score(a)
        new_focus = _image_focus_score(b)
        return {
            "diff_ratio": diff_ratio,
            "edge_diff_ratio": edge_diff_ratio,
            "old_focus": old_focus,
            "new_focus": new_focus,
        }
    except Exception:
        return None


def _latest_active_camera_capture_path():
    """Find the most recent camera capture already accepted for Claude processing."""
    try:
        with captures_lock:
            for cap in reversed(captures):
                if cap.get("skip_in_final_outputs"):
                    continue
                if cap.get("type") == "screenshot":
                    continue
                path = cap.get("path", "")
                if path and pathlib.Path(path).exists():
                    return path
    except Exception:
        pass
    return None


def _should_skip_duplicate_photo(new_img_path):
    """Decide whether a new photo is a near-duplicate that should not be sent to Claude."""
    if not LOCAL_DUPLICATE_PREFILTER:
        return False, "local duplicate prefilter disabled"

    old_img_path = _latest_active_camera_capture_path()
    if not old_img_path:
        return False, "no earlier capture to compare"

    metrics = _compare_board_images(old_img_path, new_img_path)
    if not metrics:
        return False, "comparison unavailable"

    diff = metrics["diff_ratio"]
    edge_diff = metrics["edge_diff_ratio"]
    old_focus = metrics["old_focus"]
    new_focus = metrics["new_focus"]

    looks_duplicate = diff < DUPLICATE_DIFF_THRESHOLD and edge_diff < DUPLICATE_EDGE_DIFF_THRESHOLD
    much_sharper = old_focus > 0 and new_focus > old_focus * SHARPNESS_KEEP_MULTIPLIER

    if looks_duplicate and not much_sharper:
        reason = (
            f"near-duplicate of previous capture "
            f"(diff={diff:.4f}, edge_diff={edge_diff:.4f}, "
            f"sharpness old={old_focus:.0f}, new={new_focus:.0f})"
        )
        return True, reason

    if looks_duplicate and much_sharper:
        return False, (
            f"similar board, but new photo is much sharper "
            f"(old={old_focus:.0f}, new={new_focus:.0f}); keeping for Claude"
        )

    return False, f"not duplicate enough (diff={diff:.4f}, edge_diff={edge_diff:.4f})"


def _move_duplicate_skipped_image(img_path, ts, reason):
    """Move a skipped duplicate out of images/ and write a small reason file."""
    try:
        src = pathlib.Path(img_path)
        if not src.exists():
            return
        dst = DUPLICATE_SKIPPED_DIR / src.name
        if dst.exists():
            dst = DUPLICATE_SKIPPED_DIR / f"{src.stem}_{ts}{src.suffix}"
        src.replace(dst)
        info_path = DUPLICATE_SKIPPED_DIR / f"{dst.stem}_reason.txt"
        info_path.write_text(
            f"Skipped duplicate capture at {ts}\n"
            f"Original path: {src}\n"
            f"Saved backup: {dst}\n"
            f"Reason: {reason}\n",
            encoding="utf-8"
        )
    except Exception as e:
        print(f"  [!] Could not move duplicate skipped image: {e}")

# ── Snap photo instantly and add to queue ─────────────────────────────────
def snap_photo():
    global flash_frame, last_capture_time
    if note_input_active:
        return
    now_t = time.time()
    if now_t - last_capture_time < 3:
        return  # cooldown — ignore rapid/ghost presses
    last_capture_time = now_t

    ret, frame = cam.read()
    if not ret:
        print("  [!] Camera read failed")
        return

    ts = datetime.datetime.now().strftime("%H-%M-%S")
    img_path = IMAGES_DIR / f"capture_{ts}.jpg"

    # Adaptive brightness using CLAHE — works well in dark rooms
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(img_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Free local pre-filter: if this photo is basically the same as the last
    # accepted camera capture, do not pay Claude to process it. If it is much
    # sharper, keep it so Sonnet can do the smart delete/lookback normally.
    skip_duplicate, duplicate_reason = _should_skip_duplicate_photo(str(img_path))
    if skip_duplicate:
        global duplicate_skipped_count
        duplicate_skipped_count += 1
        _move_duplicate_skipped_image(str(img_path), ts, duplicate_reason)
        print(f"  [{ts}] Skipped duplicate capture #{duplicate_skipped_count} — {duplicate_reason}")
        return
    elif LOCAL_DUPLICATE_PREFILTER and duplicate_reason:
        print(f"  [{ts}] Local pre-filter: {duplicate_reason}")

    queue_pos = capture_queue.qsize() + 1
    print(f"  [{ts}] Photo #{queue_pos + len(captures)} taken — queued for Claude...")
    flash_frame = True
    capture_queue.put({"path": str(img_path), "timestamp": ts})

# ── Smart lookback repair/delete: lets newer clearer captures replace older bad ones ──
def _looks_unclear(notes):
    """Quick local signal used only as a backup after Claude's judgment."""
    n = (notes or "").lower()
    unclear_words = ["[unclear]", "unclear", "difficult to read", "partially visible", "cannot determine", "not clearly visible", "may be"]
    return sum(n.count(w) for w in unclear_words) >= 2


def _same_topic_hint(a, b):
    """Cheap topic-overlap check so we do not auto-hide unrelated captures."""
    import re
    stop = {"the","and","for","with","from","this","that","into","notes","image","capture","system","equation","equations"}
    def words(x):
        return {w for w in re.findall(r"[a-zA-Z]{4,}", (x or "").lower()) if w not in stop}
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) >= 3


def _move_replaced_image(capture_entry):
    """Move an old replaced image into backups/replaced_captures but keep the path valid."""
    try:
        old_img = pathlib.Path(capture_entry.get("path", ""))
        if old_img.exists() and old_img.parent != REPLACED_DIR:
            moved_img = REPLACED_DIR / old_img.name
            if moved_img.exists():
                moved_img = REPLACED_DIR / f"{old_img.stem}_{capture_entry.get('timestamp','')}{old_img.suffix}"
            old_img.replace(moved_img)
            capture_entry["original_path"] = str(old_img)
            capture_entry["path"] = str(moved_img)
    except Exception as move_err:
        print(f"  [!] Could not move replaced capture image: {move_err}")


def smart_repair_recent_captures():
    """Compare the newest capture against the last few active captures.

    Goal:
    - If Capture 1 is blurry/unclear and Capture 2 is the clearer version of the same board,
      hide Capture 1 from 01_photos.html, 02_ai_notes.html, and 03_summary.html.
    - The newer clearer capture stays visible in ai_notes.html with a small Smart Lookback note.
    - Old images are moved to backups/replaced_captures instead of being permanently destroyed.
    """
    if not SMART_LOOKBACK_REPAIR:
        return

    try:
        with captures_lock:
            if len(captures) < 2:
                return
            curr_index = len(captures) - 1
            curr = dict(captures[curr_index])

            # Look back over the last few non-deleted captures, not only the immediate previous one.
            candidate_indices = []
            for idx in range(curr_index - 1, -1, -1):
                if len(candidate_indices) >= 3:
                    break
                if not captures[idx].get("skip_in_final_outputs"):
                    candidate_indices.append(idx)

            candidates = [(idx, dict(captures[idx])) for idx in reversed(candidate_indices)]

        curr_path = curr.get("path", "")
        if not curr_path or not pathlib.Path(curr_path).exists() or not candidates:
            return

        print(f"  [{curr.get('timestamp','')}] Smart lookback: checking newest capture against last {len(candidates)} active capture(s)...")

        # Build multimodal content with labels so Claude knows which image is which.
        content = []
        candidate_summary_lines = []
        for idx, cap in candidates:
            path = cap.get("path", "")
            if not path or not pathlib.Path(path).exists():
                continue
            with open(path, "rb") as f:
                img64 = base64.standard_b64encode(f.read()).decode()
            content.append({"type": "text", "text": f"Previous candidate index {idx}, timestamp {cap.get('timestamp','')}:"})
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg" if str(path).lower().endswith((".jpg", ".jpeg")) else "image/png",
                    "data": img64
                }
            })
            candidate_summary_lines.append(
                f"INDEX {idx} | TIMESTAMP {cap.get('timestamp','')} | NOTES:\n{cap.get('notes','')[:2500]}"
            )

        with open(curr_path, "rb") as f:
            curr_b64 = base64.standard_b64encode(f.read()).decode()
        content.append({"type": "text", "text": f"Newest/current capture index {curr_index}, timestamp {curr.get('timestamp','')}:"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg" if str(curr_path).lower().endswith((".jpg", ".jpeg")) else "image/png",
                "data": curr_b64
            }
        })

        repair_prompt = rf"""You are the Smart Lookback quality-control editor for LectureLens.

You are comparing the newest/current capture with the previous candidate captures.

Main task:
- If an older capture is blurry, incomplete, duplicated, has many [unclear] parts, or misread math,
  and the newest capture shows the same board/work more clearly, mark that older capture for deletion from final output.
- "Deletion" means hidden from final HTML/PDF study notes; it will still be kept in backup.
- The newest clearer capture should remain visible and should contain the corrected notes.
- When the newest/current capture has blanks, ?, [unclear], or partial entries, use the previous candidate captures plus mathematical reasoning to complete them when possible.
- For row-reduction/back-substitution, infer missing values from equations and matrix structure when the inference is strong. Example: $4x_3 = 12$ implies $x_3 = 3$; $3x_2 = 12$ implies $x_2 = 4$; then $x_1 - x_2 - x_3 = 0$ implies $x_1 = 7$.
- Replace vague notes like "various arithmetic operations" with the actual inferred operation/result when it follows from the visible work.
- Only leave [unclear] when the math cannot be determined from any capture or context. Do not invent numbers.

Important decision rule:
- If the newer capture fully replaces the older capture and the older capture adds no unique useful information,
  set delete_previous_indices to include that older index.
- If the older capture has unique content not visible in the newer one, do NOT delete it.
- If unsure, keep the older capture.

LaTeX output rules:
- Use inline math only for short expressions like $x_1$.
- Use display math $$...$$ for systems, matrices, row operations, and important equations.
- Keep each multi-line LaTeX environment inside one complete $$...$$ block.
- For cases, aligned, array, matrix, pmatrix, and bmatrix, every row must end with double backslashes \\ except the last row.
- Never use a single backslash at the end of a row.
- Do not put blank lines inside LaTeX environments.

Return ONLY valid JSON in this exact shape:
{{
  "delete_previous_indices": [list_of_integer_indices_to_hide_from_final_outputs],
  "update_current": true_or_false,
  "updated_current_notes": "replacement markdown notes for the newest/current capture, or empty string",
  "reason": "short reason explaining what was replaced or why nothing was replaced"
}}

Previous candidate notes:
<<<PREVIOUS_CANDIDATES
{chr(10).join(candidate_summary_lines)}
PREVIOUS_CANDIDATES>>>

Newest/current capture notes:
<<<CURRENT_NOTES
{curr.get('notes','')[:5000]}
CURRENT_NOTES>>>"""
        content.append({"type": "text", "text": repair_prompt})

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=6000,
            messages=[{"role": "user", "content": content}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        import json
        data = json.loads(raw)
        requested_delete = data.get("delete_previous_indices", []) or []
        reason = data.get("reason", "newer capture is clearer and replaces older capture")

        # Safety: only allow indices from the candidate list.
        allowed = {idx for idx, _ in candidates}
        delete_indices = [int(i) for i in requested_delete if int(i) in allowed]

        # Backup local heuristic: if the immediate previous capture is obviously unclear
        # and the current capture is on the same topic and is much cleaner, hide the previous one.
        # This catches the exact case where Claude is too cautious.
        if SMART_DELETE_WORSE_CAPTURES and candidates:
            prev_idx, prev_cap = candidates[-1]
            if prev_idx not in delete_indices:
                if _looks_unclear(prev_cap.get("notes", "")) and not _looks_unclear(curr.get("notes", "")) and _same_topic_hint(prev_cap.get("notes", ""), curr.get("notes", "")):
                    delete_indices.append(prev_idx)
                    reason += " Local safety heuristic also detected the previous capture was unclear and replaced by a clearer same-topic capture."

        changed = False
        with captures_lock:
            if data.get("update_current") and data.get("updated_current_notes", "").strip():
                captures[curr_index]["notes"] = data["updated_current_notes"].strip()
                captures[curr_index]["revised_by_lookback"] = curr.get("timestamp", "")
                changed = True

            replaced_ts = captures[curr_index].setdefault("replaced_previous_timestamps", [])
            replaced_reasons = captures[curr_index].setdefault("replaced_previous_reasons", [])

            if SMART_DELETE_WORSE_CAPTURES:
                for idx in delete_indices:
                    if idx == curr_index:
                        continue
                    captures[idx]["deleted_by_lookback"] = curr.get("timestamp", "")
                    captures[idx]["deleted_reason"] = reason
                    captures[idx]["skip_in_final_outputs"] = True
                    if captures[idx].get("timestamp") not in replaced_ts:
                        replaced_ts.append(captures[idx].get("timestamp", ""))
                    replaced_reasons.append(reason)
                    _move_replaced_image(captures[idx])
                    changed = True

            curr_notes_after = captures[curr_index].get("notes", curr.get("notes", ""))

        if changed:
            print(f"  [{curr.get('timestamp','')}] ✓ Smart lookback cleaned final notes: {reason}")
            try:
                for idx in delete_indices:
                    if idx in allowed:
                        old_ts = ""
                        with captures_lock:
                            old_ts = captures[idx].get("timestamp", "")
                        prev_txt = BACKUP_DIR / f"notes_{str(idx+1).zfill(2)}_{old_ts}_REPLACED_BY_CAPTURE_{curr.get('timestamp','')}.txt"
                        with open(prev_txt, "w", encoding="utf-8") as nf:
                            nf.write(f"Capture #{idx+1} — {old_ts}\n")
                            nf.write(f"Status: hidden from final HTML because capture #{curr_index+1} at {curr.get('timestamp','')} is better.\n")
                            nf.write(f"Reason: {reason}\n")

                curr_txt = BACKUP_DIR / f"notes_{str(curr_index+1).zfill(2)}_{curr.get('timestamp','')}_LOOKBACK_CLEANED.txt"
                with open(curr_txt, "w", encoding="utf-8") as nf:
                    nf.write(f"Capture #{curr_index+1} — {curr.get('timestamp','')}\n")
                    nf.write("Smart Lookback checked this capture.\n")
                    if delete_indices:
                        nf.write(f"This capture replaced earlier capture index/indices: {delete_indices}.\n")
                    nf.write(f"Reason: {reason}\n")
                    nf.write("=" * 50 + "\n\n")
                    nf.write(curr_notes_after)
            except Exception as e:
                print(f"  [!] Could not save lookback cleanup backup: {e}")
        else:
            print(f"  [{curr.get('timestamp','')}] Smart lookback: kept earlier captures.")

    except Exception as e:
        # Never let the lookback step break normal capture processing.
        print(f"  [!] Smart lookback skipped: {e}")


# ── Camera capture prep: full image + zoomed crops for Claude ─────────────
def _prep_capture_for_claude(path):
    """Prepare camera captures the same way screenshot mode works.

    Camera photos usually work already, but this helps when the board/page/math is
    small inside the frame. Claude receives the full capture for context plus
    zoomed crops for exact signs, coefficients, and constants.
    """
    import io as _io
    from PIL import Image as _PILImage
    import numpy as _np

    img = _PILImage.open(path).convert("RGB")
    W, H = img.size

    def _encode_png(im, max_long_edge=None, upscale_to=None):
        im = im.convert("RGB")
        w, h = im.size
        if max_long_edge and max(w, h) > max_long_edge:
            scale = max_long_edge / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), _PILImage.LANCZOS)
            w, h = im.size
        if upscale_to and max(w, h) < upscale_to:
            scale = upscale_to / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), _PILImage.LANCZOS)
        buf = _io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode()

    full_b64 = _encode_png(img, max_long_edge=1800)
    crops = []

    def _add_crop(label, box, upscale_to=1600):
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, min(W - 1, x0)); y0 = max(0, min(H - 1, y0))
        x1 = max(x0 + 1, min(W, x1)); y1 = max(y0 + 1, min(H, y1))
        crop = img.crop((x0, y0, x1, y1))
        max_edge = 2400 if max(crop.size) > 1900 else None
        crops.append({"label": label, "data": _encode_png(crop, max_long_edge=max_edge, upscale_to=upscale_to)})

    # General crops that help with boards, monitors, and paper/worksheet shots.
    _add_crop("CENTER MATH/BOARD AREA - high priority for exact equations", (int(W*0.10), int(H*0.08), int(W*0.90), int(H*0.92)), upscale_to=1900)
    _add_crop("TOP HALF OF CAPTURE - likely printed problem or board heading", (0, 0, W, int(H*0.55)), upscale_to=1800)
    _add_crop("TOP CENTER CAPTURE - useful for systems/matrices near top", (int(W*0.25), 0, int(W*0.82), int(H*0.55)), upscale_to=2200)
    _add_crop("LEFT/CENTER PRINTED PROBLEM AREA - useful for screens/paper", (0, 0, int(W*0.85), int(H*0.65)), upscale_to=2000)

    # Detect bright board/page/screen regions and send that as a ground-truth crop.
    try:
        gray = _np.array(img.convert("L"))
        light = gray > 190
        rows = _np.where(light.mean(axis=1) > 0.15)[0]
        cols = _np.where(light.mean(axis=0) > 0.08)[0]
        if len(rows) > 20 and len(cols) > 20:
            y0, y1 = int(rows[0]), int(rows[-1])
            x0, x1 = int(cols[0]), int(cols[-1])
            area = (x1-x0) * (y1-y0)
            if area > W * H * 0.01:
                _add_crop("DETECTED LIGHT BOARD/PAPER/SCREEN CONTENT - ground truth", (x0-45, y0-35, x1+45, y1+35), upscale_to=2100)
    except Exception as e:
        print(f"  [prep] capture light-content crop skipped: {e}")

    # Detect dense dark writing/text over the whole capture.
    try:
        arr = _np.array(img.convert("L"))
        dark = arr < 120
        row_density = dark.mean(axis=1)
        col_density = dark.mean(axis=0)
        text_rows = _np.where(row_density > 0.002)[0]
        text_cols = _np.where(col_density > 0.001)[0]
        if len(text_rows) > 10 and len(text_cols) > 10:
            y0, y1 = int(text_rows[0]), int(text_rows[-1])
            x0, x1 = int(text_cols[0]), int(text_cols[-1])
            if (x1-x0) > 30 and (y1-y0) > 20:
                _add_crop("DENSE TEXT/WRITING CROP - use for exact signs/numbers", (x0-60, y0-50, x1+60, y1+50), upscale_to=2200)
    except Exception as e:
        print(f"  [prep] capture dense-text crop skipped: {e}")

    return full_b64, crops

# ── Background worker — sends queued photos to Claude one by one ───────────
def process_queue():
    while True:
        try:
            item = capture_queue.get(timeout=1)
        except queue.Empty:
            if stopped:
                break
            continue

        img_path = item["path"]
        ts = item["timestamp"]
        capture_num = len(captures) + 1
        print(f"  [{ts}] Processing capture #{capture_num} with Claude...")

        try:
            # Camera capture mode now uses the same idea as screenshot mode:
            # send the full image for context PLUS high-resolution zoomed crops.
            full_b64, crops = _prep_capture_for_claude(img_path)
            image_blocks = [{
                "type": "text",
                "text": "IMAGE 1: FULL CAMERA CAPTURE. Use for context; focused crops below have priority for exact math."
            }, {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": full_b64}
            }]
            for i, crop in enumerate(crops, start=2):
                image_blocks.append({
                    "type": "text",
                    "text": f"IMAGE {i}: {crop.get('label','focused capture crop')}. Use this crop for exact signs, coefficients, constants, and row entries."
                })
                image_blocks.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": crop["data"]}
                })
            if crops:
                print(f"  [{ts}] Sending full camera capture + {len(crops)} high-resolution focused crop(s).")
            else:
                print(f"  [{ts}] Sending full camera capture only.")

            READ_PROMPT = r"""You are a precise mathematical reader for math screenshots and board photos.

Your job in this pass is to READ and EXTRACT, not solve yet. You may be given a full camera capture plus zoomed crops; use the focused crops for exact signs, coefficients, constants, and matrix entries.

Be especially careful with signs, coefficients, and right-hand sides.

Extract in this order:
1. PRINTED / TYPED problem statement first.
   - Treat this as the main question.
   - Transcribe every equation exactly, including plus/minus signs and constants.
   - If the page asks "Solve the system," include that instruction.
2. HANDWRITTEN work second.
   - Treat this as scratch work unless it clearly matches the printed problem.
   - Transcribe every matrix entry, row by row, and every visible row-operation label.
3. Diagrams, vectors, annotations, or other labels.

Priority rule:
- If printed text and handwritten scratch work conflict, mark the conflict and keep the printed text as the ground truth.

Rules:
- If a character is genuinely unreadable, write [unclear] for THAT entry only.
- Do not guess unreadable characters.
- Do not solve in this pass.
- Use plain LaTeX for math. Matrices as augmented \left[\begin{array}{...|c}...\end{array}\right].
- Separate sections using exactly: PRINTED:, HANDWRITTEN:, OTHER:."""

            SOLVE_PROMPT = r"""You are a careful math tutor. Use the extracted content above to answer the actual problem, like a normal Claude chat.

CRITICAL RULES:
- The PRINTED section is the ground truth for the problem.
- Use HANDWRITTEN/dark-canvas work only as supporting scratch work. Never let it override the white printed problem.
- If handwritten work conflicts with the printed problem, trust the printed problem and mention the mismatch briefly.
- Do not rename variables. If the problem uses x and y, keep x and y.
- Do not change any printed numbers or signs.
- If the printed problem asks to solve, you MUST solve it even if the handwritten solution is missing.
- Build any augmented matrix mechanically from the printed system: row i = [coefficients of variables in equation i | RHS].
- Show enough algebra that the answer can be checked.
- Verify the final answer by substituting into the original printed equations.

If something is [unclear]:
- Use the clearest printed text first.
- Only infer an unclear value when arithmetic or the rest of the printed problem makes it certain.
- State any inference in the "Reading Notes" section.

FORMATTING:
- Inline math for short expressions: $x + y = -1$
- Display math $$...$$ for systems, matrices, row operations, and checks.
- Multi-line environments use \\ between rows, never a single backslash.
- No blank lines inside LaTeX environments.

OUTPUT (clean markdown):
## Topic
## Given Problem
(write the printed problem exactly as extracted)
## Augmented Matrix
(if useful, build it from the printed system)
## Work
(solve step by step)
## Check
(substitute the answer into the original printed equations)
## Final Answer
(state the answer clearly)
## Reading Notes
(only include conflicts, unclear entries, or inferred values; otherwise write "No reading issues.")"""

            # ── PASS 1: Read — extract the math exactly as written ──────────
            print(f"  [{ts}] Pass 1: Reading board...")
            read_resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=3000,
                messages=[{
                    "role": "user",
                    "content": image_blocks + [{"type": "text", "text": READ_PROMPT}]
                }]
            )
            extracted = read_resp.content[0].text.strip()
            print(f"  [{ts}] Pass 1 done. Pass 2: Solving with full capture + crops...")

            # ── PASS 2: Solve — reason from the clean extraction ────────────
            solve_resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=6000,
                messages=[
                    {"role": "user", "content": image_blocks + [{"type": "text", "text": READ_PROMPT}]},
                    {"role": "assistant", "content": extracted},
                    {"role": "user", "content": image_blocks + [{"type": "text", "text": SOLVE_PROMPT}]}
                ]
            )
            notes = solve_resp.content[0].text

            with captures_lock:
                captures.append({
                    "path": img_path,
                    "notes": notes,
                    "timestamp": ts
                })

            # Smart lookback: if this capture makes the previous capture clearer,
            # revise/overwrite the older notes before the HTML files are generated.
            smart_repair_recent_captures()

            # ── Save notes to txt immediately so nothing is lost if laptop closes ──
            try:
                note_num = len(captures)
                with captures_lock:
                    notes_to_save = captures[-1].get("notes", notes)
                txt_path = BACKUP_DIR / f"notes_{str(note_num).zfill(2)}_{ts}.txt"
                with open(txt_path, "w", encoding="utf-8") as nf:
                    nf.write(f"Capture #{note_num} — {ts}\n")
                    nf.write("=" * 50 + "\n\n")
                    nf.write(notes_to_save)
                print(f"  [{ts}] ✓ Notes done + saved to txt — {len(captures)} complete, {capture_queue.qsize()} in queue")
            except Exception as e:
                print(f"  [{ts}] ✓ Notes done — {len(captures)} complete, {capture_queue.qsize()} in queue")
                print(f"  [{ts}] [!] Could not save txt: {e}")

        except Exception as e:
            print(f"  [{ts}] ERROR calling Claude: {e}")
            traceback.print_exc()
        finally:
            capture_queue.task_done()

# ── Instructor note dialog (Tkinter — blocks while open) ──────────────────
def show_note_dialog():
    """Open a Tkinter window for the instructor to type a note.

    Returns dict {note: str, ask_claude: bool} or None if cancelled.
    Runs on the main thread (Tkinter is not thread-safe).
    """
    
    result = {"note": None, "ask_claude": False}

    root = tk.Tk()
    root.title("LectureLens — Instructor Note")
    root.geometry("560x320")
    root.attributes("-topmost", True)  # float above the camera preview window
    root.configure(bg="#f5f5f5")

    # Header
    header = tk.Label(
        root,
        text="Add a note for this lecture",
        font=("Segoe UI", 13, "bold"),
        bg="#1e5aa0", fg="white", pady=10
    )
    header.pack(fill="x")

    hint = tk.Label(
        root,
        text="e.g. \"student asked: why does flux depend on angle?\"",
        font=("Segoe UI", 9), fg="#666", bg="#f5f5f5", pady=4
    )
    hint.pack()

    # Text box
    text_frame = tk.Frame(root, bg="#f5f5f5", padx=15, pady=5)
    text_frame.pack(fill="both", expand=True)
    text_box = tk.Text(text_frame, height=6, font=("Segoe UI", 11), wrap="word",
                      relief="solid", borderwidth=1)
    text_box.pack(fill="both", expand=True)
    text_box.focus_set()

    # Ask-Claude checkbox
    ask_var = tk.BooleanVar(value=True)
    check_frame = tk.Frame(root, bg="#f5f5f5", padx=15)
    check_frame.pack(fill="x")
    check = tk.Checkbutton(
        check_frame,
        text="Ask Claude to respond (e.g. answer a student question)",
        variable=ask_var, bg="#f5f5f5", font=("Segoe UI", 10)
    )
    check.pack(anchor="w")

    # Buttons
    btn_frame = tk.Frame(root, bg="#f5f5f5", padx=15, pady=10)
    btn_frame.pack(fill="x")

    def submit(event=None):
        text = text_box.get("1.0", "end").strip()
        if text:
            result["note"] = text
            result["ask_claude"] = ask_var.get()
        root.destroy()

    def cancel(event=None):
        result["note"] = None
        root.destroy()

    save_btn = tk.Button(
        btn_frame, text="Save (Ctrl+Enter)", command=submit,
        bg="#1e5aa0", fg="white", font=("Segoe UI", 10, "bold"),
        padx=15, pady=5, relief="flat", cursor="hand2"
    )
    save_btn.pack(side="right", padx=5)

    cancel_btn = tk.Button(
        btn_frame, text="Cancel (Esc)", command=cancel,
        bg="#ccc", fg="#333", font=("Segoe UI", 10),
        padx=15, pady=5, relief="flat", cursor="hand2"
    )
    cancel_btn.pack(side="right")

    # Keyboard shortcuts
    root.bind("<Control-Return>", submit)
    root.bind("<Escape>", cancel)

    # Center the window
    root.update_idletasks()
    w, h = root.winfo_width(), root.winfo_height()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    root.mainloop()
    return result if result["note"] else None

# ── Note opening trigger (runs on main thread via flag) ───────────────────
note_dialog_request = threading.Event()  # set by N-key, consumed by main loop

def add_unicode_fonts(pdf):
    import urllib.request, tempfile, zipfile
    font_dir = BASE_DIR / "fonts"
    font_dir.mkdir(exist_ok=True)
    regular = font_dir / "DejaVuSans.ttf"
    bold    = font_dir / "DejaVuSans-Bold.ttf"
    # Re-download if EITHER file is missing (previous version only checked regular)
    if not regular.exists() or not bold.exists():
        print("  Downloading Unicode font (first run only)...")
        url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
        tmp = tempfile.mktemp(suffix=".zip")
        urllib.request.urlretrieve(url, tmp)
        with zipfile.ZipFile(tmp) as z:
            for name in z.namelist():
                if "DejaVuSans.ttf" in name and "Bold" not in name and "Oblique" not in name:
                    with open(regular, "wb") as f: f.write(z.read(name))
                if "DejaVuSans-Bold.ttf" in name:
                    with open(bold, "wb") as f: f.write(z.read(name))
        os.remove(tmp)
    pdf.add_font("DejaVu", "",  str(regular))
    pdf.add_font("DejaVu", "B", str(bold))

# ── Helper: render markdown-ish notes to PDF, hardened ────────────────────
def render_notes_to_pdf(pdf, notes, content_width=None):
    """Render markdown-ish notes to PDF.

    Key fixes vs earlier versions:
      - Auto-detect content width from page margins if not specified
      - Check longer header prefixes FIRST (#### before ### before ## before #)
      - Use new_x="LMARGIN" so cursor returns to left margin after each line
      - Use align="L" so text isn't justified across full width (no weird gaps)
      - Handle --- (horizontal rule) and > (blockquote) gracefully
      - try/except per line so one bad line doesn't kill the PDF
    """
    if content_width is None:
        # A4 page width (210mm) minus left and right margins
        content_width = pdf.w - pdf.l_margin - pdf.r_margin
    if not notes or not notes.strip():
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(150, 150, 150)
        try:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w=content_width, h=6, text="[No notes available]",
                           new_x="LMARGIN", new_y="NEXT", align="L")
        except Exception:
            pass
        pdf.set_text_color(30, 30, 30)
        return

    for line in notes.split('\n'):
        try:
            # Always start each line at the left margin
            pdf.set_x(pdf.l_margin)

            # Strip inline bold/italic markers
            def clean_inline(s):
                return s.replace('**', '').replace('__', '').replace('*', '').replace('_', '')

            # ORDER MATTERS: check longer prefixes first
            if line.startswith('#### '):
                pdf.set_font("DejaVu", "B", 11)
                pdf.set_text_color(60, 60, 60)
                pdf.multi_cell(w=content_width, h=6, text=clean_inline(line[5:]),
                               new_x="LMARGIN", new_y="NEXT", align="L")
                pdf.set_text_color(30, 30, 30)
            elif line.startswith('### '):
                pdf.set_font("DejaVu", "B", 12)
                pdf.set_text_color(50, 50, 50)
                pdf.multi_cell(w=content_width, h=6, text=clean_inline(line[4:]),
                               new_x="LMARGIN", new_y="NEXT", align="L")
                pdf.set_text_color(30, 30, 30)
            elif line.startswith('## '):
                pdf.set_font("DejaVu", "B", 13)
                pdf.set_text_color(30, 90, 160)
                pdf.multi_cell(w=content_width, h=7, text=clean_inline(line[3:]),
                               new_x="LMARGIN", new_y="NEXT", align="L")
                pdf.set_text_color(30, 30, 30)
            elif line.startswith('# '):
                pdf.set_font("DejaVu", "B", 15)
                pdf.set_text_color(10, 60, 130)
                pdf.multi_cell(w=content_width, h=8, text=clean_inline(line[2:]),
                               new_x="LMARGIN", new_y="NEXT", align="L")
                pdf.set_text_color(30, 30, 30)
            elif line.startswith('- ') or line.startswith('* '):
                pdf.set_font("DejaVu", "", 10)
                pdf.multi_cell(w=content_width, h=6, text="  • " + clean_inline(line[2:]),
                               new_x="LMARGIN", new_y="NEXT", align="L")
            elif line.startswith('> '):
                # Blockquote — indent slightly, italic gray
                pdf.set_font("DejaVu", "", 10)
                pdf.set_text_color(100, 100, 100)
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(w=content_width - 5, h=6, text=clean_inline(line[2:]),
                               new_x="LMARGIN", new_y="NEXT", align="L")
                pdf.set_text_color(30, 30, 30)
            elif line.strip().startswith('---') or line.strip().startswith('***'):
                # Horizontal rule — draw a light gray line
                pdf.ln(2)
                y = pdf.get_y()
                pdf.set_draw_color(200, 200, 200)
                pdf.line(pdf.l_margin, y, pdf.l_margin + content_width, y)
                pdf.ln(3)
            elif line.strip() == '':
                pdf.ln(3)
            else:
                pdf.set_font("DejaVu", "", 10)
                pdf.multi_cell(w=content_width, h=6, text=clean_inline(line),
                               new_x="LMARGIN", new_y="NEXT", align="L")
        except Exception as e:
            print(f"  [!] Skipped a line ({e}): {line[:60]!r}")
            continue


# ── Parse and render LaTeX matrices as visual grids ──────────────────────
def parse_latex_matrix(latex_str):
    """Detect a pmatrix/bmatrix/vmatrix/matrix and return (bracket_type, rows)
    where rows is a list of lists of cell strings. Returns None if not a matrix."""
    import re
    s = latex_str.strip().strip('$').strip()
    # Match \begin{XXmatrix} ... \end{XXmatrix}
    m = re.search(r'\\begin\{(p|b|v|V|B)?matrix\}(.*?)\\end\{\1?matrix\}', s, re.DOTALL)
    if not m:
        return None
    bracket = m.group(1) or ''  # '' means \matrix (no brackets), 'p' is parens, 'b' is brackets
    body = m.group(2).strip()
    # Split into rows on \\
    rows = []
    for raw_row in re.split(r'\\\\', body):
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        # Split on & for cells
        cells = [c.strip() for c in raw_row.split('&')]
        rows.append(cells)
    if not rows:
        return None
    return (bracket, rows)

def render_matrix_to_pdf(pdf, bracket, rows):
    """Render a parsed matrix as a visual grid in the PDF using fpdf cells.
    bracket: 'p' (parens), 'b' (square), 'v'/'V' (bars), '' (none)."""
    if not rows:
        return
    # Compute column count and cell width
    ncols = max(len(r) for r in rows)
    nrows = len(rows)
    # Each cell ~10mm wide, ~7mm tall — but cap matrix width to 100mm
    cell_w = min(15, 100 / max(ncols, 1))
    cell_h = 7
    matrix_w = cell_w * ncols
    bracket_w = 3  # width of the bracket marks

    # Start position
    start_x = pdf.get_x() + 10
    start_y = pdf.get_y() + 2
    total_h = cell_h * nrows

    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(30, 30, 30)
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.4)

    # Draw left bracket
    lx = start_x
    ly = start_y
    if bracket == 'p':
        # Parenthesis — draw a curve (approximate with two short lines)
        pdf.line(lx + bracket_w, ly, lx, ly + total_h * 0.2)
        pdf.line(lx, ly + total_h * 0.2, lx, ly + total_h * 0.8)
        pdf.line(lx, ly + total_h * 0.8, lx + bracket_w, ly + total_h)
    elif bracket == 'b':
        # Square bracket
        pdf.line(lx, ly, lx + bracket_w, ly)
        pdf.line(lx, ly, lx, ly + total_h)
        pdf.line(lx, ly + total_h, lx + bracket_w, ly + total_h)
    elif bracket in ('v', 'V'):
        # Vertical bars
        pdf.line(lx + bracket_w / 2, ly, lx + bracket_w / 2, ly + total_h)
        if bracket == 'V':
            pdf.line(lx + bracket_w / 2 + 1.5, ly, lx + bracket_w / 2 + 1.5, ly + total_h)

    # Draw cells (text only, no borders)
    for r_idx, row in enumerate(rows):
        for c_idx in range(ncols):
            cell_text = row[c_idx] if c_idx < len(row) else ''
            # Clean cell text — strip $ signs that might wrap individual cells
            cell_text = cell_text.replace('$', '').strip()
            x = start_x + bracket_w + 1 + c_idx * cell_w
            y = start_y + r_idx * cell_h
            pdf.set_xy(x, y)
            pdf.cell(cell_w, cell_h, cell_text, align='C')

    # Draw right bracket
    rx = start_x + bracket_w + 1 + matrix_w
    if bracket == 'p':
        pdf.line(rx, ly, rx + bracket_w, ly + total_h * 0.2)
        pdf.line(rx + bracket_w, ly + total_h * 0.2, rx + bracket_w, ly + total_h * 0.8)
        pdf.line(rx + bracket_w, ly + total_h * 0.8, rx, ly + total_h)
    elif bracket == 'b':
        pdf.line(rx + bracket_w, ly, rx, ly)
        pdf.line(rx + bracket_w, ly, rx + bracket_w, ly + total_h)
        pdf.line(rx + bracket_w, ly + total_h, rx, ly + total_h)
    elif bracket in ('v', 'V'):
        pdf.line(rx + bracket_w / 2, ly, rx + bracket_w / 2, ly + total_h)
        if bracket == 'V':
            pdf.line(rx + bracket_w / 2 - 1.5, ly, rx + bracket_w / 2 - 1.5, ly + total_h)

    # Advance cursor past the matrix
    pdf.set_xy(pdf.l_margin, start_y + total_h + 4)


# ── Render a LaTeX expression to a PNG image for embedding in PDF ──────────
def render_latex_to_image(latex_str, font_size=12):
    """Render a LaTeX string to a PNG bytes object.
    Returns None if the expression uses environments matplotlib cant handle."""
    try:
        latex_str = latex_str.strip()
        # Strip outer $$ or $ markers for matplotlib
        inner = latex_str.strip('$').strip()

        # Skip environments matplotlib cant render — fall back to text box
        skip_keywords = ['\\begin', '\\end', '\\pmatrix', '\\bmatrix',
                         '\\vmatrix', '\\matrix', '\\array', '\\cases']
        if any(kw in inner for kw in skip_keywords):
            return None

        # Wrap back in $ for matplotlib
        display = f'${inner}$'

        fig = plt.figure(figsize=(5, 0.6))
        fig.patch.set_facecolor('white')
        text_obj = fig.text(0.05, 0.5, display,
                 fontsize=font_size,
                 verticalalignment='center',
                 usetex=False)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                    facecolor='white', edgecolor='none',
                    pad_inches=0.05)
        plt.close(fig)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None

def embed_latex_in_pdf(pdf, latex_str):
    """Render LaTeX and embed as image in PDF.
    Falls back to a styled text box if rendering fails."""
    # Try matrix rendering first (matplotlib can't do pmatrix/bmatrix)
    matrix = parse_latex_matrix(latex_str)
    if matrix is not None:
        try:
            bracket, rows = matrix
            render_matrix_to_pdf(pdf, bracket, rows)
            return True
        except Exception as e:
            print(f"  [!] Matrix render failed, falling back: {e}")

    img_bytes = render_latex_to_image(latex_str)
    if img_bytes:
        try:
            import tempfile, os
            tmp = tempfile.mktemp(suffix='.png')
            with open(tmp, 'wb') as f:
                f.write(img_bytes)
            pdf.image(tmp, x=pdf.get_x() + 5, w=150)
            pdf.ln(2)
            os.remove(tmp)
            return True
        except Exception:
            pass

    # ── Fallback: styled text box so equation is still readable ──
    try:
        # Clean up display — strip $$ markers for readability
        display_text = latex_str.strip().strip('$').strip()
        pdf.set_fill_color(240, 245, 255)   # light blue background
        pdf.set_draw_color(100, 140, 200)   # blue border
        pdf.set_line_width(0.5)
        x = pdf.get_x() + 5
        y = pdf.get_y()
        pdf.rect(x, y, 170, 10, 'FD')      # filled box with border
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(60, 60, 60)
        pdf.set_xy(x + 3, y + 2)
        pdf.cell(164, 6, display_text[:120], ln=True)  # truncate if too long
        pdf.set_text_color(30, 30, 30)
        pdf.set_line_width(0.2)
        pdf.set_draw_color(0, 0, 0)
        pdf.ln(3)
        return True  # fallback succeeded
    except Exception:
        # Last resort — plain text
        try:
            pdf.set_font("DejaVu", "", 10)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(0, 6, f"[Equation: {latex_str.strip()[:100]}]")
            pdf.set_text_color(30, 30, 30)
        except Exception:
            pass
        return False

def safe_multi_cell(pdf, text, height=6, font="DejaVu", style="", size=10, color=(30,30,30)):
    """Safely render text, resetting cursor to left margin first."""
    try:
        pdf.set_x(pdf.l_margin)
        pdf.set_font(font, style, size)
        pdf.set_text_color(*color)
        # Remove any characters that could break rendering
        clean = text.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, height, clean)
    except Exception:
        try:
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, height, text[:80], ln=True)
        except Exception:
            pdf.ln(height)

def render_notes_with_latex(pdf, notes):
    """Render notes to PDF, detecting and rendering LaTeX equations as images."""
    import re
    lines = notes.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        # Always reset to left margin before rendering anything
        pdf.set_x(pdf.l_margin)

        # Detect raw matrix environments WITHOUT $$ wrapping (Claude often writes these)
        if '\\begin{' in line and 'matrix}' in line:
            # Collect lines until \end{...matrix} is found
            matrix_block = line
            while '\\end{' not in matrix_block or 'matrix}' not in matrix_block.split('\\end{')[-1]:
                i += 1
                if i >= len(lines):
                    break
                matrix_block += ' ' + lines[i]
            # Extract just the matrix portion
            m = re.search(r'(\\begin\{[pbvVB]?matrix\}.*?\\end\{[pbvVB]?matrix\})',
                          matrix_block, re.DOTALL)
            if m:
                # Render any text BEFORE the matrix
                before = matrix_block[:m.start()].strip()
                if before:
                    safe_multi_cell(pdf, before.replace('**','').replace('*',''))
                # Render the matrix itself
                pdf.set_x(pdf.l_margin)
                embed_latex_in_pdf(pdf, f'$${m.group(1)}$$')
                pdf.set_x(pdf.l_margin)
                # Render any text AFTER the matrix on the same logical line
                after = matrix_block[m.end():].strip()
                if after:
                    safe_multi_cell(pdf, after.replace('**','').replace('*',''))
            else:
                # Couldn't parse — fall through to plain text
                safe_multi_cell(pdf, matrix_block.replace('**','').replace('*',''))
            i += 1
            continue

        # Detect display math blocks $$ ... $$
        if line.strip().startswith('$$'):
            latex_block = line.strip()
            while not latex_block.endswith('$$') or latex_block == '$$':
                i += 1
                if i >= len(lines):
                    break
                latex_block += ' ' + lines[i].strip()
            inner = latex_block.strip('$').strip()
            pdf.set_x(pdf.l_margin)
            embed_latex_in_pdf(pdf, f'$${inner}$$')
            pdf.set_x(pdf.l_margin)
            i += 1
            continue

        # Detect inline math $...$
        if '$' in line and line.count('$') >= 2:
            parts = re.split(r'(\$[^$]+\$)', line)
            for part in parts:
                pdf.set_x(pdf.l_margin)
                if part.startswith('$') and part.endswith('$') and len(part) > 2:
                    embed_latex_in_pdf(pdf, part)
                else:
                    if part.strip():
                        safe_multi_cell(pdf, part.replace('**','').replace('*',''))
            pdf.set_x(pdf.l_margin)
            i += 1
            continue

        # Normal markdown rendering
        pdf.set_x(pdf.l_margin)
        if line.startswith('## '):
            safe_multi_cell(pdf, line[3:], height=7, style="B", size=13, color=(30,90,160))
        elif line.startswith('# '):
            safe_multi_cell(pdf, line[2:], height=8, style="B", size=15, color=(10,60,130))
        elif line.startswith('### '):
            safe_multi_cell(pdf, line[4:], height=6, style="B", size=11, color=(50,50,50))
        elif line.startswith('- ') or line.startswith('* '):
            text = line[2:].strip()
            if text:  # skip empty bullets
                safe_multi_cell(pdf, "  - " + text)
        elif line.strip() == '':
            pdf.ln(3)
        else:
            clean = line.replace('**', '').replace('*', '')
            safe_multi_cell(pdf, clean)
        pdf.set_x(pdf.l_margin)
        i += 1

# ── PDF generation ─────────────────────────────────────────────────────────
def generate_html():
    """Generate 3 HTML files with MathJax for perfect LaTeX rendering."""
    import base64 as b64mod

    if not captures and not instructor_notes:
        print("\n  No captures to save.")
        return

    total = len(captures)
    note_count = len(instructor_notes)
    print(f"\n{'='*52}")
    print(f"  Generating HTML files for {total} captures + {note_count} notes...")
    if duplicate_skipped_count:
        print(f"  Local pre-filter skipped {duplicate_skipped_count} near-duplicate capture(s).")
    print(f"{'='*52}")

    # MathJax must be configured BEFORE loading the script.
    # Important: MathJax v3 does not always enable single-dollar inline math by default.
    # Without this config, expressions like $R_3 \leftarrow R_3 + R_1$ can show as raw text.
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
img.cimg{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:8px 0 16px}
.ans-lbl{color:#2d6a4f;font-weight:bold;margin-top:12px;font-family:Arial,sans-serif;font-size:13px}
.q-txt{font-style:italic;color:#555;margin:8px 0}
ul,ol{padding-left:1.5em}li{margin-bottom:4px}
p{line-height:1.7;margin:8px 0}
.footer{margin-top:40px;padding-top:12px;border-top:1px solid #ddd;font-size:11px;color:#aaa;font-family:Arial,sans-serif;text-align:center}
.toc{background:#f0f5ff;border:1px solid #c0d0f0;border-radius:4px;padding:16px 20px;margin:16px 0}
.toc a{color:#1e5aa0;text-decoration:none;display:block;margin:4px 0;font-family:Arial,sans-serif;font-size:13px}
.math-block{margin:12px 0; overflow-x:auto; text-align:center}
</style>"""

    def md_to_html(text):
        r"""Convert simple markdown to HTML while preserving LaTeX for MathJax.

        Important: display math blocks such as $$...$$, \[...\], and raw
        \begin{...}...\end{...} environments must NOT be wrapped line-by-line
        in <p> tags. MathJax needs the whole block kept together.
        """
        import re as _re
        import html as _html

        lines = text.split("\n")
        out = []
        in_ul = False
        in_math = False
        math_end = None
        math_lines = []

        # Environments that MathJax can render if preserved as one block.
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
            """Escape normal HTML but preserve inline $...$ math as raw text."""
            parts = _re.split(r'(\$[^$]+\$)', s)
            fixed = []
            for part in parts:
                if part.startswith('$') and part.endswith('$') and len(part) > 2:
                    fixed.append(part)
                else:
                    esc = _html.escape(part)
                    esc = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", esc)
                    fixed.append(esc)
            return "".join(fixed)

        def begin_math_block(line):
            stripped = line.strip()

            # $$ display math
            if stripped.startswith("$$"):
                if stripped.endswith("$$") and stripped != "$$":
                    return ("single", "$$")
                return ("start", "$$")

            # \[ display math
            if stripped.startswith(r"\["):
                if stripped.endswith(r"\]") and stripped != r"\[":
                    return ("single", r"\]")
                return ("start", r"\]")

            # Raw LaTeX environments without surrounding $$
            for env in latex_envs:
                if rf"\begin{{{env}}}" in line:
                    if rf"\end{{{env}}}" in line:
                        return ("single_env", env)
                    return ("start_env", env)

            return None

        def emit_math_block(block_lines, wrap_raw_env=True):
            block = "\n".join(block_lines).strip()
            # If Claude gives raw \begin{cases} without $$, wrap it for MathJax.
            if wrap_raw_env and block.startswith(r"\begin"):
                block = "$$\n" + block + "\n$$"
            out.append('<div class="math-block">\n' + block + '\n</div>')

        for line in lines:
            stripped = line.strip()

            # Continue a math block until its proper ending is reached.
            if in_math:
                math_lines.append(line)
                if math_end == "$$" and stripped.endswith("$$"):
                    emit_math_block(math_lines, wrap_raw_env=False)
                    in_math = False
                    math_lines = []
                    math_end = None
                elif math_end == r"\]" and stripped.endswith(r"\]"):
                    emit_math_block(math_lines, wrap_raw_env=False)
                    in_math = False
                    math_lines = []
                    math_end = None
                elif math_end and math_end not in ("$$", r"\]") and rf"\end{{{math_end}}}" in line:
                    emit_math_block(math_lines, wrap_raw_env=True)
                    in_math = False
                    math_lines = []
                    math_end = None
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

            # ORDER MATTERS: check ### before ## before #.
            if line.startswith("### "):
                close_ul()
                out.append(f"<h3>{escape_keep_inline_math(line[4:])}</h3>")
            elif line.startswith("## "):
                close_ul()
                out.append(f"<h2>{escape_keep_inline_math(line[3:])}</h2>")
            elif line.startswith("# "):
                close_ul()
                out.append(f"<h2>{escape_keep_inline_math(line[2:])}</h2>")
            elif line.startswith("- ") or line.startswith("* "):
                txt = escape_keep_inline_math(line[2:].strip())
                if txt:
                    if not in_ul:
                        out.append("<ul>")
                        in_ul = True
                    out.append(f"<li>{txt}</li>")
            else:
                close_ul()
                out.append(f"<p>{escape_keep_inline_math(line)}</p>")

        close_ul()

        # If the model forgot to close a math block, still keep it together.
        if in_math and math_lines:
            emit_math_block(math_lines, wrap_raw_env=True)

        return "\n".join(out)

    def img_b64_tag(path, style=""):
        try:
            p = pathlib.Path(path)
            if not p.exists(): return "<p style=\"color:#999\">[Image not available]</p>"
            with open(p, "rb") as f: data = b64mod.standard_b64encode(f.read()).decode()
            ext = p.suffix.lower().replace(".", "")
            mime = "jpeg" if ext in ("jpg","jpeg") else ext
            return f"<img class=\"cimg\" src=\"data:image/{mime};base64,{data}\" {style}>"
        except: return ""

    # Build chronological list. Captures replaced by Smart Lookback are hidden
    # from the final outputs so the better later capture appears instead.
    all_entries = []
    active_captures = [c for c in captures if not c.get("skip_in_final_outputs")]
    for c in active_captures: all_entries.append(("capture", c))
    for n in instructor_notes: all_entries.append(("note", n))
    all_entries.sort(key=lambda x: x[1].get("timestamp",""))
    replaced_count = len(captures) - len(active_captures)

    # ── 01_photos.html ─────────────────────────────────────────────────────
    try:
        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Photos | {session_name}</title>{CSS}</head><body>
<h1>Lecture Photos</h1>
<p style="font-family:Arial,sans-serif;font-size:13px;color:#666">Session: {session_name} | {len(active_captures)} active captures | {replaced_count} replaced | {duplicate_skipped_count} duplicate skipped | LectureLens</p>\n"""
        cn = 0
        for typ, e in all_entries:
            if typ != "capture" or e.get("type") == "screenshot": continue
            cn += 1
            html += f"<div class=\"cap-hdr\"><span>Capture #{cn} | {e.get('timestamp','')}</span></div>\n"
            if e.get("path"): html += img_b64_tag(e["path"])
            else: html += "<p style=\"color:#999\">[Image not available]</p>"
        html += f"<div class=\"footer\">LectureLens &mdash; {session_name}</div></body></html>"
        (SESSION_DIR / "01_photos.html").write_text(html, encoding="utf-8")
        print("  \u2713 01_photos.html")
    except Exception as e:
        print(f"  [!] Photos error: {e}")

    # ── 02_ai_notes.html ───────────────────────────────────────────────────
    try:
        toc = ""
        cn = sn = nn = 0
        for typ, e in all_entries:
            ts = e.get("timestamp","")
            if typ == "capture":
                if e.get("type") == "screenshot": sn+=1; toc+=f"<a href=\"#e{ts}\">Screenshot #{sn} &mdash; {ts}</a>\n"
                else: cn+=1; toc+=f"<a href=\"#e{ts}\">Capture #{cn} &mdash; {ts}</a>\n"
            else: nn+=1; toc+=f"<a href=\"#e{ts}\">Note #{nn} &mdash; {ts}</a>\n"

        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>AI Notes | {session_name}</title>{CSS}{MATHJAX}</head><body>
<h1>AI Lecture Notes</h1>
<p style="font-family:Arial,sans-serif;font-size:13px;color:#666">Session: {session_name} | {len(active_captures)} active captures, {replaced_count} replaced, {duplicate_skipped_count} duplicate skipped, {note_count} notes | LectureLens</p>
<div class="toc"><strong style="font-family:Arial,sans-serif;font-size:13px">Contents</strong>\n{toc}</div>\n"""

        cn = sn = nn = 0
        for typ, e in all_entries:
            ts = e.get("timestamp","")
            if typ == "capture":
                is_ss = e.get("type") == "screenshot"
                if is_ss: sn+=1; hdr=f"Screenshot #{sn}"; hc="ss-hdr"
                else: cn+=1; hdr=f"Capture #{cn}"; hc="cap-hdr"
                html += f"<div class=\"{hc}\" id=\"e{ts}\"><span>{hdr} | {ts} | LectureLens</span></div>\n"
                if is_ss and e.get("path"): html += img_b64_tag(e["path"], 'style="max-height:300px"')
                if e.get("replaced_previous_timestamps"):
                    replaced_list = ", ".join([str(x) for x in e.get("replaced_previous_timestamps", []) if x])
                    reason_text = e.get("replaced_previous_reasons", [""])[-1] if e.get("replaced_previous_reasons") else "newer capture was clearer"
                    html += f'<div class="img-type"><strong>Smart Lookback:</strong> this clearer capture replaced earlier capture(s): {replaced_list}. Reason: {reason_text}</div>\n'
                notes = e.get("notes","")
                if "Image type:" in notes or "Screenshot type:" in notes:
                    first = notes.split("\n")[0]
                    html += f"<div class=\"img-type\">{first}</div>\n"
                    notes = "\n".join(notes.split("\n")[1:])
                html += md_to_html(notes) + "\n"
            else:
                nn += 1
                is_photo = e.get("type") == "photo_note"
                html += f"<div class=\"note-hdr\" id=\"e{ts}\">{'Photo Note' if is_photo else 'Text Note'} #{nn} | {ts}</div>\n"
                if is_photo and e.get("photo_path"): html += img_b64_tag(e["photo_path"], 'style="max-height:200px"')
                html += f"<p class=\"q-txt\"><strong>Question:</strong> {e.get('note','')}</p>\n"
                if e.get("answer"):
                    html += "<p class=\"ans-lbl\">Claude Answer:</p>\n"
                    html += md_to_html(e["answer"]) + "\n"

        html += f"<div class=\"footer\">Generated by Claude Sonnet &mdash; LectureLens &mdash; {session_name}</div></body></html>"
        (SESSION_DIR / "02_ai_notes.html").write_text(html, encoding="utf-8")
        print("  \u2713 02_ai_notes.html")
    except Exception as e:
        print(f"  [!] Notes error: {e}")
        import traceback; traceback.print_exc()

    # ── 03_summary.html ────────────────────────────────────────────────────
    try:
        print("  Generating summary with Claude...")
        summary_captures = [c for c in captures if not c.get("skip_in_final_outputs")]
        all_notes_text = "\n\n---\n\n".join([
            f"[{'Screenshot' if c.get('type')=='screenshot' else 'Capture'} at {c['timestamp']}]\n{c['notes']}"
            for c in summary_captures
        ])
        instructor_text = ""
        if instructor_notes:
            instructor_text = "\n\n## Questions and Answers from Class\n\n"
            for i,n in enumerate(instructor_notes):
                instructor_text += f"**Q{i+1}:** {n.get('note','')}\n\n**A:** {n.get('answer','')}\n\n"

        summary_resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=6000,
            messages=[{"role":"user","content":rf"""Here are lecture notes from {len(summary_captures)} active captures:

{all_notes_text}{instructor_text}

Create a comprehensive lecture summary with:
1. Main topic and learning objectives
2. All key concepts clearly explained
3. Every important equation, system, row operation, and matrix in proper LaTeX
   - Use inline math only for short expressions like $x_1$
   - Use display math $$...$$ for equations, systems, matrices, row operations, and final answers
   - Keep each display math block together as one complete block
   - For multi-line environments like cases, aligned, array, matrix, pmatrix, or bmatrix, use double backslashes \\ at the end of each row, never a single backslash
   - Do not put blank lines inside LaTeX environments
4. Step-by-step processes shown
5. Major conclusions and takeaways
6. Questions and Answers section if applicable

Important math-reasoning rule:
- If the capture notes contain ?, [unclear], or partial values, try to resolve them using the surrounding equations, repeated captures, row-reduction logic, and back-substitution.
- State inferred values confidently only when the math supports them. Example: $4x_3 = 12$ gives $x_3 = 3$, then $3x_2 = 12$ gives $x_2 = 4$, and $x_1 - x_2 - x_3 = 0$ gives $x_1 = 7$.
- Do not preserve obvious uncertainty if the answer can be solved from the visible math.
- Do not invent values when the math does not determine them; use [unclear] only then.

Format as clean markdown with ## headers. Use LaTeX for ALL math."""}]
        )
        summary_md = summary_resp.content[0].text

        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>Summary | {session_name}</title>{CSS}{MATHJAX}</head><body>
<h1>Lecture Summary</h1>
<p style="font-family:Arial,sans-serif;font-size:13px;color:#666">Session: {session_name} | {total} captures | LectureLens</p>
{md_to_html(summary_md)}
<div class="footer">Generated by Claude Sonnet &mdash; LectureLens &mdash; {session_name}</div>
</body></html>"""
        (SESSION_DIR / "03_summary.html").write_text(html, encoding="utf-8")
        print("  \u2713 03_summary.html")
    except Exception as e:
        print(f"  [!] Summary error: {e}")

    # Rename folder
    print("  Renaming session based on content...")
    try:
        sample = "".join(c.get("notes","")[:300] for c in captures[:3])
        rename_resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role":"user","content":f"Based on these lecture notes, give a 3-5 word folder name. Lowercase, hyphens only. Return ONLY the name.\n\n{sample[:600]}"}]
        )
        import re as _re
        raw = rename_resp.content[0].text.strip().lower()
        safe = _re.sub(r"[^a-z0-9\-]","",raw)[:40] or "lecture-session"
        date_prefix = "_".join(session_name.split("_")[:2])
        new_name = f"{date_prefix}_{safe}"
        new_dir = SESSION_DIR.parent / new_name
        if new_dir != SESSION_DIR and not new_dir.exists():
            SESSION_DIR.rename(new_dir)
            print(f"  \u2713 Renamed to: {new_name}")
            final_dir = new_dir
        else:
            final_dir = SESSION_DIR
    except Exception as e:
        final_dir = SESSION_DIR

    print(f"\n{'='*52}")
    print(f"  All HTML files saved to:")
    print(f"  {final_dir}")
    print(f"  Open in any browser for perfect math rendering.")
    print(f"  Print to PDF: Ctrl+P -> Save as PDF in browser.")
    print(f"{'='*52}")
    input("\n  Press Enter to exit...")


# ── Live preview thread ────────────────────────────────────────────────────
def live_preview():
    global flash_frame, note_flash, stopped
    flash_counter = 0
    note_flash_counter = 0

    frozen_frame = None
    while not stopped:
        if note_input_active:
            # Freeze the preview while teacher is typing
            if frozen_frame is not None:
                display = frozen_frame.copy()
                cv2.putText(display, "INPUT MODE — type in terminal", (200, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(display, "Preview will restore automatically", (220, 310),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                cv2.imshow("LectureLens — Live View", display)
            cv2.waitKey(30)
            continue

        ret, frame = cam.read()
        if ret:
            frozen_frame = cv2.resize(frame, (960, 540))
        if not ret:
            break

        # Resize for display (keeps it manageable on screen)
        display = frozen_frame.copy()

        # Green flash border on capture
        if flash_frame:
            flash_counter = 8
            flash_frame = False
        if flash_counter > 0:
            cv2.rectangle(display, (0, 0), (959, 539), (0, 255, 0), 12)
            flash_counter -= 1

        # Blue flash border on note added
        if note_flash:
            note_flash_counter = 8
            note_flash = False
        if note_flash_counter > 0:
            cv2.rectangle(display, (0, 0), (959, 539), (255, 150, 0), 12)
            note_flash_counter -= 1

        # Overlay info
        capture_count = len(captures)
        note_count = len(instructor_notes)
        queued = capture_queue.qsize()
        status = f"Captures: {capture_count}  Notes: {note_count}  Queue: {queued}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Show active camera label
        cam_label = camera_labels[camera_index] if camera_index < len(camera_labels) else f"Camera {camera_index}"
        cv2.putText(display, cam_label, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
        hint = "SPACE=capture  S=screenshot  N=note  O=photo-note  TAB=cam  ESC=stop" if len(available_cameras) > 1 else "SPACE=capture  S=screenshot  N=note  O=photo-note  ESC=stop"
        cv2.putText(display, hint, (10, 520),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(display, "REC", (900, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("LectureLens — Live View", display)
        cv2.waitKey(30)  # pynput handles all keys

    cam.release()
    cv2.destroyAllWindows()

# ── Keyboard listener (pynput — catches remote clicker too) ───────────────

def minimize_preview():
    """Minimize the OpenCV preview window using Windows API."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, "LectureLens — Live View")
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
    except Exception:
        pass


def restore_preview():
    """Restore the OpenCV preview window."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, "LectureLens — Live View")
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def focus_terminal():
    """Bring the terminal window to front."""
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass



def timed_input(prompt_text, timeout=15):
    """
    Get input from terminal with auto-cancel after timeout seconds.
    Timer only resets when a real non-space character is typed.
    Spaces alone do NOT reset the timer or count as valid input.
    """
    import threading
    import msvcrt  # Windows only — reads individual keystrokes

    result_chars = []
    cancelled = [False]
    warned = [False]
    last_real_char_time = [time.time()]
    input_done = [False]

    def read_input():
        """Read characters one at a time so we can track real vs space input."""
        print(prompt_text, end="", flush=True)
        while not cancelled[0]:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\r", "\n"):
                    # Enter pressed — done
                    print()  # newline
                    input_done[0] = True
                    break
                elif ch == "\x08":
                    # Backspace
                    if result_chars:
                        result_chars.pop()
                        print("\b \b", end="", flush=True)
                elif ch == " ":
                    # Space — add to result but do NOT reset timer
                    result_chars.append(" ")
                    print(" ", end="", flush=True)
                else:
                    # Real character — add and reset timer
                    result_chars.append(ch)
                    print(ch, end="", flush=True)
                    last_real_char_time[0] = time.time()
                    warned[0] = False  # reset warning if they start typing
            else:
                time.sleep(0.05)

    def timeout_watcher():
        while not input_done[0] and not cancelled[0]:
            elapsed = time.time() - last_real_char_time[0]
            if elapsed >= timeout - 5 and not warned[0]:
                warned[0] = True
                print(f"\n  ⚠  Auto-cancelling in 5 seconds — type a real character to reset timer...")
                print("  > ", end="", flush=True)
            if elapsed >= timeout:
                cancelled[0] = True
                input_done[0] = True
                print(f"\n\n  ✗ No real input received — note cancelled. Preview restored.")
                break
            time.sleep(0.5)

    input_thread = threading.Thread(target=read_input, daemon=True)
    watcher_thread = threading.Thread(target=timeout_watcher, daemon=True)
    input_thread.start()
    watcher_thread.start()
    input_thread.join(timeout + 2)

    if cancelled[0]:
        return None

    # Strip — spaces-only input is empty/cancelled
    val = "".join(result_chars).strip()
    return val if val else ""


def add_text_note():
    global note_input_active, note_flash
    if note_input_active:
        return
    note_input_active = True
    try:
        # Minimize preview and bring terminal to front
        minimize_preview()
        time.sleep(0.2)
        focus_terminal()

        print("\n  ── Instructor Note ────────────────────────────")
        print("  Type your note or question and press Enter:")
        print("  (leave blank to cancel — auto-cancels in 15s)")
        text = timed_input("  > ", timeout=15)
        if text is None:
            note_input_active = False
            time.sleep(0.3)
            restore_preview()
            return
        text = text.strip()
        if not text:
            print("  Cancelled.")
            note_input_active = False
            time.sleep(0.3)
            restore_preview()
            return
        ts = datetime.datetime.now().strftime("%H-%M-%S")
        print(f"  [{ts}] Asking Claude...")

        # Ask Claude to answer the note/question
        claude_answer = ""
        try:
            note_resp = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""You are a helpful classroom assistant. A teacher or student has raised the following note or question during a lecture:

"{text}"

Please provide a clear, concise, and accurate answer or response. Keep it educational and appropriate for a classroom setting. 2-4 paragraphs maximum."""
                }]
            )
            claude_answer = note_resp.content[0].text.strip()
            print(f"  [{ts}] ✓ Claude answered!\n")
            print("  ── Claude's Answer ────────────────────────────")
            for line in claude_answer.split("\n"):
                print(f"  {line}")
            print("  ──────────────────────────────────────────────")
        except Exception as e:
            claude_answer = f"[Claude answer unavailable: {e}]"
            print(f"  [{ts}] [!] Claude answer failed: {e}")

        entry = {"timestamp": ts, "note": text, "answer": claude_answer, "type": "text", "after_capture": len(captures)}
        with notes_lock:
            instructor_notes.append(entry)

        # Save to backups
        try:
            note_num = len(instructor_notes)
            txt_path = BACKUP_DIR / f"note_{str(note_num).zfill(2)}_{ts}.txt"
            with open(txt_path, "w", encoding="utf-8") as nf:
                nf.write(f"Instructor Note #{note_num} at {ts}\n")
                nf.write("=" * 50 + "\n\n")
                nf.write(f"QUESTION/NOTE:\n{text}\n\n")
                nf.write(f"CLAUDE\'S ANSWER:\n{claude_answer}\n")
        except Exception:
            pass

        note_flash = True
        print(f"  ✓ Note + answer saved!")
        print("  ──────────────────────────────────────────────\n")
    except Exception as e:
        print(f"  [!] Note error: {e}")
    finally:
        note_input_active = False
        # Restore preview window
        time.sleep(0.3)
        restore_preview()


def add_photo_note():
    global note_input_active, note_flash
    if note_input_active:
        return
    note_input_active = True
    try:
        # Snap photo first before minimizing
        ret, frame = cam.read()
        photo_path = None
        photo_b64 = None
        if ret:
            ts_photo = datetime.datetime.now().strftime("%H-%M-%S")
            photo_path = IMAGES_DIR / f"note_photo_{ts_photo}.jpg"
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_ch = clahe.apply(l_ch)
            enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(photo_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(photo_path, "rb") as f:
                photo_b64 = base64.standard_b64encode(f.read()).decode()

        # Minimize preview and bring terminal to front
        minimize_preview()
        time.sleep(0.2)
        focus_terminal()

        print("\n  ── Note + Board Photo ─────────────────────────")
        print("  Claude will see the current board when answering.")
        print("  Type your note or question and press Enter:")
        print("  (leave blank to cancel — auto-cancels in 15s)")
        text = timed_input("  > ", timeout=15)
        if text is None:
            note_input_active = False
            time.sleep(0.3)
            restore_preview()
            return
        text = text.strip()
        if not text:
            print("  Cancelled.")
            note_input_active = False
            time.sleep(0.3)
            restore_preview()
            return

        ts = datetime.datetime.now().strftime("%H-%M-%S")
        print(f"  [{ts}] Asking Claude with board photo...")

        claude_answer = ""
        try:
            if photo_b64:
                image_block_photo = {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": photo_b64
                    }
                }

                READ_PROMPT_PHOTO = r"""You are a precise mathematical transcriber. Your ONLY job right now is to READ and EXTRACT what is written on the board in this image — do not solve, simplify, or explain anything yet.

Extract in order:
1. Every equation, system of equations, or expression — exactly as written.
2. Every matrix (augmented or not) — every entry, row by row.
3. Every row operation label visible (e.g. R2 <- R2 - 3R1).
4. Every diagram label, vector, or annotation.

Rules:
- If a value is genuinely unreadable, write [unclear] for THAT entry only.
- Do NOT solve, reduce, or infer beyond what is literally visible.
- Use plain LaTeX for math.
- Output as a clean structured list. No prose explanations."""

                # Pass 1: Read the board
                read_resp = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": [image_block_photo, {"type": "text", "text": READ_PROMPT_PHOTO}]}]
                )
                extracted_photo = read_resp.content[0].text.strip()

                # Pass 2: Answer the question using the clean extraction + image
                answer_prompt = f"""Using that board extraction as context, answer this question from the instructor/student:

"{text}"

Be clear and accurate, referencing the board content where relevant. Keep it educational and concise — 2-4 paragraphs maximum."""

                note_resp = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=600,
                    messages=[
                        {"role": "user", "content": [image_block_photo, {"type": "text", "text": READ_PROMPT_PHOTO}]},
                        {"role": "assistant", "content": extracted_photo},
                        {"role": "user", "content": answer_prompt}
                    ]
                )
            else:
                note_resp = client.messages.create(
                    model="claude-sonnet-4-5",
                    max_tokens=600,
                    messages=[{"role": "user", "content": f"""You are a helpful classroom assistant. A teacher or student has the following question:
"{text}"
Please answer clearly and concisely."""}]
                )
            claude_answer = note_resp.content[0].text.strip()
            print(f"  [{ts}] ✓ Claude answered with board context!\n")
            print("  ── Claude's Answer (with board) ───────────────")
            for line in claude_answer.split("\n"):
                print(f"  {line}")
            print("  ──────────────────────────────────────────────")
        except Exception as e:
            claude_answer = f"[Claude answer unavailable: {e}]"
            print(f"  [{ts}] [!] Claude answer failed: {e}")

        entry = {
            "timestamp": ts,
            "note": text,
            "answer": claude_answer,
            "type": "photo",
            "photo_path": str(photo_path) if photo_path else "",
            "after_capture": len(captures)
        }
        with notes_lock:
            instructor_notes.append(entry)

        # Save to backups
        try:
            note_num = len(instructor_notes)
            txt_path = BACKUP_DIR / f"note_{str(note_num).zfill(2)}_{ts}_photo.txt"
            with open(txt_path, "w", encoding="utf-8") as nf:
                nf.write(f"Photo Note #{note_num} at {ts}\n")
                nf.write("=" * 50 + "\n\n")
                nf.write(f"QUESTION/NOTE:\n{text}\n\n")
                nf.write(f"CLAUDE'S ANSWER (with board photo):\n{claude_answer}\n")
        except Exception:
            pass

        note_flash = True
        print(f"  ✓ Note + board photo + answer saved!")
        print("  ──────────────────────────────────────────────\n")
    except Exception as e:
        print(f"  [!] Photo note error: {e}")
    finally:
        note_input_active = False
        time.sleep(0.3)
        restore_preview()



# ─── BEGIN screenshot fix (patched) ─────────────────────────────────────────
MODEL_SS = "claude-sonnet-4-5-20250929"  # pinned dated snapshot


def _prep_screenshot_for_claude(path):
    """
    Prepare screenshot images for Claude.

    Why this exists:
    - A full desktop screenshot is often too wide, so the WebWork/math text gets
      shrunk and Claude may read the handwritten black canvas instead.
    - We send the full screenshot for context PLUS several high-resolution crops:
        1. top page area
        2. top-left problem area
        3. detected white webpage/problem box
        4. optional dense text crop
    """
    import io as _io
    from PIL import Image as _PILImage
    import numpy as _np

    img = _PILImage.open(path).convert("RGB")
    W, H = img.size

    def _encode_png(im, max_long_edge=None, upscale_to=None):
        im = im.convert("RGB")
        w, h = im.size
        if max_long_edge and max(w, h) > max_long_edge:
            scale = max_long_edge / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), _PILImage.LANCZOS)
            w, h = im.size
        if upscale_to and max(w, h) < upscale_to:
            scale = upscale_to / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), _PILImage.LANCZOS)
        buf = _io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return base64.standard_b64encode(buf.getvalue()).decode()

    # Full screenshot is only context. Keep it smaller so token/cost is controlled.
    full_b64 = _encode_png(img, max_long_edge=1568)

    crops = []

    def _add_crop(label, box, upscale_to=1400):
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, min(W - 1, x0)); y0 = max(0, min(H - 1, y0))
        x1 = max(x0 + 1, min(W, x1)); y1 = max(y0 + 1, min(H, y1))
        crop = img.crop((x0, y0, x1, y1))
        # Avoid huge crops; make smaller crops high-res.
        max_edge = 2200 if max(crop.size) > 1800 else None
        crops.append({"label": label, "data": _encode_png(crop, max_long_edge=max_edge, upscale_to=upscale_to)})

    # WebWork/page problems are usually near the top. In the user's screenshots,
    # the actual equation/system can be near the TOP-CENTER while the left side
    # contains labels/buttons. Send several fixed crops so Claude can zoom the
    # printed equation instead of reading the dark handwritten scratch canvas.
    _add_crop("TOP OF SCREEN - likely printed WebWork/problem statement", (0, 0, W, int(H * 0.45)), upscale_to=1700)
    _add_crop("TOP LEFT PRINTED PROBLEM AREA - high priority", (0, 0, int(W * 0.80), int(H * 0.40)), upscale_to=1900)
    _add_crop("TOP CENTER PRINTED EQUATIONS - highest priority for systems/matrices", (int(W * 0.32), int(H * 0.07), int(W * 0.78), int(H * 0.42)), upscale_to=2200)
    _add_crop("WHITE WEBWORK QUESTION AREA - title through answer boxes", (0, 0, int(W * 0.95), int(H * 0.55)), upscale_to=1900)

    # Detect the large white/light webpage area. This handles the white problem box on a dark desktop.
    try:
        gray = _np.array(img.convert("L"))
        light = gray > 205
        # Only search upper half first because WebWork question is usually there.
        upper = light[:max(1, int(H * 0.55)), :]
        rows = _np.where(upper.mean(axis=1) > 0.18)[0]
        cols = _np.where(upper.mean(axis=0) > 0.08)[0]
        if len(rows) > 20 and len(cols) > 20:
            y0, y1 = int(rows[0]), int(rows[-1])
            x0, x1 = int(cols[0]), int(cols[-1])
            # Add padding so the title/instructions/equations stay included.
            pad_x, pad_y = 35, 25
            area = (x1 - x0) * (y1 - y0)
            if area > W * H * 0.01:
                _add_crop("DETECTED WHITE PRINTED CONTENT - ground truth problem", (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y), upscale_to=1900)
    except Exception as e:
        print(f"  [prep] white problem crop skipped: {e}")

    # Detect dense dark text inside the light webpage crop. This can help if the equation itself is tiny.
    try:
        arr = _np.array(img.convert("L"))
        # Dark text on light background in upper half.
        upper_h = int(H * 0.55)
        dark = arr[:upper_h, :] < 120
        light_bg = arr[:upper_h, :] > 190
        # Slightly favor columns/rows where dark text appears, but don't require too much.
        row_density = dark.mean(axis=1)
        col_density = dark.mean(axis=0)
        text_rows = _np.where(row_density > 0.002)[0]
        text_cols = _np.where(col_density > 0.001)[0]
        if len(text_rows) > 10 and len(text_cols) > 10:
            y0, y1 = int(text_rows[0]), int(text_rows[-1])
            x0, x1 = int(text_cols[0]), int(text_cols[-1])
            # Keep this crop focused but padded.
            if (y1 - y0) > 20 and (x1 - x0) > 50:
                _add_crop("DENSE TEXT CROP - use for exact signs/numbers", (x0 - 60, y0 - 40, x1 + 60, y1 + 40), upscale_to=2000)
    except Exception as e:
        print(f"  [prep] dense text crop skipped: {e}")

    return full_b64, crops

def screenshot_and_note():
    global note_input_active, note_flash, flash_frame, active_threads
    if note_input_active:
        return
    note_input_active = True
    active_threads += 1
    try:
        ts = datetime.datetime.now().strftime("%H-%M-%S")
        screenshot_path = IMAGES_DIR / f"screenshot_{ts}.png"

        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(str(screenshot_path))
        except Exception:
            ps_cmd = f'Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Screen]::PrimaryScreen | Out-Null; $bitmap = [System.Drawing.Bitmap]::new([System.Windows.Forms.SystemInformation]::VirtualScreen.Width, [System.Windows.Forms.SystemInformation]::VirtualScreen.Height); $graphics = [System.Drawing.Graphics]::FromImage($bitmap); $graphics.CopyFromScreen([System.Windows.Forms.SystemInformation]::VirtualScreen.Location, [System.Drawing.Point]::Empty, $bitmap.Size); $bitmap.Save("{screenshot_path}")'
            subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)

        print(f"  [{ts}] Screenshot saved!")

        minimize_preview()
        time.sleep(0.2)
        focus_terminal()

        print("\n  ── Screenshot Captured ────────────────────────")
        print("  Add a note about this screenshot?")
        print("  (press Enter to skip — auto-cancels in 15s, screenshot still saved)")
        text = timed_input("  > ", timeout=15)
        if text is None:
            text = ""
        text = text.strip()

        note_input_active = False
        time.sleep(0.3)
        restore_preview()

        print(f"  [{ts}] Sending screenshot to Claude in background...")

        # Send full screenshot + a focused crop of the printed text region
        full_b64, crops = _prep_screenshot_for_claude(screenshot_path)

        image_blocks = [{
            "type": "text",
            "text": "IMAGE 1: FULL DESKTOP SCREENSHOT. Use this for context only; details may be small."
        }, {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": full_b64}
        }]

        for i, crop in enumerate(crops, start=2):
            image_blocks.append({
                "type": "text",
                "text": f"IMAGE {i}: {crop.get('label','focused crop')}. This crop has priority for reading exact printed equations."
            })
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": crop["data"]}
            })

        if crops:
            print(f"  [{ts}] Sending full screenshot + {len(crops)} high-resolution focused crop(s).")
        else:
            print(f"  [{ts}] Sending full screenshot only.")

        try:
            READ_PROMPT_SS = r"""You are a precise mathematical reader for screenshots. READ ONLY in this pass; do not solve yet.

You may be given MULTIPLE images of the same screenshot: one full desktop screenshot plus focused crops. Use the focused crops to read exact math. The full screenshot is context only.

IMPORTANT: If the screenshot contains a white WebWork/browser/page area and a dark handwritten canvas, the white printed page is the actual problem unless the instructor note says otherwise.

Extract, IN THIS ORDER:
1. PRINTED / TYPED content.
   - This is the main problem and ground truth.
   - Transcribe every equation exactly, character by character, including signs and right-hand sides.
   - Include instructions like "Solve the system" and answer-field labels if visible.
2. HANDWRITTEN content on the dark canvas.
   - Treat it as scratch work unless it clearly matches the printed problem.
   - Transcribe every matrix entry, row by row, and every visible row-operation label.
3. Any diagrams, vectors, annotations, or warning messages.

ABSOLUTE RULES:
- If you cannot clearly see a character, write [unclear] for THAT specific character. Do not guess.
- Do not solve, reduce, derive, or infer in this pass.
- Do not "clean up" coefficients. If the page says 3x - 3y, write 3x - 3y, not x - y.
- If handwritten work conflicts with printed text, write a short "CONFLICT:" note after the handwritten section.
- Separate sections using exactly: PRINTED:, HANDWRITTEN:, OTHER:.
- Use plain LaTeX. Augmented matrices: \left[\begin{array}{...|c}...\end{array}\right]."""

            solve_prompt_ss = r"""You are a careful math tutor. You are given the original screenshot/crops AND the extracted content. Answer the actual screenshot problem like a normal Claude chat.

CRITICAL RULES:
- Look back at the screenshot/crops directly while solving. Do NOT rely only on the extracted text if the extraction may be wrong.
- The PRINTED section from the white WebWork/browser/page area is the ground truth.
- Use HANDWRITTEN/dark-canvas work only as supporting scratch work. Never let it override the white printed problem.
- If handwritten work conflicts with the printed problem, trust the printed problem and mention the mismatch briefly.
- Do not rename variables. Do not change printed numbers or signs.
- If the printed problem asks to solve, you MUST solve it even if the handwritten solution is missing.
- If the PRINTED section contains a system of equations, build the augmented matrix mechanically:
    row i = [coefficients of variables in equation i | RHS of equation i]
  The matrix must match the printed system exactly.
- Show clear algebra steps and verify the final answer by substitution.
- If the answer fields already contain values, still solve/check them against the printed equations.

For [unclear] entries:
- Use the focused crop and printed text first.
- Only infer a value when arithmetic or the visible problem makes it certain.
- State the inference in "Reading Notes."

FORMATTING:
- Inline math: $x + y = -1$
- Display math $$...$$ for systems, matrices, row operations, and checks.
- Use \\ between matrix rows.
- No blank lines inside LaTeX environments.

OUTPUT:
## Topic
## Given Problem
(write the printed problem exactly as extracted)
## Augmented Matrix
(built mechanically from the printed system, if relevant)
## Work
(solve step by step)
## Check
(substitute the answer into the original printed equations)
## Final Answer
(state the answer clearly)
## Reading Notes
(mention conflicts, unclear entries, or inferred values; otherwise write "No reading issues.")"""

            if text:
                solve_prompt_ss += f"\n\nThe instructor added this note: {text}"

            print(f"  [{ts}] Pass 1: Reading screenshot...")
            read_resp = client.messages.create(
                model=MODEL_SS,
                max_tokens=3000,
                messages=[{"role": "user",
                           "content": image_blocks + [{"type": "text", "text": READ_PROMPT_SS}]}]
            )
            extracted_ss = read_resp.content[0].text.strip()
            print(f"  [{ts}] Pass 1 done. Pass 2: Formatting...")

            solve_resp = client.messages.create(
                model=MODEL_SS,
                max_tokens=6000,
                messages=[
                    {"role": "user",
                     "content": image_blocks + [{"type": "text", "text": READ_PROMPT_SS}]},
                    {"role": "assistant", "content": extracted_ss},
                    {"role": "user", "content": image_blocks + [{"type": "text", "text": solve_prompt_ss}]}
                ]
            )
            notes = solve_resp.content[0].text.strip()

            # Conservative audit pass — can only resolve [unclear], cannot overwrite confident reads
            try:
                audit_prompt = r"""You are doing a final quality check on screenshot math notes.

YOUR JOB:
1. Verify the GIVEN PROBLEM matches the PRINTED content in the screenshot. The printed problem is ground truth.
2. Verify the AUGMENTED MATRIX is built mechanically from the printed system. If it does not match, fix the matrix, not the printed system.
3. If the printed problem asks to solve, make sure the notes actually solve it and include a final answer. Add the solution if it is missing.
4. Verify the final answer by substitution into the original printed equations.
5. Fix markdown/LaTeX formatting issues, especially missing \\ between rows.
6. Resolve [unclear] entries only when the screenshot or arithmetic makes the value certain.

YOU MUST NOT:
- Change any clearly readable printed equation, coefficient, sign, or constant.
- Let handwritten scratch work override the printed problem.
- Invent unrelated row-reduction steps.

If you have nothing to change, return the draft unchanged.

Return the complete corrected markdown notes only. No preamble."""
                audit_resp = client.messages.create(
                    model=MODEL_SS,
                    max_tokens=6000,
                    messages=[{"role": "user", "content": image_blocks + [
                        {"type": "text",
                         "text": audit_prompt + "\n\nFIRST DRAFT NOTES:\n<<<DRAFT\n" + notes[:7000] + "\nDRAFT>>>"}
                    ]}]
                )
                improved_notes = audit_resp.content[0].text.strip()
                if improved_notes:
                    notes = improved_notes
                    print(f"  [{ts}] ✓ Conservative audit pass complete")
            except Exception as e:
                print(f"  [{ts}] [!] Audit skipped: {e}")

            if text:
                print(f"  [{ts}] ✓ Claude processed screenshot + note!")
            else:
                print(f"  [{ts}] ✓ Screenshot notes ready!")
            print("\n  ── Claude's Notes ─────────────────────────────")
            for line in notes.split("\n")[:8]:
                print(f"  {line}")
            print("  ──────────────────────────────────────────────")

            entry = {
                "path": str(screenshot_path),
                "notes": notes,
                "timestamp": ts,
                "type": "screenshot",
                "instructor_note": text
            }
            with captures_lock:
                captures.append(entry)

            try:
                num = len(captures)
                txt_path = BACKUP_DIR / f"notes_{str(num).zfill(2)}_{ts}_screenshot.txt"
                with open(txt_path, "w", encoding="utf-8") as nf:
                    nf.write(f"Screenshot #{num} at {ts}\n")
                    nf.write("=" * 50 + "\n\n")
                    if text:
                        nf.write(f"Instructor note: {text}\n\n")
                    nf.write(notes)
            except Exception:
                pass

            note_flash = True
            print(f"  ✓ Screenshot + notes saved!")
            print("  ──────────────────────────────────────────────\n")

        except Exception as e:
            print(f"  [{ts}] ERROR: {e}")

    except Exception as e:
        print(f"  [!] Screenshot error: {e}")
    finally:
        active_threads -= 1
        if note_input_active:
            note_input_active = False
            time.sleep(0.3)
            restore_preview()
# ─── END screenshot fix (patched) ───────────────────────────────────────────

def toggle_camera():
    global cam, camera_index
    if len(available_cameras) <= 1:
        return
    cam.release()
    camera_index = (camera_index + 1) % len(available_cameras)
    cam = cv2.VideoCapture(available_cameras[camera_index])
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"  [CAM] Switched to {camera_labels[camera_index]}")

def on_press(key):
    global stopped
    if stopped:
        return False
    try:
        if key == keyboard.Key.space:
            if not note_input_active:
                threading.Thread(target=snap_photo, daemon=True).start()
        elif key == keyboard.Key.tab:
            threading.Thread(target=toggle_camera, daemon=True).start()
        elif key == keyboard.Key.esc:
            stopped = True
            return False
        # Also support right arrow (common remote clicker button)
        elif key == keyboard.Key.right:
            if not note_input_active:
                threading.Thread(target=snap_photo, daemon=True).start()
        # Support F5 (some iClickers send this)
        elif key == keyboard.Key.f5:
            if not note_input_active:
                threading.Thread(target=snap_photo, daemon=True).start()
        # N key — add text note in terminal
        elif hasattr(key, 'char') and key.char and key.char.lower() == 'n':
            threading.Thread(target=add_text_note, daemon=True).start()
        elif hasattr(key, 'char') and key.char and key.char.lower() == 'o':
            threading.Thread(target=add_photo_note, daemon=True).start()
        elif hasattr(key, 'char') and key.char and key.char.lower() == 's':
            t = threading.Thread(target=screenshot_and_note, daemon=False)
            screenshot_threads.append(t)
            t.start()
    except Exception:
        pass

# ── Main ───────────────────────────────────────────────────────────────────
# Start background queue processor for captures
queue_thread = threading.Thread(target=process_queue, daemon=True)
queue_thread.start()

# Start live preview
preview_thread = threading.Thread(target=live_preview, daemon=True)
preview_thread.start()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

# Signal stop and wait for queue to finish processing
stopped = True
print("\n  Waiting for remaining captures to finish processing...")
capture_queue.join()

# Wait for screenshot/note threads still processing. Do not time out here:
# screenshots may need multiple Claude calls, and pressing ESC should not cancel them.
if active_threads > 0:
    print("  Waiting for screenshots/notes to finish processing...")
while active_threads > 0:
    import time as _t
    _t.sleep(0.5)
for t in list(screenshot_threads):
    try:
        t.join()
    except Exception:
        pass
if screenshot_threads:
    print("  Screenshots/notes done!")

print("  All captures processed!")
preview_thread.join(timeout=3)

# Generate PDFs
generate_html()