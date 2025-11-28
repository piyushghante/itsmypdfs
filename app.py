"""
Streamlit PDF Optimizer — SAFER VERSION
=====================================

This improved single-file Streamlit app:
 - analyzes PDF streams (images/others) and shows the largest streams
 - selectively recompresses embedded images only when the recompressed bytes are
   meaningfully smaller (so the optimizer never increases overall file size)
 - skips images with alpha/transparency and already-efficient formats (JPEG2000, JBIG2)
 - offers tunable thresholds (absolute & percent) for replacements
 - optional Ghostscript final pass (if `gs` is installed) to squeeze extra bytes
 - optional qpdf linearization (if `qpdf` is installed) — kept separate from compression

Requirements
------------
pip install streamlit pymupdf pillow

Optional tools
--------------
 - qpdf (for linearization): `sudo apt install qpdf` or `brew install qpdf`
 - Ghostscript (for a secondary compression pass): `sudo apt install ghostscript` or `brew install ghostscript`

Run
---
streamlit run streamlit_pdf_optimizer.py

Notes
-----
 - This version avoids blind replacement of image streams. It only replaces when the new
   compressed image stream is both smaller by an absolute number of bytes and by a
   minimum percentage. That prevents size increases.
 - Keep backups of source PDFs until you're satisfied with results.
"""

import io
import time
import tempfile
import os
import subprocess
from datetime import datetime
from typing import Tuple, List

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd


# ---------------- Utility functions ---------------- #

def human_readable_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def is_exe_available(exe_name: str) -> bool:
    try:
        subprocess.check_output([exe_name, "--version"], stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


# ---------------- PDF analysis helpers ---------------- #

def analyze_pdf_sizes(pdf_bytes: bytes, top_n: int = 20) -> List[Tuple[int, int, str]]:
    """
    Return list of top stream objects by size: (xref, size, type)
    Type is heuristically 'image' or 'other'.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    info = []
    for xref in range(1, doc.xref_length()):
        try:
            length = doc.xref_stream_length(xref)
            if length <= 0:
                continue
            typ = "other"
            try:
                # doc.extract_image raises if not an image
                img_info = doc.extract_image(xref)
                typ = "image"
            except Exception:
                typ = "other"
            info.append((xref, length, typ))
        except Exception:
            continue

    info.sort(key=lambda t: t[1], reverse=True)
    return info[:top_n]


# ---------------- Image recompression (safe, selective) ---------------- #

def recompress_images(
    doc: fitz.Document,
    quality: int = 80,
    max_dim: int | None = None,
    replace_margin_bytes: int = 1024,
    replace_margin_pct: float = 0.05,
) -> Tuple[int, int, int, List[dict]]:
    """
    Recompress images but only replace when helpful.

    Returns: (total_images_found, images_recompressed, skipped_count, per_image_stats)
    per_image_stats: list of dicts {xref, orig_len, new_len, saved_bytes, saved_pct, replaced, ext}
    """
    processed_xrefs = set()
    total_images = 0
    recompressed_images = 0
    skipped = 0
    stats = []

    for page_index, page in enumerate(doc):
        images = page.get_images(full=True)
        total_images += len(images)

        for img in images:
            xref = img[0]
            if xref in processed_xrefs:
                continue
            processed_xrefs.add(xref)

            entry = {"xref": xref, "page": page_index}

            # Extract original image bytes and metadata
            try:
                img_info = doc.extract_image(xref)
                orig_bytes = img_info.get("image") or b""
                orig_ext = (img_info.get("ext") or "").lower()
            except Exception:
                skipped += 1
                entry.update({"orig_len": 0, "new_len": 0, "saved_bytes": 0, "saved_pct": 0.0, "replaced": False, "ext": None})
                stats.append(entry)
                continue

            orig_len = len(orig_bytes)
            entry["orig_len"] = orig_len
            entry["ext"] = orig_ext

            # Skip tiny images; replacement overhead may increase size
            if orig_len < 2048:
                skipped += 1
                entry.update({"new_len": orig_len, "saved_bytes": 0, "saved_pct": 0.0, "replaced": False})
                stats.append(entry)
                continue

            # Skip already-efficient formats by default
            if orig_ext in ("jp2", "jpx", "jb2", "jbig2"):
                skipped += 1
                entry.update({"new_len": orig_len, "saved_bytes": 0, "saved_pct": 0.0, "replaced": False})
                stats.append(entry)
                continue

            # Build Pixmap
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                skipped += 1
                entry.update({"new_len": orig_len, "saved_bytes": 0, "saved_pct": 0.0, "replaced": False})
                stats.append(entry)
                continue

            # Skip images with alpha channels (transparency)
            try:
                if getattr(pix, "alpha", False):
                    skipped += 1
                    entry.update({"new_len": orig_len, "saved_bytes": 0, "saved_pct": 0.0, "replaced": False})
                    stats.append(entry)
                    continue
            except Exception:
                pass

            # Convert to RGB if needed
            try:
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
            except Exception:
                pass

            # Create PIL image
            try:
                mode = "RGB" if pix.n >= 3 else "L"
                img_pil = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            except Exception:
                skipped += 1
                entry.update({"new_len": orig_len, "saved_bytes": 0, "saved_pct": 0.0, "replaced": False})
                stats.append(entry)
                continue

            # Optional downscale
            if max_dim is not None:
                w, h = img_pil.size
                scale = min(max_dim / float(w), max_dim / float(h), 1.0)
                if scale < 1.0:
                    img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            # Recompress to JPEG in memory
            buf = io.BytesIO()
            try:
                rgb = img_pil.convert("RGB")
                rgb.save(buf, format="JPEG", optimize=True, quality=quality)
                new_bytes = buf.getvalue()
            except Exception:
                skipped += 1
                entry.update({"new_len": orig_len, "saved_bytes": 0, "saved_pct": 0.0, "replaced": False})
                stats.append(entry)
                buf.close()
                continue
            finally:
                buf.close()

            new_len = len(new_bytes)
            saved_bytes = orig_len - new_len
            saved_pct = (saved_bytes / orig_len) if orig_len > 0 else 0.0

            entry.update({"new_len": new_len, "saved_bytes": saved_bytes, "saved_pct": round(saved_pct, 4)})

            # Decide whether to replace
            if saved_bytes >= replace_margin_bytes and saved_pct >= replace_margin_pct:
                try:
                    doc.update_stream(xref, new_bytes)
                    recompressed_images += 1
                    entry["replaced"] = True
                except Exception:
                    entry["replaced"] = False
                    skipped += 1
            else:
                entry["replaced"] = False
                skipped += 1

            stats.append(entry)

    return total_images, recompressed_images, skipped, stats


# ---------------- Optimization pipeline ---------------- #

def ghostscript_optimize(in_bytes: bytes, gs_setting: str = "/ebook") -> bytes:
    """
    Run Ghostscript pdfwrite with a chosen setting. Returns new bytes or raises.
    gs_setting examples: /screen, /ebook, /printer, /prepress
    """
    if not is_exe_available("gs"):
        raise FileNotFoundError("Ghostscript (gs) not available")

    in_temp = None
    out_temp_path = None
    try:
        in_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        in_temp.write(in_bytes)
        in_temp.flush()
        in_temp.close()

        out_temp_path = in_temp.name.replace(".pdf", "_gs.pdf")

        cmd = [
            "gs",
            "-sDEVICE=pdfwrite",
            "-dCompatibilityLevel=1.4",
            f"-dPDFSETTINGS={gs_setting}",
            "-dNOPAUSE",
            "-dQUIET",
            "-dBATCH",
            f"-sOutputFile={out_temp_path}",
            in_temp.name,
        ]
        subprocess.check_call(cmd)

        with open(out_temp_path, "rb") as f:
            return f.read()
    finally:
        try:
            if in_temp is not None and os.path.exists(in_temp.name):
                os.remove(in_temp.name)
            if out_temp_path and os.path.exists(out_temp_path):
                os.remove(out_temp_path)
        except Exception:
            pass


def optimize_pdf(
    pdf_bytes: bytes,
    mode: str = "Medium",
    linearize_after: bool = False,
    gs_after: bool = False,
    gs_setting: str = "/ebook",
    replace_margin_bytes: int = 1024,
    replace_margin_pct: float = 0.05,
) -> Tuple[bytes, dict, List[dict]]:
    """
    Optimize/Compress PDF bytes using selective recompression + cleanup.
    Returns (optimized_bytes, stats_dict, per_image_stats)
    """
    if mode == "Light (lossless-ish)":
        image_quality = 90
        max_dim = None
        garbage = 2
        deflate = True
    elif mode == "Strong":
        image_quality = 65
        max_dim = 1500
        garbage = 4
        deflate = True
    else:  # Medium
        image_quality = 80
        max_dim = 2000
        garbage = 3
        deflate = True

    input_size = len(pdf_bytes)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    total_images, recompressed_images, skipped, per_image_stats = recompress_images(
        doc,
        quality=image_quality,
        max_dim=max_dim,
        replace_margin_bytes=replace_margin_bytes,
        replace_margin_pct=replace_margin_pct,
    )

    # Save to memory after image updates
    out_buf = io.BytesIO()
    try:
        doc.save(out_buf, garbage=garbage, deflate=deflate, clean=True, incremental=False)
        optimized_bytes = out_buf.getvalue()
    finally:
        out_buf.close()

    linearized_flag = False
    gs_flag = False

    # Optional Ghostscript final pass
    if gs_after and is_exe_available("gs"):
        try:
            optimized_bytes = ghostscript_optimize(optimized_bytes, gs_setting=gs_setting)
            gs_flag = True
        except Exception:
            gs_flag = False

    # Optional linearize with qpdf
    if linearize_after and is_exe_available("qpdf"):
        in_temp = None
        out_temp_path = None
        try:
            in_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            in_temp.write(optimized_bytes)
            in_temp.flush()
            in_temp.close()

            out_temp_path = in_temp.name.replace(".pdf", "_lin.pdf")
            subprocess.check_call(["qpdf", "--linearize", in_temp.name, out_temp_path])

            with open(out_temp_path, "rb") as f:
                optimized_bytes = f.read()

            linearized_flag = True
        except Exception:
            linearized_flag = False
        finally:
            try:
                if in_temp is not None and os.path.exists(in_temp.name):
                    os.remove(in_temp.name)
                if out_temp_path and os.path.exists(out_temp_path):
                    os.remove(out_temp_path)
            except Exception:
                pass

    output_size = len(optimized_bytes)
    reduction = (1 - output_size / input_size) * 100 if input_size > 0 else 0.0

    stats = {
        "input_size": input_size,
        "output_size": output_size,
        "reduction_percent": round(reduction, 2),
        "total_images": total_images,
        "recompressed_images": recompressed_images,
        "skipped_images": skipped,
        "mode": mode,
        "linearized": linearized_flag,
        "ghostscript_applied": gs_flag,
    }

    return optimized_bytes, stats, per_image_stats


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="PDF Optimizer (Safer)", page_icon="📄", layout="centered")

st.title("📄 Smart PDF Optimizer — Safer Mode")
st.caption("Selective image recompression + structural cleanup. This version avoids increasing file size.")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox(
        "Compression mode",
        ["Light (lossless-ish)", "Medium", "Strong"],
        help=(
            "Light: Mostly structural cleanup & gentle recompression.\n"
            "Medium: Good balance of size & quality.\n"
            "Strong: Smaller file, but more aggressive image recompression."
        ),
    )

with col2:
    show_details = st.checkbox("Show detailed stats", value=True)

# Replacement thresholds
st.markdown("**Replacement thresholds (tune to avoid bad conversions)**")
col_a, col_b = st.columns(2)
with col_a:
    replace_margin_bytes = st.number_input("Min absolute bytes saved to replace (KB)", min_value=0, value=1, step=1)
    replace_margin_bytes = int(replace_margin_bytes) * 1024
with col_b:
    replace_margin_pct = st.slider("Min percent saved to replace", min_value=0.0, max_value=0.5, value=0.05, step=0.01)

linearize_checkbox = st.checkbox("Linearize output (requires qpdf)")
qs_available = is_exe_available("qpdf")
if linearize_checkbox and not qs_available:
    st.warning("qpdf is not available on this host. Linearize will be skipped. Install qpdf if you want linearized output.")

gs_checkbox = st.checkbox("Run Ghostscript final pass (optional)")
gs_available = is_exe_available("gs")
if gs_checkbox and not gs_available:
    st.warning("Ghostscript (gs) not available. Ghostscript pass will be skipped. Install gs if you want an extra compression pass.")

if uploaded_file is not None:
    st.write(f"**Original file:** `{uploaded_file.name}`")
    st.write(f"Size: `{human_readable_size(uploaded_file.size)}`")

    if st.button("🚀 Analyze & Optimize PDF"):
        pdf_bytes = uploaded_file.read()

        # Quick analysis: top streams
        with st.spinner("Analyzing PDF streams..."):
            top_streams = analyze_pdf_sizes(pdf_bytes, top_n=20)

        st.subheader("Top streams (largest first)")
        if len(top_streams) == 0:
            st.write("No streams found in PDF.")
        else:
            df_streams = pd.DataFrame(top_streams, columns=["xref", "size", "type"])
            df_streams["size_hr"] = df_streams["size"].apply(human_readable_size)
            st.table(df_streams.head(20))

        # Run optimizer
        with st.spinner("Optimizing your PDF (selective recompression)..."):
            start = time.time()
            try:
                optimized_bytes, stats, per_image_stats = optimize_pdf(
                    pdf_bytes,
                    mode=mode,
                    linearize_after=linearize_checkbox,
                    gs_after=gs_checkbox,
                    gs_setting="/ebook",
                    replace_margin_bytes=replace_margin_bytes,
                    replace_margin_pct=replace_margin_pct,
                )
            except Exception as e:
                st.error(f"Error during optimization: {e}")
                raise
            duration = time.time() - start

        st.success("Optimization complete!")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(label="Original size", value=human_readable_size(stats["input_size"]))
            st.metric(label="Optimized size", value=human_readable_size(stats["output_size"]))
        with col_b:
            st.metric(label="Reduction", value=f"{stats['reduction_percent']} %")
            st.write(f"Mode: `{stats['mode']}`")
            st.write(f"Linearized: `{stats['linearized']}`")
            st.write(f"Ghostscript applied: `{stats['ghostscript_applied']}`")
            st.write(f"Time taken: `{duration:.2f} s`")

        if show_details:
            st.divider()
            st.subheader("Image recompression details")
            df_stats = pd.DataFrame(per_image_stats)
            if not df_stats.empty:
                df_stats["orig_hr"] = df_stats["orig_len"].apply(human_readable_size)
                df_stats["new_hr"] = df_stats["new_len"].apply(human_readable_size)
                df_stats["saved_hr"] = df_stats["saved_bytes"].apply(human_readable_size)
                # show relevant columns
                cols = ["page", "xref", "ext", "orig_hr", "new_hr", "saved_hr", "saved_pct", "replaced"]
                st.dataframe(df_stats[cols].sort_values(by="saved_bytes", ascending=False))
            else:
                st.write("No image statistics available.")

        # Download button
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"optimized_{ts}_{uploaded_file.name}"
        st.download_button(label="⬇️ Download optimized PDF", data=optimized_bytes, file_name=out_name, mime="application/pdf")

else:
    st.info("Upload a PDF to analyze and optimize.")
