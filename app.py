"""
Streamlit PDF Optimizer
======================

Single-file Streamlit app that:
 - accepts an uploaded PDF
 - recompresses embedded images (configurable quality / downscale)
 - removes unused objects, deflates streams, and performs structural cleanup
 - optionally linearizes output using `qpdf` (if installed)

Requirements
------------
pip install streamlit pymupdf pillow

Run
---
streamlit run streamlit_pdf_optimizer.py

Notes
-----
 - Uses PyMuPDF (fitz). Newer versions of PyMuPDF removed update_image() and linearization via save(), so this app uses update_stream() and delegates linearization to qpdf if requested.
 - Be careful with extremely fragile PDFs; always keep backups.
"""

import io
import time
import tempfile
import os
import subprocess
from datetime import datetime

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image


# ---------------- Utility functions ---------------- #

def human_readable_size(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def is_qpdf_available() -> bool:
    try:
        subprocess.check_output(["qpdf", "--version"], stderr=subprocess.DEVNULL)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# ---------------- Image recompression ---------------- #

def recompress_images(doc: fitz.Document, quality: int = 75, max_dim: int | None = None):
    """
    Recompress images inside a PyMuPDF document using Pillow -> JPEG.

    - doc: opened fitz.Document
    - quality: JPEG quality (1-95)
    - max_dim: max width/height to downscale images (None to keep size)

    Returns: (total_images_found, images_recompressed)
    """
    processed_xrefs = set()
    total_images = 0
    recompressed_images = 0

    for page_index, page in enumerate(doc):
        images = page.get_images(full=True)
        total_images += len(images)

        for img in images:
            xref = img[0]
            if xref in processed_xrefs:
                continue
            processed_xrefs.add(xref)

            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception as e:
                # Could not extract image pixmap; skip
                continue

            # If pix is CMYK or has transparency, convert to RGB first
            try:
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
            except Exception:
                # Fallback: keep original pix
                pass

            # Determine PIL mode and create image
            if pix.n >= 3:
                mode = "RGB"
                img_pil = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            else:
                mode = "L"
                img_pil = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

            # Optional downscale
            if max_dim is not None:
                w, h = img_pil.size
                scale = min(max_dim / float(w), max_dim / float(h), 1.0)
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    img_pil = img_pil.resize(new_size, Image.LANCZOS)

            # Convert to RGB and recompress as optimized JPEG
            buf = io.BytesIO()
            try:
                rgb = img_pil.convert("RGB")
                rgb.save(buf, format="JPEG", optimize=True, quality=quality)
                new_img_bytes = buf.getvalue()
            except Exception:
                # If JPEG conversion fails, skip this image to be safe
                continue
            finally:
                buf.close()

            # Update the raw stream for the image object in the PDF
            try:
                doc.update_stream(xref, new_img_bytes)
                recompressed_images += 1
            except Exception:
                # If update_stream fails, skip and continue
                continue

    return total_images, recompressed_images


# ---------------- Optimization pipeline ---------------- #

def optimize_pdf(pdf_bytes: bytes, mode: str = "Medium", linearize_after: bool = False) -> tuple[bytes, dict]:
    """
    Optimize/Compress PDF bytes according to selected mode.

    - mode: one of ["Light (lossless-ish)", "Medium", "Strong"]
    - linearize_after: if True and qpdf is available, run qpdf --linearize on the output

    Returns: (optimized_bytes, stats_dict)
    """
    # Mode presets
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

    total_images, recompressed_images = recompress_images(doc, quality=image_quality, max_dim=max_dim)

    # Save optimized PDF to memory (no linear option here)
    out_buf = io.BytesIO()
    try:
        doc.save(out_buf, garbage=garbage, deflate=deflate, clean=True, incremental=False)
        optimized_bytes = out_buf.getvalue()
    finally:
        out_buf.close()

    linearized_flag = False

    # Optional: linearize using qpdf if requested and available
    if linearize_after and is_qpdf_available():
        in_temp = None
        out_temp = None
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
            # if qpdf failed for any reason, keep the non-linearized output
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
        "mode": mode,
        "linearized": linearized_flag,
    }

    return optimized_bytes, stats


# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="PDF Optimizer", page_icon="📄", layout="centered")

st.title("📄 Smart PDF Optimizer")
st.caption("Recompress images + clean structure using PyMuPDF (fitz) and Pillow. Optional linearize via qpdf.")

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

linearize_checkbox = st.checkbox("Linearize output (requires qpdf)")

qpdf_ok = is_qpdf_available()
if linearize_checkbox and not qpdf_ok:
    st.warning("qpdf is not available on this host. Linearize will be skipped. Install qpdf if you want linearized output.")

if uploaded_file is not None:
    st.write(f"**Original file:** `{uploaded_file.name}`")
    st.write(f"Size: `{human_readable_size(uploaded_file.size)}`")

    if st.button("🚀 Optimize PDF"):
        pdf_bytes = uploaded_file.read()

        with st.spinner("Optimizing your PDF..."):
            start = time.time()
            try:
                optimized_bytes, stats = optimize_pdf(pdf_bytes, mode=mode, linearize_after=linearize_checkbox and qpdf_ok)
            except Exception as e:
                st.error(f"Error during optimization: {e}")
            else:
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
                    st.write(f"Time taken: `{duration:.2f} s`")

                if show_details:
                    st.divider()
                    st.subheader("Details")
                    st.write(f"- Total images found: **{stats['total_images']}**")
                    st.write(f"- Images recompressed: **{stats['recompressed_images']}**")
                    st.write("- Structural cleanup: **garbage collection + deflate streams**")

                # Download button
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_name = f"optimized_{ts}_{uploaded_file.name}"
                st.download_button(
                    label="⬇️ Download optimized PDF",
                    data=optimized_bytes,
                    file_name=out_name,
                    mime="application/pdf",
                )

else:
    st.info("Upload a PDF to get started.")
