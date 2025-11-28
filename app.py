import io
import time
from datetime import datetime

import streamlit as st
import fitz  # PyMuPDF
from PIL import Image


def recompress_images(doc, quality=75, max_dim=None):
    """
    Recompress images inside a PyMuPDF document.

    quality: JPEG quality (1-95)
    max_dim: max width/height in pixels (int) or None to keep original size
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
                continue  # same image used multiple times
            processed_xrefs.add(xref)

            pix = fitz.Pixmap(doc, xref)

            # Convert to RGB if needed
            if pix.n > 4:  # CMYK or similar
                pix = fitz.Pixmap(fitz.csRGB, pix)

            mode = "RGB" if pix.n in (3, 4) else "L"
            img_pil = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

            # Optional downscale
            if max_dim is not None:
                w, h = img_pil.size
                scale = min(max_dim / float(w), max_dim / float(h), 1.0)
                if scale < 1.0:
                    new_size = (int(w * scale), int(h * scale))
                    img_pil = img_pil.resize(new_size, Image.LANCZOS)

            buf = io.BytesIO()
            # Save as JPEG (even if original was PNG) for better compression of photos
            img_pil = img_pil.convert("RGB")
            img_pil.save(buf, format="JPEG", optimize=True, quality=quality)
            buf_val = buf.getvalue()

            # Update image stream in PDF
            doc.update_stream(xref, stream=buf_val)
            recompressed_images += 1

    return total_images, recompressed_images


def optimize_pdf(pdf_bytes, mode="Medium"):
    """
    Optimize/Compress PDF bytes according to selected mode.
    Returns (optimized_bytes, stats_dict)
    """
    # Tweak settings based on mode
    if mode == "Light (lossless-ish)":
        image_quality = 90
        max_dim = None        # no downscale
        garbage = 2           # remove unreferenced objects
        deflate = True
    elif mode == "Medium":
        image_quality = 80
        max_dim = 2000        # limit large images
        garbage = 3
        deflate = True
    else:  # "Strong"
        image_quality = 65
        max_dim = 1500
        garbage = 4           # aggressive garbage collection
        deflate = True

    input_size = len(pdf_bytes)

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Recompress images
    total_images, recompressed_images = recompress_images(
        doc, quality=image_quality, max_dim=max_dim
    )

    # Save with cleaning, deflate & linearization
    optimized_bytes = doc.tobytes(
        garbage=garbage,
        deflate=deflate,
        clean=True,
        linear=True,
    )

    output_size = len(optimized_bytes)
    ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0.0

    stats = {
        "input_size": input_size,
        "output_size": output_size,
        "reduction_percent": round(ratio, 2),
        "total_images": total_images,
        "recompressed_images": recompressed_images,
        "mode": mode,
    }

    return optimized_bytes, stats


def human_readable_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


# ------------------ Streamlit UI ------------------ #

st.set_page_config(page_title="PDF Optimizer", page_icon="📄", layout="centered")

st.title("📄 Smart PDF Optimizer")
st.caption("Recompress images + clean structure using PyMuPDF, inspired by modern compression research.")

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

if uploaded_file is not None:
    st.write(f"**Original file:** `{uploaded_file.name}`")
    st.write(f"Size: `{human_readable_size(uploaded_file.size)}`")

    if st.button("🚀 Optimize PDF"):
        pdf_bytes = uploaded_file.read()

        with st.spinner("Optimizing your PDF..."):
            start = time.time()
            try:
                optimized_bytes, stats = optimize_pdf(pdf_bytes, mode=mode)
            except Exception as e:
                st.error(f"Error during optimization: {e}")
            else:
                duration = time.time() - start

                st.success("Optimization complete!")

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric(
                        label="Original size",
                        value=human_readable_size(stats["input_size"]),
                    )
                    st.metric(
                        label="Optimized size",
                        value=human_readable_size(stats["output_size"]),
                    )
                with col_b:
                    st.metric(
                        label="Reduction",
                        value=f"{stats['reduction_percent']} %",
                    )
                    st.write(f"Mode: `{stats['mode']}`")
                    st.write(f"Time taken: `{duration:.2f} s`")

                if show_details:
                    st.divider()
                    st.subheader("Details")
                    st.write(f"- Total images found: **{stats['total_images']}**")
                    st.write(f"- Images recompressed: **{stats['recompressed_images']}**")
                    st.write("- Structural cleanup: **garbage collection + deflate streams + linearization**")

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
