import os
import re
from html import escape
from io import BytesIO
from pathlib import Path
from google.cloud import firestore

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.oauth2 import service_account

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem, KeepTogether
)

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

load_dotenv()
APP_DIR = Path(__file__).resolve().parent
HTML_FILE = APP_DIR / "documind_axon_ui.html"
MAX_FILE_SIZE = 20 * 1024 * 1024
PROJECT_ID = st.secrets.get(
    "GOOGLE_CLOUD_PROJECT",
    os.getenv("GOOGLE_CLOUD_PROJECT", "")
).strip()

LOCATION = st.secrets.get(
    "GOOGLE_CLOUD_LOCATION",
    os.getenv("GOOGLE_CLOUD_LOCATION", "global")
).strip()

MODEL = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
).strip()

FIRESTORE_DATABASE_ID = st.secrets.get(
    "FIRESTORE_DATABASE_ID",
    "(default)"
).strip()

st.set_page_config(page_title="DocuMind AI", page_icon="✳", layout="wide")

st.markdown("""
<style>
.stApp{background:#07060b;color:#eeeaf6}
[data-testid="stHeader"]{background:rgba(7,6,11,.92)}
[data-testid="stSidebar"]{background:#0c0a12;border-right:1px solid #292433}
.block-container{max-width:1500px;padding-top:.7rem;padding-bottom:2rem}
h1,h2,h3,h4,p,label,span{color:#eeeaf6}
.stButton>button,.stDownloadButton>button{background:#302344;color:#fff;border:1px solid #66518f;border-radius:7px}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#b79aff;color:#fff}
[data-testid="stFileUploader"]{background:#111019;border:1px dashed #594b73;border-radius:8px;padding:9px}
div[data-testid="stMetric"]{background:#111019;border:1px solid #292433;padding:10px;border-radius:8px}
div[data-testid="stTabs"] button{font-size:12px}
.doc-panel{background:#111019;border:1px solid #292433;border-radius:11px;padding:14px}
.panel-title{font-size:11px;font-weight:700;letter-spacing:.11em;text-transform:uppercase;color:#d9caff;margin:0 0 12px}
.muted{color:#a29aad;font-size:11px}
</style>
""", unsafe_allow_html=True)

if not HTML_FILE.is_file():
    st.error(f"Missing {HTML_FILE.name}. Keep it in the same folder as this Python file.")
    st.stop()
document_id = None
firestore_available = False
with st.sidebar:
    st.markdown("## ✳ DocuMind AI")
    st.caption("Document intelligence workspace")
    st.caption("Google Cloud · Firestore Enabled")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], key="pdf_upload")
    st.divider()
    st.caption(f"AI model: {MODEL}")
    st.caption("Google Cloud · Vertex AI")
    st.caption("Maximum PDF size: 20 MB")

pdf_bytes = None
page_text = []
doc_name = "No document"
error_message = None
if uploaded_file is not None:
    candidate = uploaded_file.getvalue()
    if not candidate:
        error_message = "The selected file is empty."
    elif len(candidate) > MAX_FILE_SIZE:
        error_message = "The PDF exceeds the 20 MB limit."
    elif not candidate.startswith(b"%PDF-"):
        error_message = "This file does not appear to be a valid PDF."
    else:
        try:
            reader = PdfReader(BytesIO(candidate))
            if reader.is_encrypted:
                error_message = "Encrypted PDFs are not supported."
            elif len(reader.pages) == 0:
                error_message = "The PDF contains no pages."
            else:
                pdf_bytes = candidate
                page_text = [(page.extract_text() or "") for page in reader.pages]
                doc_name = uploaded_file.name
        except Exception as exc:
            error_message = f"Could not read PDF: {exc}"

if error_message:
    st.error(error_message)

# =====================================
# SAVE UPLOADED DOCUMENT TO FIRESTORE
# =====================================

document_id = None
firestore_available = False

if pdf_bytes:

    try:
        document_id = save_document(
            pdf_bytes,
            doc_name,
            len(page_text)
        )

        firestore_available = True

    except Exception as exc:
        st.warning(f"Firestore error: {exc}")
        

# Top dashboard banner is authored in the separate HTML file.
template = HTML_FILE.read_text(encoding="utf-8")
safe_doc_name = doc_name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
status_text = "DOCUMENT LOADED" if pdf_bytes else "READY · WAITING FOR PDF"
template = template.replace(
    '<div class="pill" id="status-text">READY · WAITING FOR PDF</div>',
    f'<div class="pill" id="status-text">{status_text}</div>',
)
template = template.replace(
    '<div class="metric-value" id="document-name">No document</div>',
    f'<div class="metric-value" id="document-name">{safe_doc_name}</div>',
)
st.components.v1.html(template, height=430, scrolling=False)

#@st.cache_resource
# def get_vertex_client(project_id: str, location: str):
#     if not project_id:
#         raise ValueError("Set GOOGLE_CLOUD_PROJECT in your .env file.")
#     return genai.Client(vertexai=True, project=project_id, location=location)


@st.cache_resource
def get_gcp_credentials():

    if "gcp_service_account" not in st.secrets:
        raise ValueError("Google Cloud credentials are missing.")

    return service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )


@st.cache_resource
def get_vertex_client(project_id: str, location: str):

    if not project_id:
        raise ValueError("Google Cloud Project ID is missing.")

    credentials = get_gcp_credentials()

    return genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        credentials=credentials
    )


@st.cache_resource
def get_firestore_client(project_id: str):

    if not project_id:
        raise ValueError("Google Cloud Project ID is missing.")

    credentials = get_gcp_credentials()

    return firestore.Client(
        project=project_id,
        credentials=credentials,
        database=FIRESTORE_DATABASE_ID
    )


def get_database():
    return get_firestore_client(PROJECT_ID)


# =====================================
# DOCUMIND AI - FIRESTORE OPERATIONS
# =====================================

import uuid
import hashlib


def get_session_id():

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    return st.session_state.session_id


def get_document_id(data):

    return hashlib.sha256(data).hexdigest()


def save_document(data, filename, pages):

    db = get_database()

    session_id = get_session_id()
    document_id = get_document_id(data)

    ref = (
        db.collection("sessions")
        .document(session_id)
        .collection("documents")
        .document(document_id)
    )

    if not ref.get().exists:

        ref.set({
            "filename": filename,
            "document_id": document_id,
            "page_count": pages,
            "file_size": len(data),
            "uploaded_at": firestore.SERVER_TIMESTAMP,
            "status": "processed"
        })

    return document_id


def save_chat(document_id, role, message):

    db = get_database()

    db.collection("sessions") \
        .document(get_session_id()) \
        .collection("documents") \
        .document(document_id) \
        .collection("chat_history") \
        .add({
            "role": role,
            "message": message,
            "timestamp": firestore.SERVER_TIMESTAMP
        })


def save_ai_result(document_id, result_type, content):

    db = get_database()

    db.collection("sessions") \
        .document(get_session_id()) \
        .collection("documents") \
        .document(document_id) \
        .collection("ai_results") \
        .document(result_type) \
        .set({
            "content": content,
            "generated_at": firestore.SERVER_TIMESTAMP
        })
    


def ask_gemini(data: bytes, prompt: str) -> str:
    client = get_vertex_client(PROJECT_ID, LOCATION)
    pdf_part = types.Part.from_bytes(data=data, mime_type="application/pdf")
    response = client.models.generate_content(model=MODEL, contents=[pdf_part, prompt])
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text


def markdown_to_pdf(markdown_text: str, title: str = "DocuMind Report") -> bytes:
    """Convert generated Markdown text into a readable, styled PDF."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="DocuMind AI",
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "DocTitle", parent=base_styles["Title"], fontName="Helvetica-Bold",
            fontSize=19, leading=24, textColor=colors.HexColor("#34244d"),
            alignment=TA_LEFT, spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "DocH1", parent=base_styles["Heading1"], fontName="Helvetica-Bold",
            fontSize=16, leading=20, textColor=colors.HexColor("#34244d"),
            spaceBefore=12, spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "DocH2", parent=base_styles["Heading2"], fontName="Helvetica-Bold",
            fontSize=13, leading=17, textColor=colors.HexColor("#574078"),
            spaceBefore=10, spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "DocH3", parent=base_styles["Heading3"], fontName="Helvetica-Bold",
            fontSize=11, leading=14, textColor=colors.HexColor("#574078"),
            spaceBefore=8, spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "DocBody", parent=base_styles["BodyText"], fontName="Helvetica",
            fontSize=9.5, leading=14, textColor=colors.HexColor("#25212b"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "DocSmall", parent=base_styles["BodyText"], fontSize=8,
            leading=11, textColor=colors.HexColor("#716b78"), spaceAfter=5,
        ),
    }

    def inline_markup(text: str) -> str:
        text = escape(text, quote=False)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
        text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
        text = re.sub(r"\[(.+?)\]\((https?://[^)]+)\)", r'<u>\1</u> (\2)', text)
        return text

    story = [
        Paragraph(escape(title), styles["title"]),
        Paragraph("Generated by DocuMind AI", styles["small"]),
        Spacer(1, 5),
    ]

    lines = markdown_text.replace("\r\n", "\n").split("\n")
    paragraph_lines = []
    bullet_items = []

    def flush_paragraph():
        if paragraph_lines:
            joined = " ".join(part.strip() for part in paragraph_lines)
            story.append(Paragraph(inline_markup(joined), styles["body"]))
            paragraph_lines.clear()

    def flush_bullets():
        if bullet_items:
            items = [
                ListItem(Paragraph(inline_markup(item), styles["body"]), leftIndent=10)
                for item in bullet_items
            ]
            story.append(ListFlowable(
                items, bulletType="bullet", start="circle",
                leftIndent=18, bulletFontSize=7, spaceAfter=5
            ))
            bullet_items.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_bullets()
            continue

        # Ignore Markdown horizontal rules.
        if re.fullmatch(r"[-*_]{3,}", line):
            flush_paragraph()
            flush_bullets()
            story.append(Spacer(1, 5))
            continue

        # Render simple Markdown tables.
        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            flush_bullets()
            table_rows = []
            while lines and line.startswith("|") and line.endswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell or "-") for cell in cells):
                    table_rows.append([Paragraph(inline_markup(c), styles["small"]) for c in cells])
                if len(table_rows) >= 30:
                    break
                # Move to the next line in the source list.
                break
            # Tables are handled by the general line parser below when not contiguous.
            # Keep this line readable as a paragraph if a full table parser is not possible.
            cells_text = "  |  ".join(cell.strip() for cell in line.strip("|").split("|"))
            if cells_text and not all(re.fullmatch(r":?-{3,}:?", c or "-") for c in cells_text.split("|")):
                story.append(Paragraph(inline_markup(cells_text), styles["small"]))
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_bullets()
            level = len(heading.group(1))
            story.append(Paragraph(inline_markup(heading.group(2)), styles[f"h{level}"]))
            continue

        bullet = re.match(r"^(?:[-*+]|\d+[.)])\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            bullet_items.append(bullet.group(1))
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    flush_bullets()
    doc.build(story)
    return buffer.getvalue()

def render_pdf_page(data: bytes, page_index: int, scale: float = 1.35):
    """Render a PDF page to a PNG image. This avoids relying on the browser PDF plugin."""
    if pdfium is None:
        return None
    document = pdfium.PdfDocument(data)
    page = document[page_index]
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()

# Two-column work area: actual PDF preview on the left, document tools on the right.
left, right = st.columns([1.35, 1], gap="large")

with left:
    st.markdown('<div class="panel-title">DOCUMENT PREVIEW</div>', unsafe_allow_html=True)
    if pdf_bytes:
        total_pages = len(page_text)
        page_number = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"page_number_{len(pdf_bytes)}",
        )
        # Use a consistent preview scale; no user-facing preview-size control.
        if pdfium is not None:
            try:
                with st.spinner("Rendering PDF page..."):
                    page_image = render_pdf_page(pdf_bytes, int(page_number)-1, 1.25)
                st.image(page_image, use_container_width=True)
            except Exception as exc:
                st.error(f"Could not render this PDF page: {exc}")
                st.info("You can still download the original PDF below.")
        else:
            st.warning("Install pypdfium2 to enable the in-app page preview.")
        st.download_button(
            "Download original PDF",
            pdf_bytes,
            file_name=doc_name,
            mime="application/pdf",
            key="download_original_pdf",
        )
        m1, m2 = st.columns(2)
        m1.metric("Pages", total_pages)
        m2.metric("File size", f"{len(pdf_bytes)/1024:.1f} KB")
    else:
        st.markdown("""
        <div style="min-height:390px;border:1px dashed #594b73;border-radius:10px;background:#0d0b13;display:flex;align-items:center;justify-content:center;text-align:center;padding:28px">
          <div><div style="font-size:38px;color:#b79aff">▤</div>
          <div style="font-weight:700;font-size:13px;margin:10px 0">YOUR DOCUMENT WILL APPEAR HERE</div>
          <div style="color:#a29aad;font-size:11px;line-height:1.7">Upload a PDF from the left sidebar.<br>The selected page will render in this preview panel.</div></div>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-title">DOCUMENT WORKSPACE</div>', unsafe_allow_html=True)
    if not pdf_bytes:
        st.info("Upload a PDF to activate chat, summaries, extraction and search.")
    else:
        st.caption(f"CURRENT FILE · {doc_name}")
        tabs = st.tabs(["Ask AI", "Summarize", "Extract", "Search"])

        with tabs[0]:
            st.caption("Ask questions using the contents of the uploaded PDF.")
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            for message in st.session_state.chat_messages[-8:]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            with st.form("chat_form", clear_on_submit=True):
                question = st.text_input("Your question", placeholder="What is the main objective?")
                send = st.form_submit_button("Send question", use_container_width=True)
            if send and question.strip():
                st.session_state.chat_messages.append({"role": "user", "content": question})
                try:
                    with st.spinner("Gemini is reading the document..."):
                        history = "\n".join(
                            f'{m["role"]}: {m["content"]}'
                            for m in st.session_state.chat_messages[-8:]
                        )
                        answer = ask_gemini(pdf_bytes, f"""You are DocuMind Assistant.
Answer using only the attached PDF.
Conversation:
{history}
Question: {question}
Be clear and accurate. Cite page numbers only if reliable. If the PDF does not contain the answer, say so. Do not invent facts.""")
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    st.rerun()
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()

        with tabs[1]:
            summary_type = st.selectbox("Summary format", ["Short summary", "Detailed summary", "Key points"])
            if st.button("Generate summary", use_container_width=True):
                try:
                    with st.spinner("Generating summary..."):
                        result = ask_gemini(pdf_bytes, f"""Create a {summary_type} of this PDF.
Include topic, purpose, methodology, key findings and conclusion where available.
Use only supported information and identify missing details.""")
                    st.session_state["latest_summary"] = result
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
            if st.session_state.get("latest_summary"):
                st.markdown("#### Generated summary")
                with st.container(height=480, border=True):
                    st.markdown(st.session_state["latest_summary"])
                summary_pdf = markdown_to_pdf(
                    st.session_state["latest_summary"], "DocuMind Summary"
                )
                st.download_button(
                    "Download summary as PDF",
                    data=summary_pdf,
                    file_name="DocuMind_Summary.pdf",
                    mime="application/pdf",
                    key="download_summary_pdf",
                    use_container_width=True,
                )

        with tabs[2]:
            extraction_type = st.selectbox("Information to extract", [
                "Research objectives and methodology",
                "Key findings and conclusions",
                "Important dates, names and entities",
                "Complete document insights",
            ])
            if st.button("Extract information", use_container_width=True):
                try:
                    with st.spinner("Extracting information..."):
                        result = ask_gemini(pdf_bytes, f"""Extract {extraction_type} from this PDF.
Organise with headings and tables where useful. Do not invent missing information.""")
                    st.session_state["latest_extraction"] = result
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
            if st.session_state.get("latest_extraction"):
                st.markdown("#### Extracted information")
                with st.container(height=480, border=True):
                    st.markdown(st.session_state["latest_extraction"])
                extraction_pdf = markdown_to_pdf(
                    st.session_state["latest_extraction"], "DocuMind Extracted Information"
                )
                st.download_button(
                    "Download extraction as PDF",
                    data=extraction_pdf,
                    file_name="DocuMind_Extraction.pdf",
                    mime="application/pdf",
                    key="download_extraction_pdf",
                    use_container_width=True,
                )

        with tabs[3]:
            query = st.text_input("Search document text", placeholder="Enter a word or phrase")
            if query:
                matches = [
                    (i+1, text) for i, text in enumerate(page_text)
                    if query.casefold() in text.casefold()
                ]
                if not matches:
                    st.warning("No matches found. Scanned PDFs may require OCR.")
                else:
                    st.success(f"Found matches on {len(matches)} page(s).")
                    for page_no, text in matches:
                        pos = text.casefold().find(query.casefold())
                        excerpt = text[max(0, pos-160):pos+len(query)+220].replace("\n", " ")
                        with st.expander(f"Page {page_no}"):
                            st.write(excerpt)

st.markdown("---")
st.caption(f"DocuMind AI · Vertex AI · {MODEL} · Session-based document processing")
