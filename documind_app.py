
import os
import re
import uuid
import hashlib
from html import escape
from io import BytesIO
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types
from google.oauth2 import service_account
from google.cloud import firestore

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    ListFlowable, ListItem
)

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

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
    os.getenv("GOOGLE_CLOUD_LOCATION", "asia-south1")
).strip()

MODEL = st.secrets.get(
    "GEMINI_MODEL",
    os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
).strip()

FIRESTORE_DATABASE_ID = st.secrets.get(
    "FIRESTORE_DATABASE_ID",
    "(default)"
).strip()

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="✳",
    layout="wide"
)


# ==========================================
# 2. DASHBOARD STYLING
# ==========================================

st.markdown("""
<style>
.stApp {
    background: #07060b;
    color: #eeeaf6;
}

[data-testid="stHeader"] {
    background: rgba(7,6,11,.92);
}

[data-testid="stSidebar"] {
    background: #0c0a12;
    border-right: 1px solid #292433;
}

.block-container {
    max-width: 1500px;
    padding-top: .7rem;
    padding-bottom: 2rem;
}

h1,h2,h3,h4,p,label,span {
    color: #eeeaf6;
}

.stButton>button,
.stDownloadButton>button {
    background: #302344;
    color: #fff;
    border: 1px solid #66518f;
    border-radius: 7px;
}

.stButton>button:hover,
.stDownloadButton>button:hover {
    border-color: #b79aff;
    color: #fff;
}

[data-testid="stFileUploader"] {
    background: #111019;
    border: 1px dashed #594b73;
    border-radius: 8px;
    padding: 9px;
}

div[data-testid="stMetric"] {
    background: #111019;
    border: 1px solid #292433;
    padding: 10px;
    border-radius: 8px;
}

div[data-testid="stTabs"] button {
    font-size: 12px;
}

.doc-panel {
    background: #111019;
    border: 1px solid #292433;
    border-radius: 11px;
    padding: 14px;
}

.panel-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .11em;
    text-transform: uppercase;
    color: #d9caff;
    margin: 0 0 12px;
}

.muted {
    color: #a29aad;
    font-size: 11px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. GOOGLE CLOUD AUTHENTICATION
# ==========================================

@st.cache_resource
def get_gcp_credentials():

    if "gcp_service_account" not in st.secrets:
        raise ValueError(
            "Google Cloud service account credentials are missing."
        )

    return service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]),
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform"
        ]
    )


@st.cache_resource
def get_vertex_client(project_id, location):

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
def get_firestore_client(project_id):

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


# ==========================================
# 4. FIRESTORE SESSION MANAGEMENT
# ==========================================

def get_session_id():

    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex

    return st.session_state.session_id


def get_document_id(data):

    return hashlib.sha256(data).hexdigest()


# ==========================================
# 5. FIRESTORE DOCUMENT OPERATIONS
# ==========================================

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


# ==========================================
# 6. FIRESTORE CHAT HISTORY
# ==========================================

def save_chat(document_id, role, message):

    db = get_database()

    (
        db.collection("sessions")
        .document(get_session_id())
        .collection("documents")
        .document(document_id)
        .collection("chat_history")
        .add({
            "role": role,
            "message": message,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
    )


def load_chat(document_id):

    db = get_database()

    messages_ref = (
        db.collection("sessions")
        .document(get_session_id())
        .collection("documents")
        .document(document_id)
        .collection("chat_history")
    )

    messages = (
        messages_ref
        .order_by("timestamp")
        .stream()
    )

    return [
        {
            "role": item.to_dict()["role"],
            "content": item.to_dict()["message"]
        }
        for item in messages
    ]


# ==========================================
# 7. FIRESTORE AI RESULTS
# ==========================================

def save_ai_result(document_id, result_type, content):

    db = get_database()

    (
        db.collection("sessions")
        .document(get_session_id())
        .collection("documents")
        .document(document_id)
        .collection("ai_results")
        .document(result_type)
        .set({
            "content": content,
            "generated_at": firestore.SERVER_TIMESTAMP
        })
    )


def load_ai_result(document_id, result_type):

    db = get_database()

    ref = (
        db.collection("sessions")
        .document(get_session_id())
        .collection("documents")
        .document(document_id)
        .collection("ai_results")
        .document(result_type)
    )

    result = ref.get()

    if result.exists:
        return result.to_dict().get("content", "")

    return ""


# ==========================================
# 8. GEMINI AI FUNCTION
# ==========================================

def ask_gemini(data: bytes, prompt: str) -> str:

    client = get_vertex_client(PROJECT_ID, LOCATION)

    pdf_part = types.Part.from_bytes(
        data=data,
        mime_type="application/pdf"
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[pdf_part, prompt]
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text


# ==========================================
# 9. MARKDOWN TO PDF CONVERTER
# ==========================================

def markdown_to_pdf(
    markdown_text: str,
    title: str = "DocuMind Report"
) -> bytes:

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=19 * mm,
        leftMargin=19 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="DocuMind AI"
    )

    base = getSampleStyleSheet()

    styles = {
        "title": ParagraphStyle(
            "DocTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=colors.HexColor("#34244d"),
            alignment=TA_LEFT,
            spaceAfter=12
        ),
        "h1": ParagraphStyle(
            "DocH1",
            parent=base["Heading1"],
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#34244d"),
            spaceBefore=12,
            spaceAfter=7
        ),
        "h2": ParagraphStyle(
            "DocH2",
            parent=base["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#574078"),
            spaceBefore=10,
            spaceAfter=5
        ),
        "h3": ParagraphStyle(
            "DocH3",
            parent=base["Heading3"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#574078"),
            spaceBefore=8,
            spaceAfter=4
        ),
        "body": ParagraphStyle(
            "DocBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#25212b"),
            spaceAfter=6
        ),
        "small": ParagraphStyle(
            "DocSmall",
            parent=base["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#716b78"),
            spaceAfter=5
        )
    }

    def inline_markup(text):

        text = escape(text, quote=False)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(
            r"(?<!\*)\*([^*]+)\*(?!\*)",
            r"<i>\1</i>",
            text
        )
        text = re.sub(
            r"`([^`]+)`",
            r'<font name="Courier">\1</font>',
            text
        )

        return text

    story = [
        Paragraph(escape(title), styles["title"]),
        Paragraph("Generated by DocuMind AI", styles["small"]),
        Spacer(1, 5)
    ]

    paragraph_lines = []
    bullet_items = []

    def flush_paragraph():

        if paragraph_lines:
            joined = " ".join(
                part.strip() for part in paragraph_lines
            )

            story.append(
                Paragraph(
                    inline_markup(joined),
                    styles["body"]
                )
            )

            paragraph_lines.clear()

    def flush_bullets():

        if bullet_items:

            items = [
                ListItem(
                    Paragraph(
                        inline_markup(item),
                        styles["body"]
                    ),
                    leftIndent=10
                )
                for item in bullet_items
            ]

            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                    bulletFontSize=7,
                    spaceAfter=5
                )
            )

            bullet_items.clear()

    for raw_line in markdown_text.replace(
        "\r\n", "\n"
    ).split("\n"):

        line = raw_line.strip()

        if not line:
            flush_paragraph()
            flush_bullets()
            continue

        if re.fullmatch(r"[-*_]{3,}", line):
            flush_paragraph()
            flush_bullets()
            story.append(Spacer(1, 5))
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)

        if heading:

            flush_paragraph()
            flush_bullets()

            level = len(heading.group(1))

            story.append(
                Paragraph(
                    inline_markup(heading.group(2)),
                    styles[f"h{level}"]
                )
            )

            continue

        bullet = re.match(
            r"^(?:[-*+]|\d+[.)])\s+(.+)$",
            line
        )

        if bullet:
            flush_paragraph()
            bullet_items.append(bullet.group(1))
            continue

        if line.startswith("|") and line.endswith("|"):

            flush_paragraph()
            flush_bullets()

            cells = [
                cell.strip()
                for cell in line.strip("|").split("|")
            ]

            table_text = " | ".join(cells)

            if not all(
                re.fullmatch(r":?-{3,}:?", cell or "-")
                for cell in cells
            ):
                story.append(
                    Paragraph(
                        inline_markup(table_text),
                        styles["small"]
                    )
                )

            continue

        paragraph_lines.append(line)

    flush_paragraph()
    flush_bullets()

    doc.build(story)

    return buffer.getvalue()


# ==========================================
# 10. PDF PAGE RENDERING
# ==========================================

def render_pdf_page(
    data: bytes,
    page_index: int,
    scale: float = 1.35
):

    if pdfium is None:
        return None

    document = pdfium.PdfDocument(data)

    page = document[page_index]

    bitmap = page.render(scale=scale)

    return bitmap.to_pil()


# ==========================================
# 11. CHECK HTML DASHBOARD
# ==========================================

if not HTML_FILE.is_file():

    st.error(
        f"Missing {HTML_FILE.name}. "
        "Keep it in the same folder as this Python file."
    )

    st.stop()


# ==========================================
# 12. SIDEBAR AND PDF UPLOAD
# ==========================================

with st.sidebar:

    st.markdown("## ✳ DocuMind AI")

    st.caption("Document intelligence workspace")

    st.caption("Google Cloud · Firestore Enabled")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key="pdf_upload"
    )

    st.divider()

    st.caption(f"AI model: {MODEL}")

    st.caption("Google Cloud · Vertex AI")

    st.caption("Maximum PDF size: 20 MB")


# ==========================================
# 13. PDF VALIDATION
# ==========================================

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

                page_text = [
                    page.extract_text() or ""
                    for page in reader.pages
                ]

                doc_name = uploaded_file.name

        except Exception as exc:

            error_message = f"Could not read PDF: {exc}"


if error_message:
    st.error(error_message)


# ==========================================
# 14. SAVE DOCUMENT TO FIRESTORE
# ==========================================

document_id = None
firestore_available = False

if pdf_bytes:

    document_id = get_document_id(pdf_bytes)

    try:

        document_id = save_document(
            pdf_bytes,
            doc_name,
            len(page_text)
        )

        firestore_available = True

    except Exception as exc:

        st.warning(f"Firestore error: {exc}")


# ==========================================
# 15. LOAD PREVIOUS DOCUMENT RESULTS
# ==========================================

if pdf_bytes:

    if st.session_state.get(
        "active_document_id"
    ) != document_id:

        st.session_state.active_document_id = document_id

        st.session_state.chat_messages = []
        st.session_state.latest_summary = ""
        st.session_state.latest_extraction = ""

        if firestore_available:

            try:

                st.session_state.chat_messages = load_chat(
                    document_id
                )

                st.session_state.latest_summary = (
                    load_ai_result(document_id, "summary")
                )

                st.session_state.latest_extraction = (
                    load_ai_result(document_id, "extraction")
                )

            except Exception as exc:

                st.warning(
                    f"Could not retrieve previous data: {exc}"
                )


# ==========================================
# 16. DASHBOARD BANNER
# ==========================================

template = HTML_FILE.read_text(encoding="utf-8")

safe_doc_name = (
    escape(doc_name)
)

status_text = (
    "DOCUMENT LOADED"
    if pdf_bytes
    else "READY · WAITING FOR PDF"
)

template = template.replace(
    '<div class="pill" id="status-text">READY · WAITING FOR PDF</div>',
    f'<div class="pill" id="status-text">{status_text}</div>'
)

template = template.replace(
    '<div class="metric-value" id="document-name">No document</div>',
    f'<div class="metric-value" id="document-name">{safe_doc_name}</div>'
)

st.components.v1.html(
    template,
    height=430,
    scrolling=False
)


# ==========================================
# 17. TWO-COLUMN WORKSPACE
# ==========================================

left, right = st.columns(
    [1.35, 1],
    gap="large"
)


# ==========================================
# 18. DOCUMENT PREVIEW
# ==========================================

with left:

    st.markdown(
        '<div class="panel-title">DOCUMENT PREVIEW</div>',
        unsafe_allow_html=True
    )

    if pdf_bytes:

        total_pages = len(page_text)

        page_number = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            key=f"page_number_{len(pdf_bytes)}"
        )

        if pdfium is not None:

            try:

                with st.spinner("Rendering PDF page..."):

                    page_image = render_pdf_page(
                        pdf_bytes,
                        int(page_number) - 1,
                        1.25
                    )

                st.image(
                    page_image,
                    use_container_width=True
                )

            except Exception as exc:

                st.error(
                    f"Could not render this PDF page: {exc}"
                )

                st.info(
                    "You can still download the original PDF below."
                )

        else:

            st.warning(
                "Install pypdfium2 to enable the in-app page preview."
            )

        st.download_button(
            "Download original PDF",
            pdf_bytes,
            file_name=doc_name,
            mime="application/pdf",
            key="download_original_pdf"
        )

        m1, m2 = st.columns(2)

        m1.metric("Pages", total_pages)

        m2.metric(
            "File size",
            f"{len(pdf_bytes)/1024:.1f} KB"
        )

    else:

        st.markdown("""
        <div style="
            min-height:390px;
            border:1px dashed #594b73;
            border-radius:10px;
            background:#0d0b13;
            display:flex;
            align-items
