import io
import json
import os
import re
import sqlite3
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape as xml_escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, OpenAI, RateLimitError
import mysql.connector
from mysql.connector import pooling

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# -----------------------------------------------------------------------------
# Configuration / secrets
# -----------------------------------------------------------------------------
load_dotenv()

def get_setting(name: str, default: str = "") -> str:
    """Read a setting from the environment first, then Streamlit secrets (the secrets file may not exist)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


def safe_link(url: str) -> str:
    """Only allow http(s) and mailto links for the contact button."""
    url = (url or "").strip()
    return url if url.lower().startswith(("https://", "http://", "mailto:")) else ""


APP_TITLE = get_setting("APP_TITLE", "QueryPilot AI")
GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = get_setting("GROQ_API_KEY")
CONTACT_URL = safe_link(get_setting("CONTACT_URL"))
CONTACT_LABEL = get_setting("CONTACT_LABEL", "Book a call")
PROMO_TEXT = get_setting("PROMO_TEXT", "Custom dashboards, your own database, your branding and scheduled reports.")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_ROWS_PER_FILE = int(os.getenv("MAX_ROWS_PER_FILE", "100000"))
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "5000"))
MAX_SCHEMA_TABLES = int(os.getenv("MAX_SCHEMA_TABLES", "40"))
MAX_SCHEMA_COLUMNS = int(os.getenv("MAX_SCHEMA_COLUMNS", "60"))
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "25"))

st.set_page_config(page_title=APP_TITLE, page_icon="✦", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Premium UI: design tokens + component styles
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --paper:#F5F6F8; --surface:#FFFFFF; --ink:#101828; --slate:#5B6475; --mute:#8A92A3;
  --line:#E3E6EC; --line-soft:#EEF0F4;
  --accent:#3D5AFE; --accent-ink:#2A3FD1; --accent-soft:#EEF1FF;
  --ok:#12805C; --ok-soft:#E8F6F0; --warn:#9A5B00; --warn-soft:#FFF4E0;
  --shadow-sm:0 1px 2px rgba(16,24,40,.05);
  --shadow-md:0 12px 28px -12px rgba(16,24,40,.18), 0 2px 6px rgba(16,24,40,.05);
}
html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button, h1, h2, h3, h4, h5, h6 {
  font-family:'Manrope', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer, [data-testid="stDeployButton"] { visibility:hidden; }
.block-container { max-width:1160px; padding-top:2.2rem; padding-bottom:7.5rem; }
:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

/* Sidebar */
[data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }
.side-brand { display:flex; align-items:center; gap:12px; padding:2px 0 6px; }
.logo { width:36px; height:36px; border-radius:10px; background:var(--accent); color:#fff; display:grid; place-items:center; font-size:1.1rem; box-shadow:0 8px 16px -6px rgba(61,90,254,.55); }
.side-name { font-weight:800; letter-spacing:-.02em; color:var(--ink); font-size:1.05rem; line-height:1.2; }
.side-tag { color:var(--mute); font-size:.76rem; }
.pill { display:inline-block; padding:3px 10px; margin:0 6px 6px 0; border-radius:8px; background:var(--paper); border:1px solid var(--line); color:var(--ink); font-family:'JetBrains Mono', ui-monospace, monospace; font-size:.76rem; }

/* Hero */
.hero { padding:.2rem 0 1.5rem; }
.hero h1 { font-size:3rem; line-height:1.04; letter-spacing:-.045em; font-weight:800; color:var(--ink); margin:0 0 .75rem; padding:0; }
.hero p { color:var(--slate); max-width:640px; font-size:1.05rem; line-height:1.65; margin:0 0 1.15rem; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; background:var(--surface); border:1px solid var(--line); color:var(--slate); font-size:.8rem; font-weight:600; }
.chip i { width:7px; height:7px; border-radius:50%; background:var(--mute); display:inline-block; }
.chip.ok { background:var(--ok-soft); border-color:#CDEBDD; color:var(--ok); }
.chip.ok i { background:#1FAE7C; box-shadow:0 0 0 3px rgba(31,174,124,.2); }
.chip.warn { background:var(--warn-soft); border-color:#F6DDB0; color:var(--warn); }
.chip.warn i { background:#E59A1B; }

/* Stat strip */
.statbar { display:grid; grid-template-columns:repeat(4,1fr); background:var(--surface); border:1px solid var(--line); border-radius:16px; box-shadow:var(--shadow-sm); overflow:hidden; margin:.2rem 0 1.5rem; }
.stat { padding:16px 22px; }
.stat + .stat { border-left:1px solid var(--line); }
.stat-label { color:var(--mute); font-size:.78rem; font-weight:600; margin-bottom:4px; }
.stat-value { color:var(--ink); font-size:1.6rem; font-weight:800; letter-spacing:-.02em; font-variant-numeric:tabular-nums; }
@media (max-width:760px) {
  .statbar { grid-template-columns:repeat(2,1fr); }
  .stat:nth-child(3) { border-left:none; }
  .stat:nth-child(n+3) { border-top:1px solid var(--line); }
  .hero h1 { font-size:2.2rem; }
}

/* Cards, titles, empty states */
.card { background:var(--surface); border:1px solid var(--line); border-radius:16px; padding:18px 20px; box-shadow:var(--shadow-sm); }
.metric-label { color:var(--mute); font-size:.78rem; font-weight:600; }
.metric-value { color:var(--ink); font-size:1.6rem; font-weight:800; letter-spacing:-.02em; margin-top:3px; font-variant-numeric:tabular-nums; }
.section-title { font-size:1.15rem; font-weight:700; color:var(--ink); letter-spacing:-.015em; margin:.2rem 0 .2rem; }
.sub-title { display:flex; align-items:baseline; gap:10px; font-size:.95rem; font-weight:700; color:var(--ink); margin:1.3rem 0 .6rem; }
.sub-title .meta { font-weight:500; color:var(--mute); font-size:.8rem; }
.small-muted { color:var(--mute); font-size:.8rem; }
.empty { background:var(--surface); border:1px dashed #C5CBD6; border-radius:20px; padding:34px 36px; }
.empty h3 { margin:0 0 .4rem; padding:0; font-size:1.4rem; letter-spacing:-.025em; color:var(--ink); }
.empty p { color:var(--slate); max-width:560px; line-height:1.65; margin:0 0 1.2rem; }
.steps { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0; padding:0; list-style:none; counter-reset:step; }
.steps li { counter-increment:step; background:var(--paper); border-radius:14px; padding:16px 18px; color:var(--slate); font-size:.9rem; line-height:1.5; }
.steps li::before { content:counter(step); display:grid; place-items:center; width:24px; height:24px; border-radius:50%; background:var(--ink); color:#fff; font-size:.74rem; font-weight:700; margin-bottom:10px; }
.steps li b { display:block; color:var(--ink); font-size:.95rem; margin-bottom:2px; }
@media (max-width:760px) { .steps { grid-template-columns:1fr; } }
.hist { display:flex; gap:14px; align-items:flex-start; padding:12px 0; border-bottom:1px solid var(--line-soft); color:var(--ink); line-height:1.5; }
.hist-n { flex:none; width:26px; height:26px; border-radius:8px; background:var(--paper); border:1px solid var(--line); display:grid; place-items:center; font-size:.75rem; font-weight:700; color:var(--slate); }
.footer { color:var(--mute); font-size:.78rem; padding:1.4rem 0 0; }

.cta, .cta:visited { display:inline-flex; align-items:center; padding:9px 16px; border-radius:12px; background:var(--ink); color:#fff !important; font-weight:700; font-size:.88rem; text-decoration:none !important; box-shadow:var(--shadow-sm); }
.cta:hover { background:#1D2939; }
.hero-actions { display:flex; flex-wrap:wrap; align-items:center; gap:14px; margin:0 0 1.1rem; }
.promo { margin-top:.4rem; }
.promo-title { font-weight:800; color:var(--ink); letter-spacing:-.01em; margin-bottom:4px; }
.promo .small-muted { margin-bottom:12px; line-height:1.5; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap:6px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height:44px; padding:0 14px; color:var(--slate); font-weight:600; background:transparent; }
.stTabs [aria-selected="true"] { color:var(--ink); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--accent); height:2px; }
.stTabs [data-baseweb="tab-border"] { display:none; }

/* Buttons */
.stButton > button, .stDownloadButton > button { border-radius:12px; border:1px solid var(--line); background:var(--surface); color:var(--ink); font-weight:600; padding:.55rem 1rem; box-shadow:var(--shadow-sm); transition:border-color .15s ease, box-shadow .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--accent); color:var(--accent-ink); box-shadow:0 6px 16px -8px rgba(61,90,254,.45); }
.stButton > button p, .stDownloadButton > button p { color:inherit; font-weight:600; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] { background:var(--ink); border-color:var(--ink); color:#fff; }
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background:#1D2939; border-color:#1D2939; color:#fff; }

/* Inputs and uploader */
.stTextInput input, .stNumberInput input { border-radius:10px; }
[data-testid="stFileUploaderDropzone"] { background:var(--paper); border:1px dashed #C5CBD6; border-radius:14px; }
[data-testid="stAlert"] { border-radius:14px; }

/* Expanders, tables, code */
[data-testid="stExpander"] { border:none; background:transparent; box-shadow:none; }
[data-testid="stExpander"] details { border:1px solid var(--line); border-radius:14px; background:var(--surface); box-shadow:var(--shadow-sm); }
[data-testid="stExpander"] summary { font-weight:600; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stCode"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
[data-testid="stCode"] pre, [data-testid="stCode"] code { font-family:'JetBrains Mono', ui-monospace, monospace; font-size:.82rem; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }

/* Chat */
div[data-testid="stChatMessage"] { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.25rem; margin-bottom:.9rem; box-shadow:var(--shadow-sm); }
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background:var(--accent-soft); border-color:#DCE2FF; box-shadow:none; }
[data-testid="stBottom"] > div { background:rgba(245,246,248,.92); backdrop-filter:blur(14px); }
[data-testid="stChatInput"] { padding-top:.5rem; }
[data-testid="stChatInput"] textarea { border-radius:16px !important; border:1px solid var(--line) !important; background:var(--surface) !important; min-height:58px !important; padding:16px 52px 16px 18px !important; font-size:1rem !important; line-height:1.45 !important; box-shadow:var(--shadow-md) !important; }
[data-testid="stChatInput"] textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px rgba(61,90,254,.15), var(--shadow-md) !important; }
[data-testid="stChatInput"] button { width:38px !important; height:38px !important; border-radius:12px !important; margin-right:7px !important; }
</style>
""",
    unsafe_allow_html=True,
)


def render_html(markup: str) -> None:
    """Render custom HTML on a single line so Markdown never mistakes it for a code block."""
    compact = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(compact, unsafe_allow_html=True)


def cta_link(label: Optional[str] = None) -> str:
    """HTML for the contact button. Empty when no CONTACT_URL is configured."""
    if not CONTACT_URL:
        return ""
    attrs = ' target="_blank" rel="noopener noreferrer"' if CONTACT_URL.lower().startswith("http") else ""
    return f'<a class="cta" href="{escape(CONTACT_URL, quote=True)}"{attrs}>{escape(label or CONTACT_LABEL)}</a>'


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
for key, default in {
    "history": [],
    "messages": [],
    "uploaded_db": None,
    "uploaded_tables": [],
    "uploaded_files": [],
    "mysql_pool": None,
    "mysql_config_fingerprint": None,
    "last_sql": "",
    "last_result": None,
    "last_question": "",
    "last_chart": None,
    "current_answer": None,
    "db_mode": "Uploaded files (SQLite)",
    "pending_question": None,
    "is_sample": False,
    "question_count": 0,
    "data_version": 0,
    "metadata_cache": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------------------------------------------------
# AI client
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_ai_client(api_key: str):
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1", timeout=45.0, max_retries=2)


def ask_ai(prompt: str) -> str:
    client = get_ai_client(GROQ_API_KEY)
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured. Add it to Streamlit Secrets or .env.")
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a production-grade analytics assistant. "
                        "Never invent tables or columns. For SQL generation, output exactly one read-only SQL statement. "
                        "Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, ATTACH, PRAGMA, "
                        "LOAD DATA, OUTFILE, or other write/administrative statements."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
    except RateLimitError:
        raise RuntimeError("The AI service is rate-limited right now. Please wait a few seconds and try again.")
    except APIConnectionError:
        raise RuntimeError("Couldn't reach the AI service. Check the connection and try again.")
    return (response.choices[0].message.content or "").strip()


# -----------------------------------------------------------------------------
# Safe identifiers / file loaders
# -----------------------------------------------------------------------------
def safe_identifier(value: str, fallback: str = "table") -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value).strip().lower()).strip("_")
    if not value:
        value = fallback
    if value[0].isdigit():
        value = f"t_{value}"
    return value[:63]


def unique_table_name(name: str, existing: set) -> str:
    base = safe_identifier(name)
    candidate = base
    i = 2
    while candidate in existing:
        candidate = f"{base}_{i}"
        i += 1
    return candidate


def unique_columns(columns) -> List[str]:
    """Clean column names and make them unique instead of silently dropping duplicates."""
    used: set = set()
    result = []
    for i, column in enumerate(columns):
        base = safe_identifier(column, f"column_{i}")
        candidate, n = base, 2
        while candidate in used:
            candidate = f"{base}_{n}"
            n += 1
        used.add(candidate)
        result.append(candidate)
    return result


def load_uploaded_files(files) -> Tuple[sqlite3.Connection, List[str], List[str]]:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    tables = []
    file_names = []
    existing = set()

    for uploaded in files:
        raw = uploaded.getvalue()
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            raise ValueError(f"{uploaded.name} is larger than the {MAX_UPLOAD_MB} MB upload limit.")

        suffix = Path(uploaded.name).suffix.lower()
        stem = Path(uploaded.name).stem
        table = unique_table_name(stem, existing)

        if suffix == ".csv":
            df = pd.read_csv(io.BytesIO(raw), nrows=MAX_ROWS_PER_FILE)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(io.BytesIO(raw), nrows=MAX_ROWS_PER_FILE)
        elif suffix == ".json":
            df = pd.read_json(io.BytesIO(raw))
            df = df.head(MAX_ROWS_PER_FILE)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use CSV, XLSX/XLS, or JSON.")

        df.columns = unique_columns(df.columns)
        df.to_sql(table, conn, if_exists="replace", index=False)
        existing.add(table)
        tables.append(table)
        file_names.append(uploaded.name)

    return conn, tables, file_names

SAMPLE_QUESTIONS = [
    "What is the total revenue and average order value?",
    "What is the total revenue by product category?",
    "Show the monthly revenue trend",
    "Who are the top 10 customers by revenue?",
    "Which sales channel has the highest average order value?",
    "Which city generates the most revenue?",
]


def build_sample_tables() -> Dict[str, pd.DataFrame]:
    """Deterministic synthetic store data so anyone can try the app without uploading a file."""
    rng = np.random.default_rng(42)

    first = ["Aarav", "Diya", "Vihaan", "Ananya", "Kabir", "Meera", "Rohan", "Isha", "Arjun", "Sara", "Neel", "Priya", "Karan", "Tara", "Dev", "Zoya"]
    last = ["Sharma", "Verma", "Khan", "Patel", "Singh", "Iyer", "Reddy", "Gupta", "Mehta", "Das"]
    cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Pune", "Jaipur", "Kolkata", "Ahmedabad", "Gurugram"]
    n_customers = 300
    names = [f"{first[i % 16]} {last[(i // 16) % 10]}" + (" Jr." if i >= 160 else "") for i in range(n_customers)]
    signup = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 540, n_customers), unit="D")
    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers + 1),
        "customer_name": names,
        "city": rng.choice(cities, n_customers),
        "segment": rng.choice(["Consumer", "Small Business", "Enterprise"], n_customers, p=[0.6, 0.3, 0.1]),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })

    catalog = {
        "Electronics": [("Wireless Earbuds", 2499), ("Bluetooth Speaker", 3299), ("Smartwatch", 5999), ("Power Bank 20000mAh", 1999), ("Mechanical Keyboard", 4499)],
        "Home & Kitchen": [("Air Fryer", 6499), ("Mixer Grinder", 3799), ("Cookware Set", 2899), ("Vacuum Flask", 899), ("LED Desk Lamp", 1299)],
        "Fashion": [("Running Shoes", 3499), ("Denim Jacket", 2799), ("Cotton Kurta", 1599), ("Backpack", 1899), ("Sunglasses", 1499)],
        "Beauty": [("Vitamin C Serum", 799), ("Sunscreen SPF 50", 599), ("Hair Dryer", 1799), ("Beard Trimmer", 1499)],
        "Sports": [("Yoga Mat", 999), ("Dumbbell Set", 3299), ("Cricket Bat", 2499), ("Badminton Racket", 1799), ("Resistance Bands", 699)],
    }
    rows = [(cat, name, price) for cat, items in catalog.items() for name, price in items]
    products = pd.DataFrame({
        "product_id": np.arange(1, len(rows) + 1),
        "product_name": [r[1] for r in rows],
        "category": [r[0] for r in rows],
        "unit_price": [r[2] for r in rows],
    })

    n_orders = 3000
    month_w = np.array([6, 6, 7, 7, 8, 8, 8, 9, 9, 11, 14, 15], dtype=float)
    months = rng.choice(np.arange(1, 13), n_orders, p=month_w / month_w.sum())
    days = rng.integers(1, 29, n_orders)
    dates = pd.to_datetime(pd.DataFrame({"year": 2025, "month": months, "day": days}))
    cust_w = rng.pareto(1.6, n_customers) + 1
    prod_w = rng.uniform(0.5, 1.5, len(products))
    product_ids = rng.choice(products["product_id"], n_orders, p=prod_w / prod_w.sum())
    quantity = rng.choice([1, 2, 3, 4], n_orders, p=[0.62, 0.25, 0.09, 0.04])
    discount = rng.choice([0, 5, 10, 15, 20], n_orders, p=[0.5, 0.15, 0.15, 0.1, 0.1])
    unit_price = products.set_index("product_id").loc[product_ids, "unit_price"].to_numpy()
    orders = pd.DataFrame({
        "order_date": dates.dt.strftime("%Y-%m-%d"),
        "customer_id": rng.choice(customers["customer_id"], n_orders, p=cust_w / cust_w.sum()),
        "product_id": product_ids,
        "quantity": quantity,
        "discount_pct": discount,
        "revenue": np.round(quantity * unit_price * (1 - discount / 100), 2),
        "channel": rng.choice(["Website", "Mobile App", "Marketplace", "Retail Store"], n_orders, p=[0.34, 0.30, 0.24, 0.12]),
        "status": rng.choice(["Delivered", "Shipped", "Returned"], n_orders, p=[0.86, 0.07, 0.07]),
    }).sort_values("order_date", kind="stable").reset_index(drop=True)
    orders.insert(0, "order_id", np.arange(1001, 1001 + n_orders))
    return {"customers": customers, "products": products, "orders": orders}


def load_sample_data() -> Tuple[sqlite3.Connection, List[str], List[str]]:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    tables = []
    for name, frame in build_sample_tables().items():
        frame.to_sql(name, conn, if_exists="replace", index=False)
        tables.append(name)
    return conn, tables, ["sample_store_data"]


def activate_sqlite(conn: sqlite3.Connection, tables: List[str], names: List[str], is_sample: bool = False) -> None:
    """Make a loaded SQLite database the active data source and reset the conversation."""
    st.session_state.uploaded_db = conn
    st.session_state.uploaded_tables = tables
    st.session_state.uploaded_files = names
    st.session_state.is_sample = is_sample
    st.session_state.history = []
    st.session_state.messages = []
    st.session_state.last_result = None
    st.session_state.current_answer = None
    st.session_state.metadata_cache = None
    st.session_state.data_version += 1


def load_sample_data_cb() -> None:
    conn, tables, names = load_sample_data()
    activate_sqlite(conn, tables, names, is_sample=True)


# -----------------------------------------------------------------------------
# MySQL connection pooling
# -----------------------------------------------------------------------------
def mysql_fingerprint(config: Dict[str, Any]) -> str:
    import hashlib
    public = {k: v for k, v in config.items() if k != "password"}
    return hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest()


@st.cache_resource(show_spinner=False)
def create_mysql_pool(host: str, port: int, user: str, password: str, database: str, require_tls: bool):
    # Credentials exist only in process memory. They are never written to disk/logs.
    return pooling.MySQLConnectionPool(
        pool_name="querypilot_pool",
        pool_size=5,
        pool_reset_session=True,
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True,
        ssl_disabled=not require_tls,
    )


def get_mysql_connection(config: Dict[str, Any]):
    return create_mysql_pool(
        config["host"], int(config["port"]), config["user"], config["password"],
        config["database"], bool(config["require_tls"])
    ).get_connection()

# -----------------------------------------------------------------------------
# Schema metadata + relevance pruning
# -----------------------------------------------------------------------------
def quote_sqlite_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def quote_mysql_identifier(name: str) -> str:
    return "`" + str(name).replace("`", "``") + "`"


def _text(value: Any) -> str:
    """MySQL connectors sometimes return information_schema text as bytes."""
    return value.decode() if isinstance(value, (bytes, bytearray)) else str(value)


def get_schema_metadata(db_type: str, sqlite_db=None, mysql_config=None) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    if db_type == "SQLite":
        conn = sqlite_db
        if conn is None:
            raise ValueError("SQLite database is not loaded.")
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for (table,) in rows:
            cols = conn.execute(f"PRAGMA table_info({quote_sqlite_identifier(table)})").fetchall()
            count = conn.execute(f"SELECT COUNT(*) FROM {quote_sqlite_identifier(table)}").fetchone()[0]
            columns = []
            for c in cols:
                col_type = c[2] or "TEXT"
                values = None
                if col_type.upper().startswith("TEXT"):
                    # A few example values help the model use real labels (e.g. status = 'Returned').
                    found = conn.execute(
                        f"SELECT DISTINCT {quote_sqlite_identifier(c[1])} FROM {quote_sqlite_identifier(table)} "
                        f"WHERE {quote_sqlite_identifier(c[1])} IS NOT NULL LIMIT 7"
                    ).fetchall()
                    if 0 < len(found) <= 6:
                        values = [str(r[0])[:40] for r in found]
                columns.append({"name": c[1], "type": col_type, "pk": bool(c[5]), "not_null": bool(c[3]), "values": values})
            metadata[table] = {"rows": int(count), "columns": columns}
    else:
        if mysql_config is None:
            raise ValueError("MySQL is not connected.")
        # Two information_schema queries instead of DESCRIBE + COUNT(*) per table. Row counts are estimates.
        conn = get_mysql_connection(mysql_config)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT table_name, table_rows FROM information_schema.tables "
                "WHERE table_schema = DATABASE() ORDER BY table_name"
            )
            table_rows = {_text(r[0]): int(r[1] or 0) for r in cur.fetchall()}
            cur.execute(
                "SELECT table_name, column_name, column_type, is_nullable, column_key "
                "FROM information_schema.columns WHERE table_schema = DATABASE() "
                "ORDER BY table_name, ordinal_position"
            )
            for table, column, col_type, nullable, key in cur.fetchall():
                table = _text(table)
                entry = metadata.setdefault(table, {"rows": table_rows.get(table, 0), "columns": []})
                entry["columns"].append({
                    "name": _text(column), "type": _text(col_type),
                    "pk": _text(key) == "PRI", "not_null": _text(nullable) == "NO",
                })
        finally:
            conn.close()
    return metadata


def get_cached_metadata(db_type: str, sqlite_db=None, mysql_config=None) -> Dict[str, Dict[str, Any]]:
    """Schema is read once per data source instead of on every Streamlit rerun."""
    if db_type == "SQLite":
        key = f"sqlite:{st.session_state.data_version}"
    else:
        key = f"mysql:{mysql_fingerprint(mysql_config)}"
    cache = st.session_state.get("metadata_cache")
    if cache and cache["key"] == key:
        return cache["value"]
    value = get_schema_metadata(db_type, sqlite_db, mysql_config)
    st.session_state.metadata_cache = {"key": key, "value": value}
    return value


def _metadata_signature(metadata: Dict[str, Dict[str, Any]]) -> str:
    payload = json.dumps(metadata, sort_keys=True, default=str)
    import hashlib
    return hashlib.sha256(payload.encode()).hexdigest()


@st.cache_data(ttl=600, show_spinner=False)
def build_schema_index(metadata_json: str) -> List[Dict[str, Any]]:
    metadata = json.loads(metadata_json)
    index = []
    for table, info in metadata.items():
        columns = [c["name"] for c in info["columns"]]
        tokens = set(re.findall(r"[a-z0-9]+", f"{table} {' '.join(columns)}".lower()))
        index.append({"table": table, "columns": columns, "tokens": sorted(tokens), "rows": info["rows"]})
    return index


def rank_schema(question: str, metadata: Dict[str, Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
    signature = _metadata_signature(metadata)
    index = build_schema_index(json.dumps(metadata, sort_keys=True, default=str))
    q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    scored = []
    for item in index:
        table_tokens = set(item["tokens"])
        overlap = len(q_tokens & table_tokens)
        phrase_bonus = 3 if item["table"].lower() in question.lower() else 0
        column_bonus = sum(1 for c in item["columns"] if c.lower() in question.lower())
        score = overlap + phrase_bonus + column_bonus
        scored.append((score, item))
    scored.sort(key=lambda x: (x[0], x[1]["rows"]), reverse=True)
    selected = [item for score, item in scored if score > 0][:top_k]
    if not selected:
        selected = [item for _, item in scored[:min(top_k, len(scored))]]
    return selected


def schema_text(metadata: Dict[str, Dict[str, Any]], selected: Optional[List[Dict[str, Any]]] = None) -> str:
    tables = [item["table"] for item in selected] if selected else list(metadata)[:MAX_SCHEMA_TABLES]
    chunks = []
    for table in tables:
        info = metadata[table]
        cols = []
        for c in info["columns"][:MAX_SCHEMA_COLUMNS]:
            desc = f"{c['name']} ({c['type']})"
            if c.get("values"):
                desc += " [values: " + ", ".join(c["values"]) + "]"
            cols.append(desc)
        chunks.append(f"Table: {table}\nColumns: {'; '.join(cols)}\nApprox rows: {info['rows']}")
    return "\n\n".join(chunks)


# -----------------------------------------------------------------------------
# Read-only SQL validation / execution
# -----------------------------------------------------------------------------
FORBIDDEN_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|LOAD|OUTFILE|INTO\s+OUTFILE|INTO\s+DUMPFILE|CALL|SET|USE|HANDLER)\b",
    re.IGNORECASE,
)
COMMENT_SQL = re.compile(r"(--|/\*|\*/|#)")


def extract_sql(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    return text.strip().rstrip(";").strip() + ";"


def validate_read_only_sql(sql: str) -> Tuple[bool, str]:
    sql = extract_sql(sql)
    body = sql.rstrip(";").strip()
    if not re.match(r"^(SELECT|WITH)\b", body, re.IGNORECASE):
        return False, "Only read-only SELECT/WITH queries are allowed."
    if FORBIDDEN_SQL.search(body):
        return False, "The query contains a blocked write or administrative SQL operation."
    if COMMENT_SQL.search(body):
        return False, "SQL comments are blocked to reduce injection and obfuscation risks."
    if ";" in body:
        return False, "Multiple SQL statements are not allowed."
    return True, ""


def execute_query(db_type: str, sql: str, sqlite_db=None, mysql_config=None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    valid, reason = validate_read_only_sql(sql)
    if not valid:
        return None, reason
    sql = extract_sql(sql)
    try:
        if db_type == "SQLite":
            df = pd.read_sql_query(sql, sqlite_db)
        else:
            conn = get_mysql_connection(mysql_config)
            try:
                df = pd.read_sql_query(sql, conn)
            finally:
                conn.close()
        if len(df) > MAX_RESULT_ROWS:
            df = df.head(MAX_RESULT_ROWS)
        return df, None
    except Exception as exc:
        return None, str(exc)

# -----------------------------------------------------------------------------
# AI functions
# -----------------------------------------------------------------------------
def generate_sql(question: str, db_type: str, metadata: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    selected = rank_schema(question, metadata, top_k=8)
    relevant_schema = schema_text(metadata, selected)
    prompt = f"""
Generate exactly one read-only SQL query for the user's analytics question.

Database dialect: {db_type}

Relevant schema retrieved from the metadata index:
{relevant_schema}

User question:
{question}

Rules:
- Use only tables and columns present in the relevant schema.
- If a needed table/column is not present, do not invent it.
- Return only one SELECT or WITH query.
- Never modify data or database structure.
- Do not use SQL comments.
"""
    sql = extract_sql(ask_ai(prompt))
    return sql, relevant_schema


def repair_sql(question: str, sql: str, error: str, relevant_schema: str, db_type: str) -> str:
    prompt = f"""
Correct this failed read-only SQL query.

Database dialect: {db_type}
Schema:
{relevant_schema}

User question:
{question}

Failed SQL:
{sql}

Database error:
{error}

Return exactly one corrected SELECT or WITH query and nothing else.
Do not use comments or any write/DDL/administrative SQL.
"""
    return extract_sql(ask_ai(prompt))


def explain_result(question: str, sql: str, df: pd.DataFrame) -> str:
    sample = df.head(20).to_json(orient="records", date_format="iso")
    prompt = f"""
Act as an executive analytics assistant.

Question: {question}
SQL: {sql}
Result sample: {sample}
Rows returned: {len(df)}

Give a concise, actionable summary in 3-5 bullet points.
Use only evidence present in the result. Do not invent causes, trends, or numbers.
Mention a useful next analysis only if it follows naturally from the data.
"""
    return ask_ai(prompt)


def recommend_chart(question: str, df: pd.DataFrame) -> Dict[str, Any]:
    columns = [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]
    prompt = f"""
Choose the most useful chart for this analytics result.
Question: {question}
Columns: {json.dumps(columns)}

Return JSON only with this structure:
{{"chart":"bar|line|scatter|pie|histogram|none","x":"column or null","y":"column or null","title":"short title"}}
If the result is unsuitable for a chart, choose none.
"""
    raw = ask_ai(prompt)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"chart": "none", "x": None, "y": None, "title": ""}
    try:
        result = json.loads(match.group(0))
        valid_columns = {str(c) for c in df.columns}
        if result.get("x") not in valid_columns:
            result["x"] = None
        if result.get("y") not in valid_columns:
            result["y"] = None
        if result.get("chart") not in {"bar", "line", "scatter", "pie", "histogram", "none"}:
            result["chart"] = "none"
        return result
    except Exception:
        return {"chart": "none", "x": None, "y": None, "title": ""}


PALETTE = ["#3D5AFE", "#18A999", "#F2A33A", "#8E6CF0", "#E5546B", "#5B6475"]


def style_figure(fig, chart: str):
    """Apply the QueryPilot look to a Plotly figure."""
    font = "Manrope, system-ui, sans-serif"
    fig.update_layout(
        template="simple_white",
        colorway=PALETTE,
        font=dict(family=font, size=13, color="#344054"),
        title=dict(font=dict(size=16, color="#101828"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=56, b=8),
        hoverlabel=dict(bgcolor="#101828", bordercolor="#101828", font=dict(family=font, color="#FFFFFF")),
        legend=dict(title_text=""),
        xaxis=dict(showgrid=False, linecolor="#D0D5DD", tickcolor="#D0D5DD", title_font=dict(size=12, color="#8A92A3")),
        yaxis=dict(gridcolor="#EEF0F4", zeroline=False, linecolor="rgba(0,0,0,0)", title_font=dict(size=12, color="#8A92A3")),
    )
    if chart == "bar":
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.35)
    elif chart == "line":
        fig.update_traces(line=dict(width=3), marker=dict(size=7))
    elif chart == "scatter":
        fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0)))
    elif chart == "pie":
        fig.update_traces(hole=0.58, textposition="inside", textinfo="percent", marker=dict(line=dict(color="#FFFFFF", width=2)))
    elif chart == "histogram":
        fig.update_traces(marker_line_width=0)
        fig.update_layout(bargap=0.06)
    return fig


def make_chart(spec: Dict[str, Any], df: pd.DataFrame):
    chart = spec.get("chart")
    x, y = spec.get("x"), spec.get("y")
    common = dict(title=spec.get("title") or "Analytics", color_discrete_sequence=PALETTE)
    fig = None
    if chart == "bar" and x and y:
        fig = px.bar(df, x=x, y=y, **common)
    elif chart == "line" and x and y:
        fig = px.line(df, x=x, y=y, markers=True, **common)
    elif chart == "scatter" and x and y:
        fig = px.scatter(df, x=x, y=y, **common)
    elif chart == "pie" and x and y:
        fig = px.pie(df, names=x, values=y, **common)
    elif chart == "histogram" and x:
        fig = px.histogram(df, x=x, **common)
    if fig is None:
        return None
    return style_figure(fig, chart)


def render_result_summary(df: pd.DataFrame):
    """Render compact KPI cards for one-row results instead of a meaningless chart."""
    if df is None or df.empty or len(df) != 1:
        return

    numeric_columns = list(df.select_dtypes(include="number").columns)
    if not numeric_columns:
        return

    render_html('<div class="sub-title">Key metrics</div>')
    cols = st.columns(min(len(numeric_columns), 4))
    for i, column in enumerate(numeric_columns[:4]):
        value = df.iloc[0][column]
        if pd.isna(value):
            display_value = "—"
        elif pd.api.types.is_integer(value):
            display_value = f"{int(value):,}"
        else:
            display_value = f"{float(value):,.2f}"
        with cols[i]:
            render_html(
                f'<div class="card"><div class="metric-label">{escape(str(column))}</div>'
                f'<div class="metric-value">{escape(display_value)}</div></div>'
            )


def render_visualization(df: pd.DataFrame, spec: Optional[Dict[str, Any]]):
    """Show KPI cards for single-row results, otherwise the chart chosen when the answer was built."""
    if df is None or df.empty:
        st.info("The query returned no rows.")
        return

    if len(df) == 1:
        render_result_summary(df)
        return

    chart = make_chart(spec or {}, df)
    if chart is not None:
        render_html('<div class="sub-title">Visualization</div>')
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})

# -----------------------------------------------------------------------------
# Export helpers
# -----------------------------------------------------------------------------
def _pdf_plain(value: Any) -> str:
    """ReportLab's built-in Helvetica uses the Windows-1252 character set, so map or replace the rest."""
    text = str(value).replace("₹", "Rs. ")
    return text.encode("cp1252", "replace").decode("cp1252")


def _pdf_markup(value: Any) -> str:
    """Escape text for ReportLab Paragraphs (they parse an XML-like markup) and keep **bold**."""
    escaped = xml_escape(_pdf_plain(value))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def make_pdf(question: str, sql: str, explanation: str, df: pd.DataFrame) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab is not installed.")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"{_pdf_markup(APP_TITLE)} - Analytics Report", styles["Title"]), Spacer(1, 12)]
    story += [Paragraph(f"<b>Question:</b> {_pdf_markup(question)}", styles["BodyText"]), Spacer(1, 8)]
    story += [Paragraph("<b>SQL:</b>", styles["Heading3"]), Paragraph(xml_escape(_pdf_plain(sql)), styles["Code"]), Spacer(1, 8)]
    story += [Paragraph("<b>AI Summary:</b>", styles["Heading3"])]
    for line in explanation.splitlines():
        clean = re.sub(r"^[-•*]\s*", "", line).strip()
        if clean:
            story.append(Paragraph("• " + _pdf_markup(clean), styles["BodyText"]))
    story.append(Spacer(1, 12))

    max_cols, max_rows = 8, 30
    shown = df.iloc[:max_rows, :max_cols]
    table_data = [[_pdf_plain(c)[:28] for c in shown.columns]]
    table_data += [[_pdf_plain(v)[:28] for v in row] for row in shown.itertuples(index=False)]
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeea")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ddddd5")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    if len(df) > max_rows or len(df.columns) > max_cols:
        story += [Spacer(1, 6), Paragraph(
            f"Showing the first {min(len(df), max_rows)} of {len(df):,} rows and {min(len(df.columns), max_cols)} of {len(df.columns)} columns. Export the CSV for the full result.",
            styles["BodyText"],
        )]
    doc.build(story)
    return buffer.getvalue()

# -----------------------------------------------------------------------------
# Sidebar — connection + upload
# -----------------------------------------------------------------------------
with st.sidebar:
    render_html(
        f"""
        <div class="side-brand">
          <div class="logo">✦</div>
          <div><div class="side-name">{escape(APP_TITLE)}</div><div class="side-tag">Secure natural-language analytics</div></div>
        </div>
        """
    )
    st.divider()

    mode = st.radio("Data source", ["Uploaded files (SQLite)", "MySQL database"], index=0)
    st.session_state.db_mode = mode

    if mode == "Uploaded files (SQLite)":
        st.markdown("**Upload your data**")
        files = st.file_uploader(
            "CSV, Excel or JSON",
            type=["csv", "xlsx", "xls", "json"],
            accept_multiple_files=True,
            help=f"Files are loaded into a temporary in-memory SQLite database. Limit: {MAX_UPLOAD_MB} MB per file.",
        )
        if files and st.button("Load files", use_container_width=True, type="primary"):
            try:
                conn, tables, names = load_uploaded_files(files)
                activate_sqlite(conn, tables, names)
                st.success(f"Loaded {len(tables)} table(s).")
            except Exception as exc:
                st.error(str(exc))

        st.button("Try with sample data", use_container_width=True, on_click=load_sample_data_cb)

        if st.session_state.uploaded_tables:
            st.markdown("**Loaded tables**")
            render_html("".join(f'<span class="pill">{escape(t)}</span>' for t in st.session_state.uploaded_tables))

    else:
        st.markdown("**MySQL connection**")
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", min_value=1, max_value=65535, value=3306)
        user = st.text_input("Username")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database")
        require_tls = st.checkbox("Require encrypted TLS", value=True)

        if st.button("Test connection", use_container_width=True):
            if not all([host, user, password, database]):
                st.warning("Enter host, username, password and database.")
            else:
                config = {"host": host, "port": int(port), "user": user, "password": password, "database": database, "require_tls": require_tls}
                try:
                    conn = get_mysql_connection(config)
                    conn.close()
                    st.session_state.mysql_config_fingerprint = mysql_fingerprint(config)
                    st.session_state.mysql_pool = config
                    st.success("Secure connection established.")
                except Exception as exc:
                    st.error(f"Connection failed: {exc}")

    st.divider()
    with st.expander("Security controls"):
        for control in [
            "Read-only SQL enforcement",
            "Single-statement validation",
            "Injection/DDL keyword blocking",
            "TLS option for MySQL",
            "Credentials kept in memory only",
            "Schema metadata pruning",
        ]:
            st.caption(f"✓ {control}")

    if CONTACT_URL:
        render_html(
            f'<div class="card promo"><div class="promo-title">Want this on your own data?</div>'
            f'<div class="small-muted">{escape(PROMO_TEXT)}</div>{cta_link()}</div>'
        )

# -----------------------------------------------------------------------------
# Resolve active database
# -----------------------------------------------------------------------------
if mode == "Uploaded files (SQLite)":
    active_db_type = "SQLite"
    active_db = st.session_state.uploaded_db
    mysql_config = None
else:
    active_db_type = "MySQL"
    active_db = None
    mysql_config = st.session_state.mysql_pool

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
if active_db_type == "SQLite":
    source_ready = active_db is not None
    table_count = len(st.session_state.uploaded_tables)
    status_label = f"{table_count} table{'s' if table_count != 1 else ''} loaded" if source_ready else "Waiting for files"
else:
    source_ready = mysql_config is not None
    status_label = "MySQL connected" if source_ready else "Not connected"

render_html(
    f"""
    <div class="hero">
      <h1>Ask your data. Get answers.</h1>
      <p>Upload a spreadsheet or connect a database, ask questions in plain English, and get the SQL, a chart and a written summary in seconds. No SQL knowledge needed.</p>
      {('<div class="hero-actions">' + cta_link() + '</div>') if CONTACT_URL else ''}
      <div class="chips">
        <span class="chip {'ok' if source_ready else ''}"><i></i>{escape(status_label)}</span>
        <span class="chip"><i></i>Model: {escape(GROQ_MODEL.split('/')[-1])}</span>
        <span class="chip"><i></i>Read-only access</span>
      </div>
    </div>
    """
)

if not GROQ_API_KEY:
    st.warning("Add GROQ_API_KEY to Streamlit Secrets before using AI features.")

# -----------------------------------------------------------------------------
# Empty state / database metadata
# -----------------------------------------------------------------------------
if active_db_type == "SQLite" and active_db is None:
    render_html(
        """
        <div class="empty">
          <h3>Bring your data to start asking</h3>
          <p>Upload CSV, Excel or JSON files from the sidebar, or load the sample store data to see how it works. Each file becomes a read-only table in temporary memory.</p>
          <ul class="steps">
            <li><b>Upload files</b>Choose one or more files in the sidebar.</li>
            <li><b>Load them</b>Press Load files to build your tables.</li>
            <li><b>Ask questions</b>Type in plain English, then export the answer.</li>
          </ul>
        </div>
        """
    )
    st.write("")
    st.button("Try with sample data", type="primary", on_click=load_sample_data_cb)
    st.stop()

if active_db_type == "MySQL" and mysql_config is None:
    render_html(
        """
        <div class="empty">
          <h3>Connect your database</h3>
          <p>Enter your host, credentials and database name in the sidebar, then press Test connection. Every generated query is validated as read-only before it runs.</p>
        </div>
        """
    )
    st.stop()

try:
    metadata = get_cached_metadata(active_db_type, active_db, mysql_config)
except Exception as exc:
    st.error(f"Unable to read database metadata: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# Stat strip
# -----------------------------------------------------------------------------
stats = [
    ("Source", active_db_type),
    ("Tables", f"{len(metadata):,}"),
    ("Columns", f"{sum(len(v['columns']) for v in metadata.values()):,}"),
    ("Security", "Read-only"),
]
render_html(
    '<div class="statbar">'
    + "".join(
        f'<div class="stat"><div class="stat-label">{escape(label)}</div>'
        f'<div class="stat-value">{escape(str(value))}</div></div>'
        for label, value in stats
    )
    + "</div>"
)

# -----------------------------------------------------------------------------
# Analysis pipeline + starter questions
# -----------------------------------------------------------------------------
def queue_question(text: str) -> None:
    """Button callback: runs before the next rerun, so the question is picked up at the top of the script."""
    st.session_state.pending_question = text


def starter_questions(meta: Dict[str, Dict[str, Any]]) -> List[str]:
    """Suggest first questions: curated ones for the sample data, schema-based ones otherwise."""
    if st.session_state.is_sample:
        return SAMPLE_QUESTIONS
    if not meta:
        return []
    table, info = next(iter(meta.items()))
    cols = info["columns"]
    numeric = [
        c["name"] for c in cols
        if any(t in c["type"].upper() for t in ("INT", "REAL", "FLOA", "DOUB", "DEC", "NUM"))
        and not re.search(r"(^|_)id$", c["name"])
    ]
    text = [
        c["name"] for c in cols
        if any(t in c["type"].upper() for t in ("CHAR", "TEXT", "CLOB", "ENUM"))
        and not re.search(r"(^|_)id$", c["name"])
    ]
    prompts = [f"How many rows are in {table}?"]
    if text:
        prompts.append(f"Which {text[0]} values appear most often in {table}?")
    if numeric:
        prompts.append(f"What is the average {numeric[0]} in {table}?")
    if len(prompts) < 3:
        prompts.append(f"Show me the first 10 rows of {table}")
    return prompts[:3]


def run_analysis(question: str, status) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Question -> SQL -> result -> summary -> chart choice. Returns (answer, error_message)."""
    status.update(label="Writing the SQL query…")
    sql, relevant_schema = generate_sql(question, active_db_type, metadata)
    valid, reason = validate_read_only_sql(sql)
    if not valid:
        return None, reason

    status.update(label="Running the query…")
    df, error = execute_query(active_db_type, sql, active_db, mysql_config)
    was_corrected = False
    if error:
        status.update(label="Fixing the query…")
        corrected = repair_sql(question, sql, error, relevant_schema, active_db_type)
        valid2, reason2 = validate_read_only_sql(corrected)
        if not valid2:
            return None, f"The generated correction was blocked: {reason2}"
        df, error2 = execute_query(active_db_type, corrected, active_db, mysql_config)
        if error2:
            return None, f"The query could not be executed safely: {error2}"
        sql = corrected
        was_corrected = True

    status.update(label="Summarizing the result…")
    explanation = explain_result(question, sql, df)

    # Pick the chart once, here, so earlier answers never call the model again on rerun.
    chart_spec = None
    if df is not None and len(df) > 1:
        status.update(label="Choosing a chart…")
        chart_spec = recommend_chart(question, df)

    answer = {
        "id": len(st.session_state.messages),
        "question": question,
        "sql": sql,
        "df": df,
        "explanation": explanation,
        "was_corrected": was_corrected,
        "chart_spec": chart_spec,
    }
    return answer, None


# Created before the tabs so a question can be answered inside the chat tab.
# At the root of the script, Streamlit still pins the composer to the bottom.
question = st.chat_input("Ask something about your data…") or st.session_state.pop("pending_question", None)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_schema, tab_history = st.tabs(["Ask data", "Schema", "History"])


def render_answer_detail(answer):
    """Render one complete result: summary, chart, table, SQL, exports."""
    df = answer["df"]

    if answer.get("was_corrected"):
        render_html('<div class="chips"><span class="chip warn"><i></i>SQL was corrected automatically after the first attempt failed</span></div>')

    render_html('<div class="sub-title">Summary</div>')
    with st.container(border=True):
        st.markdown(answer["explanation"])

    render_visualization(df, answer.get("chart_spec"))

    render_html(f'<div class="sub-title">Results<span class="meta">{len(df):,} rows, {len(df.columns)} columns</span></div>')
    st.dataframe(df, use_container_width=True, height=min(360, 35 * (len(df) + 1) + 3))

    with st.expander("View generated SQL", expanded=False):
        st.code(answer["sql"], language="sql")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    export_cols = st.columns(2 if REPORTLAB_AVAILABLE else 1)
    with export_cols[0]:
        st.download_button(
            "Export CSV",
            data=csv_bytes,
            file_name="query_result.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"csv_{answer['id']}",
        )

    if REPORTLAB_AVAILABLE:
        # Build the PDF once per answer instead of on every rerun.
        if "pdf_bytes" not in answer:
            answer["pdf_bytes"] = make_pdf(answer["question"], answer["sql"], answer["explanation"], df)
        with export_cols[1]:
            st.download_button(
                "Export PDF report",
                data=answer["pdf_bytes"],
                file_name="analytics_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_{answer['id']}",
            )


with tab_chat:
    if not st.session_state.messages and not question:
        render_html(
            '<div class="section-title">What would you like to know?</div>'
            '<div class="small-muted">Pick a starting point, or type your own question below.</div>'
        )
        starters = starter_questions(metadata)
        if starters:
            for start in range(0, len(starters), 3):
                row = starters[start:start + 3]
                for (offset, prompt), col in zip(enumerate(row), st.columns(len(row))):
                    with col:
                        st.button(prompt, key=f"starter_{start + offset}", on_click=queue_question, args=(prompt,), use_container_width=True)

    # Completed answers live in session state, so earlier results stay visible like a chat thread.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["content"])
            elif message["role"] == "user":
                st.markdown(message["content"])
            elif message.get("answer"):
                render_answer_detail(message["answer"])

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.history.append(question)
        if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            st.session_state.messages.append({
                "role": "assistant", "error": True,
                "content": f"This demo is limited to {MAX_QUESTIONS_PER_SESSION} question{'s' if MAX_QUESTIONS_PER_SESSION != 1 else ''} per session. Refresh the page to start a new session, or get in touch to run this on your own data.",
            })
            st.rerun()
        st.session_state.question_count += 1
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.status("Understanding your question…", expanded=True) as status:
                try:
                    answer, failure = run_analysis(question, status)
                except Exception as exc:
                    answer, failure = None, str(exc)
                if failure:
                    status.update(label="This analysis could not be completed", state="error", expanded=False)
                else:
                    status.update(label="Analysis ready", state="complete", expanded=False)

        if failure:
            st.session_state.messages.append({"role": "assistant", "content": failure, "error": True})
        else:
            st.session_state.messages.append({"role": "assistant", "content": "", "answer": answer})
        st.rerun()

with tab_schema:
    st.markdown('<div class="section-title">Schema intelligence</div>', unsafe_allow_html=True)
    st.caption("Metadata is indexed and pruned so only relevant tables are sent to the LLM for each question.")

    if active_db_type == "MySQL":
        st.button("Refresh schema", on_click=lambda: st.session_state.update(metadata_cache=None))

    approx = "~" if active_db_type == "MySQL" else ""
    for table, info in metadata.items():
        with st.expander(f"{table}  ·  {approx}{info['rows']:,} rows  ·  {len(info['columns'])} columns"):
            schema_df = pd.DataFrame(info["columns"])
            if "values" in schema_df:
                schema_df["values"] = schema_df["values"].apply(lambda v: ", ".join(v) if isinstance(v, list) else "")
            st.dataframe(schema_df, use_container_width=True, hide_index=True)
            try:
                sample = None
                if active_db_type == "SQLite":
                    sample = pd.read_sql_query(f"SELECT * FROM {quote_sqlite_identifier(table)} LIMIT 5", active_db)
                else:
                    conn = get_mysql_connection(mysql_config)
                    try:
                        sample = pd.read_sql_query(f"SELECT * FROM {quote_mysql_identifier(table)} LIMIT 5", conn)
                    finally:
                        conn.close()
                st.caption("Sample rows")
                st.dataframe(sample, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.caption(f"Sample unavailable: {exc}")

with tab_history:
    st.markdown('<div class="section-title">Query history</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.caption("Your questions will appear here during this session.")
    else:
        render_html(
            "".join(
                f'<div class="hist"><span class="hist-n">{i}</span><span>{escape(" ".join(item.split()))}</span></div>'
                for i, item in enumerate(reversed(st.session_state.history), 1)
            )
        )

render_html(
    f'<div class="footer">{escape(APP_TITLE)}. Read-only analytics. Uploaded files stay in temporary SQLite memory, '
    'and database credentials are never persisted by this application.</div>'
)
