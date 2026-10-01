# QueryPilot AI

Ask questions about your data in plain English. Upload a CSV, Excel or JSON file (or connect a MySQL database) and get the SQL, an interactive chart, a written summary, and CSV/PDF exports in seconds. No SQL knowledge needed.

> Add a screenshot or a 60-second demo GIF here. It is the first thing a client looks at.

**Live demo:** _add your Streamlit Cloud link_

## What it does

- Natural language to SQL (Groq LLM), shown transparently so results can be verified
- Auto-chosen chart plus an executive summary for every answer
- Upload CSV / Excel / JSON, or connect MySQL
- One-click sample store data for instant demos
- Read-only query validation, CSV and PDF export

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then add your GROQ_API_KEY
streamlit run app.py
```

Put `config.toml` in a `.streamlit/` folder next to `app.py` for the theme.

## Deploy free on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, `README.md` and `.streamlit/config.toml` to a GitHub repo (not `.env`).
2. Go to share.streamlit.io, sign in with GitHub, choose **Create app**, pick the repo and `app.py`.
3. Open **Advanced settings > Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_key"
   CONTACT_URL = "https://calendly.com/your-name/intro"
   ```
4. Deploy, then open the link in a private window and run the six sample questions once.

## Configuration

| Setting | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | LLM access | required |
| `GROQ_MODEL` | Model name | `openai/gpt-oss-120b` |
| `CONTACT_URL` | Adds the Book a call button (http, https or mailto) | none |
| `CONTACT_LABEL` | Button text | Book a call |
| `APP_TITLE` | Name shown in the app and PDF | QueryPilot AI |
| `PROMO_TEXT` | Sidebar pitch text | custom dashboards... |
| `MAX_QUESTIONS_PER_SESSION` | Demo question cap | 25 |

## Sample questions

Load the sample data and try: total revenue and average order value, revenue by product category, monthly revenue trend, top 10 customers, average order value by channel, revenue by city.

## Available for custom work

Connect to your own database, branded dashboards, scheduled email reports, and more. Get in touch via the button in the app.
