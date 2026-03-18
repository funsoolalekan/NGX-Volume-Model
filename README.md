# 📊 NGX Volume Surge Tracker — Streamlit Dashboard

A real-time volume surge detection dashboard for all equities listed on the
Nigerian Stock Exchange (NGX), with Telegram alerts and interactive charts.

---

## 🚀 Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Configure Telegram
```bash
cp .env.example .env
# Edit .env with your bot token and chat ID
```
You can also enter your credentials directly in the app sidebar.

### 3. Launch the app
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`

---

## 🎯 Features

| Feature | Details |
|---|---|
| **Live scraping** | Pulls all NGX equities from ngxgroup.com |
| **Surge detection** | Flags tickers where today's volume > 3× the 20-day average |
| **Interactive table** | Searchable, sortable, with surge highlighting |
| **Volume charts** | Multi-ticker history + today vs average bar chart |
| **Telegram alerts** | Instant notification when surges are found |
| **Configurable** | Adjust multiplier and rolling window from the sidebar |
| **Auto-refresh** | Optional 30-minute auto-refresh during market hours |

---

## 📁 Files

```
ngx_streamlit/
├── app.py                       # Main Streamlit app
├── requirements.txt             # Dependencies
├── .env.example                 # Credentials template
├── .env                         # Your credentials (create this)
├── ngx_volume_history.json      # Auto-created: volume history cache
```

---

## ⚠️ Scraper Note

If NGX changes their site layout or uses JavaScript rendering, the scraper
may return empty results. Check the error message shown in the app and
update the column mapping in `scrape_ngx()` if needed.
