"""
NGX Volume Surge Tracker
Run with: streamlit run app.py

Requires:
    pip install streamlit plotly pandas numpy requests beautifulsoup4
                python-dotenv scikit-learn joblib apscheduler pytz lxml
"""

import os, json, time, logging, threading
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import pytz

# ── Optional ML imports ───────────────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    ML_OK = True
except ImportError:
    ML_OK = False

load_dotenv()

# ─────────────────────────── Page Setup ──────────────────────────────────────
st.set_page_config(
    page_title="NGX Volume Tracker",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────── Constants ───────────────────────────────────────
_HERE        = Path(__file__).parent.resolve()
DATA_FILE    = _HERE / "ngx_volume_history.json"
CSV_FILE     = _HERE / "ngx_clean.csv"
NGX_URL      = "https://afx.kwayisi.org/ngx/"
NGX_TZ       = pytz.timezone("Africa/Lagos")
SCAN_MINS    = 30
MC_PATHS     = 1000
MC_HORIZON   = 20
SURGE_MULT   = 3.0
MODEL_DIR    = _HERE

REGIME_META = {
    "Uptrend":      {"icon": "🟢", "hex": "#00C853", "meaning": "Breakout likely ✅",      "action": "BUY / HOLD"},
    "Accumulation": {"icon": "🔵", "hex": "#2196F3", "meaning": "Smart money entering ✅", "action": "WATCH / ACCUMULATE"},
    "Downtrend":    {"icon": "🟠", "hex": "#FF6D00", "meaning": "Exit liquidity ❌",        "action": "REDUCE / AVOID"},
    "Panic":        {"icon": "🔴", "hex": "#D50000", "meaning": "Forced selling ❌",        "action": "STAY OUT"},
}

ML_FEATURES = [
    'vol_ratio_log', 'day_ret', 'day_ret_abs',
    'pre20_ret', 'pre5_ret', 'pre_vol_avg',
    'vs_ma20', 'vs_ma50', 'trend_accel',
]

# ─────────────────────────── History Helpers ─────────────────────────────────
def load_history():
    if DATA_FILE.exists():
        with open(DATA_FILE) as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(DATA_FILE, "w") as f:
        json.dump(history, f, indent=2)

# ─────────────────────────── Scraper ─────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def scrape_ngx():
    def fetch_page(url):
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
        try:
            session = requests.Session()
            session.headers.update(headers)
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception:
            return None

    def clean_number(series):
        return (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^0-9.]", "", regex=True)
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

    def parse_page(html):
        try:
            tables = pd.read_html(html)
        except Exception:
            return pd.DataFrame()
        for table in sorted(tables, key=len, reverse=True):
            cols = [str(c).lower() for c in table.columns]
            if any("ticker" in c for c in cols) and any("vol" in c for c in cols):
                df = table.copy()
                rename = {}
                for col in df.columns:
                    lc = col.lower()
                    if "ticker" in lc:                   rename[col] = "ticker"
                    elif "name" in lc:                   rename[col] = "name"
                    elif "vol" in lc:                    rename[col] = "volume"
                    elif "price" in lc or "close" in lc: rename[col] = "close"
                df = df.rename(columns=rename)
                return pd.DataFrame({
                    "ticker": df["ticker"].astype(str).str.strip(),
                    "name":   df["name"].astype(str).str.strip() if "name" in df.columns else "",
                    "volume": clean_number(df["volume"]).astype(int),
                    "close":  clean_number(df["close"]) if "close" in df.columns else 0.0,
                    "date":   str(date.today()),
                })
        return pd.DataFrame()

    def get_total_pages(html):
        import re
        text = BeautifulSoup(html, "html.parser").get_text()
        match = re.search(r"Page\s+\d+\s+of\s+(\d+)", text, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    html1 = fetch_page(NGX_URL)
    if not html1:
        st.session_state["error"] = "Could not connect to afx.kwayisi.org"
        return pd.DataFrame()

    total_pages = get_total_pages(html1)
    all_pages   = [parse_page(html1)]
    for page_num in range(2, total_pages + 1):
        html = fetch_page(f"{NGX_URL}?page={page_num}")
        if html:
            all_pages.append(parse_page(html))

    combined = pd.concat([p for p in all_pages if not p.empty], ignore_index=True)
    if combined.empty:
        st.session_state["error"] = "No data found on any page."
        return pd.DataFrame()

    combined = combined[combined["ticker"].str.match(r"^[A-Z]{2,12}$", na=False)]
    combined = combined[combined["volume"] > 0].reset_index(drop=True)
    st.session_state["error"] = None
    return combined


# ─────────────────────────── Surge Detection ─────────────────────────────────
def compute_surges(df, history, multiplier, window):
    today   = str(date.today())
    records = []
    for _, row in df.iterrows():
        ticker    = row["ticker"]
        vol_today = int(row["volume"])
        if ticker not in history:
            history[ticker] = {}
        history[ticker][today] = vol_today
        past_vols = [v for d, v in sorted(history[ticker].items()) if d != today][-window:]
        avg_vol   = sum(past_vols) / len(past_vols) if past_vols else 0
        ratio     = round(vol_today / avg_vol, 2) if avg_vol > 0 else 0
        records.append({
            "ticker":       ticker,
            "name":         row.get("name", ticker),
            "close":        row.get("close", 0),
            "volume":       vol_today,
            "avg_volume":   round(avg_vol),
            "ratio":        ratio,
            "days_tracked": len(past_vols),
            "is_surge":     ratio >= multiplier and len(past_vols) >= 5,
        })
    return pd.DataFrame(records)


    # ── Walk-Forward Backtest ─────────────────────────────────────────────────
# ─────────────────────────── ML Engine ───────────────────────────────────────
@st.cache_resource(show_spinner="Training regime classifier on historical data…")
def load_ml_engine():
    """Load or train the ML regime classifier from ngx_clean.csv."""
    import traceback
    if not ML_OK:
        st.error("scikit-learn not installed. Run: pip install scikit-learn joblib")
        return None
    if not CSV_FILE.exists():
        st.error(f"ngx_clean.csv not found at: {CSV_FILE}")
        return None
    try:
        engine = _MLEngine(str(CSV_FILE), str(MODEL_DIR))
        engine.train()
        return engine
    except Exception as e:
        st.error(f"ML engine error: {e}")
        st.code(traceback.format_exc())
        return None


class _MLEngine:
    def __init__(self, data_path, model_dir):
        self.model_dir = model_dir
        df = pd.read_csv(data_path)
        df['date'] = pd.to_datetime(df['date'], dayfirst=True)
        self.df = df.sort_values(['ticker','date']).reset_index(drop=True)
        self.rf = self.gbm = self.le = None
        self.bt = {}   # backtest params per regime

    def _eng(self, t):
        t = t.copy().sort_values('date').reset_index(drop=True)
        t['log_ret']       = np.log(t['close'] / t['close'].shift(1))
        t['vol_ma20']      = t['volume'].rolling(20).mean()
        t['vol_ratio']     = t['volume'] / t['vol_ma20'].replace(0, np.nan)
        t['price_ma20']    = t['close'].rolling(20).mean()
        t['price_ma50']    = t['close'].rolling(50).mean()
        t['pre20_ret']     = t['log_ret'].rolling(20).sum().shift(1)
        t['pre5_ret']      = t['log_ret'].rolling(5).sum().shift(1)
        t['pre_vol_avg']   = t['vol_ratio'].rolling(5).mean().shift(1)
        t['vs_ma20']       = t['close'] / t['price_ma20'] - 1
        t['vs_ma50']       = t['close'] / t['price_ma50'].replace(0, np.nan) - 1
        t['vol_ratio_log'] = np.log(t['vol_ratio'].clip(lower=1))
        t['day_ret']       = t['log_ret']
        t['day_ret_abs']   = t['log_ret'].abs()
        t['trend_accel']   = t['pre5_ret'] - (t['log_ret'].rolling(20).sum().shift(6).fillna(0) / 4)
        return t

    @staticmethod
    def _label(fwd20, fwd5, pre20, day):
        if fwd20 > 0.07:                      return 'Uptrend'
        if fwd20 < -0.07 and day < -0.02:     return 'Panic'
        if fwd20 < -0.12:                     return 'Panic'
        if abs(fwd20) < 0.05:                 return 'Accumulation'
        if fwd20 < -0.02:                     return 'Downtrend'
        return 'Accumulation'

    def train(self, force=False):
        rf_p  = os.path.join(self.model_dir, 'ngx_rf.pkl')
        gbm_p = os.path.join(self.model_dir, 'ngx_gbm.pkl')
        le_p  = os.path.join(self.model_dir, 'ngx_le.pkl')
        bt_p  = os.path.join(self.model_dir, 'ngx_bt.json')

        # Build surge events for backtest stats
        records = []
        for ticker, grp in self.df.groupby('ticker'):
            t = self._eng(grp)
            mask = (t['vol_ratio'] >= SURGE_MULT) & (t['vol_ma20'] > 0)
            for idx in t.index[mask]:
                pos = t.index.get_loc(idx)
                if pos < 55 or pos + 21 >= len(t): continue
                row = t.loc[idx]
                f5  = t.iloc[pos+1:pos+6]['log_ret'].sum()
                f10 = t.iloc[pos+1:pos+11]['log_ret'].sum()
                f20 = t.iloc[pos+1:pos+21]['log_ret'].sum()
                records.append({**{f: float(row.get(f,0) or 0) for f in ML_FEATURES},
                                 'ticker': ticker, 'date': row.get('date', None),
                                 'fwd5':f5,'fwd10':f10,'fwd20':f20})
        ev = pd.DataFrame(records).dropna(subset=['pre20_ret','vs_ma20'])
        ev['regime'] = ev.apply(lambda r: self._label(r['fwd20'],r['fwd5'],r['pre20_ret'],r['day_ret']),axis=1)
        self.surge_events = ev  # expose for backtest tab

        # Backtest stats
        for reg in ['Uptrend','Accumulation','Downtrend','Panic']:
            sub = ev[ev['regime']==reg]
            self.bt[reg] = {
                'n': len(sub),
                'fwd5_win':  float((sub['fwd5']>0).mean()),
                'fwd10_win': float((sub['fwd10']>0).mean()),
                'fwd20_win': float((sub['fwd20']>0).mean()),
                'fwd5_avg':  float(sub['fwd5'].mean()),
                'fwd10_avg': float(sub['fwd10'].mean()),
                'fwd20_avg': float(sub['fwd20'].mean()),
                'daily_mu':  float(sub['fwd20'].mean()/MC_HORIZON),
                'daily_sig': float(sub['fwd20'].std()/MC_HORIZON**0.5),
                'p5':  float(sub['fwd20'].quantile(0.05)),
                'p50': float(sub['fwd20'].median()),
                'p95': float(sub['fwd20'].quantile(0.95)),
            }
        with open(bt_p,'w') as f: json.dump(self.bt, f)

        # Load cached model or retrain
        if not force and all(os.path.exists(p) for p in [rf_p, gbm_p, le_p]):
            self.rf  = joblib.load(rf_p)
            self.gbm = joblib.load(gbm_p)
            self.le  = joblib.load(le_p)
            return

        X   = ev[ML_FEATURES].fillna(0).values
        y   = ev['regime'].values
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y)
        self.rf  = RandomForestClassifier(n_estimators=600,max_depth=10,
            min_samples_leaf=12,class_weight='balanced',random_state=42,n_jobs=-1)
        self.gbm = GradientBoostingClassifier(n_estimators=300,max_depth=5,
            learning_rate=0.05,subsample=0.8,random_state=42)
        self.rf.fit(X, y_enc)
        self.gbm.fit(X, y_enc)
        joblib.dump(self.rf, rf_p); joblib.dump(self.gbm, gbm_p); joblib.dump(self.le, le_p)

    def classify(self, ticker: str, live_close: float = None, live_vol: float = None):
        """Classify regime for a ticker. Optionally inject live price/volume."""
        t_df = self.df[self.df['ticker']==ticker]
        if len(t_df) < 60: return None
        t = self._eng(t_df)

        # If live data provided, append a synthetic last row
        if live_close and live_vol:
            last = t.iloc[-1].copy()
            last['close']  = live_close
            last['volume'] = live_vol
            # Recompute key features for the injected row
            vm20 = t['volume'].tail(20).mean()
            vr   = live_vol / vm20 if vm20 > 0 else 1
            last['vol_ratio']     = vr
            last['vol_ratio_log'] = np.log(max(vr, 1))
            last['day_ret']       = np.log(live_close / float(t.iloc[-1]['close'])) if float(t.iloc[-1]['close']) > 0 else 0
            last['day_ret_abs']   = abs(last['day_ret'])
            last['vs_ma20']       = live_close / float(t['price_ma20'].iloc[-1]) - 1 if t['price_ma20'].iloc[-1] > 0 else 0
            last['vs_ma50']       = live_close / float(t['price_ma50'].iloc[-1]) - 1 if t['price_ma50'].iloc[-1] > 0 else 0
            last['pre20_ret']     = float(t['pre20_ret'].iloc[-1] or 0)
            last['pre5_ret']      = float(t['pre5_ret'].iloc[-1] or 0)
            last['pre_vol_avg']   = float(t['pre_vol_avg'].iloc[-1] or 0)
            last['trend_accel']   = float(t['trend_accel'].iloc[-1] or 0)
            feat_row = last
        else:
            feat_row = t.iloc[-1]

        feat = np.array([[float(feat_row.get(f, 0) or 0) for f in ML_FEATURES]])
        proba = (self.rf.predict_proba(feat)[0] + self.gbm.predict_proba(feat)[0]) / 2
        classes = self.le.classes_
        regime  = classes[np.argmax(proba)]
        probs   = {c: float(p) for c, p in zip(classes, proba)}

        # Monte Carlo
        bp = self.bt[regime]
        log_rets = np.log(t_df['close']/t_df['close'].shift(1)).dropna().tail(252).values
        t_mu  = float(log_rets.mean()) if len(log_rets) > 20 else 0
        t_sig = float(log_rets.std())  if len(log_rets) > 20 else 0.02
        mu    = 0.4*t_mu  + 0.6*bp['daily_mu']
        sigma = max(0.4*t_sig + 0.6*bp['daily_sig'], 0.005)
        shocks = np.random.normal(mu, sigma, (MC_PATHS, MC_HORIZON))
        ratios = np.hstack([np.ones((MC_PATHS,1)), np.cumprod(1+shocks, axis=1)])
        up_pct = float((ratios[:,-1]>1).mean())
        si     = np.argsort(ratios[:,-1])
        close  = float(live_close or feat_row['close'])
        pct    = {p: close*ratios[si[int(MC_PATHS*p/100)]] for p in [5,25,50,75,95]}

        return {
            'regime': regime, 'probs': probs,
            'ratios': ratios, 'up_pct': up_pct, 'pct': pct,
            'bt': bp, 'close': close,
        }


    def walk_forward_backtest(self, lookback_days=30, min_surge=3.0):
        """
        True walk-forward backtest — covers the full lookback window.
        - Completed: surges with >= 18 days of outcome data (verifiable)
        - Still Unfolding: surges from recent days where 20d window not yet closed
        All predictions made using ONLY data available on the surge day. Zero lookahead.
        """
        import warnings; warnings.filterwarnings("ignore")
        FEATS = [
            'vol_ratio_log','day_ret','day_ret_abs',
            'pre20_ret','pre5_ret','pre_vol_avg',
            'vs_ma20','vs_ma50','trend_accel',
        ]
        latest_date  = self.df['date'].max()
        window_start = latest_date - pd.tseries.offsets.BDay(lookback_days)
        records = []

        for ticker, grp in self.df.groupby('ticker'):
            t = self._eng(grp)

            # ALL surges in the full lookback window — no cutoff
            surge_rows = t[
                (t['vol_ratio'] >= min_surge) &
                (t['date'] >= window_start) &
                (t['vol_ma20'] > 0)
            ]

            for _, row in surge_rows.iterrows():
                idx = t[t['date'] == row['date']].index[0]
                pos = t.index.get_loc(idx)
                if pos < 55: continue

                # ── Predict using ONLY surge-day features — strict no lookahead ──
                feat    = np.array([[float(row.get(f, 0) or 0) for f in FEATS]])
                proba   = (self.rf.predict_proba(feat)[0] + self.gbm.predict_proba(feat)[0]) / 2
                classes = self.le.classes_
                regime  = classes[np.argmax(proba)]
                conf    = float(np.max(proba))
                probs   = {c: float(p) for c, p in zip(classes, proba)}

                # ── How many trading days have elapsed since the surge ──
                days_since   = len(t) - 1 - pos          # trading days in our dataset
                fwd_rows     = t.iloc[pos+1:pos+21]       # up to 20 days of actual data
                days_elapsed = min(days_since, 20)
                complete     = len(fwd_rows) >= 18        # need 18+ days to call it done

                # ── Actual outcome (only what has happened so far) ──
                actual_20d = float(fwd_rows['log_ret'].sum()) if len(fwd_rows) >= 1 else None

                # Direction correct? Only meaningful when near-complete
                if actual_20d is not None and complete:
                    correct = actual_20d > 0 if regime in ['Uptrend','Accumulation'] else actual_20d < 0
                else:
                    correct = None   # still unfolding — no verdict yet

                # Return so far (even for partial periods)
                ret_so_far = float(fwd_rows['log_ret'].sum()) if len(fwd_rows) >= 1 else 0.0

                records.append({
                    'ticker':        ticker,
                    'surge_date':    row['date'],
                    'surge_price':   float(row['close']),
                    'vol_ratio':     float(row['vol_ratio']),
                    'regime':        regime,
                    'confidence':    conf,
                    'probs':         probs,
                    'days_elapsed':  days_elapsed,
                    'days_left':     max(0, 20 - days_elapsed),
                    'current_price': float(t.iloc[-1]['close']),
                    'current_date':  t.iloc[-1]['date'],
                    'actual_20d':    actual_20d if complete else None,
                    'ret_so_far':    ret_so_far,
                    'correct':       correct,
                    'complete':      complete,
                })

        return pd.DataFrame(records).sort_values('surge_date', ascending=False).reset_index(drop=True)


# ─────────────────────────── Telegram ────────────────────────────────────────
def send_telegram(token, chat_id, message, parse_mode="Markdown"):
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": chat_id, "text": message,
            "parse_mode": parse_mode,
        }, timeout=10)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def build_telegram_message(surges_df, multiplier, window, ml_results=None):
    """Build Telegram message — includes ML regime and Monte Carlo if available."""
    lines = [
        f"🚨 *NGX Volume Surge Alert* — {date.today().strftime('%d %b %Y')}",
        f"_{len(surges_df)} ticker(s) exceeded {multiplier}× their {window}-day average_\n",
    ]
    for _, row in surges_df.sort_values('ratio', ascending=False).iterrows():
        ticker = row['ticker']
        lines.append(
            f"📈 *{ticker}* — {row['name']}\n"
            f"   Volume: {int(row['volume']):,}  (×{row['ratio']} the avg)\n"
            f"   {window}d avg: {int(row['avg_volume']):,}   Price: ₦{row['close']}\n"
        )
        # Attach ML regime if available
        if ml_results and ticker in ml_results:
            ml = ml_results[ticker]
            if ml:
                meta  = REGIME_META[ml['regime']]
                bp    = ml['bt']
                pct   = ml['pct']
                close = ml['close']
                lines.append(
                    f"   {meta['icon']} *Regime: {ml['regime']}* — {meta['meaning']}\n"
                    f"   🎯 {meta['action']}\n"
                    f"   MC 20d: {ml['up_pct']*100:.0f}% paths up\n"
                    f"   📊 Bull(95th): {(pct[95][-1]/close-1)*100:+.1f}% | "
                    f"Base(50th): {(pct[50][-1]/close-1)*100:+.1f}% | "
                    f"Bear(5th): {(pct[5][-1]/close-1)*100:+.1f}%\n"
                    f"   Backtest 20d: win {bp['fwd20_win']*100:.0f}% | avg {bp['fwd20_avg']*100:+.1f}%\n"
                )
    return "\n".join(lines)


# ─────────────────────────── Auto-scan Background Thread ─────────────────────
def _is_market_hours():
    now  = datetime.now(NGX_TZ)
    if now.weekday() >= 5: return False
    mins = now.hour*60 + now.minute
    return 600 <= mins <= 870  # 10:00–14:30 WAT


def start_auto_scan():
    """
    Runs in a background daemon thread.
    Every 30 minutes during NGX trading hours:
      1. Scrapes live data
      2. Detects surges
      3. Runs ML classification
      4. Fires Telegram alert if surges found
    """
    def _loop():
        while True:
            time.sleep(SCAN_MINS * 60)
            if not _is_market_hours():
                continue
            token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
            if not token or not chat_id:
                continue
            try:
                # Fetch live data
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(NGX_URL, headers=headers, timeout=30)
                if not resp.ok: continue

                # Quick parse (page 1 only for speed)
                tables = pd.read_html(resp.text)
                raw = pd.DataFrame()
                for tbl in sorted(tables, key=len, reverse=True):
                    cols = [str(c).lower() for c in tbl.columns]
                    if any("ticker" in c for c in cols) and any("vol" in c for c in cols):
                        tbl.columns = [str(c) for c in tbl.columns]
                        rename = {}
                        for col in tbl.columns:
                            lc = col.lower()
                            if "ticker" in lc:                    rename[col]="ticker"
                            elif "name" in lc:                    rename[col]="name"
                            elif "vol" in lc:                     rename[col]="volume"
                            elif "price" in lc or "close" in lc:  rename[col]="close"
                        tbl = tbl.rename(columns=rename)
                        raw = pd.DataFrame({
                            "ticker": tbl["ticker"].astype(str).str.strip(),
                            "name":   tbl.get("name", pd.Series([""] * len(tbl))).astype(str),
                            "volume": pd.to_numeric(tbl["volume"].astype(str).str.replace(",","").str.replace(r"[^0-9.]","",regex=True), errors="coerce").fillna(0).astype(int),
                            "close":  pd.to_numeric(tbl["close"].astype(str).str.replace(",","").str.replace(r"[^0-9.]","",regex=True), errors="coerce").fillna(0) if "close" in tbl.columns else 0.0,
                            "date":   str(date.today()),
                        })
                        raw = raw[raw["ticker"].str.match(r"^[A-Z]{2,12}$", na=False)]
                        raw = raw[raw["volume"] > 0].reset_index(drop=True)
                        break

                if raw.empty: continue

                history = load_history()
                multiplier = float(os.getenv("SURGE_MULTIPLIER", "3.0"))
                window     = int(os.getenv("SURGE_WINDOW", "20"))
                results    = compute_surges(raw, history, multiplier, window)
                save_history(history)

                surges = results[results["is_surge"]]
                if surges.empty: continue

                # ML classification for each surge
                engine = st.session_state.get("ml_engine")
                ml_results = {}
                if engine:
                    for _, row in surges.iterrows():
                        try:
                            ml_results[row['ticker']] = engine.classify(
                                row['ticker'],
                                live_close=row['close'],
                                live_vol=row['volume'],
                            )
                        except Exception:
                            ml_results[row['ticker']] = None

                msg = build_telegram_message(surges, multiplier, window, ml_results)
                send_telegram(token, chat_id, msg)

            except Exception as e:
                logging.error(f"[AutoScan] {e}")

    t = threading.Thread(target=_loop, daemon=True)
    t.start()


# ─────────────────────────── Session State ───────────────────────────────────
if "results"        not in st.session_state: st.session_state["results"]        = pd.DataFrame()
if "history"        not in st.session_state: st.session_state["history"]        = load_history()
if "last_scan"      not in st.session_state: st.session_state["last_scan"]      = None
if "error"          not in st.session_state: st.session_state["error"]          = None
if "auto_scan_on"   not in st.session_state: st.session_state["auto_scan_on"]   = False
if "alerted_today"  not in st.session_state: st.session_state["alerted_today"]  = set()
if "last_scan_day"  not in st.session_state: st.session_state["last_scan_day"]  = None

# Load ML engine once
if "ml_engine" not in st.session_state:
    st.session_state["ml_engine"] = load_ml_engine()

# Reset daily alert tracker at start of new day
today_str = str(date.today())
if st.session_state["last_scan_day"] != today_str:
    st.session_state["alerted_today"]  = set()
    st.session_state["last_scan_day"]  = today_str

# Start background auto-scan thread (only once per app session)
if not st.session_state["auto_scan_on"]:
    start_auto_scan()
    st.session_state["auto_scan_on"] = True


# ─────────────────────────── Sidebar ─────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    st.subheader("Surge Detection")
    multiplier = st.slider("Surge Multiplier (×)", 1.5, 10.0, 3.0, 0.5,
        help="Alert when volume exceeds this multiple of the rolling average")
    window = st.slider("Rolling Window (days)", 5, 50, 20, 5,
        help="Days used to calculate the average volume")

    st.divider()
    st.subheader("📬 Telegram Alerts")
    tg_on      = st.checkbox("Enable alerts", value=bool(os.getenv("TELEGRAM_BOT_TOKEN")))
    tg_token   = st.text_input("Bot Token",  value=os.getenv("TELEGRAM_BOT_TOKEN",""), type="password")
    tg_chat_id = st.text_input("Chat ID",    value=os.getenv("TELEGRAM_CHAT_ID",""))

    st.divider()
    st.subheader("🤖 Auto-Scan")
    now_wat = datetime.now(NGX_TZ)
    market_open = _is_market_hours()
    st.markdown(f"**WAT Time:** {now_wat.strftime('%H:%M')}  "
                f"{'🟢 Market Open' if market_open else '🔴 Market Closed'}")
    st.caption(f"Auto-scan fires every **{SCAN_MINS} min** during NGX hours (10:00–14:30 WAT, Mon–Fri) and sends Telegram alerts automatically.")

    if st.session_state.get("ml_engine"):
        st.success("✅ ML Engine loaded")
    else:
        st.warning("⚠️ ML Engine not available\n(ngx_clean.csv not found)")

    st.divider()
    if st.button("🔄 Retrain ML Model"):
        st.cache_resource.clear()
        st.session_state.pop("ml_engine", None)
        st.rerun()

    st.caption("NGX hours: Mon–Fri, 10:00–14:30 WAT\nData: afx.kwayisi.org/ngx")


# ─────────────────────────── Main Page ───────────────────────────────────────
st.title("📈 NGX Volume Surge Tracker")
st.caption("Nigerian Stock Exchange — Live scrape · ML regime classification · Monte Carlo simulation")

st.divider()

col1, col2 = st.columns([1, 4])
with col1:
    run_scan = st.button("⚡ Run Scan Now", use_container_width=True)
with col2:
    if st.session_state["last_scan"]:
        st.success(f"Last scan: {st.session_state['last_scan'].strftime('%H:%M:%S')}")
    else:
        st.info("No scan yet — click Run Scan Now or wait for auto-scan")

# ─────────────────────────── Scan Logic ──────────────────────────────────────
if run_scan:
    with st.spinner("Fetching NGX data from afx.kwayisi.org…"):
        scrape_ngx.clear()
        raw_df = scrape_ngx()

    if raw_df.empty:
        st.error(f"Could not fetch data: {st.session_state.get('error','Unknown error')}")
    else:
        history = st.session_state["history"]
        results = compute_surges(raw_df, history, multiplier, window)
        save_history(history)

        st.session_state["results"]   = results
        st.session_state["last_scan"] = datetime.now()

        # ── Run ML classification on all surges ───────────────────────────
        surges     = results[results["is_surge"]]
        engine     = st.session_state.get("ml_engine")
        ml_results = {}

        if engine and not surges.empty:
            for _, row in surges.iterrows():
                try:
                    ml_results[row['ticker']] = engine.classify(
                        row['ticker'],
                        live_close=float(row['close']),
                        live_vol=float(row['volume']),
                    )
                except Exception:
                    ml_results[row['ticker']] = None
        st.session_state["ml_results"] = ml_results

        # ── Telegram alert (deduped per day) ──────────────────────────────
        new_surges = surges[~surges['ticker'].isin(st.session_state["alerted_today"])]
        if tg_on and tg_token and tg_chat_id and not new_surges.empty:
            msg  = build_telegram_message(new_surges, multiplier, window, ml_results)
            sent = send_telegram(tg_token, tg_chat_id, msg)
            if sent:
                st.success(f"📬 Telegram alert sent for {len(new_surges)} surge(s)!")
                st.session_state["alerted_today"].update(new_surges['ticker'].tolist())
            else:
                st.warning("Telegram failed — check token and chat ID.")

        st.rerun()


# ─────────────────────────── Results ─────────────────────────────────────────
results    = st.session_state["results"]
ml_results = st.session_state.get("ml_results", {})

if results.empty:
    st.markdown("### Run a scan to see results")
else:
    n_tickers = len(results)
    n_surges  = int(results["is_surge"].sum())
    n_warning = int(((results["ratio"] >= multiplier*0.7) & ~results["is_surge"]).sum())
    top_row   = results.loc[results["ratio"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickers Scanned", n_tickers)
    c2.metric("🚨 Surges",       n_surges)
    c3.metric("⚠️ Near-Surge",   n_warning)
    c4.metric("Top Ratio",       f"{top_row['ratio']}× ({top_row['ticker']})")

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 All Tickers", "🚨 Surge Alerts", "🤖 ML Analysis", "📈 Charts", "🧪 Backtest"])

    # ── Tab 1: All Tickers ────────────────────────────────────────────────────
    with tab1:
        col_a, col_b, col_c = st.columns([3, 1, 1])
        with col_a: search    = st.text_input("🔍 Search", placeholder="e.g. DANGCEM, Access…")
        with col_b: sort_by   = st.selectbox("Sort by", ["ratio","volume","ticker"])
        with col_c: surges_only = st.checkbox("Surges only")

        filtered = results.copy()
        if search:
            filtered = filtered[
                filtered["ticker"].str.contains(search.upper(), na=False) |
                filtered["name"].str.contains(search, case=False, na=False)]
        if surges_only:
            filtered = filtered[filtered["is_surge"]]
        filtered = filtered.sort_values(sort_by, ascending=(sort_by=="ticker"))

        st.caption(f"{len(filtered)} equities shown")
        display = pd.DataFrame({
            "Ticker":         filtered["ticker"],
            "Company":        filtered["name"],
            "Close (₦)":     filtered["close"].apply(lambda x: f"₦{x:,.2f}" if x>0 else "—"),
            "Volume Today":   filtered["volume"].apply(lambda x: f"{x:,}"),
            f"{window}d Avg": filtered["avg_volume"].apply(lambda x: f"{x:,}"),
            "Ratio":          filtered["ratio"].apply(lambda x: f"{x}×"),
            "Alert":          filtered["is_surge"].apply(lambda x: "🚨 SURGE" if x else ""),
            "Days Tracked":   filtered["days_tracked"].apply(lambda x: f"{x}d"),
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    # ── Tab 2: Surge Alerts ───────────────────────────────────────────────────
    with tab2:
        surges = results[results["is_surge"]]
        if surges.empty:
            st.success("✅ No surges detected in this scan.")
        else:
            st.warning(f"🚨 {len(surges)} surge(s) detected!")
            for _, row in surges.sort_values("ratio", ascending=False).iterrows():
                with st.expander(f"🚨 {row['ticker']} — {row['name']}  |  {row['ratio']}× average"):
                    ca, cb, cc = st.columns(3)
                    ca.metric("Today's Volume",   f"{int(row['volume']):,}")
                    cb.metric(f"{window}d Average", f"{int(row['avg_volume']):,}")
                    cc.metric("Surge Ratio",      f"{row['ratio']}×")
                    # ML regime badge
                    ml = ml_results.get(row['ticker'])
                    if ml:
                        meta = REGIME_META[ml['regime']]
                        st.markdown(
                            f"<div style='background:{meta['hex']}18;"
                            f"border-left:4px solid {meta['hex']};"
                            f"border-radius:6px;padding:8px 12px;margin-top:6px'>"
                            f"{meta['icon']} <b>{ml['regime']}</b> — {meta['meaning']} &nbsp;|&nbsp; "
                            f"🎯 {meta['action']} &nbsp;|&nbsp; "
                            f"MC: <b>{ml['up_pct']*100:.0f}%</b> paths up"
                            f"</div>", unsafe_allow_html=True)

            st.divider()
            if tg_on and tg_token and tg_chat_id:
                if st.button("📬 Send Telegram Alert Now"):
                    msg  = build_telegram_message(surges, multiplier, window, ml_results)
                    sent = send_telegram(tg_token, tg_chat_id, msg)
                    st.success("✅ Sent!" if sent else "❌ Failed — check credentials.")
            else:
                st.info("Set up Telegram in the sidebar to enable alerts.")

    # ── Tab 3: ML Analysis ────────────────────────────────────────────────────
    with tab3:
        engine = st.session_state.get("ml_engine")
        if not engine:
            st.error(f"ML Engine not loaded.")
            st.info(f"Looking for CSV at: `{CSV_FILE}`  |  Exists: {CSV_FILE.exists()}")
            st.info(f"scikit-learn available: {ML_OK}")


        elif not ml_results:
            st.info("Run a scan first. ML analysis will appear here for any surges detected.")
        else:
            st.subheader(f"Regime Analysis — {len(ml_results)} surge ticker(s)")

            for ticker, ml in ml_results.items():
                if not ml: continue
                meta  = REGIME_META[ml['regime']]
                bp    = ml['bt']
                pct   = ml['pct']
                close = ml['close']
                paths = ml['ratios']

                with st.expander(f"{meta['icon']} {ticker}  —  {ml['regime']}  |  {ml['up_pct']*100:.0f}% paths up", expanded=True):
                    st.markdown(
                        f"<div style='background:{meta['hex']}18;"
                        f"border-left:5px solid {meta['hex']};"
                        f"border-radius:8px;padding:12px 16px'>"
                        f"<b style='font-size:17px'>{meta['icon']} {ml['regime']}</b>"
                        f"<span style='color:#aaa;margin-left:10px'>— {meta['meaning']}</span><br>"
                        f"<span style='color:#888'>Action: <b style='color:white'>{meta['action']}</b></span>"
                        f"</div>", unsafe_allow_html=True)

                    col_l, col_r = st.columns(2)

                    # Regime probability bar
                    with col_l:
                        st.caption("Regime Probabilities")
                        regs = list(ml['probs'])
                        vals = [ml['probs'][r]*100 for r in regs]
                        rc   = {r: m['hex'] for r,m in REGIME_META.items()}
                        fig  = go.Figure(go.Bar(x=vals, y=regs, orientation='h',
                            marker_color=[rc.get(r,'#888') for r in regs],
                            text=[f"{v:.1f}%" for v in vals], textposition='outside'))
                        fig.update_layout(xaxis=dict(range=[0,100]),
                            yaxis=dict(autorange="reversed"), height=180,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'), margin=dict(l=10,r=50,t=5,b=5))
                        st.plotly_chart(fig, use_container_width=True)

                    # Monte Carlo fan chart
                    with col_r:
                        st.caption(f"Monte Carlo — 20 days, {MC_PATHS} paths")
                        x   = list(range(MC_HORIZON+1))
                        p5  = (pct[5]/close  - 1)*100
                        p25 = (pct[25]/close - 1)*100
                        p50 = (pct[50]/close - 1)*100
                        p75 = (pct[75]/close - 1)*100
                        p95 = (pct[95]/close - 1)*100
                        fig2 = go.Figure()
                        for i in np.random.choice(MC_PATHS, 50, replace=False):
                            fig2.add_trace(go.Scatter(x=x, y=(paths[i]-1)*100,
                                line=dict(color='rgba(255,255,255,0.04)', width=0.7),
                                showlegend=False, hoverinfo='skip'))
                        fig2.add_trace(go.Scatter(x=x+x[::-1],
                            y=list(p95)+list(p5[::-1]),
                            fill='toself', fillcolor='rgba(33,150,243,0.10)',
                            line=dict(color='rgba(0,0,0,0)'), name='5–95th'))
                        fig2.add_trace(go.Scatter(x=x+x[::-1],
                            y=list(p75)+list(p25[::-1]),
                            fill='toself', fillcolor='rgba(33,150,243,0.22)',
                            line=dict(color='rgba(0,0,0,0)'), name='25–75th'))
                        fig2.add_trace(go.Scatter(x=x, y=list(p50),
                            line=dict(color='#2196F3', width=2.5), name='Median'))
                        fig2.add_hline(y=0, line=dict(color='white',width=1,dash='dot'))
                        fig2.update_layout(height=200,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            xaxis=dict(title="Days",color='white',gridcolor='#333'),
                            yaxis=dict(title="Return %",color='white',gridcolor='#333'),
                            legend=dict(orientation='h',y=1.1,bgcolor='rgba(0,0,0,0)',
                                        font=dict(color='white',size=10)),
                            margin=dict(l=10,r=10,t=10,b=10))
                        st.plotly_chart(fig2, use_container_width=True)

                    # Scenario table
                    labels = ['🐂 Bull (95th)','📈 Good (75th)','➡ Base (50th)','📉 Weak (25th)','🐻 Bear (5th)']
                    st.table(pd.DataFrame({
                        'Scenario': labels,
                        'Return 20d': [f"{(pct[p][-1]/close-1)*100:+.1f}%" for p in [95,75,50,25,5]],
                        'Target ₦':  [f"₦{pct[p][-1]:.2f}" for p in [95,75,50,25,5]],
                    }))

                    # Backtest strip
                    ba, bb, bc = st.columns(3)
                    ba.metric("Win Rate 5d",  f"{bp['fwd5_win']*100:.0f}%",  f"avg {bp['fwd5_avg']*100:+.2f}%")
                    bb.metric("Win Rate 10d", f"{bp['fwd10_win']*100:.0f}%", f"avg {bp['fwd10_avg']*100:+.2f}%")
                    bc.metric("Win Rate 20d", f"{bp['fwd20_win']*100:.0f}%", f"avg {bp['fwd20_avg']*100:+.2f}%")

    # ── Tab 4: Charts (your original) ─────────────────────────────────────────
    with tab4:
        history = st.session_state["history"]
        if not history:
            st.info("Run a few scans to build up history for charts.")
        else:
            all_tickers = sorted(history.keys())
            default     = list(results[results["is_surge"]]["ticker"])[:3] or all_tickers[:3]
            selected    = st.multiselect("Select tickers to chart", options=all_tickers,
                default=[t for t in default if t in all_tickers], max_selections=8)

            if not selected:
                st.info("Select at least one ticker above.")
            else:
                st.subheader("Volume History")
                fig = go.Figure()
                colors = ["#00e5a0","#ff4d6d","#60a5fa","#ffb347","#a78bfa","#fb7185","#34d399","#f472b6"]
                for i, ticker in enumerate(selected):
                    dates = sorted(history[ticker].keys())
                    vols  = [history[ticker][d] for d in dates]
                    fig.add_trace(go.Scatter(x=dates, y=vols, name=ticker, mode="lines+markers",
                        line=dict(color=colors[i%len(colors)], width=2), marker=dict(size=5)))
                fig.update_layout(height=400, hovermode="x unified",
                    yaxis=dict(tickformat=","), margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)

                if not results.empty:
                    st.subheader("Today vs Average")
                    chart_data = results[results["ticker"].isin(selected)]
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(y=chart_data["ticker"], x=chart_data["avg_volume"],
                        name=f"{window}d Average", orientation="h", marker_color="steelblue"))
                    fig2.add_trace(go.Bar(y=chart_data["ticker"], x=chart_data["volume"],
                        name="Today", orientation="h",
                        marker_color=["red" if s else "green" for s in chart_data["is_surge"]]))
                    fig2.update_layout(barmode="overlay",
                        height=max(300, len(chart_data)*45),
                        xaxis=dict(tickformat=","), margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 5: Backtest ───────────────────────────────────────────────────────
    with tab5:
        engine = st.session_state.get("ml_engine")
        if not engine:
            st.warning("ML Engine not loaded.")
        else:
            RC = {r: m["hex"] for r, m in REGIME_META.items()}

            st.subheader("🧪 Walk-Forward Backtest")
            st.caption(
                "Finds every volume surge from the past 30 trading days · "
                "Predicts regime using **only data available on the surge day** · "
                "Compares prediction to what actually happened afterward · "
                "Zero lookahead bias."
            )

            # ── Controls ──────────────────────────────────────────────────────
            bc1, bc2 = st.columns(2)
            with bc1:
                bt_lookback = st.slider("Lookback window (trading days)", 10, 60, 30, 5,
                    key="bt_lookback",
                    help="How far back to search for surges. 30 = last 30 trading days (~6 weeks)")
            with bc2:
                bt_min_surge = st.slider("Min surge multiplier (×)", 2.0, 5.0, 3.0, 0.5,
                    key="bt_surge",
                    help="Only include surges above this volume threshold")

            if st.button("▶ Run Walk-Forward Backtest", type="primary"):
                with st.spinner("Running walk-forward backtest — predicting without future data…"):
                    wf = engine.walk_forward_backtest(
                        lookback_days=bt_lookback,
                        min_surge=bt_min_surge,
                    )
                    # ── Overwrite current_price with live scrape data if available ──
                    live = st.session_state.get("results", pd.DataFrame())
                    if not live.empty and not wf.empty:
                        live_prices = live.set_index("ticker")["close"].to_dict()
                        wf["current_price"] = wf["ticker"].map(
                            lambda t: live_prices.get(t, None)
                        ).fillna(wf["current_price"])
                        # Recompute ret_so_far using live price
                        wf["ret_so_far"] = np.log(
                            wf["current_price"] / wf["surge_price"]
                        )
                st.session_state["wf_results"] = wf

            wf = st.session_state.get("wf_results", pd.DataFrame())

            if wf.empty:
                st.info("Click **Run Walk-Forward Backtest** to start.")
            else:
                completed = wf[wf["complete"] == True].copy()
                pending   = wf[wf["complete"] == False].copy()

                # ── Headline metrics ──────────────────────────────────────────
                st.markdown("---")
                n_total   = len(wf)
                n_done    = len(completed)
                n_correct = int(completed["correct"].sum()) if not completed.empty else 0
                acc       = n_correct / n_done if n_done > 0 else 0
                avg_ret   = completed["actual_20d"].mean() * 100 if not completed.empty else 0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Surges Found",       n_total)
                m2.metric("Completed (≥18d)",   n_done)
                m3.metric("Correct Calls",       f"{n_correct}/{n_done}")
                m4.metric("Directional Accuracy",f"{acc*100:.0f}%",
                           "vs 50% coin flip" if acc > 0.5 else "below coin flip")
                m5.metric("Avg Actual Return",   f"{avg_ret:+.2f}%")

                st.markdown("---")

                # ── Per-regime accuracy ───────────────────────────────────────
                st.markdown("#### Accuracy by Regime (no lookahead)")

                if not completed.empty:
                    reg_cols = st.columns(4)
                    for i, reg in enumerate(["Uptrend","Accumulation","Downtrend","Panic"]):
                        sub = completed[completed["regime"] == reg]
                        meta = REGIME_META[reg]
                        with reg_cols[i]:
                            if len(sub) == 0:
                                st.markdown(f"""
                                <div style="background:{meta['hex']}18;border-radius:10px;
                                     border-left:4px solid {meta['hex']};padding:14px;min-height:160px">
                                  <h4 style='margin:0'>{meta['icon']} {reg}</h4>
                                  <p style='color:#888'>No calls in this window</p>
                                </div>""", unsafe_allow_html=True)
                            else:
                                n_sub  = len(sub)
                                n_corr = int(sub["correct"].sum())
                                acc_r  = n_corr / n_sub
                                avg_r  = sub["actual_20d"].mean() * 100
                                # direction label
                                if reg in ["Uptrend","Accumulation"]:
                                    dir_label = "Predicted: UP"
                                else:
                                    dir_label = "Predicted: DOWN"
                                st.markdown(f"""
                                <div style="background:{meta['hex']}18;border-radius:10px;
                                     border-left:4px solid {meta['hex']};padding:14px;min-height:160px">
                                  <h4 style='margin:0'>{meta['icon']} {reg}</h4>
                                  <p style='color:#aaa;font-size:12px;margin:2px 0'>{dir_label}</p>
                                  <b>Calls:</b> {n_sub}<br>
                                  <b>Correct:</b> {n_corr}/{n_sub} = <b>{acc_r*100:.0f}%</b><br>
                                  <b>Avg actual ret:</b> {avg_r:+.2f}%<br>
                                  <b>Action:</b> {meta['action']}
                                </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # ── Accuracy vs return scatter ────────────────────────────────
                if not completed.empty:
                    st.markdown("#### Every Call — Prediction vs Actual Outcome")

                    fig_sc = go.Figure()
                    for reg in ["Uptrend","Accumulation","Downtrend","Panic"]:
                        sub = completed[completed["regime"] == reg]
                        if sub.empty: continue
                        fig_sc.add_trace(go.Scatter(
                            x=sub["surge_date"].dt.strftime("%d/%m/%Y"),
                            y=sub["actual_20d"] * 100,
                            mode="markers",
                            name=f"{REGIME_META[reg]['icon']} {reg}",
                            marker=dict(
                                color=RC[reg], size=14, opacity=0.85,
                                symbol=["circle" if c else "x" for c in sub["correct"]],
                                line=dict(color="white", width=1),
                            ),
                            hovertemplate=(
                                "<b>%{customdata[0]}</b><br>"
                                "Surge: %{x}<br>"
                                "Vol: %{customdata[1]}×<br>"
                                "Predicted: " + reg + "<br>"
                                "Actual 20d: %{y:.1f}%<br>"
                                "Entry: ₦%{customdata[2]}<br>"
                                "Now: ₦%{customdata[3]}<br>"
                                "Correct: %{customdata[4]}<extra></extra>"
                            ),
                            customdata=np.column_stack([
                                sub["ticker"],
                                sub["vol_ratio"].round(1),
                                sub["surge_price"].round(2),
                                sub["current_price"].round(2),
                                sub["correct"].map({True:"✅ Yes", False:"❌ No"}),
                            ]),
                        ))
                    fig_sc.add_hline(y=0, line=dict(color="white", width=1, dash="dot"))
                    fig_sc.update_layout(
                        height=420,
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font=dict(color="white"),
                        xaxis=dict(title="Surge Date", color="white", gridcolor="#2a2a3a"),
                        yaxis=dict(title="Actual 20d Return (%)", color="white", gridcolor="#2a2a3a"),
                        legend=dict(orientation="h", y=1.08, bgcolor="rgba(0,0,0,0)"),
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)
                    st.caption("● = correct call   ✕ = wrong call   Colour = predicted regime")

                st.markdown("---")

                # ── Full results table ────────────────────────────────────────
                st.markdown("#### Full Results Table")

                tab_done, tab_pending = st.tabs([
                    f"✅ Completed ({len(completed)})",
                    f"⏳ Still Unfolding ({len(pending)})",
                ])

                def format_table(df_in, show_outcome=True):
                    rows = []
                    for _, r in df_in.iterrows():
                        meta = REGIME_META[r["regime"]]
                        row = {
                            "Ticker":       r["ticker"],
                            "Surge Date":   r["surge_date"].strftime("%d/%m/%Y"),
                            "Entry ₦":      f"₦{r['surge_price']:.2f}",
                            "Vol":          f"{r['vol_ratio']:.1f}×",
                            "Prediction":   f"{meta['icon']} {r['regime']}",
                            "Confidence":   f"{r['confidence']*100:.0f}%",
                            "Action":       meta["action"],
                            "Days":         int(r["days_elapsed"]),
                            "Now ₦":        f"₦{r['current_price']:.2f}",
                        }
                        if show_outcome and r["actual_20d"] is not None:
                            ret = r["actual_20d"] * 100
                            row["Actual Ret"] = f"{ret:+.2f}%"
                            row["Result"] = "✅ Correct" if r["correct"] else "❌ Wrong"
                        rows.append(row)
                    return pd.DataFrame(rows)

                with tab_done:
                    if completed.empty:
                        st.info("No completed events yet.")
                    else:
                        st.dataframe(format_table(completed), use_container_width=True, hide_index=True)

                with tab_pending:
                    if pending.empty:
                        st.info("No pending events.")
                    else:
                        st.markdown("*These surges happened recently — outcome not yet complete. "
                                    "This is what the model predicted **before knowing the result.***")
                        pend_rows = []
                        for _, r in pending.sort_values("surge_date", ascending=False).iterrows():
                            meta = REGIME_META[r["regime"]]
                            ret_so_far = r["ret_so_far"] * 100
                            # Direction tracking even while unfolding
                            if r["regime"] in ["Uptrend", "Accumulation"]:
                                on_track = "✅ On track" if ret_so_far > 0 else "⚠️ Against prediction"
                            else:
                                on_track = "✅ On track" if ret_so_far < 0 else "⚠️ Against prediction"
                            pend_rows.append({
                                "Ticker":          r["ticker"],
                                "Surge Date":      r["surge_date"].strftime("%d/%m/%Y"),
                                "Entry ₦":         f"₦{r['surge_price']:.2f}",
                                "Vol":             f"{r['vol_ratio']:.1f}×",
                                "Prediction":      f"{meta['icon']} {r['regime']}",
                                "Confidence":      f"{r['confidence']*100:.0f}%",
                                "Action":          meta["action"],
                                "Days Done":       int(r["days_elapsed"]),
                                "Days Left":       int(r["days_left"]),
                                "Now ₦":           f"₦{r['current_price']:.2f}",
                                "Return So Far":   f"{ret_so_far:+.2f}%",
                                "Tracking":        on_track,
                            })
                        st.dataframe(pd.DataFrame(pend_rows), use_container_width=True, hide_index=True)

                # ── Disclaimer ────────────────────────────────────────────────
                st.markdown("---")
                st.caption(
                    "⚠️ **Disclaimer:** This backtest uses a 30-day window — results will vary "
                    "with different time periods and market conditions. Past accuracy does not "
                    "guarantee future performance. Always apply your own judgement."
                )