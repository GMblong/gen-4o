import os
import sys
import re
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import ta
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT & CUSTOM CSS
# =============================================================================
# st.set_page_config(
#     page_title="Dashboard Trading",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

st.markdown(
    """
    <style>
    .header-container {
        /* background-color: #FFCC00FF; */
        /* padding: 20px; */
        border-radius: 8px;
        color: #FFFFFFFF;
        text-align: center;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #FFFFFFFF;
        padding: 10px;
        height: 100%;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .reason-box {
        background-color: #FFFFFFFF;
        padding: 10px;
        color: #000000FF;
        height: 100%;
        border-radius: 8px;
        margin-top: 5px;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .title {
        font-size: 2rem;
        font-weight: bold;
    }
    .subtitle {
        font-size: 1.2rem;
        font-weight: normal;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# KONFIGURASI LOGGING & SESSION REQUESTS
# =============================================================================
# logging.basicConfig(
#     filename="trading_analysis.log",
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S"
# )
session = requests.Session()

# =============================================================================
# FUNGSI UTILITAS
# =============================================================================
@st.cache_data(ttl=10)
def get_google_time():
    """
    Ambil waktu dari header respons Google untuk mendapatkan waktu yang akurat.
    """
    try:
        response = session.get("https://www.google.com", timeout=3)
        date_header = response.headers.get("Date")
        if date_header:
            return datetime.strptime(date_header, "%a, %d %b %Y %H:%M:%S GMT")
    except requests.RequestException:
        logging.error("Gagal mengambil waktu dari Google.")
    return datetime.utcnow()

# =============================================================================
# FUNGSI ANALISIS DATA
# =============================================================================
def fetch_price_data():
    """
    Ambil data harga candlestick dari API Binomo.
    """
    base_url = "https://api.binomo2.com/candles/v1"
    symbol = "Z-CRY%2FIDX"
    interval = "60"  # Data 1 menit
    locale = "en"
    current_time = get_google_time()
    formatted_time = current_time.strftime('%Y-%m-%dT00:00:00')
    url = f"{base_url}/{symbol}/{formatted_time}/{interval}?locale={locale}"
    
    try:
        response = session.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get('candles') or data.get('data', [])
    except requests.RequestException:
        logging.error("Gagal mengambil data harga.")
    return []


def calculate_indicators(df):
    """
    Hitung indikator teknikal: ATR, ADX, EMA, RSI, StochRSI, Bollinger Bands, MACD.
    Optimalkan untuk data 1 menit dengan parameter lebih sensitif.
    """
    df = df.copy()
    
    # Gunakan window lebih kecil untuk data 1 menit
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=3)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=3)
    df['EMA_3'] = ta.trend.ema_indicator(df['close'], window=3)
    df['EMA_5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['RSI'] = ta.momentum.rsi(df['close'], window=3)
    
    # StochRSI dengan window 7 (lebih cepat) daripada default 14
    min_rsi = df['RSI'].rolling(window=7).min()
    max_rsi = df['RSI'].rolling(window=7).max()
    df['StochRSI_K'] = np.where(
        (max_rsi - min_rsi) == 0,
        0.5,
        (df['RSI'] - min_rsi) / (max_rsi - min_rsi)
    ) * 100
    
    # Bollinger Bands: gunakan rolling window 3 dan multiplier 1.2
    rolling_mean = df['close'].rolling(3).mean()
    rolling_std = df['close'].rolling(3).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * 1.2)
    df['BB_Lower'] = rolling_mean - (rolling_std * 1.2)
    
    # MACD dengan parameter lebih sensitif: window_slow=6, window_fast=3, window_sign=4
    df['MACD'] = ta.trend.macd(df['close'], window_slow=6, window_fast=3)
    df['MACD_signal'] = ta.trend.macd_signal(df['close'], window_slow=6, window_fast=3, window_sign=4)
    
    return df


def detect_candlestick_patterns(df):
    """
    Deteksi pola candlestick pada data.
    Optimalkan untuk data 1 menit dengan penyesuaian toleransi.
    """
    df = df.copy()
    
    # Pola dasar
    df['Hammer'] = ((df['high'] - df['low']) > 2 * abs(df['close'] - df['open'])) & (df['close'] > df['open'])
    df['Shooting_Star'] = ((df['high'] - df['low']) > 2 * abs(df['close'] - df['open'])) & (df['open'] > df['close'])
    df['Bullish_Engulfing'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
    df['Bearish_Engulfing'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])
    df['Three_White_Soldiers'] = df['close'].diff().rolling(3).sum() > 0
    df['Three_Black_Crows'] = df['close'].diff().rolling(3).sum() < 0
    df['Doji'] = abs(df['close'] - df['open']) <= 0.1 * (df['high'] - df['low'])
    
    # Morning Star
    bearish_first = df['close'].shift(2) < df['open'].shift(2)
    small_body_second = abs(df['close'].shift(1) - df['open'].shift(1)) <= 0.1 * (df['high'].shift(1) - df['low'].shift(1))
    bullish_third = df['close'] > df['open']
    close_above_mid_first = df['close'] > ((df['open'].shift(2) + df['close'].shift(2)) / 2)
    df['Morning_Star'] = bearish_first & small_body_second & bullish_third & close_above_mid_first
    
    # Evening Star
    bullish_first = df['close'].shift(2) > df['open'].shift(2)
    small_body_second = abs(df['close'].shift(1) - df['open'].shift(1)) <= 0.1 * (df['high'].shift(1) - df['low'].shift(1))
    bearish_third = df['close'] < df['open']
    close_below_mid_first = df['close'] < ((df['open'].shift(2) + df['close'].shift(2)) / 2)
    df['Evening_Star'] = bullish_first & small_body_second & bearish_third & close_below_mid_first
    
    # Spinning Top
    df['Spinning_Top'] = (
        (abs(df['close'] - df['open']) > 0.1 * (df['high'] - df['low'])) &
        (abs(df['close'] - df['open']) <= 0.3 * (df['high'] - df['low']))
    )
    
    # Marubozu dengan toleransi dikurangi (0.03)
    tolerance = 0.03 * (df['high'] - df['low'])
    bullish_marubozu = (df['close'] > df['open']) & ((df['open'] - df['low']) <= tolerance) & ((df['high'] - df['close']) <= tolerance)
    bearish_marubozu = (df['close'] < df['open']) & ((df['high'] - df['open']) <= tolerance) & ((df['close'] - df['low']) <= tolerance)
    df['Marubozu'] = bullish_marubozu | bearish_marubozu

    # Pola tambahan dengan toleransi disesuaikan
    # 1. Tweezer Top & Bottom (toleransi 0.03)
    tol = 0.03 * (df['high'] - df['low'])
    df['Tweezer_Top'] = (
        (abs(df['high'] - df['high'].shift(1)) <= tol) &
        (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])
    )
    df['Tweezer_Bottom'] = (
        (abs(df['low'] - df['low'].shift(1)) <= tol) &
        (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
    )
    
    # 2. Railroad Tracks (toleransi 0.03)
    tol_rt = 0.03 * (df['high'] - df['low'])
    df['Railroad_Tracks'] = (
        (abs(df['open'] - df['open'].shift(1)) <= tol_rt) &
        (abs(df['close'] - df['close'].shift(1)) <= tol_rt) &
        (
            ((df['close'].shift(1) > df['open'].shift(1)) & (df['close'] > df['open'])) |
            ((df['close'].shift(1) < df['open'].shift(1)) & (df['close'] < df['open']))
        )
    )
    
    # 3. Three Inside (Up & Down)
    df['Three_Inside_Up'] = (
        (df['close'].shift(2) < df['open'].shift(2)) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['high'].shift(1) < df['high'].shift(2)) & (df['low'].shift(1) > df['low'].shift(2)) &
        (df['close'] > df['open']) &
        (df['open'] < df['open'].shift(1)) & (df['close'] > df['close'].shift(1))
    )
    df['Three_Inside_Down'] = (
        (df['close'].shift(2) > df['open'].shift(2)) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['high'].shift(1) < df['high'].shift(2)) & (df['low'].shift(1) > df['low'].shift(2)) &
        (df['close'] < df['open']) &
        (df['open'] > df['open'].shift(1)) & (df['close'] < df['close'].shift(1))
    )
    df['Three_Inside'] = df['Three_Inside_Up'] | df['Three_Inside_Down']
    
    # 4. Fakey Pattern
    body_prev = abs(df['close'].shift(1) - df['open'].shift(1))
    df['Fakey_Bullish'] = (
        ((df['high'].shift(1) - np.maximum(df['open'].shift(1), df['close'].shift(1))) > 2 * body_prev) &
        (df['close'] < df['low'].shift(1))
    )
    df['Fakey_Bearish'] = (
        ((np.minimum(df['open'].shift(1), df['close'].shift(1)) - df['low'].shift(1)) > 2 * body_prev) &
        (df['close'] > df['high'].shift(1))
    )
    df['Fakey_Pattern'] = df['Fakey_Bullish'] | df['Fakey_Bearish']
    
    # 5. Rising & Falling Wedge (window 5)
    df['Rising_Wedge'] = False
    df['Falling_Wedge'] = False
    window = 5
    for i in range(window - 1, len(df)):
        highs = df['high'].iloc[i - window + 1: i + 1]
        lows = df['low'].iloc[i - window + 1: i + 1]
        if len(highs) < window or len(lows) < window:
            continue
        x = np.arange(window)
        slope_high = np.polyfit(x, highs, 1)[0]
        slope_low = np.polyfit(x, lows, 1)[0]
        # Rising Wedge: kedua slope positif dan slope_low lebih besar daripada slope_high
        if (slope_high > 0) and (slope_low > 0) and (slope_low > slope_high):
            df.at[df.index[i], 'Rising_Wedge'] = True
        # Falling Wedge: kedua slope negatif dan slope_high lebih besar daripada slope_low
        if (slope_high < 0) and (slope_low < 0) and (slope_high > slope_low):
            df.at[df.index[i], 'Falling_Wedge'] = True
    
    # 6. Dragonfly & Gravestone Doji
    df['Dragonfly_Doji'] = (
        df['Doji'] &
        ((df['high'] - np.maximum(df['open'], df['close'])) <= 0.1 * (df['high'] - df['low'])) &
        ((np.minimum(df['open'], df['close']) - df['low']) >= 0.6 * (df['high'] - df['low']))
    )
    df['Gravestone_Doji'] = (
        df['Doji'] &
        ((np.minimum(df['open'], df['close']) - df['low']) <= 0.1 * (df['high'] - df['low'])) &
        ((df['high'] - np.maximum(df['open'], df['close'])) >= 0.6 * (df['high'] - df['low']))
    )
    
    return df


def check_entry_signals(df):
    """
    Tentukan sinyal entry berdasarkan pola candlestick dan indikator teknikal.
    df yang diterima sudah TIDAK berisi candle terakhir yang belum selesai.
    """
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    
    adx_threshold = 20
    atr_threshold = df['ATR'].mean() * 0.5
    
    bullish_primary = (
        last_candle['Bullish_Engulfing'] or
        last_candle['Hammer'] or
        last_candle['Three_White_Soldiers'] or
        (last_candle['EMA_3'] > last_candle['EMA_5'] and prev_candle['EMA_3'] < prev_candle['EMA_5'])
    )
    bearish_primary = (
        last_candle['Bearish_Engulfing'] or
        last_candle['Shooting_Star'] or
        last_candle['Three_Black_Crows'] or
        (last_candle['EMA_3'] < last_candle['EMA_5'] and prev_candle['EMA_3'] > prev_candle['EMA_5'])
    )
    if last_candle['Morning_Star']:
        bullish_primary = True
    if last_candle['Evening_Star']:
        bearish_primary = True
    if last_candle['Marubozu']:
        bullish_primary = (last_candle['close'] > last_candle['open'])
    
    confirmations_bull = sum([
        last_candle['RSI'] < 50,
        last_candle['StochRSI_K'] < 20,
        (last_candle['MACD'] > last_candle['MACD_signal'] and prev_candle['MACD'] < prev_candle['MACD_signal']),
        (last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold)
    ])
    confirmations_bear = sum([
        last_candle['RSI'] > 50,
        last_candle['StochRSI_K'] > 80,
        (last_candle['MACD'] < last_candle['MACD_signal'] and prev_candle['MACD'] > prev_candle['MACD_signal']),
        (last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold)
    ])
    
    breakout_bull = (last_candle['close'] < last_candle['BB_Lower'])
    breakout_bear = (last_candle['close'] > last_candle['BB_Upper'])
    
    divergence_warning = ""
    if (last_candle['close'] < prev_candle['close']) and (last_candle['MACD'] > prev_candle['MACD']):
        divergence_warning = "Terdeteksi bullish divergence, waspada pembalikan naik. "
    if (last_candle['close'] > prev_candle['close']) and (last_candle['MACD'] < prev_candle['MACD']):
        divergence_warning = "Terdeteksi bearish divergence, waspada pembalikan turun. "
    
    reason = []
    if last_candle['Bullish_Engulfing']:
        reason.append("Bullish Engulfing")
    if last_candle['Hammer']:
        reason.append("Hammer")
    if last_candle['Three_White_Soldiers']:
        reason.append("Three White Soldiers")
    if (last_candle['EMA_3'] > last_candle['EMA_5'] and prev_candle['EMA_3'] < prev_candle['EMA_3']):
        reason.append("EMA3 cross EMA5 ke atas")
    if last_candle['Morning_Star']:
        reason.append("Morning Star")
    if last_candle['Marubozu'] and last_candle['close'] > last_candle['open']:
        reason.append("Marubozu bullish")
    if last_candle['Bearish_Engulfing']:
        reason.append("Bearish Engulfing")
    if last_candle['Shooting_Star']:
        reason.append("Shooting Star")
    if last_candle['Three_Black_Crows']:
        reason.append("Three Black Crows")
    if (last_candle['EMA_3'] < last_candle['EMA_5'] and prev_candle['EMA_3'] > prev_candle['EMA_5']):
        reason.append("EMA3 cross EMA5 ke bawah")
    if last_candle['Evening_Star']:
        reason.append("Evening Star")
    if last_candle['Marubozu'] and last_candle['close'] < last_candle['open']:
        reason.append("Marubozu bearish")
    if last_candle['Doji']:
        reason.append("Doji terdeteksi")
    if last_candle['Spinning_Top']:
        reason.append("Spinning Top terdeteksi")
    if breakout_bull:
        reason.append("Breakout bullish pada BB_Lower")
    if breakout_bear:
        reason.append("Breakout bearish pada BB_Upper")
    if divergence_warning:
        reason.append(divergence_warning.strip())
    
    # Pola tambahan
    if last_candle['Tweezer_Top']:
        reason.append("Tweezer Top")
    if last_candle['Tweezer_Bottom']:
        reason.append("Tweezer Bottom")
    if last_candle['Railroad_Tracks']:
        reason.append("Railroad Tracks")
    if last_candle['Three_Inside']:
        reason.append("Three Inside")
    if last_candle['Fakey_Pattern']:
        reason.append("Fakey Pattern")
    if last_candle['Rising_Wedge']:
        reason.append("Rising Wedge")
    if last_candle['Falling_Wedge']:
        reason.append("Falling Wedge")
    if last_candle['Dragonfly_Doji']:
        reason.append("Dragonfly Doji")
    if last_candle['Gravestone_Doji']:
        reason.append("Gravestone Doji")
    
    reason_str = ", ".join(reason) + ", " if reason else ""
    
    if bullish_primary:
        effective = confirmations_bull + (1 if breakout_bull else 0)
        max_possible = 5 if breakout_bull else 4
        strength_percent = (effective / max_possible) * 100
        signal = "BUY KUAT 📈" if (confirmations_bull >= 3 or breakout_bull) else "BUY LEMAH 📈"
        return signal, "Konfirmasi bullish: " + reason_str, strength_percent
    
    if bearish_primary:
        effective = confirmations_bear + (1 if breakout_bear else 0)
        max_possible = 5 if breakout_bear else 4
        strength_percent = (effective / max_possible) * 100
        signal = "SELL KUAT 📉" if (confirmations_bear >= 3 or breakout_bear) else "SELL LEMAH 📉"
        return signal, "Konfirmasi bearish: " + reason_str, strength_percent
    
    if last_candle['RSI'] < 50:
        return ("BUY 📈", "RSI di bawah 50, peluang kenaikan.", 50)
    else:
        return ("SELL 📉", "RSI di atas 50, peluang penurunan.", 50)


def process_data():
    """
    Ambil data harga, lalu buang candle terakhir (yang dianggap masih berjalan).
    Setelah itu, hitung indikator dan pola candlestick, lalu evaluasi sinyal.
    Optimalkan untuk data 1 menit.
    """
    candles = fetch_price_data()
    if not candles:
        return None, None, None, None
    
    columns = ['time', 'open', 'close', 'high', 'low']
    if 'volume' in candles[0]:
        columns.append('volume')
    
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['created_at'])
    df = df[columns]
    
    # Pastikan kolom numerik memiliki tipe data float64
    for col in ['open', 'close', 'high', 'low']:
        df[col] = df[col].astype(np.float64).round(8)
    
    # Opsional: Buang candle terakhir yang belum lengkap
    # df = df.iloc[:-1].reset_index(drop=True)
    
    df = calculate_indicators(df)
    df = detect_candlestick_patterns(df)
    
    signal, reason, strength = check_entry_signals(df)
    
    # Simulasi outcome trading
    trade_open = df.iloc[-1]['open']   # Candle terakhir (close)
    trade_close = df.iloc[-2]['close'] # Candle sebelumnya
    if "BUY" in signal:
        trade_success = (trade_open > trade_close)
    elif "SELL" in signal:
        trade_success = (trade_open < trade_close)
    else:
        trade_success = False
    
    st.session_state.last_trade_success = trade_success
    log_message = (
        f"\n{get_google_time().strftime('%H:%M:%S')} Sinyal Trading : {signal} | "
        f"Kekuatan: {strength:.1f}% | Trade Success : {trade_success}\n"
        f"Alasan: {reason} | Close Candle Sebelumnya : {trade_close} | Open Candle Terakhir : {trade_open}"
    )
    print(log_message)
    
    return df, signal, reason, strength

def init_driver(twofa_code="", account_type="Demo", username_input="", password_input=""):
    """
    Inisialisasi driver Selenium dan lakukan login ke platform trading
    dalam mode headless.
    """
    online_env = os.environ.get("STREAMLIT_CLOUD", "false").lower() == "true"
    local_run = os.environ.get("LOCAL_RUN", "false").lower() == "true" or (
        sys.platform.startswith("win") or sys.platform.startswith("linux") or sys.platform.startswith("darwin")
    )
    driver = None

    if online_env:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService
        options = ChromeOptions()
        # Tentukan lokasi binary Chromium
        if os.path.exists("/usr/bin/chromium-browser"):
            options.binary_location = "/usr/bin/chromium-browser"
        elif os.path.exists("/usr/bin/chromium"):
            options.binary_location = "/usr/bin/chromium"
        else:
            logging.error("Binary Chromium tidak ditemukan di path yang diharapkan.")
            return None

        # Opsi Headless dan optimasi
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--blink-settings=imagesEnabled=false")  # nonaktifkan loading gambar
        options.add_argument("--window-size=1920x1080")
        options.add_argument('--ignore-certificate-errors')
        
        try:
            service = ChromeService()
            driver = webdriver.Chrome(service=service, options=options)
            logging.info("Online run: Menggunakan Chromium dan chromium-driver.")
        except Exception as e:
            logging.error(f"Online run: Gagal inisialisasi Chromium driver: {e}")
            return None
    elif local_run:
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.chrome.service import Service as ChromeService
        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_argument("--window-size=1920x1080")
        
        try:
            import chromedriver_autoinstaller
            new_driver_path = chromedriver_autoinstaller.install(cwd='/tmp')
            service = ChromeService(new_driver_path)
            driver = webdriver.Chrome(service=service, options=options)
            logging.info("Local run: Menggunakan Chrome driver otomatis.")
        except Exception as e:
            fallback_driver_path = r'D:\aplikasi\gen-4o\133\chromedriver.exe'
            if not os.path.exists(fallback_driver_path):
                logging.error(f"Fallback driver path tidak valid: {fallback_driver_path}")
                return None
            try:
                service = ChromeService(fallback_driver_path)
                driver = webdriver.Chrome(service=service, options=options)
                logging.info("Local run: Menggunakan fallback Chrome driver.")
            except Exception as ex:
                logging.error(f"Gagal inisialisasi fallback driver: {ex}")
                return None
    else:
        logging.error("Lingkungan tidak dikenali. Tidak dapat inisialisasi driver.")
        return None

    driver.implicitly_wait(10)
    wait = WebDriverWait(driver, 20)
    driver.get("https://binomo2.com/trading")
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception as e:
        driver.quit()
        return None

    # Field Username dengan XPath sederhana
    try:
        username_field = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='text']")))
        username_val = username_input if username_input.strip() != "" else "andiarifrahmatullah@gmail.com"
        username_field.send_keys(username_val)
    except Exception as e:
        logging.error(f"Error menemukan field username: {e}")
        driver.quit()
        return None

    # Field Password dengan XPath sederhana
    try:
        password_field = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='password']")))
        password_val = password_input if password_input.strip() != "" else "@Rahmatullah07"
        password_field.send_keys(password_val)
    except Exception as e:
        logging.error(f"Error menemukan field password: {e}")
        driver.quit()
        return None

    # Tombol Login dengan XPath sederhana
    try:
        login_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[@type='submit']")))
        login_button.click()
    except Exception as e:
        logging.error(f"Error pada tombol login: {e}")
        driver.quit()
        return None

    # Proses 2FA (jika diperlukan)
    try:
        twofa_field = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH,
            '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-two-factor-auth-validation/app-otp-validation-form/form/platform-forms-input/way-input/div/div/way-input-text/input'))
        )
        if twofa_field:
            if twofa_code:
                twofa_field.send_keys(twofa_code)
                twofa_submit = wait.until(EC.element_to_be_clickable((By.XPATH,
                    '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-two-factor-auth-validation/app-otp-validation-form/form/vui-button/button')))
                twofa_submit.click()
            else:
                logging.info("2FA muncul, namun kode tidak diberikan. Menganggap 2FA tidak diperlukan.")
    except Exception as e:
        logging.info("2FA tidak diperlukan atau tidak muncul dalam waktu 5 detik.")

    try:
        account_switcher = wait.until(EC.presence_of_element_located((By.XPATH, '//*[@id="account"]')))
        account_switcher.click()
    except Exception as e:
        logging.error(f"Error saat menunggu account switcher: {e}")

    try:
        account_types = {
            'Real': "//div[normalize-space()='Real account']",
            'Demo': "//div[normalize-space()='Demo account']",
            'Tournament': "//div[normalize-space()='Tournament account']"
        }
        chosen_xpath = account_types.get(account_type)
        if not chosen_xpath:
            logging.error("Tipe akun tidak dikenali.")
        else:
            account_element = wait.until(EC.presence_of_element_located((By.XPATH, chosen_xpath)))
            account_element.click()
    except Exception as e:
        logging.error(f"Error saat memilih akun: {e}")

    try:
        popup_xpath = "/html/body/ng-component/vui-modal/div/div/div/ng-component/div/div/vui-button[1]/button"
        popup_button = wait.until(EC.presence_of_element_located((By.XPATH, popup_xpath)))
        popup_button.click()
        logging.info("Popup dengan xpath telah diklik.")
    except Exception as e:
        logging.info("Popup tidak muncul, lanjutkan proses.")

    return driver


def get_cached_balance_element(driver):
    """
    Dapatkan elemen balance dengan caching. Jika elemen sudah disimpan dan masih valid,
    gunakan elemen tersebut. Jika tidak, cari ulang dengan WebDriverWait.
    """
    if not hasattr(driver, "cached_balance_element"):
        driver.cached_balance_element = None

    if driver.cached_balance_element is not None:
        try:
            # Pastikan elemen masih tampil
            driver.cached_balance_element.is_displayed()
        except Exception:
            driver.cached_balance_element = None

    if driver.cached_balance_element is None:
        balance_xpath = "//*[@id='qa_trading_balance']"
        try:
            balance_element = WebDriverWait(driver, 0.5).until(
                EC.presence_of_element_located((By.XPATH, balance_xpath))
            )
            driver.cached_balance_element = balance_element
        except Exception as e:
            return None, f"Error menemukan elemen balance: {e}"
    return driver.cached_balance_element, None


def check_balance(driver):
    """
    Periksa balance di halaman trading dengan menggunakan elemen yang dicache.
    """
    balance_element, error = get_cached_balance_element(driver)
    if error:
        return None
    try:
        balance_text = balance_element.text
        balance_numeric = int(re.sub(r"[^\d]", "", balance_text))
        return balance_numeric
    except Exception as e:
        return None


def get_cached_bid_input(driver):
    """
    Dapatkan elemen input bid dengan caching. Jika elemen sudah disimpan dan masih valid,
    gunakan elemen tersebut. Jika tidak, cari ulang.
    """
    if not hasattr(driver, "cached_bid_input"):
        driver.cached_bid_input = None

    if driver.cached_bid_input is not None:
        try:
            driver.cached_bid_input.is_enabled()
        except Exception:
            driver.cached_bid_input = None

    if driver.cached_bid_input is None:
        bid_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/div/way-input-controls/div/input'
        try:
            bid_element = WebDriverWait(driver, 0.5).until(
                EC.presence_of_element_located((By.XPATH, bid_xpath))
            )
            driver.cached_bid_input = bid_element
        except Exception as e:
            return None, f"Error menemukan elemen bid: {e}"
    return driver.cached_bid_input, None


def set_bid(driver, bid_amount):
    """
    Tetapkan nilai bid pada input field.
    Menggunakan caching pada elemen bid untuk mempercepat eksekusi.
    """
    if bid_amount <= 0:
        raise ValueError("Bid amount must be greater than 0")
    
    bid_value_str = f"Rp{bid_amount}"
    bid_element, error = get_cached_bid_input(driver)
    if error:
        logging.error(error)
        return False

    def attempt_set_bid():
        try:
            bid_element.clear()
            bid_element.send_keys(bid_value_str)
            return True
        except Exception as ex:
            logging.error(f"Error saat clear atau mengirim bid: {ex}")
            return False

    entered_bid = bid_element.get_attribute('value')
    entered_bid_numeric = int(re.sub(r"[^\d]", "", entered_bid))
    expected_bid_numeric = int(re.sub(r"[^\d]", "", bid_value_str))
    if entered_bid_numeric != expected_bid_numeric:
        current_time = get_google_time()
        if current_time.second <= 20:
            if not attempt_set_bid():
                return False
            entered_bid = bid_element.get_attribute('value')
            entered_bid_numeric = int(re.sub(r"[^\d]", "", entered_bid))
            if entered_bid_numeric != expected_bid_numeric:
                logging.warning(f"Retry: Bid numeric {entered_bid_numeric} tidak sama dengan {expected_bid_numeric}.")
                return False
        else:
            return False
    return True


def get_cached_trade_button(driver, signal):
    """
    Caching tombol trade (BUY/SELL) agar tidak mencari ulang setiap eksekusi.
    Jika elemen sudah stale, akan dicari ulang.
    """
    if not hasattr(driver, "cached_trade_buttons"):
        driver.cached_trade_buttons = {}
    trade_buttons = driver.cached_trade_buttons

    if "BUY" in signal.upper():
        key = "BUY"
        xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/binary-info/div[2]/div/trading-buttons/vui-button[1]/button'
    elif "SELL" in signal.upper():
        key = "SELL"
        xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/binary-info/div[2]/div/trading-buttons/vui-button[2]/button'
    else:
        return None, "Signal tidak dikenali."

    button_element = trade_buttons.get(key, None)
    if button_element:
        try:
            button_element.is_enabled()
        except Exception:
            button_element = None

    if button_element is None:
        try:
            button_element = WebDriverWait(driver, 0.5).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            trade_buttons[key] = button_element
        except Exception as e:
            return None, f"Error menemukan tombol {key}: {e}"
    return button_element, None


def execute_trade_action(driver, signal, bid_amount):
    """
    Eksekusi trade berdasarkan sinyal BUY/SELL:
      1. Tetapkan bid.
      2. Dapatkan tombol dari cache dan langsung klik dengan JavaScript.
    """
    if not set_bid(driver, bid_amount):
        return f"Bid Rp{bid_amount} gagal ditetapkan."

    button_element, error = get_cached_trade_button(driver, signal)
    if error:
        return error

    try:
        driver.execute_script("arguments[0].click();", button_element)
    except Exception as e:
        return f"Error pada eksekusi trade: {e}"

    return "Trade dieksekusi."

def display_dashboard(df, signal, reason, strength, trade_msg=""):
    current_time = get_google_time().strftime('%H:%M:%S')
    if "BUY" in signal.upper():
        signal_color = "#28a745"
    elif "SELL" in signal.upper():
        signal_color = "#dc3545"
    else:
        signal_color = "#15508CFF"
    
    with st.container():
        st.markdown(
            f"<div class='header-container'><span class='title'>Analisis Trading</span><br>"
            f"<span class='subtitle'>{current_time}</span></div>",
            unsafe_allow_html=True
        )
    
    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                f"<div class='info-box' style='background-color: {signal_color};'><span class='subheader'>Sinyal :</span><br>{signal}</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div class='info-box' style='background-color: #FFFFFFFF; color: #000000FF;'><span class='subheader'>Kekuatan :</span><br>{strength:.1f}%</div>",
                unsafe_allow_html=True
            )
        if trade_msg:
            st.info(f"Info Trade : {trade_msg}")
    
    with st.container():
        st.markdown(
            f"<div class='reason-box mt-5'><span class='subheader'>Alasan :</span><br>{reason}</div>",
            unsafe_allow_html=True
        )
    
    if "new_balance" in st.session_state and st.session_state.new_balance is not None:
        balance_value = st.session_state.new_balance / 100.0
        balance_str = f"{balance_value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        st.markdown(
            f"<div class='info-box' style='background-color: #FFAA0000; color: #FFFFFFFF; border-radius: 8px; border: 1px solid #FFFFFF4F; text-align: left; margin-bottom: 20px;'>"
            f"<span class='header'>Saldo Sekarang :</span><br><span class='title'>Rp. {balance_str}</span></div>",
            unsafe_allow_html=True
        )
    
    with st.expander("Lihat Data Candle (30 Menit Terakhir)"):
        time_threshold = df['time'].max() - pd.Timedelta(minutes=30)
        df_last30 = df[df['time'] >= time_threshold]
        st.dataframe(df_last30)
    
    with st.expander("### Grafik Candlestick dan Indikator"):
        time_threshold = df['time'].max() - pd.Timedelta(minutes=15)
        df_last30 = df[df['time'] >= time_threshold]
        fig = go.Figure(data=[go.Candlestick(
            x=df_last30['time'],
            open=df_last30['open'],
            high=df_last30['high'],
            low=df_last30['low'],
            close=df_last30['close'],
            name="Candlestick"
        )])
        fig.add_trace(go.Scatter(x=df_last30['time'], y=df_last30['EMA_3'], mode='lines', name='EMA 3'))
        fig.add_trace(go.Scatter(x=df_last30['time'], y=df_last30['EMA_5'], mode='lines', name='EMA 5'))
        fig.add_trace(go.Scatter(x=df_last30['time'], y=df_last30['BB_Upper'], mode='lines', name='BB Upper', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=df_last30['time'], y=df_last30['BB_Lower'], mode='lines', name='BB Lower', line=dict(dash='dash')))
        fig.update_layout(
            title="Grafik Harga dan Indikator (30 Menit Terakhir)",
            xaxis_title="Waktu",
            yaxis_title="Harga",
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def main():
    """
    Fungsi utama untuk menjalankan dashboard dan auto trade.
    Eksekusi trade dilakukan segera setelah pengecekan saldo dan kompensasi.
    """
    if "auto_trade" not in st.session_state:
        st.session_state.auto_trade = False
    if "driver" not in st.session_state:
        st.session_state.driver = None
    if "login_time" not in st.session_state:
        st.session_state.login_time = None
    if "prev_balance" not in st.session_state:
        st.session_state.prev_balance = None
    if "current_bid" not in st.session_state:
        st.session_state.current_bid = None
    if "new_balance" not in st.session_state:
        st.session_state.new_balance = None
    if "trade_executed_minute" not in st.session_state:
        st.session_state.trade_executed_minute = None

    st.sidebar.title("Menu")
    _ = st.sidebar.button("Refresh Data")
    start_auto = st.sidebar.button("Start Auto Trade")
    stop_auto = st.sidebar.button("Stop Auto Trade")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    st.sidebar.write(f"Auto Trade : {'Aktif' if st.session_state.auto_trade else 'Nonaktif'}")

    account_type = st.sidebar.selectbox("Pilih Tipe Akun", ["Real", "Demo", "Tournament"], index=1)
    initial_bid = st.sidebar.number_input("Open Order Awal (Rp)", value=15000)
    compensation_factor = st.sidebar.number_input("Faktor Kompensasi", value=2.2, min_value=1.0, step=0.1, format="%.1f")
    username_input = st.sidebar.text_input("Username", value="")
    password_input = st.sidebar.text_input("Password", value="", type="password")
    twofa_code = st.sidebar.text_input("Masukkan kode 2FA (jika diperlukan):", value="")

    if start_auto:
        st.session_state.auto_trade = True
    if stop_auto:
        st.session_state.auto_trade = False
        if st.session_state.driver is not None:
            st.session_state.driver.quit()
            st.session_state.driver = None
            st.session_state.login_time = None

    if auto_refresh:
        current_google_time = get_google_time()
        remaining_ms = int((60 - current_google_time.second) * 1000 - current_google_time.microsecond / 1000)
        if remaining_ms < 1000:
            remaining_ms = 100
        st_autorefresh(interval=remaining_ms, limit=1000, key="auto_refresh")

    df, signal, reason, strength = process_data()
    if df is not None:
        display_dashboard(df, signal, reason, strength)

        if st.session_state.auto_trade:
            current_time = get_google_time()
            if st.session_state.trade_executed_minute is not None and current_time.minute != st.session_state.trade_executed_minute:
                st.session_state.trade_executed_minute = None

            if st.session_state.driver is None:
                driver = init_driver(twofa_code, account_type=account_type, username_input=username_input, password_input=password_input)
                if driver is None:
                    st.error("Login gagal. Pastikan kode 2FA dan kredensial telah dimasukkan.")
                    return
                st.session_state.driver = driver
                st.session_state.login_time = current_time
                balance = check_balance(driver)
                if balance is not None:
                    st.session_state.prev_balance = balance
                    st.session_state.new_balance = balance
                st.session_state.current_bid = initial_bid
                st.info("Login berhasil. Menunggu pergantian menit untuk trade pertama.")
            else:
                current_balance = check_balance(st.session_state.driver)
                if current_balance is not None:
                    st.session_state.new_balance = current_balance

                if current_time.second <= 20 and st.session_state.trade_executed_minute is None:
                    if current_balance is not None:
                        if current_balance > st.session_state.prev_balance:
                            st.session_state.current_bid = initial_bid
                            st.success(f"Profit terjadi. Reset order ke nilai awal: Rp{initial_bid}")
                            trade_msg = execute_trade_action(st.session_state.driver, signal, initial_bid)
                        elif current_balance < st.session_state.prev_balance:
                            compensated_bid = int(st.session_state.current_bid * compensation_factor)
                            st.session_state.current_bid = compensated_bid
                            st.error(f"Loss atau break-even terdeteksi. Menjalankan trade kompensasi dengan order: Rp{compensated_bid}")
                            trade_msg = execute_trade_action(st.session_state.driver, signal, compensated_bid)
                        else:
                            st.warning(f"Tidak ada perubahan pada saldo. Eksekusi trade dengan order: Rp{st.session_state.current_bid}")
                            trade_msg = execute_trade_action(st.session_state.driver, signal, st.session_state.current_bid)
                    else:
                        st.info(f"Tidak dapat memverifikasi saldo. Eksekusi trade dengan order: Rp{st.session_state.current_bid}")
                        trade_msg = execute_trade_action(st.session_state.driver, signal, st.session_state.current_bid)
                    
                    st.session_state.trade_executed_minute = current_time.minute
                    if current_balance is not None:
                        st.session_state.prev_balance = current_balance
                    st.success(f"Eksekusi perdagangan otomatis berhasil: {trade_msg}")
                else:
                    if st.session_state.login_time and current_time.minute == st.session_state.login_time.minute:
                        st.warning("Menunggu pergantian menit untuk eksekusi trade pertama.")
                    else:
                        st.warning("Menunggu pergantian menit berikutnya. Eksekusi trade hanya dilakukan pada detik 0-20 dan jika trade belum dieksekusi pada menit ini.")
    else:
        st.error("Tidak ada data harga yang tersedia.")
        
if __name__ == "__main__":
    main()
