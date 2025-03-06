import os
import re
import time
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import chromedriver_autoinstaller
import ta
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT & CUSTOM CSS
# =============================================================================
st.set_page_config(
    page_title="Dashboard Trading",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        background-color: #15508CFF;
        padding: 10px;
        height: 100%;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .reason-box {
        background-color: #D68C03FF;
        padding: 10px;
        height: 100%;
        border-radius: 8px;
        margin-top: 10px;
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
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =============================================================================
# KONFIGURASI LOGGING & SESSION REQUESTS
# =============================================================================
logging.basicConfig(
    filename="trading_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
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
    interval = "60"
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
    """
    df = df.copy()
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=5)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=5)
    df['EMA_3'] = ta.trend.ema_indicator(df['close'], window=3)
    df['EMA_5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['RSI'] = ta.momentum.rsi(df['close'], window=5)
    
    # Hitung StochRSI
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    df['StochRSI_K'] = np.where((max_rsi - min_rsi)==0, 0.5, (df['RSI']-min_rsi)/(max_rsi-min_rsi)) * 100
    
    # Bollinger Bands
    rolling_mean = df['close'].rolling(5).mean()
    rolling_std = df['close'].rolling(5).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * 1.5)
    df['BB_Lower'] = rolling_mean - (rolling_std * 1.5)
    
    # MACD
    df['MACD'] = ta.trend.macd(df['close'], window_slow=12, window_fast=6)
    df['MACD_signal'] = ta.trend.macd_signal(df['close'], window_slow=12, window_fast=6, window_sign=9)
    
    last_candle = df.iloc[-1]
    logging.info(f"ATR: {last_candle['ATR']}, ADX: {last_candle['ADX']}, EMA_3: {last_candle['EMA_3']}, "
                 f"EMA_5: {last_candle['EMA_5']}, RSI: {last_candle['RSI']}, StochRSI_K: {last_candle['StochRSI_K']}, "
                 f"MACD: {last_candle['MACD']}, MACD_signal: {last_candle['MACD_signal']}, "
                 f"BB_Upper: {last_candle['BB_Upper']}, BB_Lower: {last_candle['BB_Lower']}")
    return df

def detect_candlestick_patterns(df):
    """
    Deteksi pola candlestick pada data.
    """
    df = df.copy()
    # Pola-pola dasar
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
    df['Spinning_Top'] = (abs(df['close'] - df['open']) > 0.1 * (df['high'] - df['low'])) & \
                         (abs(df['close'] - df['open']) <= 0.3 * (df['high'] - df['low']))
    
    # Marubozu
    tolerance = 0.05 * (df['high'] - df['low'])
    bullish_marubozu = (df['close'] > df['open']) & ((df['open'] - df['low']) <= tolerance) & ((df['high'] - df['close']) <= tolerance)
    bearish_marubozu = (df['close'] < df['open']) & ((df['high'] - df['open']) <= tolerance) & ((df['close'] - df['low']) <= tolerance)
    df['Marubozu'] = bullish_marubozu | bearish_marubozu

    # === Pola tambahan ===
    # 1. Tweezer Top dan Tweezer Bottom
    tol = 0.05 * (df['high'] - df['low'])
    df['Tweezer_Top'] = (abs(df['high'] - df['high'].shift(1)) <= tol) & \
                        (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])
    df['Tweezer_Bottom'] = (abs(df['low'] - df['low'].shift(1)) <= tol) & \
                           (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
    
    # 2. Railroad Tracks
    tol_rt = 0.05 * (df['high'] - df['low'])
    df['Railroad_Tracks'] = (abs(df['open'] - df['open'].shift(1)) <= tol_rt) & \
                            (abs(df['close'] - df['close'].shift(1)) <= tol_rt) & \
                            (((df['close'].shift(1) > df['open'].shift(1)) & (df['close'] > df['open'])) | \
                             ((df['close'].shift(1) < df['open'].shift(1)) & (df['close'] < df['open'])))
    
    # 3. Three Inside (Up & Down)
    df['Three_Inside_Up'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                            (df['close'].shift(1) > df['open'].shift(1)) & \
                            (df['high'].shift(1) < df['high'].shift(2)) & (df['low'].shift(1) > df['low'].shift(2)) & \
                            (df['close'] > df['open']) & \
                            (df['open'] < df['open'].shift(1)) & (df['close'] > df['close'].shift(1))
    df['Three_Inside_Down'] = (df['close'].shift(2) > df['open'].shift(2)) & \
                              (df['close'].shift(1) < df['open'].shift(1)) & \
                              (df['high'].shift(1) < df['high'].shift(2)) & (df['low'].shift(1) > df['low'].shift(2)) & \
                              (df['close'] < df['open']) & \
                              (df['open'] > df['open'].shift(1)) & (df['close'] < df['close'].shift(1))
    df['Three_Inside'] = df['Three_Inside_Up'] | df['Three_Inside_Down']
    
    # 4. Fakey Pattern (deteksi false breakout)
    body_prev = abs(df['close'].shift(1) - df['open'].shift(1))
    df['Fakey_Bullish'] = ((df['high'].shift(1) - np.maximum(df['open'].shift(1), df['close'].shift(1))) > 2 * body_prev) & \
                          (df['close'] < df['low'].shift(1))
    df['Fakey_Bearish'] = ((np.minimum(df['open'].shift(1), df['close'].shift(1)) - df['low'].shift(1)) > 2 * body_prev) & \
                          (df['close'] > df['high'].shift(1))
    df['Fakey_Pattern'] = df['Fakey_Bullish'] | df['Fakey_Bearish']
    
    # 5. Rising & Falling Wedge (menggunakan window 5 candle)
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
        # Rising Wedge: kedua slope positif dan slope_low lebih besar (naik lebih cepat) daripada slope_high
        if (slope_high > 0) and (slope_low > 0) and (slope_low > slope_high):
            df.at[df.index[i], 'Rising_Wedge'] = True
        # Falling Wedge: kedua slope negatif dan slope_high lebih besar (naik kurang cepat) daripada slope_low
        if (slope_high < 0) and (slope_low < 0) and (slope_high > slope_low):
            df.at[df.index[i], 'Falling_Wedge'] = True
    
    # 6. Dragonfly & Gravestone Doji
    # Dragonfly Doji: tubuh sangat kecil, bayangan atas minimal, bayangan bawah panjang.
    df['Dragonfly_Doji'] = df['Doji'] & \
                           ((df['high'] - np.maximum(df['open'], df['close'])) <= 0.1 * (df['high'] - df['low'])) & \
                           ((np.minimum(df['open'], df['close']) - df['low']) >= 0.6 * (df['high'] - df['low']))
    # Gravestone Doji: tubuh sangat kecil, bayangan bawah minimal, bayangan atas panjang.
    df['Gravestone_Doji'] = df['Doji'] & \
                            ((np.minimum(df['open'], df['close']) - df['low']) <= 0.1 * (df['high'] - df['low'])) & \
                            ((df['high'] - np.maximum(df['open'], df['close'])) >= 0.6 * (df['high'] - df['low']))
    
    return df

def check_entry_signals(df):
    """
    Tentukan sinyal entry berdasarkan pola candlestick dan indikator teknikal.
    """
    df = detect_candlestick_patterns(df)
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
        bullish_primary = last_candle['close'] > last_candle['open']

    confirmations_bull = sum([
        last_candle['RSI'] < 50,
        last_candle['StochRSI_K'] < 20,
        last_candle['MACD'] > last_candle['MACD_signal'] and prev_candle['MACD'] < prev_candle['MACD_signal'],
        last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold
    ])
    confirmations_bear = sum([
        last_candle['RSI'] > 50,
        last_candle['StochRSI_K'] > 80,
        last_candle['MACD'] < last_candle['MACD_signal'] and prev_candle['MACD'] > prev_candle['MACD_signal'],
        last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold
    ])

    breakout_bull = last_candle['close'] < last_candle['BB_Lower']
    breakout_bear = last_candle['close'] > last_candle['BB_Upper']

    divergence_warning = ""
    if last_candle['close'] < prev_candle['close'] and last_candle['MACD'] > prev_candle['MACD']:
        divergence_warning = "Terdeteksi bullish divergence, waspada pembalikan naik. "
    if last_candle['close'] > prev_candle['close'] and last_candle['MACD'] < prev_candle['MACD']:
        divergence_warning = "Terdeteksi bearish divergence, waspada pembalikan turun. "

    reason = ""
    if last_candle['Bullish_Engulfing']:
        reason += "Bullish Engulfing, "
    if last_candle['Hammer']:
        reason += "Hammer, "
    if last_candle['Three_White_Soldiers']:
        reason += "Three White Soldiers, "
    if (last_candle['EMA_3'] > last_candle['EMA_5'] and prev_candle['EMA_3'] < prev_candle['EMA_5']):
        reason += "EMA3 cross EMA5 ke atas, "
    if last_candle['Morning_Star']:
        reason += "Morning Star, "
    if last_candle['Marubozu'] and last_candle['close'] > last_candle['open']:
        reason += "Marubozu bullish, "
    if last_candle['Bearish_Engulfing']:
        reason += "Bearish Engulfing, "
    if last_candle['Shooting_Star']:
        reason += "Shooting Star, "
    if last_candle['Three_Black_Crows']:
        reason += "Three Black Crows, "
    if (last_candle['EMA_3'] < last_candle['EMA_5'] and prev_candle['EMA_3'] > prev_candle['EMA_5']):
        reason += "EMA3 cross EMA5 ke bawah, "
    if last_candle['Evening_Star']:
        reason += "Evening Star, "
    if last_candle['Marubozu'] and last_candle['close'] < last_candle['open']:
        reason += "Marubozu bearish, "
    if last_candle['Doji']:
        reason += "Doji terdeteksi, "
    if last_candle['Spinning_Top']:
        reason += "Spinning Top terdeteksi, "
    if breakout_bull:
        reason += "Breakout bullish pada BB_Lower, "
    if breakout_bear:
        reason += "Breakout bearish pada BB_Upper, "
    if divergence_warning:
        reason += divergence_warning

    # Tambahan pola candlestick
    if last_candle['Tweezer_Top']:
        reason += "Tweezer Top, "
    if last_candle['Tweezer_Bottom']:
        reason += "Tweezer Bottom, "
    if last_candle['Railroad_Tracks']:
        reason += "Railroad Tracks, "
    if last_candle['Three_Inside']:
        reason += "Three Inside, "
    if last_candle['Fakey_Pattern']:
        reason += "Fakey Pattern, "
    if last_candle['Rising_Wedge']:
        reason += "Rising Wedge, "
    if last_candle['Falling_Wedge']:
        reason += "Falling Wedge, "
    if last_candle['Dragonfly_Doji']:
        reason += "Dragonfly Doji, "
    if last_candle['Gravestone_Doji']:
        reason += "Gravestone Doji, "

    if bullish_primary:
        effective = confirmations_bull + (1 if breakout_bull else 0)
        max_possible = 5 if breakout_bull else 4
        strength_percent = (effective / max_possible) * 100
        signal = "BUY KUAT 📈" if (confirmations_bull >= 3 or breakout_bull) else "BUY LEMAH 📈"
        return signal, "Konfirmasi bullish: " + reason, strength_percent

    if bearish_primary:
        effective = confirmations_bear + (1 if breakout_bear else 0)
        max_possible = 5 if breakout_bear else 4
        strength_percent = (effective / max_possible) * 100
        signal = "SELL KUAT 📉" if (confirmations_bear >= 3 or breakout_bear) else "SELL LEMAH 📉"
        return signal, "Konfirmasi bearish: " + reason, strength_percent

    return ("BUY 📈", "RSI di bawah 50, peluang kenaikan.", 50) if last_candle['RSI'] < 50 else ("SELL 📉", "RSI di atas 50, peluang penurunan.", 50)

def process_data():
    """
    Ambil data harga, hitung indikator, dan tentukan sinyal trading.
    """
    candles = fetch_price_data()
    if candles:
        columns = ['time', 'open', 'close', 'high', 'low']
        if 'volume' in candles[0]:
            columns.append('volume')
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['created_at'])
        df = df[columns]
        for col in ['open', 'close', 'high', 'low']:
            df[col] = df[col].astype(np.float64).round(8)
        df = calculate_indicators(df)
        signal, reason, strength = check_entry_signals(df)
        
        trade_open = df.iloc[-1]['open']
        trade_close = df.iloc[-2]['close']
        if "BUY" in signal:
            trade_success = trade_open > trade_close
        elif "SELL" in signal:
            trade_success = trade_open < trade_close
        else:
            trade_success = False
        
        st.session_state.last_trade_success = trade_success
        log_message = (
            f"\n{get_google_time().strftime('%H:%M:%S')} Sinyal Trading: {signal} | "
            f"Kekuatan: {strength:.1f}% | Trade Success: {trade_success}\n"
            f"Alasan: {reason} | Close Trade: {trade_close} | Open Trade: {trade_open}"
        )
        logging.info(log_message)
        print(log_message)
        return df, signal, reason, strength
    return None, None, None, None

# =============================================================================
# FUNGSI UNTUK EKSEKUSI PERDAGANGAN (TRADING)
# =============================================================================
def set_bid(driver, bid_amount):
    """
    Tetapkan nilai bid pada input field di halaman trading.
    """
    if bid_amount <= 0:
        raise ValueError("Bid amount must be greater than 0")
    bid_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/div/way-input-controls/div/input'
    try:
        bid_element = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, bid_xpath))
        )
    except Exception as e:
        logging.error(f"Error menemukan elemen bid: {e}")
        return False

    bid_value_str = f"Rp{bid_amount}"
    try:
        bid_element.clear()
    except Exception as e:
        logging.error(f"Error saat clear bid: {e}")
        return False
    time.sleep(1)
    try:
        bid_element.send_keys(bid_value_str)
    except Exception as e:
        logging.error(f"Error saat mengirim bid: {e}")
        return False
    time.sleep(1)
    entered_bid = bid_element.get_attribute('value')
    logging.info(f"Bid yang dimasukkan: {entered_bid}, seharusnya: {bid_value_str}")
    entered_bid_numeric = int(re.sub(r"[^\d]", "", entered_bid))
    expected_bid_numeric = int(re.sub(r"[^\d]", "", bid_value_str))
    if entered_bid_numeric != expected_bid_numeric:
        logging.warning(f"Bid numeric yang dimasukkan {entered_bid_numeric} tidak sama dengan {expected_bid_numeric}.")
        return False
    return True

def init_driver(twofa_code="", account_type="Demo", username_input="", password_input=""):
    """
    Inisialisasi driver Selenium dan lakukan login ke platform trading.
    """
    driver_path = chromedriver_autoinstaller.install(cwd='/tmp')
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Nonaktifkan headless jika perlu debugging
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 Chrome/114.0.5735.90 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    if os.path.exists('/usr/bin/chromium-browser'):
        options.binary_location = '/usr/bin/chromium-browser'
    elif os.path.exists('/usr/bin/google-chrome'):
        options.binary_location = '/usr/bin/google-chrome'
    
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    
    wait = WebDriverWait(driver, 10)
    driver.get("https://binomo2.com/trading")
    
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception as e:
        logging.error(f"Error menunggu body muncul: {e}")
        driver.quit()
        return None
    
    time.sleep(10)
    
    username_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/div[1]/platform-forms-input/way-input/div/div[1]/way-input-text/input'
    password_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/div[2]/platform-forms-input/way-input/div/div/way-input-password/input'
    login_button_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/vui-button/button'
    
    username = username_input if username_input.strip() != "" else "andiarifrahmatullah@gmail.com"
    password = password_input if password_input.strip() != "" else "@Rahmatullah07"
    
    try:
        driver.find_element(By.XPATH, username_xpath).send_keys(username)
    except Exception as e:
        logging.error(f"Error menemukan field username: {e}")
        driver.quit()
        return None
    
    try:
        driver.find_element(By.XPATH, password_xpath).send_keys(password)
    except Exception as e:
        logging.error(f"Error menemukan field password: {e}")
        driver.quit()
        return None
    
    try:
        wait.until(EC.element_to_be_clickable((By.XPATH, login_button_xpath))).click()
    except Exception as e:
        logging.error(f"Error pada tombol login: {e}")
        driver.quit()
        return None
    
    time.sleep(2)
    
    try:
        account_switcher = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, '//*[@id="account"]'))
        )
        account_switcher.click()
        time.sleep(1)
        account_types = {
            'Real': '/html/body/vui-popover/div[2]/account-list/div[1]',
            'Demo': '/html/body/vui-popover/div[2]/account-list/div[2]',
            'Tournament': '/html/body/vui-popover/div[2]/account-list/div[3]'
        }
        chosen_xpath = account_types.get(account_type)
        if not chosen_xpath:
            logging.error("Tipe akun tidak dikenali.")
        else:
            account_element = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, chosen_xpath))
            )
            account_element.click()
            time.sleep(1)
    except Exception as e:
        logging.error(f"Error saat memilih akun: {e}")
    
    try:
        popup_xpath = "/ng-component/vui-modal/div/button/vui-icon/svg/use"
        popup_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, popup_xpath))
        )
        popup_button.click()
        logging.info("Popup dengan xpath telah diklik.")
    except Exception as e:
        logging.info("Popup tidak muncul, lanjutkan proses.")
    
    twofa_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-two-factor-auth-validation/app-otp-validation-form/form/platform-forms-input/way-input/div/div/way-input-text/input'
    try:
        twofa_input = driver.find_element(By.XPATH, twofa_xpath)
        if not twofa_code:
            st.info("Autentikasi 2FA terdeteksi. Masukkan kode 2FA di sidebar dan tekan 'Refresh Data'.")
            driver.quit()
            return None
        else:
            twofa_input.send_keys(twofa_code)
            twofa_submit_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-two-factor-auth-validation/app-otp-validation-form/form/vui-button/button'
            wait.until(EC.element_to_be_clickable((By.XPATH, twofa_submit_xpath))).click()
            time.sleep(2)
    except Exception as e:
        logging.info("2FA tidak diperlukan atau sudah ditangani.")
    
    return driver

def check_balance(driver):
    """
    Periksa balance yang ada pada halaman trading menggunakan xpath:
    //*[@id="qa_trading_balance"]
    Mengembalikan nilai balance dalam bentuk integer atau None jika gagal.
    """
    try:
        balance_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[@id='qa_trading_balance']"))
        )
        balance_text = balance_element.text
        balance_numeric = int(re.sub(r"[^\d]", "", balance_text))
        logging.info(f"Balance ditemukan: {balance_numeric}")
        return balance_numeric
    except Exception as e:
        logging.error(f"Gagal memeriksa balance: {e}")
        return None

def execute_trade_action(driver, signal, bid_amount):
    """
    Eksekusi trade berdasarkan sinyal BUY atau SELL:
      1. Periksa balance terlebih dahulu.
      2. Tetapkan bid dengan nilai bid_amount.
      3. Klik tombol trade sesuai sinyal.
    """
    current_balance = check_balance(driver)
    if current_balance is None:
        return "Gagal mengambil balance."
    if current_balance < bid_amount:
        return f"Balance tidak mencukupi: Balance saat ini Rp{current_balance}, dibutuhkan Rp{bid_amount}."

    if not set_bid(driver, bid_amount):
        logging.warning(f"Bid Rp{bid_amount} gagal ditetapkan.")
        return f"Bid Rp{bid_amount} gagal ditetapkan."
    
    wait = WebDriverWait(driver, 10)
    if "BUY" in signal.upper():
        button_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/binary-info/div[2]/div/trading-buttons/vui-button[1]/button'
    elif "SELL" in signal.upper():
        button_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/binary-info/div[2]/div/trading-buttons/vui-button[2]/button'
    else:
        logging.error("Signal tidak dikenali.")
        return "Signal tidak dikenali."
    
    try:
        wait.until(EC.element_to_be_clickable((By.XPATH, button_xpath))).click()
        logging.info(f"Trade {signal} dengan bid Rp{bid_amount} dikirim.")
    except Exception as e:
        error_msg = f"Error pada eksekusi trade: {e}"
        logging.error(error_msg)
        return error_msg
    
    return "Trade dieksekusi."

# =============================================================================
# FUNGSI UNTUK MENAMPILKAN DASHBOARD
# =============================================================================
def display_dashboard(df, signal, reason, strength, trade_msg=""):
    """
    Tampilkan dashboard analisa dan grafik dengan tampilan yang lebih rapi.
    Warna box sinyal akan disesuaikan:
      - BUY: Hijau
      - SELL: Merah
    """
    current_time = get_google_time().strftime('%H:%M:%S')
    
    if "BUY" in signal.upper():
        signal_color = "#28a745"
    elif "SELL" in signal.upper():
        signal_color = "#dc3545"
    else:
        signal_color = "#15508CFF"
    
    with st.container():
        st.markdown(
            f"<div class='header-container'><span class='title'>Dashboard Analisis Trading</span> <br>"
            f"<span class='subtitle'>Auto Trade : {'<b> Aktif </b>' if st.session_state.auto_trade else '<b>Nonaktif</b>'} <br> {current_time}</span></div>",
            unsafe_allow_html=True
        )
    
    with st.container():
        st.markdown("### Sinyal Trading")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(
                f"<div class='info-box' style='background-color: {signal_color};'><span class='subheader'>Sinyal:</span><br> {signal}</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div class='info-box' style='background-color: #FFFFFFFF; color: #000000;'><span class='subheader'>Kekuatan:</span><br> {strength:.1f}%</div>",
                unsafe_allow_html=True
            )
        if trade_msg:
            st.info(f"Info Trade: {trade_msg}")
    
    with st.container():
        st.markdown(
            f"<div class='reason-box mt-5'><span class='subheader'>Alasan:</span><br> {reason}</div>",
            unsafe_allow_html=True
        )
    
    with st.expander("Lihat Data Candle (30 Menit Terakhir)"):
        time_threshold = df['time'].max() - pd.Timedelta(minutes=30)
        df_last30 = df[df['time'] >= time_threshold]
        st.dataframe(df_last30)
    
    with st.container():
        st.markdown("### Grafik Candlestick dan Indikator")
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

# =============================================================================
# FUNGSI UTAMA
# =============================================================================
def main():
    """
    Fungsi utama untuk menjalankan dashboard dan auto trade secara sistematis.
    Eksekusi trade hanya dilakukan jika waktu server (detik) berada di antara 0-20.
    Setelah login pertama kali, trade pertama hanya akan dieksekusi setelah pergantian menit.
    Fitur kompensasi: jika trade menghasilkan loss atau break-even, bid berikutnya dikalikan dengan faktor 2.2,
    dan jika profit, bid di-reset ke nilai awal.
    Trade baru akan dieksekusi setelah trade sebelumnya selesai, yang ditandai dengan pergantian menit.
    """
    # Inisialisasi status session
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
    # Variabel untuk menandai bahwa trade telah dieksekusi pada menit tertentu
    if "trade_executed_minute" not in st.session_state:
        st.session_state.trade_executed_minute = None

    compensation_factor = 2.2

    st.sidebar.title("Menu")
    _ = st.sidebar.button("Refresh Data")
    start_auto = st.sidebar.button("Start Auto Trade")
    stop_auto = st.sidebar.button("Stop Auto Trade")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (per menit)", value=True)
    
    account_type = st.sidebar.selectbox("Pilih Tipe Akun", ["Real", "Demo", "Tournament"], index=1)
    initial_bid = st.sidebar.number_input("Bid Awal (Rp)", value=15000)
    username_input = st.sidebar.text_input("Username", value="")
    password_input = st.sidebar.text_input("Password", value="", type="password")
    
    if start_auto:
        st.session_state.auto_trade = True
    if stop_auto:
        st.session_state.auto_trade = False
        if st.session_state.driver is not None:
            st.session_state.driver.quit()
            st.session_state.driver = None
            st.session_state.login_time = None

    st.sidebar.write(f"Auto Trade: {'Aktif' if st.session_state.auto_trade else 'Nonaktif'}")
    
    if auto_refresh:
        current_google_time = get_google_time()
        remaining_ms = int((60 - current_google_time.second) * 1000 - current_google_time.microsecond / 1000)
        if remaining_ms < 1000:
            remaining_ms = 60000
        st_autorefresh(interval=remaining_ms, limit=1000, key="auto_refresh")
    
    twofa_code = st.sidebar.text_input("Masukkan kode 2FA (jika diperlukan):", value="")
    
    df, signal, reason, strength = process_data()
    if df is not None:
        display_dashboard(df, signal, reason, strength)
        
        if st.session_state.auto_trade:
            current_time = get_google_time()
            
            # Jika driver belum ada, lakukan login dan inisialisasi
            if st.session_state.driver is None:
                driver = init_driver(twofa_code, account_type=account_type, username_input=username_input, password_input=password_input)
                if driver is None:
                    st.error("Login gagal. Pastikan kode 2FA dan kredensial telah dimasukkan.")
                    return
                st.session_state.driver = driver
                st.session_state.login_time = get_google_time()
                st.session_state.prev_balance = check_balance(driver)
                st.session_state.current_bid = initial_bid
                st.info("Login berhasil. Menunggu pergantian menit untuk trade pertama.")
            else:
                # Jika trade sudah dieksekusi sebelumnya dan menit telah berganti, cek hasil trade dan langsung eksekusi trade baru
                if st.session_state.trade_executed_minute is not None and current_time.minute != st.session_state.trade_executed_minute:
                    new_balance = check_balance(st.session_state.driver)
                    if new_balance is not None:
                        if new_balance > st.session_state.prev_balance:
                            st.session_state.current_bid = initial_bid
                            st.info(f"Profit terjadi. Reset bid ke nilai awal: Rp{initial_bid}")
                        elif new_balance < st.session_state.prev_balance:
                            st.session_state.current_bid = int(st.session_state.current_bid * compensation_factor)
                            st.info(f"Loss atau break-even. Bid selanjutnya: Rp{st.session_state.current_bid}")
                        else:
                            st.info(f"Tidak ada perubahan pada saldo. Bid tetap: Rp{st.session_state.current_bid}")

                        st.session_state.prev_balance = new_balance
                        st.session_state.trade_executed_minute = None
                        # Setelah update bid, jika kondisi waktu terpenuhi, langsung eksekusi trade baru
                        if current_time.second <= 20:
                            st.write("Eksekusi trade baru: Waktu server berada di antara 0-20 detik.")
                            trade_msg = execute_trade_action(st.session_state.driver, signal, st.session_state.current_bid)
                            if "Error" in trade_msg or "Gagal" in trade_msg:
                                st.error(f"Eksekusi perdagangan otomatis gagal: {trade_msg}")
                            else:
                                st.success(f"Eksekusi perdagangan otomatis berhasil: {trade_msg}")
                                st.session_state.trade_executed_minute = get_google_time().minute
                else:
                    # Jika belum ada trade yang dieksekusi pada menit ini, periksa waktu dan eksekusi trade jika syarat terpenuhi
                    if st.session_state.login_time and current_time.minute == st.session_state.login_time.minute:
                        st.info("Menunggu pergantian menit untuk eksekusi trade pertama.")
                    else:
                        if current_time.second <= 20 and st.session_state.trade_executed_minute is None:
                            st.write("Eksekusi trade: Waktu server berada di antara 0-20 detik.")
                            trade_msg = execute_trade_action(st.session_state.driver, signal, st.session_state.current_bid)
                            if "Error" in trade_msg or "Gagal" in trade_msg:
                                st.error(f"Eksekusi perdagangan otomatis gagal: {trade_msg}")
                            else:
                                st.success(f"Eksekusi perdagangan otomatis berhasil: {trade_msg}")
                                st.session_state.trade_executed_minute = get_google_time().minute
                        else:
                            st.warning("Menunggu pergantian menit berikutnya. Eksekusi trade hanya dilakukan pada detik 0-20 dan setelah trade sebelumnya selesai.")
    else:
        st.error("Tidak ada data harga yang tersedia.")

if __name__ == "__main__":
    main()
