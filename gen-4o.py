import requests
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime
import ta
import streamlit as st
import plotly.graph_objects as go
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from streamlit_autorefresh import st_autorefresh
import chromedriver_autoinstaller
import os

# Konfigurasi logging
logging.basicConfig(
    filename="trading_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Buat session requests agar koneksi reuse
session = requests.Session()

#####################
# Fungsi Analisis
#####################

@st.cache_data(ttl=10)  # cache selama 10 detik, sesuaikan jika perlu
def get_google_time():
    try:
        response = session.get("https://www.google.com", timeout=3)
        date_header = response.headers.get("Date")
        if date_header:
            return datetime.strptime(date_header, "%a, %d %b %Y %H:%M:%S GMT")
    except requests.RequestException:
        logging.error("Gagal mengambil waktu dari Google.")
    return datetime.utcnow()

@st.cache_data(ttl=10)
def fetch_price_data():
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
    df = df.copy()
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=5)
    df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'], window=5)
    df['EMA_3'] = ta.trend.ema_indicator(df['close'], window=3)
    df['EMA_5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['RSI'] = ta.momentum.rsi(df['close'], window=5)
    df['StochRSI_K'] = ta.momentum.stochrsi_k(df['close'], window=14, smooth1=3, smooth2=3)
    min_rsi = df['RSI'].rolling(window=14).min()
    max_rsi = df['RSI'].rolling(window=14).max()
    df['StochRSI_K'] = np.where((max_rsi - min_rsi) == 0, 0.5, (df['RSI'] - min_rsi) / (max_rsi - min_rsi)) * 100
    rolling_mean = df['close'].rolling(5).mean()
    rolling_std = df['close'].rolling(5).std()
    df['BB_Upper'] = rolling_mean + (rolling_std * 1.5)
    df['BB_Lower'] = rolling_mean - (rolling_std * 1.5)
    df['MACD'] = ta.trend.macd(df['close'], window_slow=12, window_fast=6)
    df['MACD_signal'] = ta.trend.macd_signal(df['close'], window_slow=12, window_fast=6, window_sign=9)
    last_candle = df.iloc[-1]
    logging.info(f"ATR: {last_candle['ATR']}, ADX: {last_candle['ADX']}, EMA_3: {last_candle['EMA_3']}, "
                 f"EMA_5: {last_candle['EMA_5']}, RSI: {last_candle['RSI']}, StochRSI_K: {last_candle['StochRSI_K']}, "
                 f"MACD: {last_candle['MACD']}, MACD_signal: {last_candle['MACD_signal']}, "
                 f"BB_Upper: {last_candle['BB_Upper']}, BB_Lower: {last_candle['BB_Lower']}")
    return df

def detect_candlestick_patterns(df):
    df = df.copy()
    # Pola-pola dasar
    df['Hammer'] = ((df['high'] - df['low']) > 2 * abs(df['close'] - df['open'])) & (df['close'] > df['open'])
    df['Shooting_Star'] = ((df['high'] - df['low']) > 2 * abs(df['close'] - df['open'])) & (df['open'] > df['close'])
    df['Bullish_Engulfing'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
    df['Bearish_Engulfing'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])
    df['Three_White_Soldiers'] = df['close'].diff().rolling(3).sum() > 0
    df['Three_Black_Crows'] = df['close'].diff().rolling(3).sum() < 0

    # Doji: badan kecil relatif terhadap range (threshold 10%)
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

    return df

def check_entry_signals(df):
    df = detect_candlestick_patterns(df)
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    adx_threshold = 20
    atr_threshold = df['ATR'].mean() * 0.5
    volume_available = 'volume' in df.columns
    volume_condition = (last_candle['volume'] >= df['volume'].rolling(5).mean().iloc[-1]) if volume_available else True

    # Sinyal utama berdasarkan pola dan EMA
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

    breakout_bull = last_candle['close'] < last_candle['BB_Lower'] and volume_condition
    breakout_bear = last_candle['close'] > last_candle['BB_Upper'] and volume_condition

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
        reason += "Breakout bullish pada BB_Lower dengan volume mendukung, "
    if breakout_bear:
        reason += "Breakout bearish pada BB_Upper dengan volume mendukung, "
    if divergence_warning:
        reason += divergence_warning

    # Hitung kekuatan sinyal
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

    return ("BUY 📈", "RSI di bawah 50, peluang kenaikan.", 50) if last_candle['RSI'] < 50 \
           else ("SELL 📉", "RSI di atas 50, peluang penurunan.", 50)

def process_data():
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
        log_message = f"\n{get_google_time().strftime('%H:%M:%S')} Sinyal Trading: {signal} | Kekuatan: {strength:.1f}%\nAlasan: {reason}"
        logging.info(log_message)
        print(log_message)
        return df, signal, reason, strength
    return None, None, None, None

###############################
# Fungsi Otomasi dengan Selenium
###############################

def init_driver(twofa_code=""):
    driver_path = chromedriver_autoinstaller.install(cwd='/tmp')
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/114.0.5735.90 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    if os.path.exists('/usr/bin/chromium-browser'):
        options.binary_location = '/usr/bin/chromium-browser'
    elif os.path.exists('/usr/bin/google-chrome'):
        options.binary_location = '/usr/bin/google-chrome'
    
    service = Service(driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    
    wait = WebDriverWait(driver, 10)  # timeout dikurangi
    driver.get("https://binomo2.com/trading")
    
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    except Exception as e:
        logging.error(f"Error menunggu body muncul: {e}")
        driver.quit()
        return None
    
    time.sleep(20)  # delay diminimalisir
    
    username_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/div[1]/platform-forms-input/way-input/div/div[1]/way-input-text/input'
    password_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/div[2]/platform-forms-input/way-input/div/div/way-input-password/input'
    login_button_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/vui-button/button'
    
    username = st.secrets.get("username", "andiarifrahmatullah@gmail.com")
    password = st.secrets.get("password", "@Rahmatullah07")
    
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

def execute_trade_action(driver, signal):
    wait = WebDriverWait(driver, 10)
    result_msg = ""
    try:
        if "BUY" in signal:
            buy_button_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/binary-info/div[2]/div/trading-buttons/vui-button[1]/button'
            wait.until(EC.element_to_be_clickable((By.XPATH, buy_button_xpath))).click()
            logging.info("Melakukan aksi BUY")
            result_msg = "Aksi BUY berhasil dieksekusi."
        elif "SELL" in signal:
            sell_button_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/main/div/app-panel/ng-component/section/binary-info/div[2]/div/trading-buttons/vui-button[2]/button'
            wait.until(EC.element_to_be_clickable((By.XPATH, sell_button_xpath))).click()
            logging.info("Melakukan aksi SELL")
            result_msg = "Aksi SELL berhasil dieksekusi."
    except Exception as e:
        error_msg = f"Error pada eksekusi trade: {e}"
        logging.error(error_msg)
        result_msg = error_msg
    return result_msg

###############################
# Dashboard Streamlit & Otomasi Trading
###############################

def display_dashboard(df, signal, reason, strength, trade_msg=""):
    current_time = get_google_time().strftime('%H:%M:%S')
    if trade_msg:
        st.success(f"Eksekusi perdagangan otomatis berhasil: {trade_msg}")
    
    st.markdown(f"### Auto Trade: **{'Aktif' if st.session_state.auto_trade else 'Nonaktif'}** ( {current_time} )")
    st.markdown("### Sinyal Trading")
    st.write(f"**Sinyal:** {signal}")
    st.write(f"**Kekuatan Sinyal:** {strength:.1f}%")
    st.write(f"**Alasan:** {reason}")
    st.markdown("### Data Candle Terakhir")
    st.dataframe(df.tail(20))
    st.markdown("### Grafik Candlestick dan Indikator")
    fig = go.Figure(data=[go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Candlestick"
    )])
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_3'], mode='lines', name='EMA 3'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['EMA_5'], mode='lines', name='EMA 5'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['BB_Upper'], mode='lines', name='BB Upper', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=df['time'], y=df['BB_Lower'], mode='lines', name='BB Lower', line=dict(dash='dash')))
    fig.update_layout(title="Grafik Harga dan Indikator", xaxis_title="Waktu", yaxis_title="Harga")
    st.plotly_chart(fig, use_container_width=True)

def main():
    if "auto_trade" not in st.session_state:
        st.session_state.auto_trade = False
    if "driver" not in st.session_state:
        st.session_state.driver = None

    st.sidebar.title("Menu")
    _ = st.sidebar.button("Refresh Data")
    start_auto = st.sidebar.button("Start Auto Trade")
    stop_auto = st.sidebar.button("Stop Auto Trade")
    auto_refresh = st.sidebar.checkbox("Auto Refresh (per menit)", value=True)
    
    if start_auto:
        st.session_state.auto_trade = True
    if stop_auto:
        st.session_state.auto_trade = False
        if st.session_state.driver is not None:
            st.session_state.driver.quit()
            st.session_state.driver = None

    st.sidebar.write(f"Auto Trade : {'Aktif' if st.session_state.auto_trade else 'Nonaktif'}")
    
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
            st.write("Auto Trade aktif: Eksekusi perdagangan otomatis sedang berjalan...")
            if st.session_state.driver is None:
                driver = init_driver(twofa_code)
                if driver is None:
                    st.error("Login gagal. Pastikan kode 2FA sudah dimasukkan jika diperlukan.")
                    return
                st.session_state.driver = driver
            trade_msg = execute_trade_action(st.session_state.driver, signal)
            if "Error" in trade_msg:
                st.error(f"Eksekusi perdagangan otomatis gagal: {trade_msg}")
            else:
                st.success(f"Eksekusi perdagangan otomatis berhasil: {trade_msg}")
    else:
        st.error("Tidak ada data harga yang tersedia.")

if __name__ == "__main__":
    main()
