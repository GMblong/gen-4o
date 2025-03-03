import requests
import logging
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import ta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from streamlit_autorefresh import st_autorefresh

# Konfigurasi logging
logging.basicConfig(
    filename="trading_analysis.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

#####################
# Fungsi Analisis
#####################

def get_google_time():
    try:
        response = requests.get("https://www.google.com", timeout=5)
        date_header = response.headers.get("Date")
        if date_header:
            return datetime.strptime(date_header, "%a, %d %b %Y %H:%M:%S GMT")
    except requests.RequestException:
        logging.error("Gagal mengambil waktu dari Google.")
    return datetime.utcnow()

def fetch_price_data():
    base_url = "https://api.binomo2.com/candles/v1"
    symbol = "Z-CRY%2FIDX"
    interval = "60"
    locale = "en"
    current_time = get_google_time()
    formatted_time = current_time.strftime('%Y-%m-%dT00:00:00')
    url = f"{base_url}/{symbol}/{formatted_time}/{interval}?locale={locale}"
    try:
        response = requests.get(url, timeout=10)
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
    logging.info("Hasil perhitungan indikator:")
    logging.info(
        f"ATR: {last_candle['ATR']}, ADX: {last_candle['ADX']}, EMA_3: {last_candle['EMA_3']}, "
        f"EMA_5: {last_candle['EMA_5']}, RSI: {last_candle['RSI']}, StochRSI_K: {last_candle['StochRSI_K']}, "
        f"MACD: {last_candle['MACD']}, MACD_signal: {last_candle['MACD_signal']}, "
        f"BB_Upper: {last_candle['BB_Upper']}, BB_Lower: {last_candle['BB_Lower']}"
    )
    return df

def detect_candlestick_patterns(df):
    df = df.copy()
    df['Hammer'] = ((df['high'] - df['low']) > 2 * abs(df['close'] - df['open'])) & (df['close'] > df['open'])
    df['Shooting_Star'] = ((df['high'] - df['low']) > 2 * abs(df['close'] - df['open'])) & (df['open'] > df['close'])
    df['Bullish_Engulfing'] = (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'])
    df['Bearish_Engulfing'] = (df['close'].shift(1) > df['open'].shift(1)) & (df['close'] < df['open'])
    df['Three_White_Soldiers'] = df['close'].diff().rolling(3).sum() > 0
    df['Three_Black_Crows'] = df['close'].diff().rolling(3).sum() < 0
    return df

def check_entry_signals(df):
    df = detect_candlestick_patterns(df)
    last_candle = df.iloc[-1]
    prev_candle = df.iloc[-2]
    adx_threshold = 20
    atr_threshold = df['ATR'].mean() * 0.5
    volume_available = 'volume' in df.columns
    if volume_available:
        avg_volume = df['volume'].rolling(5).mean().iloc[-1]
        volume_condition = last_candle['volume'] >= avg_volume
    else:
        volume_condition = True
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
    confirmations_bull = 0
    if last_candle['RSI'] < 50:
        confirmations_bull += 1
    if last_candle['StochRSI_K'] < 20:
        confirmations_bull += 1
    if last_candle['MACD'] > last_candle['MACD_signal'] and prev_candle['MACD'] < prev_candle['MACD_signal']:
        confirmations_bull += 1
    if last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold:
        confirmations_bull += 1
    confirmations_bear = 0
    if last_candle['RSI'] > 50:
        confirmations_bear += 1
    if last_candle['StochRSI_K'] > 80:
        confirmations_bear += 1
    if last_candle['MACD'] < last_candle['MACD_signal'] and prev_candle['MACD'] > prev_candle['MACD_signal']:
        confirmations_bear += 1
    if last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold:
        confirmations_bear += 1
    breakout_bull = last_candle['close'] < last_candle['BB_Lower'] and volume_condition
    breakout_bear = last_candle['close'] > last_candle['BB_Upper'] and volume_condition
    divergence_warning = ""
    if last_candle['close'] < prev_candle['close'] and last_candle['MACD'] > prev_candle['MACD']:
        divergence_warning = "Terdeteksi bullish divergence, waspada pembalikan naik. "
    if last_candle['close'] > prev_candle['close'] and last_candle['MACD'] < prev_candle['MACD']:
        divergence_warning = "Terdeteksi bearish divergence, waspada pembalikan turun. "
    if bullish_primary:
        signal = "BUY KUAT 📈" if (confirmations_bull >= 3 or breakout_bull) else "BUY LEMAH 📈"
        reason = "Konfirmasi bullish: "
        if last_candle['Bullish_Engulfing']:
            reason += "Bullish Engulfing, "
        if last_candle['Hammer']:
            reason += "Hammer, "
        if last_candle['Three_White_Soldiers']:
            reason += "Three White Soldiers, "
        if (last_candle['EMA_3'] > last_candle['EMA_5'] and prev_candle['EMA_3'] < prev_candle['EMA_5']):
            reason += "EMA3 cross EMA5 ke atas, "
        if last_candle['RSI'] < 50:
            reason += "RSI < 50, "
        if last_candle['StochRSI_K'] < 20:
            reason += "StochRSI oversold, "
        if (last_candle['MACD'] > last_candle['MACD_signal'] and prev_candle['MACD'] < prev_candle['MACD_signal']):
            reason += "MACD bullish crossover, "
        if last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold:
            reason += "ATR meningkat dan ADX > 20, "
        if breakout_bull:
            reason += "Breakout bullish pada BB_Lower dengan volume mendukung, "
        if divergence_warning:
            reason += divergence_warning
        return signal, reason
    if bearish_primary:
        signal = "SELL KUAT 📉" if (confirmations_bear >= 3 or breakout_bear) else "SELL LEMAH 📉"
        reason = "Konfirmasi bearish: "
        if last_candle['Bearish_Engulfing']:
            reason += "Bearish Engulfing, "
        if last_candle['Shooting_Star']:
            reason += "Shooting Star, "
        if last_candle['Three_Black_Crows']:
            reason += "Three Black Crows, "
        if (last_candle['EMA_3'] < last_candle['EMA_5'] and prev_candle['EMA_3'] > prev_candle['EMA_5']):
            reason += "EMA3 cross EMA5 ke bawah, "
        if last_candle['RSI'] > 50:
            reason += "RSI > 50, "
        if last_candle['StochRSI_K'] > 80:
            reason += "StochRSI overbought, "
        if (last_candle['MACD'] < last_candle['MACD_signal'] and prev_candle['MACD'] > prev_candle['MACD_signal']):
            reason += "MACD bearish crossover, "
        if last_candle['ATR'] > atr_threshold and last_candle['ADX'] > adx_threshold:
            reason += "ATR meningkat dan ADX > 20, "
        if breakout_bear:
            reason += "Breakout bearish pada BB_Upper dengan volume mendukung, "
        if divergence_warning:
            reason += divergence_warning
        return signal, reason
    return ("BUY 📈", "RSI di bawah 50, peluang kenaikan.") if last_candle['RSI'] < 50 else ("SELL 📉", "RSI di atas 50, peluang penurunan.")

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
        signal, reason = check_entry_signals(df)
        log_message = f"\n{get_google_time().strftime('%H:%M:%S')} 📢 Sinyal Trading: {signal}\nAlasan: {reason}"
        logging.info(log_message)
        print(log_message)
        return df, signal, reason
    return None, None, None

###############################
# Fungsi Otomasi dengan Selenium
###############################

def init_driver(twofa_code=""):
    """
    Inisialisasi driver Selenium secara headless, login, dan tangani 2FA (jika muncul).
    Driver disimpan di st.session_state agar tetap aktif selama auto trade.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Jalankan dalam mode headless (tanpa tampilan)
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 20)
    driver.get("https://binomo2.com/trading")
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    time.sleep(5)
    
    # Proses login
    username_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/div[1]/platform-forms-input/way-input/div/div[1]/way-input-text/input'
    password_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/div[2]/platform-forms-input/way-input/div/div/way-input-password/input'
    login_button_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-sign-in/div/form/vui-button/button'
    
    username = st.secrets.get("username", "username_default")
    password = st.secrets.get("password", "password_default")
    
    wait.until(EC.presence_of_element_located((By.XPATH, username_xpath))).send_keys(username)
    wait.until(EC.presence_of_element_located((By.XPATH, password_xpath))).send_keys(password)
    wait.until(EC.element_to_be_clickable((By.XPATH, login_button_xpath))).click()
    time.sleep(5)
    
    # Tangani 2FA jika muncul
    twofa_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-two-factor-auth-validation/app-otp-validation-form/form/platform-forms-input/way-input/div/div/way-input-text/input'
    try:
        twofa_input = driver.find_element(By.XPATH, twofa_xpath)
        if not twofa_code:
            st.info("Autentikasi 2FA terdeteksi. Masukkan kode 2FA di sidebar dan tekan 'Refresh Data'.")
            return None
        else:
            twofa_input.send_keys(twofa_code)
            twofa_submit_xpath = '/html/body/binomo-root/platform-ui-scroll/div/div/ng-component/ng-component/div/div/auth-form/sa-auth-form/div[2]/div/app-two-factor-auth-validation/app-otp-validation-form/form/vui-button/button'
            wait.until(EC.element_to_be_clickable((By.XPATH, twofa_submit_xpath))).click()
            time.sleep(5)
    except Exception as e:
        logging.info("Autentikasi dua faktor tidak terdeteksi atau sudah tidak diperlukan.")
    return driver

def execute_trade_action(driver, signal):
    """
    Menggunakan driver yang sudah login untuk mengeksekusi trade berdasarkan sinyal.
    Driver tidak akan ditutup setelah eksekusi, agar dapat terus digunakan.
    """
    wait = WebDriverWait(driver, 20)
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

def display_dashboard(df, signal, reason):
    current_time = get_google_time().strftime('%H:%M:%S')
    st.markdown(f"### Auto Trade: **{'Aktif' if st.session_state.auto_trade else 'Nonaktif'}** ( **{current_time}** )")
    
    st.markdown("### Sinyal Trading")
    st.write(f"**Sinyal:** {signal}")
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
    
    # st.markdown("### Grafik RSI")
    # fig_rsi = px.line(df, x='time', y='RSI', title="RSI")
    # st.plotly_chart(fig_rsi, use_container_width=True)
    
    # st.markdown("### Grafik MACD")
    # fig_macd = go.Figure()
    # fig_macd.add_trace(go.Scatter(x=df['time'], y=df['MACD'], mode='lines', name='MACD'))
    # fig_macd.add_trace(go.Scatter(x=df['time'], y=df['MACD_signal'], mode='lines', name='MACD Signal'))
    # fig_macd.update_layout(title="MACD", xaxis_title="Waktu", yaxis_title="Nilai")
    # st.plotly_chart(fig_macd, use_container_width=True)

###############################
# Main: Integrasi Dashboard & Otomasi Trading
###############################

def main():
    if "auto_trade" not in st.session_state:
        st.session_state.auto_trade = False
    if "driver" not in st.session_state:
        st.session_state.driver = None

    st.sidebar.title("Menu")
    refresh = st.sidebar.button("Refresh Data")
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
    
    # Jika auto refresh aktif, hitung sisa waktu hingga pergantian menit berdasarkan waktu Google
    if auto_refresh:
        current_google_time = get_google_time()
        remaining_ms = int((60 - current_google_time.second) * 1000 - current_google_time.microsecond / 1000)
        if remaining_ms < 1000:
            remaining_ms = 60000
        st_autorefresh(interval=remaining_ms, limit=1000, key="auto_refresh")
    
    # Input kode 2FA, jika diperlukan
    twofa_code = st.sidebar.text_input("Masukkan kode 2FA (jika diperlukan):", value="")
    
    df, signal, reason = process_data()
    if df is not None:
        display_dashboard(df, signal, reason)
        
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
