import zipfile
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import mplfinance as mpf

# Extract the zip files if not already extracted
print("Extracting data...")
if not os.path.exists('extracted_data'):
    with zipfile.ZipFile('nifty data.zip', 'r') as zip_ref:
        zip_ref.extractall('extracted_data')
    
    with zipfile.ZipFile('extracted_data/parquet_out (2).zip', 'r') as zip_ref:
        zip_ref.extractall('extracted_data')

# Load all parquet files
print("Loading parquet files...")
parquet_files = glob.glob('extracted_data/parquet_out/symbol=NIFTY 50/date=*/data.parquet')

dataframes = []
for file in parquet_files:
    try:
        df = pd.read_parquet(file)
        # Extract date from file path
        date_str = os.path.basename(os.path.dirname(file)).replace('date=', '')
        df['date'] = pd.to_datetime(date_str)
        dataframes.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Combine all dataframes
print(f"Combining {len(dataframes)} files...")
df = pd.concat(dataframes, ignore_index=True)

# Ensure time column is proper datetime and sort
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

print(f"Total records: {len(df)}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")
print(f"\nData preview:")
print(df.head())

# Create two separate 15-minute candlestick charts for different time spans using mplfinance
print("\nCreating 15-minute candlestick charts for 2024-04-15 to 2024-04-19 and for 2025...")

# Set time as index for resampling (keep original df for prints above)
df_time = df.set_index('time')

def build_ohlc(df_idx, rule: str) -> pd.DataFrame:
    """Resample tick data into OHLC bars at the given frequency, formatted for mplfinance."""
    ohlc = df_idx['ltp'].resample(rule).agg(['first', 'max', 'min', 'last']).dropna()
    ohlc.columns = ['Open', 'High', 'Low', 'Close']
    return ohlc
def detect_swings(df, lookback=1):
    df = df.copy()
    df['swing_high'] = False
    df['swing_low'] = False

    for i in range(lookback, len(df) - lookback):
        prev_highs = df['High'].iloc[i - lookback:i]
        next_highs = df['High'].iloc[i + 1:i + lookback + 1]

        prev_lows = df['Low'].iloc[i - lookback:i]
        next_lows = df['Low'].iloc[i + 1:i + lookback + 1]

        # Strict swing high
        if df['High'].iloc[i] > prev_highs.max() and df['High'].iloc[i] > next_highs.max():
            df.at[df.index[i], 'swing_high'] = True

        # Strict swing low
        if df['Low'].iloc[i] < prev_lows.min() and df['Low'].iloc[i] < next_lows.min():
            df.at[df.index[i], 'swing_low'] = True

    return df

def detect_bos_choch(df):
    df = df.copy()
    df['BOS_bullish'] = False
    df['CHOCH_bearish'] = False

    last_swing_high = None
    last_swing_low = None
    trend = None  # 'bullish' or 'bearish'

    for i in range(len(df)):
        # Update swings
        if df['swing_high'].iloc[i]:
            last_swing_high = df['High'].iloc[i]
            if last_swing_low is not None:
                trend = 'bullish'

        if df['swing_low'].iloc[i]:
            last_swing_low = df['Low'].iloc[i]
            if last_swing_high is not None:
                trend = 'bearish'

        # Bullish BOS (continuation)
        if (
            trend == 'bearish'
            and last_swing_high is not None
            and df['Close'].iloc[i] > last_swing_high
        ):
            df.at[df.index[i], 'BOS_bullish'] = True
            trend = 'bullish'
            last_swing_high = None

        # Bearish CHOCH (reversal)
        if (
            trend == 'bullish'
            and last_swing_low is not None
            and df['Close'].iloc[i] < last_swing_low
        ):
            df.at[df.index[i], 'CHOCH_bearish'] = True
            trend = 'bearish'
            last_swing_low = None

    return df
def build_structure_plots(df):
    """
    Creates mplfinance addplot objects for:
    - Bullish BOS
    - Bearish CHOCH
    """
    bos_prices = np.where(
        df['BOS_bullish'],
        df['Low'] * 0.999,  # slightly below candle
        np.nan
    )

    choch_prices = np.where(
        df['CHOCH_bearish'],
        df['High'] * 1.001,  # slightly above candle
        np.nan
    )

    bos_plot = mpf.make_addplot(
        bos_prices,
        type='scatter',
        marker='^',
        markersize=100,
        color='green'
    )

    choch_plot = mpf.make_addplot(
        choch_prices,
        type='scatter',
        marker='v',
        markersize=100,
        color='red'
    )

    return [bos_plot, choch_plot]

# --- Plot 1: 15th to 19th April 2024 (inclusive) ---
start_2024 = pd.Timestamp('2024-04-15 00:00:00')
end_2024 = pd.Timestamp('2024-04-19 23:59:59')
df_2024_range = df_time.loc[start_2024:end_2024]

if not df_2024_range.empty:
    ohlc_15m_2024 = build_ohlc(df_2024_range, '15T')
    ohlc_15m_2024 = detect_swings(ohlc_15m_2024, lookback=2)
    ohlc_15m_2024 = detect_bos_choch(ohlc_15m_2024) # Add BOS and CHOCH detection
    addplots_2024 = build_structure_plots(ohlc_15m_2024)

    mpf.plot(
        ohlc_15m_2024,
        type='candle',
        style='yahoo',
        title='NIFTY 50 - 15-Minute Candlestick with BOS & CHOCH (2024-04-15 to 2024-04-19)',
        ylabel='Price',
        datetime_format='%Y-%m-%d %H:%M',
        show_nontrading=False,
        addplot=addplots_2024,
        savefig='nifty_candlestick_15m_2024_BOS_CHOCH.png',
    )

    print("\nCandlestick visualization saved as 'nifty_candlestick_15m_2024_04_15_19.png'")
else:
    print("\nNo data found for 2024-04-15 to 2024-04-19.")

# --- Plot 2: All data in 2025 ---
df_2025 = df_time[df_time.index.year == 2025]

if not df_2025.empty:
    ohlc_15m_2025 = build_ohlc(df_2025, '15T')
    ohlc_15m_2025 = detect_swings(ohlc_15m_2025, lookback=2)
    ohlc_15m_2025 = detect_bos_choch(ohlc_15m_2025) # Add BOS and CHOCH detection

    addplots_2025 = build_structure_plots(ohlc_15m_2025)

    mpf.plot(
        ohlc_15m_2025,
        type='candle',
        style='yahoo',
        title='NIFTY 50 - 15-Minute Candlestick with BOS & CHOCH (2025)',
        ylabel='Price',
        datetime_format='%Y-%m-%d %H:%M',
        show_nontrading=False,
        addplot=addplots_2025,
        savefig='nifty_candlestick_15m_2025_BOS_CHOCH.png',
    )

    print("\nCandlestick visualization saved as 'nifty_candlestick_15m_2025.png'")
else:
    print("\nNo data found for year 2025.")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"Total Records: {len(df):,}")
print(f"Date Range: {df['time'].min()} to {df['time'].max()}")
print(f"\nPrice Statistics (LTP):")
print(f"  Mean: {df['ltp'].mean():.2f}")
print(f"  Median: {df['ltp'].median():.2f}")
print(f"  Std Dev: {df['ltp'].std():.2f}")
print(f"  Min: {df['ltp'].min():.2f}")
print(f"  Max: {df['ltp'].max():.2f}")
print("\n(Note: Trading volume and buy/sell quantity visualizations and stats have been omitted.)")

