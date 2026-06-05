import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import matplotlib.ticker as mticker
from smartmoneyconcepts import smc as _smc

# ============================================================
# 1. ASCTrend1i — exact Python port
# ============================================================
RANGE_FACTOR = 4.6
ALT_PERIOD   = 4

def calculate_asc_trend(df, risk=6):
    df = df.copy()
    n = len(df)
    wpr_period = 3 + risk * 2   # 15
    high_level = 67 + risk       # 73
    low_level  = 33 - risk       # 27

    hi = df['High'].values
    lo = df['Low'].values
    cl = df['Close'].values

    raw_range = np.abs(hi - lo)
    avg_range = np.full(n, np.nan)
    for i in range(9, n):
        avg_range[i] = raw_range[i-9:i+1].mean()

    def wpr_at(i, period):
        if i < period - 1:
            return np.nan
        h_max = hi[i-period+1:i+1].max()
        l_min = lo[i-period+1:i+1].min()
        if h_max == l_min:
            return 50.0
        return 100.0 - abs(-100.0 * (h_max - cl[i]) / (h_max - l_min))

    signal = np.zeros(n)
    dn_sig  = np.zeros(n)
    up_sig  = np.zeros(n)

    for i in range(n):
        ar = avg_range[i]
        if np.isnan(ar):
            continue
        use_alt = False
        for lag in range(6):
            j, j3 = i - lag, i - lag - 3
            if j >= 0 and j3 >= 0 and abs(cl[j] - cl[j3]) >= ar * RANGE_FACTOR:
                use_alt = True
                break
        wv = wpr_at(i, ALT_PERIOD if use_alt else wpr_period)
        if np.isnan(wv):
            continue
        prev = signal[i-1] if i > 0 else 0
        if   wv >= high_level:              signal[i] =  1
        elif wv <= low_level:               signal[i] = -1
        elif prev ==  1 and wv > low_level: signal[i] =  1
        elif prev == -1 and wv < high_level:signal[i] = -1
        else:                               signal[i] =  0

        if signal[i] == -1 and prev ==  1:
            dn_sig[i] = hi[i] + ar * 0.5
        if signal[i] ==  1 and prev == -1:
            up_sig[i] = lo[i] - ar * 0.5

    df['Signal']  = signal
    df['UpSignal'] = up_sig > 0
    df['DnSignal'] = dn_sig > 0
    return df

# ============================================================
# 2. Exit indicators — ATR 22, EMA 50, Chandelier Exit
# ============================================================
def compute_indicators(df, atr_period=22, ema_period=50, ce_mult=3.0):
    df = df.copy()
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low']  - prev_close).abs()
    ], axis=1).max(axis=1)
    df['ATR']      = tr.rolling(atr_period).mean()
    df['EMA50']    = df['Close'].ewm(span=ema_period, adjust=False).mean()
    # Chandelier Exit: 3 × ATR from the rolling high/low
    df['CE_long']  = df['High'].rolling(atr_period).max() - ce_mult * df['ATR']
    df['CE_short'] = df['Low'].rolling(atr_period).min()  + ce_mult * df['ATR']
    return df

# ============================================================
# 3. Oscillators — RSI, Stochastic, Drake Delayed Stochastic
# ============================================================
def _rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return (100 - 100 / (1 + ag / al.replace(0, np.nan))).clip(0, 100)

def _stoch_k(df, k_period, smooth_k):
    hi    = df['High'].rolling(k_period).max()
    lo    = df['Low'].rolling(k_period).min()
    raw_k = (df['Close'] - lo) / (hi - lo + 1e-9) * 100
    return raw_k.rolling(smooth_k).mean().clip(0, 100)

def compute_oscillators(base_df):
    """M15 RSI/Stoch болон H4/H1 Drake Delayed Stoch нэмнэ."""
    df = base_df[['Open','High','Low','Close']].copy()
    out = pd.DataFrame(index=df.index)

    # ── M15 RSI(14) ─────────────────────────────────────────
    rsi = _rsi(df['Close'], 14)
    out['RSI14']    = rsi
    out['RSI14_up'] = rsi > rsi.shift(1)
    out['RSI14_dn'] = rsi < rsi.shift(1)

    # ── M15 Stochastics ─────────────────────────────────────
    for (kp, sk), name in [
        ((8,   3),  'Sto833'),
        ((20,  10), 'Sto20'),
        ((100, 10), 'Sto100'),
    ]:
        k = _stoch_k(df, kp, sk)
        out[f'{name}_K']  = k
        out[f'{name}_up'] = k > k.shift(1)
        out[f'{name}_dn'] = k < k.shift(1)

    # ── H4 / H1 Drake Delayed Stochastic (period=8, delay=13, smooth=9) ──
    for htf, col in [('240min', 'H4_DDS'), ('60min', 'H1_DDS')]:
        h   = base_df.resample(htf).agg(
                {'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
        hi  = h['High'].rolling(8).max()
        lo  = h['Low'].rolling(8).min()
        rng = (hi - lo).replace(0, np.nan)
        raw_k = (h['Close'].shift(13) - lo) / rng * 100
        k   = raw_k.rolling(9).mean().clip(0, 100)
        dds = k.reindex(df.index, method='ffill')
        out[col]           = dds
        out[f'{col}_lt10'] = dds < 10
        out[f'{col}_gt90'] = dds > 90

    return out

# ============================================================
# 4. H1 Engulfing — map to M15 exit signal
#    Bear engulfing → exit long | Bull engulfing → exit short
#    Signal fires at first M15 bar of the NEXT H1 candle (H1 candle just closed)
# ============================================================
def compute_h1_engulfing(base_df, htf='60min'):
    h1 = base_df.resample(htf).agg(
        {'Open':'first','High':'max','Low':'min','Close':'last'}
    ).dropna()

    po, pc = h1['Open'].shift(1),  h1['Close'].shift(1)
    ph, pl = h1['High'].shift(1),  h1['Low'].shift(1)
    co, cc = h1['Open'],           h1['Close']

    # Swing high/low: previous candle must be the swing high/low of the last N H1 bars
    swing_n = 5
    is_swing_high = ph >= h1['High'].shift(1).rolling(swing_n).max()
    is_swing_low  = pl <= h1['Low'].shift(1).rolling(swing_n).min()

    # Bearish engulfing:
    #   - prev candle bullish AND is the local swing HIGH (past swing_n H1 bars)
    #   - curr candle bearish, body engulfs prev full candle (including wicks)
    bear = (pc > po) & (cc < co) & (co >= ph) & (cc <= pl) & is_swing_high

    # Bullish engulfing: symmetric — prev is swing LOW
    bull = (pc < po) & (cc > co) & (co <= pl) & (cc >= ph) & is_swing_low

    dt = pd.Timedelta(htf)
    bear_times = set(h1.index[bear] + dt)
    bull_times = set(h1.index[bull] + dt)

    out = base_df.copy()
    out['H1_Bear_Engulf'] = out.index.isin(bear_times)
    out['H1_Bull_Engulf'] = out.index.isin(bull_times)
    return out[['H1_Bear_Engulf', 'H1_Bull_Engulf']]

# ============================================================
# 4. M5 CHoCH mapped to M15
#    CHoCH computed on M5 bars; break/swing timestamps mapped
#    back to M15 integer indices so run_backtest can use them.
# ============================================================
def compute_m5_choch_on_m15(m5_df, m15_df, swing_length=5):
    ohlc  = m5_df[['Open','High','Low','Close']].rename(columns=str.lower).reset_index(drop=True)
    swing = _smc.swing_highs_lows(ohlc, swing_length=swing_length)
    bc    = _smc.bos_choch(ohlc, swing, close_break=True)

    m5_times  = m5_df.index.values          # numpy datetime64 array — fast searchsorted
    m15_times = m15_df.index.values

    choch5_dir   = np.zeros(len(m15_df))
    choch5_swing = np.full(len(m15_df), -1, dtype=int)

    for swing_idx, r in bc.dropna(subset=['CHOCH']).iterrows():
        bi = int(r['BrokenIndex'])
        if bi >= len(m5_times):
            continue

        break_ts = m5_times[bi]
        swing_ts = m5_times[int(swing_idx)]

        # M5 break timestamp → which M15 bar contains it
        m15_bi = int(np.searchsorted(m15_times, break_ts, side='right')) - 1
        if not (0 <= m15_bi < len(m15_df)):
            continue

        # M5 swing formation timestamp → M15 bar index (for after-entry filter)
        m15_si = max(0, int(np.searchsorted(m15_times, swing_ts, side='right')) - 1)

        choch5_dir[m15_bi]   = r['CHOCH']
        choch5_swing[m15_bi] = m15_si

    result = pd.DataFrame(
        {'CHOCH5': choch5_dir, 'CHOCH5_swing': choch5_swing},
        index=m15_df.index
    )
    return result

# ============================================================
# 5. CHoCH — Change of Character via smartmoneyconcepts
#    Returns df with 'CHOCH' column aligned to the BREAK bar:
#      -1 = bearish CHoCH (broke below swing low → exit long)
#      +1 = bullish CHoCH (broke above swing high → exit short)
#       0 = no CHoCH on this bar
# ============================================================
def compute_choch(df, swing_length=10):
    ohlc  = df[['Open','High','Low','Close']].rename(columns=str.lower).reset_index(drop=True)
    swing = _smc.swing_highs_lows(ohlc, swing_length=swing_length)
    bc    = _smc.bos_choch(ohlc, swing, close_break=True)

    choch_dir   = np.zeros(len(df))
    choch_swing = np.full(len(df), -1, dtype=int)  # index of swing that was broken

    for swing_idx, r in bc.dropna(subset=['CHOCH']).iterrows():
        bi = int(r['BrokenIndex'])
        if 0 <= bi < len(df):
            choch_dir[bi]   = r['CHOCH']      # -1 or +1
            choch_swing[bi] = int(swing_idx)  # when the source swing was FORMED

    out = df.copy()
    out['CHOCH']       = choch_dir
    out['CHOCH_swing'] = choch_swing           # used to filter pre-entry swings
    return out

# ============================================================
# 4. MTF Arrow — map HTF signals to M15
# ============================================================
def mtf_arrow_local(file_path, higher_tf='60min', risk=6, year=2025, m5_file=None):
    print(f"[INFO] Loading: {file_path}")
    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')
    df.columns = [col.capitalize() for col in df.columns]
    base_df = df[['Open','High','Low','Close']].dropna()
    base_df = base_df[base_df.index.year == year]
    print(f"[INFO] M15 bars ({year}): {len(base_df)}")

    higher_df = base_df.resample(higher_tf).agg(
        {'Open':'first','High':'max','Low':'min','Close':'last'}).dropna()
    print(f"[INFO] HTF bars: {len(higher_df)} — running ASCTrend1i ...")
    higher_df = calculate_asc_trend(higher_df, risk=risk)

    higher_arrows = higher_df[['UpSignal','DnSignal']].shift(1).fillna(False)
    up_times = set(higher_arrows[higher_arrows['UpSignal']].index)
    dn_times = set(higher_arrows[higher_arrows['DnSignal']].index)
    print(f"[INFO] HTF signals — Up: {len(up_times)}  Dn: {len(dn_times)}")

    final_df = base_df.copy()
    final_df['HTF_bucket'] = final_df.index.floor(higher_tf)
    avg_off = (final_df['High'] - final_df['Low']).mean() * 0.6
    final_df['UpArrow'] = np.where(final_df['HTF_bucket'].isin(up_times),
                                   final_df['Low']  - avg_off, np.nan)
    final_df['DnArrow'] = np.where(final_df['HTF_bucket'].isin(dn_times),
                                   final_df['High'] + avg_off, np.nan)

    final_df = compute_indicators(final_df)   # ATR, EMA50 (for chart display)

    print("[INFO] Computing oscillators (RSI14, Stoch, Drake DDS H4/H1) ...")
    osc = compute_oscillators(base_df)
    final_df = final_df.join(osc, how='left')
    bool_cols = [c for c in osc.columns if c.endswith(('_up', '_dn', '_lt10', '_gt90'))]
    for c in bool_cols:
        final_df[c] = final_df[c].fillna(False)

    engulf = compute_h1_engulfing(base_df, htf=higher_tf)
    final_df = final_df.join(engulf, how='left')
    final_df['H1_Bear_Engulf'] = final_df['H1_Bear_Engulf'].fillna(False)
    final_df['H1_Bull_Engulf'] = final_df['H1_Bull_Engulf'].fillna(False)
    n_be = int(final_df['H1_Bear_Engulf'].sum())
    n_bu = int(final_df['H1_Bull_Engulf'].sum())
    print(f"[INFO] H1 engulfing — Bear: {n_be}  Bull: {n_bu}")

    print("[INFO] Computing CHoCH swing=10 (M15) ...")
    final_df = compute_choch(final_df, swing_length=10)
    n_choch = int((final_df['CHOCH'] != 0).sum())

    # CHoCH5: compute on M5 data if available, else fall back to M15
    if m5_file:
        print(f"[INFO] Loading M5 data: {m5_file}")
        m5_raw = pd.read_csv(m5_file, parse_dates=['time'], index_col='time')
        m5_raw.columns = [col.capitalize() for col in m5_raw.columns]
        m5_df = m5_raw[['Open','High','Low','Close']].dropna()
        m5_df = m5_df[m5_df.index.year == year]
        print(f"[INFO] M5 bars ({year}): {len(m5_df)}")
        print("[INFO] Computing CHoCH swing=5 (M5, 3R+ sensitive) ...")
        tmp5 = compute_m5_choch_on_m15(m5_df, final_df, swing_length=5)
    else:
        print("[INFO] Computing CHoCH swing=5 (M15 fallback, 3R+ sensitive) ...")
        tmp5 = compute_choch(final_df, swing_length=5).rename(
            columns={'CHOCH': 'CHOCH5', 'CHOCH_swing': 'CHOCH5_swing'})

    final_df['CHOCH5']       = tmp5['CHOCH5']
    final_df['CHOCH5_swing'] = tmp5['CHOCH5_swing']
    n5 = int((final_df['CHOCH5'] != 0).sum())
    print(f"[INFO] CHoCH signals — swing10: {n_choch}  swing5(M5): {n5}")
    return final_df

# ============================================================
# 5. Backtest
#    Entry   : Open of bar AFTER HTF signal bar
#    Hard SL : swing low/high of last sl_lookback bars
#    Exit    : (1) Hard SL  (2) CHoCH on post-entry swing
#    P&L     : (exit_price − entry_price) × lot_size
# ============================================================
def run_backtest(final_df, initial_capital=10_000, lot_size=1.0, sl_lookback=10, be_r=1.5):
    df = final_df.reset_index()
    n  = len(df)

    trades          = []
    capital         = initial_capital
    eq_curve        = [capital]
    position        = None        # shared trade state for all 3 legs
    last_sig_bucket = None
    leg_lot         = lot_size          # each leg uses full lot_size

    def record_leg(p, leg, exit_price, result, exit_time):
        nonlocal capital
        side       = p['side']
        ep         = p['entry_price']
        mfep       = p['mfe_price']
        price_diff = exit_price - ep
        if side == 'short':
            price_diff = -price_diff
        mfe_pts = (mfep - ep) if side == 'long' else (ep - mfep)
        pnl     = round(price_diff * leg_lot, 2)
        capital += pnl
        trades.append({
            'leg':           leg,
            'side':          side,
            'entry_time':    p['entry_time'],
            'signal_time':   p['signal_time'],
            'entry_price':   ep,
            'sl':            p['sl'],
            'tp':            p['tp1'] if leg == 1 else (p['tp2'] if leg == 2 else np.nan),
            'exit_price':    exit_price,
            'exit_time':     exit_time,
            'result':        result,
            'pnl':           pnl,
            'risk_dist':     p['risk_dist'],
            'entry_bar_idx': p['entry_bar_idx'],
            'mfe_price':     mfep,
            'mfe':           round(mfe_pts, 2),
        })
        p['legs_open'].discard(leg)

    for i in range(1, n):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── Check open position ──────────────────────────────
        if position is not None:
            p    = position
            side = p['side']

            # Track MFE (shared across all legs)
            if side == 'long':
                p['mfe_price'] = max(p['mfe_price'], row['High'])
            else:
                p['mfe_price'] = min(p['mfe_price'], row['Low'])

            ep  = p['entry_price']
            mfe = (p['mfe_price'] - ep) if side == 'long' else (ep - p['mfe_price'])

            # Breakeven trail (shared SL)
            if be_r > 0 and mfe >= be_r * p['risk_dist']:
                if side == 'long':
                    p['sl'] = max(p['sl'], ep)
                else:
                    p['sl'] = min(p['sl'], ep)

            entry_idx = p['entry_bar_idx']
            sl        = p['sl']

            # ── SL hit → close all remaining legs ───────────
            sl_hit = ((side == 'long'  and row['Low']  <= sl) or
                      (side == 'short' and row['High'] >= sl))
            if sl_hit:
                for leg in list(p['legs_open']):
                    record_leg(p, leg, sl, 'SL', row['time'])
            else:
                # ── Reversal exits (CHoCH / ENGULF) → close all ─
                choch       = row.get('CHOCH', 0.0)
                choch_swing = int(row.get('CHOCH_swing', -1))
                choch_ok    = choch_swing > entry_idx

                choch5       = row.get('CHOCH5', 0.0)
                choch5_swing = int(row.get('CHOCH5_swing', -1))
                choch5_ok    = (choch5_swing > entry_idx) and (mfe >= 3.0 * p['risk_dist'])

                h1_bear = bool(row.get('H1_Bear_Engulf', False))
                h1_bull = bool(row.get('H1_Bull_Engulf', False))

                rev_price, rev_result = None, None
                if side == 'long':
                    if   choch == -1 and choch_ok:   rev_price, rev_result = row['Close'], 'CHoCH'
                    elif choch5 == -1 and choch5_ok: rev_price, rev_result = row['Close'], 'CHoCH5'
                    elif h1_bear:                    rev_price, rev_result = row['Open'],  'ENGULF'
                else:
                    if   choch == 1 and choch_ok:    rev_price, rev_result = row['Close'], 'CHoCH'
                    elif choch5 == 1 and choch5_ok:  rev_price, rev_result = row['Close'], 'CHoCH5'
                    elif h1_bull:                    rev_price, rev_result = row['Open'],  'ENGULF'

                if rev_result is not None:
                    for leg in list(p['legs_open']):
                        record_leg(p, leg, rev_price, rev_result, row['time'])
                else:
                    # ── TP exits (leg-specific) ──────────────────
                    # Leg 1 → TP1 (1R)
                    if 1 in p['legs_open']:
                        tp1_hit = ((side == 'long'  and row['High'] >= p['tp1']) or
                                   (side == 'short' and row['Low']  <= p['tp1']))
                        if tp1_hit:
                            record_leg(p, 1, p['tp1'], 'TP1', row['time'])

                    # Leg 2 → TP2 (2R)
                    if 2 in p['legs_open']:
                        tp2_hit = ((side == 'long'  and row['High'] >= p['tp2']) or
                                   (side == 'short' and row['Low']  <= p['tp2']))
                        if tp2_hit:
                            record_leg(p, 2, p['tp2'], 'TP2', row['time'])

                    # Leg 3 — DDS exit (Leg 1, 2 хаагдсаны дараа л)
                    if (3 in p['legs_open']
                            and 1 not in p['legs_open']
                            and 2 not in p['legs_open']):
                        if side == 'long':
                            dds_exit = (row.get('H4_DDS_lt10', False) and
                                        row.get('H1_DDS_lt10', False) and
                                        row.get('RSI14_up',    False) and
                                        row.get('Sto833_up',   False) and
                                        row.get('Sto20_up',    False) and
                                        row.get('Sto100_up',   False))
                        else:  # short
                            dds_exit = (row.get('H4_DDS_gt90', False) and
                                        row.get('H1_DDS_gt90', False) and
                                        row.get('RSI14_dn',    False) and
                                        row.get('Sto833_dn',   False) and
                                        row.get('Sto20_dn',    False) and
                                        row.get('Sto100_dn',   False))
                        if dds_exit:
                            record_leg(p, 3, row['Close'], 'DDS', row['time'])

            if not p['legs_open']:
                position = None

        eq_curve.append(capital)

        # ── Open new trade if flat ───────────────────────────
        if position is None:
            up = not np.isnan(prev['UpArrow'])
            dn = not np.isnan(prev['DnArrow'])

            if up or dn:
                sig_bucket = prev['HTF_bucket']
                if sig_bucket == last_sig_bucket:
                    continue
                # 4 дэх M15 bar хаагдсаны дараа орно:
                # prev сүүлийн M15 bar байх ёстой → row өөр bucket-т байна
                if prev['HTF_bucket'] == row['HTF_bucket']:
                    continue
                last_sig_bucket = sig_bucket

                sig_idx  = i - 1
                lookback = df.iloc[max(0, sig_idx - sl_lookback + 1) : sig_idx + 1]
                entry    = row['Open']

                if up:
                    sl, side = lookback['Low'].min(),  'long'
                else:
                    sl, side = lookback['High'].max(), 'short'

                risk_dist = abs(entry - sl)
                if risk_dist < 1e-6:
                    continue

                # EMA50 entry filter
                ema50_now = row.get('EMA50', np.nan)
                if not np.isnan(ema50_now):
                    if side == 'long'  and row['Close'] < ema50_now:
                        continue
                    if side == 'short' and row['Close'] > ema50_now:
                        continue

                # Pre-move filter: signal-аас өмнөх 10 bar-д
                # trade чиглэлд 3×ATR-аас их хөдөлсөн бол skip
                atr_now = prev.get('ATR', np.nan)
                if not np.isnan(atr_now) and atr_now > 0:
                    pre_w      = df.iloc[max(0, sig_idx - 9) : sig_idx + 1]
                    close_now  = pre_w['Close'].iloc[-1]
                    close_prev = pre_w['Close'].iloc[0]
                    if side == 'long':
                        pre_move = max(0.0, close_now - close_prev)
                    else:
                        pre_move = max(0.0, close_prev - close_now)
                    if pre_move > 3.0 * atr_now:
                        continue


                tp1 = entry + risk_dist       if side == 'long' else entry - risk_dist
                tp2 = entry + 2.0 * risk_dist if side == 'long' else entry - 2.0 * risk_dist
                position = {
                    'side':          side,
                    'entry_time':    row['time'],
                    'signal_time':   prev['time'],
                    'entry_price':   entry,
                    'sl':            sl,
                    'tp1':           tp1,
                    'tp2':           tp2,
                    'risk_dist':     risk_dist,
                    'entry_bar_idx': i,
                    'mfe_price':     entry,
                    'legs_open':     {1, 2, 3},
                }

    # Close any still-open legs at last bar close
    if position is not None:
        last = df.iloc[-1]
        for leg in list(position['legs_open']):
            record_leg(position, leg, last['Close'], 'OPEN', last['time'], n - 1)
        eq_curve.append(capital)

    return pd.DataFrame(trades), eq_curve, initial_capital

# ============================================================
# 5. Statistics
# ============================================================
def calc_stats(trades_df, eq_curve, initial_capital):
    if trades_df.empty:
        return {}

    closed = trades_df[trades_df['result'] != 'OPEN']
    wins   = closed[closed['pnl'] > 0]
    losses = closed[closed['pnl'] < 0]

    gross_profit = wins['pnl'].sum()
    gross_loss   = abs(losses['pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    eq = np.array(eq_curve)
    running_max = np.maximum.accumulate(eq)
    dd = running_max - eq
    max_dd = dd.max()
    max_dd_pct = max_dd / running_max[dd.argmax()] * 100 if running_max[dd.argmax()] > 0 else 0

    by_result = closed['result'].value_counts()
    exit_str  = '  '.join(f"{k}:{v}" for k, v in by_result.items())

    n_signals = closed['entry_time'].nunique() if 'entry_time' in closed.columns else len(closed) // 3
    return {
        'Total signals':    n_signals,
        'Total legs':       len(closed),
        'Winners':          len(wins),
        'Losers':           len(losses),
        'Win rate (legs)':  f"{len(wins)/len(closed)*100:.1f}%" if len(closed) else "0%",
        'Profit factor':    f"{pf:.2f}",
        'Net P&L':          f"${closed['pnl'].sum():+,.0f}",
        'Gross profit':     f"${gross_profit:,.0f}",
        'Gross loss':       f"${gross_loss:,.0f}",
        'Avg win':          f"${wins['pnl'].mean():,.0f}" if len(wins) else "$0",
        'Avg loss':         f"${losses['pnl'].mean():,.0f}" if len(losses) else "$0",
        'Best trade':       f"${closed['pnl'].max():+,.0f}" if len(closed) else "$0",
        'Worst trade':      f"${closed['pnl'].min():+,.0f}" if len(closed) else "$0",
        'Max drawdown':     f"${max_dd:,.0f}  ({max_dd_pct:.1f}%)",
        'Final capital':    f"${eq_curve[-1]:,.0f}",
        'Return':           f"{(eq_curve[-1]-initial_capital)/initial_capital*100:+.1f}%",
        'Exit types':       exit_str,
    }

# ============================================================
# 5. Per-trade detail print
# ============================================================
def print_trade_detail(trades_df, save_path=None):
    closed = trades_df[trades_df['result'] != 'OPEN'].copy()
    if closed.empty:
        print("  No closed trades.")
        return

    lines = []
    hdr = (f"{'#':>4}  {'Lg':<2}  {'Side':<5}  {'Entry time':<17}  {'Entry':>8}  "
           f"{'Exit':>8}  {'Result':<6}  {'PnL($)':>8}  {'MFE(pts)':>9}  {'Walked back':>12}")
    sep = '-' * len(hdr)
    lines += [sep, hdr, sep]

    for n, (_, t) in enumerate(closed.iterrows(), 1):
        side   = t['side'].upper()[:1]
        leg    = int(t.get('leg', 0))
        etime  = str(t['entry_time'])[:16]
        entry  = t['entry_price']
        exit_p = t['exit_price']
        result = t['result']
        pnl    = t['pnl']
        mfe    = t.get('mfe', np.nan)

        if t['side'] == 'long':
            walkback = (t['mfe_price'] - exit_p) if 'mfe_price' in t else np.nan
        else:
            walkback = (exit_p - t['mfe_price']) if 'mfe_price' in t else np.nan

        pnl_s = f"{pnl:+.1f}"
        # MFE/WalkBack нь зөвхөн leg 3-т утгатай (TP1/TP2 fixed exit)
        if result in ('TP1', 'TP2'):
            mfe_s = "     ---"
            wb_s  = "         ---"
        else:
            mfe_s = f"+{mfe:.1f}" if not np.isnan(mfe) else "  n/a"
            wb_s  = f"-{walkback:.1f}" if (not isinstance(walkback, float) or not np.isnan(walkback)) else "  n/a"
        lines.append(f"{n:>4}  {leg:<2}  {side:<5}  {etime:<17}  {entry:>8.2f}  "
                     f"{exit_p:>8.2f}  {result:<6}  {pnl_s:>8}  {mfe_s:>9}  {wb_s:>12}")

    lines.append(sep)
    for tag in ['SL', 'CHoCH', 'CHoCH5', 'ENGULF', 'DDS']:
        grp = closed[closed['result'] == tag]['mfe']
        if len(grp):
            lines.append(f"  Avg MFE before {tag:<6}: {grp.mean():+.2f} pts  (n={len(grp)})")

    output = '\n'.join(lines)
    print(output)

    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(output + '\n')
        print(f"\n[INFO] Trade detail saved: {save_path}")

# ============================================================
# 6. Plot backtest results
# ============================================================
def plot_backtest(trades_df, eq_curve, initial_capital, stats, year=2025):
    BG   = '#131722'
    FG   = '#d1d4dc'
    GRN  = '#26a69a'
    RED  = '#ef5350'
    BLUE = '#2962ff'

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.canvas.manager.set_window_title(f'MTFArrow Backtest — XAUUSD M15 {year}')
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.07,
                           hspace=0.45, wspace=0.35)

    # ── Equity curve ─────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor(BG)
    xs = np.arange(len(eq_curve))
    ax_eq.plot(xs, eq_curve, color=BLUE, linewidth=1.4, zorder=3)
    ax_eq.axhline(initial_capital, color='#555', linewidth=0.8, linestyle='--')
    ax_eq.fill_between(xs, initial_capital, eq_curve,
                        where=np.array(eq_curve) >= initial_capital,
                        color=GRN, alpha=0.18)
    ax_eq.fill_between(xs, initial_capital, eq_curve,
                        where=np.array(eq_curve) <  initial_capital,
                        color=RED, alpha=0.25)
    ax_eq.set_title('Equity Curve', color=FG, fontsize=10, pad=4)
    ax_eq.tick_params(colors=FG, labelsize=7)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    for sp in ax_eq.spines.values(): sp.set_edgecolor('#363c4e')
    ax_eq.grid(color='#363c4e', linewidth=0.4, linestyle='--')

    # ── Monthly P&L ──────────────────────────────────────────
    ax_mo = fig.add_subplot(gs[1, 0])
    ax_mo.set_facecolor(BG)
    closed = trades_df[trades_df['result'] != 'OPEN'].copy()
    if not closed.empty:
        closed['month'] = pd.to_datetime(closed['exit_time']).dt.to_period('M')
        monthly = closed.groupby('month')['pnl'].sum()
        cols_m  = [GRN if v >= 0 else RED for v in monthly.values]
        ax_mo.bar(range(len(monthly)), monthly.values, color=cols_m, width=0.6)
        ax_mo.set_xticks(range(len(monthly)))
        ax_mo.set_xticklabels([str(m) for m in monthly.index],
                               rotation=45, ha='right', fontsize=6.5)
        ax_mo.axhline(0, color='#555', linewidth=0.7)
        ax_mo.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_mo.set_title('Monthly P&L', color=FG, fontsize=10, pad=4)
    ax_mo.tick_params(colors=FG, labelsize=7)
    for sp in ax_mo.spines.values(): sp.set_edgecolor('#363c4e')

    # ── Trade distribution ────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.set_facecolor(BG)
    if not closed.empty:
        wins_vals   = closed[closed['pnl'] > 0]['pnl'].values
        losses_vals = closed[closed['pnl'] < 0]['pnl'].values
        if len(wins_vals):
            ax_hist.hist(wins_vals,   bins=20, color=GRN, alpha=0.75, label='Win')
        if len(losses_vals):
            ax_hist.hist(losses_vals, bins=20, color=RED, alpha=0.75, label='Loss')
        ax_hist.legend(facecolor='#1e222d', edgecolor='#363c4e',
                       labelcolor=FG, fontsize=8)
    ax_hist.set_title('P&L Distribution', color=FG, fontsize=10, pad=4)
    ax_hist.tick_params(colors=FG, labelsize=7)
    for sp in ax_hist.spines.values(): sp.set_edgecolor('#363c4e')

    # ── Stats table ──────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('off')

    # Separate 'Exit types' — show as text row below the table
    all_items  = list(stats.items())
    exit_str   = stats.get('Exit types', '')
    main_items = [(k, v) for k, v in all_items if k != 'Exit types']

    half       = (len(main_items) + 1) // 2
    row1       = main_items[:half]
    row2       = main_items[half:]
    while len(row2) < len(row1):          # pad shorter row
        row2.append(('', ''))

    col_labels = [k for k, _ in row1]
    cell_text  = [
        [v for _, v in row1],
        [v for _, v in row2],
    ]

    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center', loc='upper center',
        bbox=[0, 0.28, 1, 0.72]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_facecolor('#1e222d' if r == 0 else BG)
        cell.set_edgecolor('#363c4e')
        cell.set_text_props(color=FG if r > 0 else '#aaaaaa')

    # Exit types — full-width text row below table
    if exit_str:
        ax_tbl.text(0.5, 0.10,
                    f'Exit types:  {exit_str}',
                    ha='center', va='center',
                    color='#aaaaaa', fontsize=8,
                    transform=ax_tbl.transAxes)

    fig.suptitle(f'MTFArrow Backtest — XAUUSD M15 {year}  |  Exit: CHoCH / SL',
                 color=FG, fontsize=12, fontweight='bold')
    plt.show()

# ============================================================
# 6. Interactive chart
# ============================================================
def draw_bars(ax, df_window, trades_window=None, bar_width=0.4):
    ax.cla()
    n   = len(df_window)
    idx = df_window.index   # DatetimeIndex

    # ── OHLC bars ────────────────────────────────────────────
    segs_hl, segs_op, segs_cl, colors = [], [], [], []
    for i, (_, row) in enumerate(df_window.iterrows()):
        o, h, l, c = row['Open'], row['High'], row['Low'], row['Close']
        col = '#26a69a' if c >= o else '#ef5350'
        colors.append(col)
        segs_hl.append([(i, l), (i, h)])
        segs_op.append([(i - bar_width, o), (i, o)])
        segs_cl.append([(i, c), (i + bar_width, c)])

    ax.add_collection(LineCollection(segs_hl, colors=colors, linewidths=1.2))
    ax.add_collection(LineCollection(segs_op, colors=colors, linewidths=1.2))
    ax.add_collection(LineCollection(segs_cl, colors=colors, linewidths=1.2))

    # ── EMA 50 ───────────────────────────────────────────────
    if 'EMA50' in df_window.columns:
        ema_vals = df_window['EMA50'].values
        valid = ~np.isnan(ema_vals)
        if valid.any():
            ax.plot(np.arange(n)[valid], ema_vals[valid],
                    color='#ff9800', linewidth=1.1, alpha=0.85,
                    zorder=2, label='EMA 50')

    # ── H1 Engulfing markers ──────────────────────────────────
    if 'H1_Bear_Engulf' in df_window.columns:
        be_pts = [(i, df_window.iloc[i]['High'] * 1.0006)
                  for i in range(n) if df_window.iloc[i]['H1_Bear_Engulf']]
        bu_pts = [(i, df_window.iloc[i]['Low']  * 0.9994)
                  for i in range(n) if df_window.iloc[i]['H1_Bull_Engulf']]
        if be_pts:
            ax.plot(*zip(*be_pts), 's', color='#f97316', markersize=6,
                    alpha=0.85, zorder=4, label='H1 Bear Engulf')
        if bu_pts:
            ax.plot(*zip(*bu_pts), 's', color='#38bdf8', markersize=6,
                    alpha=0.85, zorder=4, label='H1 Bull Engulf')

    # ── CHoCH markers ─────────────────────────────────────────
    if 'CHOCH' in df_window.columns:
        choch_vals = df_window['CHOCH'].values
        bull_choch = [(i, df_window.iloc[i]['Low']  * 0.9994)
                      for i in range(n) if choch_vals[i] == 1]
        bear_choch = [(i, df_window.iloc[i]['High'] * 1.0006)
                      for i in range(n) if choch_vals[i] == -1]
        if bull_choch:
            ax.plot(*zip(*bull_choch), 'D', color='#a855f7', markersize=5,
                    alpha=0.8, zorder=4, label='CHoCH Bull')
        if bear_choch:
            ax.plot(*zip(*bear_choch), 'D', color='#f59e0b', markersize=5,
                    alpha=0.8, zorder=4, label='CHoCH Bear')

    # ── Trade lines + exit markers ───────────────────────────
    EXIT_COLOR = {'SL': '#ff1744', 'CHoCH': '#a855f7', 'CHoCH5': '#c084fc',
                  'ENGULF': '#f97316', 'TP': '#00e676',
                  'TP1': '#00e676', 'TP2': '#69f0ae',
                  'DDS': '#38bdf8', 'OPEN': '#aaaaaa'}

    if trades_window is not None and len(trades_window):
        for _, t in trades_window.iterrows():
            et = pd.Timestamp(t['entry_time'])
            xt = pd.Timestamp(t['exit_time'])

            x0 = int(np.clip(idx.searchsorted(et, side='left'),  0, n - 1))
            x1 = int(np.clip(idx.searchsorted(xt, side='right'), 0, n - 1))
            if x1 <= x0:
                x1 = min(x0 + 1, n - 1)

            sl, ep   = t['sl'], t['entry_price']
            result   = t.get('result', 'OPEN')
            xprice   = t.get('exit_price', np.nan)
            ec       = EXIT_COLOR.get(result, '#aaaaaa')
            pnl      = t.get('pnl', 0.0)
            pnl_sign = '+' if pnl >= 0 else ''

            # Entry price — white dotted
            ax.hlines(ep, x0, x1, colors='#ffffff',
                      linewidths=0.9, linestyles=':', alpha=0.55, zorder=3)
            # Hard SL — red dashed (always reference)
            ax.hlines(sl, x0, x1, colors='#ff1744',
                      linewidths=0.9, linestyles='--', alpha=0.5, zorder=3)

            tp = t.get('tp', np.nan)
            if not (tp is None or (isinstance(tp, float) and np.isnan(tp))):
                ax.hlines(tp, x0, x1, colors='#00e676',
                          linewidths=1.1, linestyles='--', alpha=0.75, zorder=3)
                ax.annotate(f'TP {tp:.1f}', xy=(x1, tp), xytext=(3, 0),
                            textcoords='offset points', color='#00e676',
                            fontsize=6, va='center')

            # Exit price marker — diamond at exit bar, colour = exit reason
            if not (xprice is None or (isinstance(xprice, float) and np.isnan(xprice))):
                ax.plot(x1, xprice, 'D', color=ec, markersize=5, zorder=6)
                ax.annotate(f'{result} {xprice:.1f}  ({pnl_sign}{pnl:.0f}$)',
                            xy=(x1, xprice),
                            xytext=(-4, 6 if t['side'] == 'long' else -8),
                            textcoords='offset points',
                            color=ec, fontsize=6.5, fontweight='bold',
                            va='bottom' if t['side'] == 'long' else 'top')

    # ── Signal arrows ─────────────────────────────────────────
    up_pts = [(i, r['UpArrow']) for i, (_, r) in enumerate(df_window.iterrows())
              if not np.isnan(r['UpArrow'])]
    dn_pts = [(i, r['DnArrow']) for i, (_, r) in enumerate(df_window.iterrows())
              if not np.isnan(r['DnArrow'])]
    if up_pts:
        ax.plot(*zip(*up_pts), '^', color='#00e676', markersize=9, zorder=5)
    if dn_pts:
        ax.plot(*zip(*dn_pts), 'v', color='#ff1744', markersize=9, zorder=5)

    # ── Axes styling ──────────────────────────────────────────
    step = max(1, n // 10)
    xs = np.arange(n)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([idx[i].strftime('%m/%d %H:%M') for i in xs[::step]],
                       rotation=30, ha='right', fontsize=7)
    ax.set_xlim(-1, n)
    pmin, pmax = df_window['Low'].min(), df_window['High'].max()
    pad = (pmax - pmin) * 0.12
    ax.set_ylim(pmin - pad, pmax + pad)
    ax.set_facecolor('#131722')
    ax.tick_params(colors='#d1d4dc', labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    for sp in ax.spines.values(): sp.set_edgecolor('#363c4e')
    ax.grid(axis='y', color='#363c4e', linewidth=0.5, linestyle='--')

    handles = []
    if up_pts: handles.append(mpatches.Patch(color='#00e676', label='Up ▲'))
    if dn_pts: handles.append(mpatches.Patch(color='#ff1744', label='Down ▼'))
    if 'EMA50' in df_window.columns:
        handles.append(mpatches.Patch(color='#ff9800', label='EMA 50'))
    if 'H1_Bear_Engulf' in df_window.columns:
        handles.append(mpatches.Patch(color='#f97316', label='H1 Bear Engulf ■'))
        handles.append(mpatches.Patch(color='#38bdf8', label='H1 Bull Engulf ■'))
    if 'CHOCH' in df_window.columns:
        handles.append(mpatches.Patch(color='#a855f7', label='CHoCH Bull ◆'))
        handles.append(mpatches.Patch(color='#f59e0b', label='CHoCH Bear ◆'))
    if handles:
        ax.legend(handles=handles, loc='upper left',
                  facecolor='#1e222d', edgecolor='#363c4e',
                  labelcolor='#d1d4dc', fontsize=8)


def plot_chart(final_df, trades_df=None, window_size=120, higher_tf='H1', year=2025):
    total = len(final_df)
    state = {'pos': max(0, total - window_size)}

    # Pre-process trade timestamps once
    if trades_df is not None and not trades_df.empty:
        tdf = trades_df[trades_df['result'] != 'OPEN'].copy()
        tdf['entry_time'] = pd.to_datetime(tdf['entry_time'])
        tdf['exit_time']  = pd.to_datetime(tdf['exit_time'])
    else:
        tdf = None

    fig = plt.figure(figsize=(16, 7), facecolor='#131722')
    fig.canvas.manager.set_window_title(f'MTFArrow — XAUUSD M15 ({year})')
    ax    = fig.add_axes([0.05, 0.18, 0.93, 0.76])
    ax_sl = fig.add_axes([0.05, 0.05, 0.93, 0.04], facecolor='#1e222d')
    slider = Slider(ax_sl, '', 0, max(0, total - window_size),
                    valinit=state['pos'], valstep=1, color='#2962ff')
    slider.label.set_color('#d1d4dc')
    slider.valtext.set_color('#d1d4dc')
    tf_label = higher_tf.replace('min','m').replace('60m','H1').replace('240m','H4')
    fig.text(0.5, 0.96, f'XAUUSD  M15  |  MTFArrow ({tf_label})  |  {year}',
             ha='center', va='top', color='#d1d4dc', fontsize=11, fontweight='bold')

    def refresh(pos):
        pos = int(max(0, min(pos, total - window_size)))
        state['pos'] = pos
        win = final_df.iloc[pos: pos + window_size]

        visible = None
        if tdf is not None:
            w0, w1 = win.index[0], win.index[-1]
            visible = tdf[(tdf['entry_time'] <= w1) & (tdf['exit_time'] >= w0)]

        draw_bars(ax, win, trades_window=visible)
        fig.canvas.draw_idle()

    slider.on_changed(refresh)

    def on_key(e):
        step = max(1, window_size // 4)
        if   e.key == 'right': refresh(state['pos'] + step); slider.set_val(state['pos'])
        elif e.key == 'left':  refresh(state['pos'] - step); slider.set_val(state['pos'])
        elif e.key == 'end':   refresh(total - window_size); slider.set_val(state['pos'])
        elif e.key == 'home':  refresh(0);                   slider.set_val(state['pos'])

    def on_scroll(e):
        if e.inaxes == ax:
            step = max(1, window_size // 8)
            refresh(state['pos'] + (-step if e.button == 'up' else step))
            slider.set_val(state['pos'])

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event',    on_scroll)
    refresh(state['pos'])
    plt.show()

# ============================================================
# 7. Main
# ============================================================
if __name__ == "__main__":
    FILE           = "D:/Meta5/data/XAUUSD_M15.csv"
    M5_FILE        = "D:/Meta5/data/XAUUSD_M5.csv"   # CHoCH5 (3R+ sensitive)
    HTF            = "60min"      # H1 signal
    RISK           = 6            # ASCTrend1i Risk
    WINDOW         = 120          # bars per screen
    INITIAL_CAP    = 10_000       # $
    LOT_SIZE       = 1.0          # P&L = price_move × lot_size (XAUUSD: 1.0 ≈ 0.01 lot)
    SL_LOOKBACK    = 10           # swing SL-д харах барын тоо
    BE_R           = 2.0          # MFE энэ R-д хүрвэл SL → entry (0 = идэвхгүй)
    YEAR           = 2025         # ← жилийг энд өөрчилнө

    # 1. Compute signals
    df_result = mtf_arrow_local(FILE, higher_tf=HTF, risk=RISK, year=YEAR, m5_file=M5_FILE)

    # 2. Backtest (before chart so SL/TP lines are available)
    print("\n[INFO] Running backtest ...")
    trades_df, eq_curve, cap0 = run_backtest(
        df_result, initial_capital=INITIAL_CAP,
        lot_size=LOT_SIZE, sl_lookback=SL_LOOKBACK, be_r=BE_R)

    # 3. Statistics
    stats = calc_stats(trades_df, eq_curve, cap0)
    print("\n===== BACKTEST RESULTS =====")
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")

    # 3b. Per-trade MFE detail  (also saved to txt file)
    detail_txt = f"D:/Meta5/data/MTFArrow_detail_{YEAR}.txt"
    print("\n===== TRADE DETAIL (MFE) =====")
    print_trade_detail(trades_df, save_path=detail_txt)

    # 4. Save trade log
    out_csv = f"D:/Meta5/data/MTFArrow_trades_{YEAR}.csv"
    trades_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Trade log saved: {out_csv}")

    # 5. Interactive chart with SL/TP lines
    plot_chart(df_result, trades_df=trades_df,
               window_size=WINDOW, higher_tf=HTF, year=YEAR)

    # 6. Plot backtest results
    plot_backtest(trades_df, eq_curve, cap0, stats, year=YEAR)
