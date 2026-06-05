import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import matplotlib.ticker as mticker

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
# 2. MTF Arrow — map HTF signals to M15
# ============================================================
def mtf_arrow_local(file_path, higher_tf='60min', risk=6, year=2025):
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
    return final_df

# ============================================================
# 3. Backtest
#    Entry   : Open of bar AFTER signal bar
#    SL Long : lowest Low  of last sl_lookback bars (swing low)
#    SL Short: highest High of last sl_lookback bars (swing high)
#    TP      : Entry ± 2 × SL_distance  (1:2 RR)
#    P&L     : price_move × lot_size  → динамик (SL зайнаас хамаарна)
# ============================================================
def run_backtest(final_df, initial_capital=10_000, lot_size=1.0, sl_lookback=10):
    """
    lot_size    : XAUUSD-д 1.0 = $1 per $1 price move (0.01 lot).
                  Lot_size ихэсгэхэд position томорно.
    sl_lookback : swing SL-д хэдэн бар харах
    """
    df = final_df.reset_index()
    n  = len(df)

    trades          = []
    capital         = initial_capital
    eq_curve        = [capital]
    position        = None
    last_sig_bucket = None

    for i in range(1, n):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        # ── Check open position ──────────────────────────────
        if position is not None:
            p = position
            if p['side'] == 'long':
                sl_hit = row['Low']  <= p['sl']
                tp_hit = row['High'] >= p['tp']
            else:
                sl_hit = row['High'] >= p['sl']
                tp_hit = row['Low']  <= p['tp']

            if sl_hit or tp_hit:
                # P&L = actual price distance × lot_size  (dynamic)
                sl_dist = p['risk_dist']
                if sl_hit:
                    p['exit_price'] = p['sl']
                    p['pnl']        = -sl_dist * lot_size
                    p['result']     = 'SL'
                else:
                    p['exit_price'] = p['tp']
                    p['pnl']        = sl_dist * 2 * lot_size   # TP = 2× SL dist
                    p['result']     = 'TP'
                p['exit_time'] = row['time']
                capital       += p['pnl']
                trades.append(p)
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
                last_sig_bucket = sig_bucket

                # Swing SL: recent N-bar low/high up to and including signal bar
                sig_idx   = i - 1
                lookback  = df.iloc[max(0, sig_idx - sl_lookback + 1) : sig_idx + 1]
                entry     = row['Open']

                if up:
                    sl   = lookback['Low'].min()
                    side = 'long'
                else:
                    sl   = lookback['High'].max()
                    side = 'short'

                risk_dist = abs(entry - sl)
                if risk_dist < 1e-6:
                    continue

                tp = (entry + 2 * risk_dist) if side == 'long' else (entry - 2 * risk_dist)
                position = {
                    'side':        side,
                    'entry_time':  row['time'],
                    'signal_time': prev['time'],
                    'entry_price': entry,
                    'sl':          sl,
                    'tp':          tp,
                    'risk_dist':   risk_dist,
                }

    # Close any still-open trade at last bar close
    if position is not None:
        last = df.iloc[-1]
        ep   = last['Close']
        price_diff = ep - position['entry_price']
        if position['side'] == 'short':
            price_diff = -price_diff
        pnl = price_diff * lot_size
        position.update({'exit_price': ep, 'exit_time': last['time'],
                         'pnl': pnl, 'result': 'OPEN'})
        capital += pnl
        trades.append(position)
        eq_curve.append(capital)

    return pd.DataFrame(trades), eq_curve, initial_capital

# ============================================================
# 4. Statistics
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

    return {
        'Total trades':     len(closed),
        'Winners':          len(wins),
        'Losers':           len(losses),
        'Win rate':         f"{len(wins)/len(closed)*100:.1f}%" if len(closed) else "0%",
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
    }

# ============================================================
# 5. Plot backtest results
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
    items = list(stats.items())
    # split into 2 rows for display
    half = (len(items) + 1) // 2
    rows  = [items[i] for i in range(half)]
    rows2 = [items[i] for i in range(half, len(items))]
    col_labels = [k for k, _ in rows]  + [k for k, _ in rows2]
    col_vals   = [v for _, v in rows]  + [v for _, v in rows2]

    tbl = ax_tbl.table(
        cellText=[col_vals],
        colLabels=col_labels,
        cellLoc='center', loc='center',
        bbox=[0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_facecolor('#1e222d' if r == 0 else BG)
        cell.set_edgecolor('#363c4e')
        cell.set_text_props(color=FG if r > 0 else '#aaaaaa')

    fig.suptitle(f'MTFArrow Backtest — XAUUSD M15 {year}  |  1:2 Risk-Reward',
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

    # ── SL / TP lines for visible trades ─────────────────────
    if trades_window is not None and len(trades_window):
        for _, t in trades_window.iterrows():
            et = pd.Timestamp(t['entry_time'])
            xt = pd.Timestamp(t['exit_time'])

            # Map timestamps → bar indices, clip to window
            x0 = int(np.clip(idx.searchsorted(et, side='left'),  0, n - 1))
            x1 = int(np.clip(idx.searchsorted(xt, side='right'), 0, n - 1))
            if x1 <= x0:
                x1 = min(x0 + 1, n - 1)

            sl, tp, ep = t['sl'], t['tp'], t['entry_price']

            # Entry price — white dotted
            ax.hlines(ep, x0, x1, colors='#ffffff',
                      linewidths=0.9, linestyles=':', alpha=0.55, zorder=3)
            # SL — red dashed
            ax.hlines(sl, x0, x1, colors='#ff1744',
                      linewidths=1.1, linestyles='--', alpha=0.75, zorder=3)
            # TP — green dashed
            ax.hlines(tp, x0, x1, colors='#00e676',
                      linewidths=1.1, linestyles='--', alpha=0.75, zorder=3)

            # Small label at the right edge
            ax.annotate(f'SL {sl:.1f}', xy=(x1, sl), xytext=(2, 0),
                        textcoords='offset points', color='#ff1744',
                        fontsize=6, va='center')
            ax.annotate(f'TP {tp:.1f}', xy=(x1, tp), xytext=(2, 0),
                        textcoords='offset points', color='#00e676',
                        fontsize=6, va='center')

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
    HTF            = "60min"      # H1 signal
    RISK           = 6            # ASCTrend1i Risk
    WINDOW         = 120          # bars per screen
    INITIAL_CAP    = 10_000       # $
    LOT_SIZE       = 1.0          # P&L = price_move × lot_size (XAUUSD: 1.0 ≈ 0.01 lot)
    SL_LOOKBACK    = 10           # swing SL-д харах барын тоо
    YEAR           = 2025         # ← жилийг энд өөрчилнө

    # 1. Compute signals
    df_result = mtf_arrow_local(FILE, higher_tf=HTF, risk=RISK, year=YEAR)

    # 2. Backtest (before chart so SL/TP lines are available)
    print("\n[INFO] Running backtest ...")
    trades_df, eq_curve, cap0 = run_backtest(
        df_result, initial_capital=INITIAL_CAP,
        lot_size=LOT_SIZE, sl_lookback=SL_LOOKBACK)

    # 3. Statistics
    stats = calc_stats(trades_df, eq_curve, cap0)
    print("\n===== BACKTEST RESULTS =====")
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")

    # 4. Save trade log
    out_csv = f"D:/Meta5/data/MTFArrow_trades_{YEAR}.csv"
    trades_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Trade log saved: {out_csv}")

    # 5. Interactive chart with SL/TP lines
    plot_chart(df_result, trades_df=trades_df,
               window_size=WINDOW, higher_tf=HTF, year=YEAR)

    # 6. Plot backtest results
    plot_backtest(trades_df, eq_curve, cap0, stats, year=YEAR)
