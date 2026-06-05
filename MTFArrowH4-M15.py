"""
MTFArrow H4→M15 Strategy
  - H4 ASCTrend state machine → overall direction filter (+1 / -1)
  - M15 ASCTrend crossing signals → entry points
  - Signal only valid when M15 direction matches H4 direction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider
import matplotlib.ticker as mticker

# ============================================================
# 1. ASCTrend1i — exact MT4 port (shared for H4 and M15)
# ============================================================
RANGE_FACTOR = 4.6
ALT_PERIOD   = 4

def calculate_asc_trend(df, risk=6):
    """
    Returns df with:
      Signal   : state machine value (-1, 0, +1) — for direction filter
      UpSignal : True only at bar where state changes -1 → +1
      DnSignal : True only at bar where state changes +1 → -1
    """
    df = df.copy()
    n = len(df)
    wpr_period = 3 + risk * 2
    high_level = 67 + risk
    low_level  = 33 - risk

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
        if   wv >= high_level:               signal[i] =  1
        elif wv <= low_level:                signal[i] = -1
        elif prev ==  1 and wv > low_level:  signal[i] =  1
        elif prev == -1 and wv < high_level: signal[i] = -1
        else:                                signal[i] =  0

        if signal[i] == -1 and prev ==  1:
            dn_sig[i] = hi[i] + ar * 0.5
        if signal[i] ==  1 and prev == -1:
            up_sig[i] = lo[i] - ar * 0.5

    df['Signal']   = signal
    df['UpSignal'] = up_sig > 0
    df['DnSignal'] = dn_sig > 0
    return df

# ============================================================
# 2. Load & Filter
#    H4 → direction state  (+1 bull / -1 bear)
#    M15 → crossing signals, filtered by H4 direction
# ============================================================
def load_csv(path):
    df = pd.read_csv(path, parse_dates=['time'], index_col='time')
    df.columns = [col.capitalize() for col in df.columns]
    return df[['Open', 'High', 'Low', 'Close']].dropna()

def build_signals(m15_file, h4_file, year=2025, risk=6):
    print(f"[INFO] Loading M15: {m15_file}")
    m15 = load_csv(m15_file)
    m15 = m15[m15.index.year == year]
    print(f"[INFO] M15 bars ({year}): {len(m15)}")

    print(f"[INFO] Loading H4:  {h4_file}")
    h4  = load_csv(h4_file)
    h4  = h4[h4.index.year == year]
    print(f"[INFO] H4  bars ({year}): {len(h4)}")

    # M15 crossing signals (entry triggers)
    print("[INFO] ASCTrend1i on M15 ...")
    m15_calc = calculate_asc_trend(m15, risk=risk)

    # H4 direction state — shift(1): use only fully-closed H4 bars
    print("[INFO] ASCTrend1i on H4 ...")
    h4_calc     = calculate_asc_trend(h4, risk=risk)
    h4_dir      = h4_calc[['Signal']].shift(1)
    h4_dir.columns = ['H4_dir']

    # Map each M15 bar to the latest known H4 direction (forward-fill)
    h4_dir_m15 = h4_dir.reindex(m15_calc.index, method='ffill')
    m15_calc['H4_dir'] = h4_dir_m15['H4_dir'].fillna(0)

    h4_up = (m15_calc['H4_dir'] == 1)
    h4_dn = (m15_calc['H4_dir'] == -1)

    up_cnt = (m15_calc['UpSignal'] & h4_up).sum()
    dn_cnt = (m15_calc['DnSignal'] & h4_dn).sum()
    print(f"[INFO] M15 signals after H4 filter — Up: {up_cnt}  Dn: {dn_cnt}")

    avg_off = (m15_calc['High'] - m15_calc['Low']).mean() * 0.6
    m15_calc['UpArrow'] = np.where(
        m15_calc['UpSignal'] & h4_up, m15_calc['Low']  - avg_off, np.nan)
    m15_calc['DnArrow'] = np.where(
        m15_calc['DnSignal'] & h4_dn, m15_calc['High'] + avg_off, np.nan)

    # HTF_bucket needed by backtest (H4 = 240min)
    m15_calc['HTF_bucket'] = m15_calc.index.floor('240min')
    return m15_calc

# ============================================================
# 3. Backtest  (1:2 RR, fixed risk)
#    M15 signals are already single-bar events → no bucket dedup
# ============================================================
def run_backtest(final_df, initial_capital=10_000, risk_usd=100):
    df = final_df.reset_index()
    n  = len(df)

    trades   = []
    capital  = initial_capital
    eq_curve = [capital]
    position = None

    for i in range(1, n):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]

        if position is not None:
            p = position
            if p['side'] == 'long':
                sl_hit = row['Low']  <= p['sl']
                tp_hit = row['High'] >= p['tp']
            else:
                sl_hit = row['High'] >= p['sl']
                tp_hit = row['Low']  <= p['tp']

            if sl_hit or tp_hit:
                if sl_hit:
                    p['exit_price'] = p['sl']
                    p['pnl']        = -risk_usd
                    p['result']     = 'SL'
                else:
                    p['exit_price'] = p['tp']
                    p['pnl']        = 2 * risk_usd
                    p['result']     = 'TP'
                p['exit_time'] = row['time']
                capital       += p['pnl']
                trades.append(p)
                position = None

        eq_curve.append(capital)

        if position is None:
            up = not np.isnan(prev['UpArrow'])
            dn = not np.isnan(prev['DnArrow'])

            if up or dn:
                entry = row['Open']
                if up:
                    sl, side = prev['Low'],  'long'
                else:
                    sl, side = prev['High'], 'short'

                risk_dist = abs(entry - sl)
                if risk_dist < 1e-6:
                    continue

                tp = (entry + 2*risk_dist) if side == 'long' else (entry - 2*risk_dist)
                position = {
                    'side':        side,
                    'entry_time':  row['time'],
                    'signal_time': prev['time'],
                    'entry_price': entry,
                    'sl':          sl,
                    'tp':          tp,
                    'risk_dist':   risk_dist,
                    'H4_dir':      prev['H4_dir'],
                }

    if position is not None:
        last = df.iloc[-1]
        ep   = last['Close']
        full = abs(position['tp'] - position['entry_price'])
        dist = abs(ep - position['entry_price'])
        raw  = risk_usd * 2 * (dist / full) if full > 0 else 0
        pnl  = raw if ((position['side']=='long' and ep > position['entry_price']) or
                       (position['side']=='short' and ep < position['entry_price'])) else -raw
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
    gp     = wins['pnl'].sum()
    gl     = abs(losses['pnl'].sum())
    pf     = gp / gl if gl > 0 else np.inf
    eq     = np.array(eq_curve)
    rm     = np.maximum.accumulate(eq)
    dd     = rm - eq
    mdd    = dd.max()
    mdd_pct= mdd / rm[dd.argmax()] * 100 if rm[dd.argmax()] > 0 else 0
    return {
        'Total trades':   len(closed),
        'Winners':        len(wins),
        'Losers':         len(losses),
        'Win rate':       f"{len(wins)/len(closed)*100:.1f}%" if len(closed) else "0%",
        'Profit factor':  f"{pf:.2f}",
        'Net P&L':        f"${closed['pnl'].sum():+,.0f}",
        'Gross profit':   f"${gp:,.0f}",
        'Gross loss':     f"${gl:,.0f}",
        'Avg win':        f"${wins['pnl'].mean():,.0f}"   if len(wins)   else "$0",
        'Avg loss':       f"${losses['pnl'].mean():,.0f}" if len(losses) else "$0",
        'Best trade':     f"${closed['pnl'].max():+,.0f}" if len(closed) else "$0",
        'Worst trade':    f"${closed['pnl'].min():+,.0f}" if len(closed) else "$0",
        'Max drawdown':   f"${mdd:,.0f}  ({mdd_pct:.1f}%)",
        'Final capital':  f"${eq_curve[-1]:,.0f}",
        'Return':         f"{(eq_curve[-1]-initial_capital)/initial_capital*100:+.1f}%",
    }

# ============================================================
# 5. Plot backtest
# ============================================================
def plot_backtest(trades_df, eq_curve, initial_capital, stats, year=2025):
    BG, FG, GRN, RED, BLUE = '#131722','#d1d4dc','#26a69a','#ef5350','#2962ff'
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    fig.canvas.manager.set_window_title(f'MTFArrow H4→M15 Backtest {year}')
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.06, right=0.97, top=0.93, bottom=0.07,
                           hspace=0.45, wspace=0.35)

    ax_eq = fig.add_subplot(gs[0, :])
    ax_eq.set_facecolor(BG)
    xs = np.arange(len(eq_curve))
    ax_eq.plot(xs, eq_curve, color=BLUE, linewidth=1.4, zorder=3)
    ax_eq.axhline(initial_capital, color='#555', linewidth=0.8, linestyle='--')
    ax_eq.fill_between(xs, initial_capital, eq_curve,
                        where=np.array(eq_curve) >= initial_capital, color=GRN, alpha=0.18)
    ax_eq.fill_between(xs, initial_capital, eq_curve,
                        where=np.array(eq_curve) <  initial_capital, color=RED, alpha=0.25)
    ax_eq.set_title('Equity Curve', color=FG, fontsize=10, pad=4)
    ax_eq.tick_params(colors=FG, labelsize=7)
    ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    for sp in ax_eq.spines.values(): sp.set_edgecolor('#363c4e')
    ax_eq.grid(color='#363c4e', linewidth=0.4, linestyle='--')

    ax_mo = fig.add_subplot(gs[1, 0])
    ax_mo.set_facecolor(BG)
    closed = trades_df[trades_df['result'] != 'OPEN'].copy()
    if not closed.empty:
        closed['month'] = pd.to_datetime(closed['exit_time']).dt.to_period('M')
        monthly = closed.groupby('month')['pnl'].sum()
        ax_mo.bar(range(len(monthly)), monthly.values,
                  color=[GRN if v >= 0 else RED for v in monthly.values], width=0.6)
        ax_mo.set_xticks(range(len(monthly)))
        ax_mo.set_xticklabels([str(m) for m in monthly.index],
                               rotation=45, ha='right', fontsize=6.5)
        ax_mo.axhline(0, color='#555', linewidth=0.7)
        ax_mo.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax_mo.set_title('Monthly P&L', color=FG, fontsize=10, pad=4)
    ax_mo.tick_params(colors=FG, labelsize=7)
    for sp in ax_mo.spines.values(): sp.set_edgecolor('#363c4e')

    ax_hist = fig.add_subplot(gs[1, 1])
    ax_hist.set_facecolor(BG)
    if not closed.empty:
        w = closed[closed['pnl'] > 0]['pnl'].values
        l = closed[closed['pnl'] < 0]['pnl'].values
        if len(w): ax_hist.hist(w, bins=20, color=GRN, alpha=0.75, label='Win')
        if len(l): ax_hist.hist(l, bins=20, color=RED, alpha=0.75, label='Loss')
        ax_hist.legend(facecolor='#1e222d', edgecolor='#363c4e',
                       labelcolor=FG, fontsize=8)
    ax_hist.set_title('P&L Distribution', color=FG, fontsize=10, pad=4)
    ax_hist.tick_params(colors=FG, labelsize=7)
    for sp in ax_hist.spines.values(): sp.set_edgecolor('#363c4e')

    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis('off')
    items = list(stats.items())
    half  = (len(items) + 1) // 2
    col_labels = [k for k, _ in items[:half]] + [k for k, _ in items[half:]]
    col_vals   = [v for _, v in items[:half]] + [v for _, v in items[half:]]
    tbl = ax_tbl.table(cellText=[col_vals], colLabels=col_labels,
                       cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, _), cell in tbl.get_celld().items():
        cell.set_facecolor('#1e222d' if r == 0 else BG)
        cell.set_edgecolor('#363c4e')
        cell.set_text_props(color=FG if r > 0 else '#aaaaaa')

    fig.suptitle(f'MTFArrow H4→M15 Backtest {year}  |  1:2 Risk-Reward',
                 color=FG, fontsize=12, fontweight='bold')
    plt.show()

# ============================================================
# 6. Interactive chart
# ============================================================
def draw_bars(ax, df_window, bar_width=0.4):
    ax.cla()
    n = len(df_window)
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

    up_pts = [(i, r['UpArrow']) for i, (_, r) in enumerate(df_window.iterrows())
              if not np.isnan(r['UpArrow'])]
    dn_pts = [(i, r['DnArrow']) for i, (_, r) in enumerate(df_window.iterrows())
              if not np.isnan(r['DnArrow'])]
    if up_pts:
        ax.plot(*zip(*up_pts), '^', color='#00e676', markersize=10, zorder=5)
    if dn_pts:
        ax.plot(*zip(*dn_pts), 'v', color='#ff1744', markersize=10, zorder=5)

    step = max(1, n // 10)
    xs = np.arange(n)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([df_window.index[i].strftime('%m/%d %H:%M') for i in xs[::step]],
                       rotation=30, ha='right', fontsize=7)
    ax.set_xlim(-1, n)
    pmin, pmax = df_window['Low'].min(), df_window['High'].max()
    pad = (pmax - pmin) * 0.10
    ax.set_ylim(pmin - pad, pmax + pad)
    ax.set_facecolor('#131722')
    ax.tick_params(colors='#d1d4dc', labelsize=7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    for sp in ax.spines.values(): sp.set_edgecolor('#363c4e')
    ax.grid(axis='y', color='#363c4e', linewidth=0.5, linestyle='--')

    handles = []
    if up_pts: handles.append(mpatches.Patch(color='#00e676', label='Up (H4 Bull) ▲'))
    if dn_pts: handles.append(mpatches.Patch(color='#ff1744', label='Down (H4 Bear) ▼'))
    if handles:
        ax.legend(handles=handles, loc='upper left',
                  facecolor='#1e222d', edgecolor='#363c4e',
                  labelcolor='#d1d4dc', fontsize=8)

def plot_chart(final_df, window_size=120, year=2025):
    total = len(final_df)
    state = {'pos': max(0, total - window_size)}
    fig = plt.figure(figsize=(16, 7), facecolor='#131722')
    fig.canvas.manager.set_window_title(f'MTFArrow H4→M15 ({year})')
    ax    = fig.add_axes([0.05, 0.18, 0.93, 0.76])
    ax_sl = fig.add_axes([0.05, 0.05, 0.93, 0.04], facecolor='#1e222d')
    slider = Slider(ax_sl, '', 0, max(0, total - window_size),
                    valinit=state['pos'], valstep=1, color='#2962ff')
    slider.label.set_color('#d1d4dc')
    slider.valtext.set_color('#d1d4dc')
    fig.text(0.5, 0.96, f'XAUUSD  M15  |  H4 Direction Filter  |  {year}',
             ha='center', va='top', color='#d1d4dc', fontsize=11, fontweight='bold')

    def refresh(pos):
        pos = int(max(0, min(pos, total - window_size)))
        state['pos'] = pos
        draw_bars(ax, final_df.iloc[pos: pos + window_size])
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
    M15_FILE       = "D:/Meta5/data/XAUUSD_M15.csv"
    H4_FILE        = "D:/Meta5/data/XAUUSD_H4.csv"
    RISK           = 6
    WINDOW         = 120
    INITIAL_CAP    = 10_000
    RISK_PER_TRADE = 100
    YEAR           = 2025     # ← жилийг энд өөрчилнө

    # 1. Build signals (M15 filtered by H4 direction)
    df_result = build_signals(M15_FILE, H4_FILE, year=YEAR, risk=RISK)

    # 2. Interactive chart
    plot_chart(df_result, window_size=WINDOW, year=YEAR)

    # 3. Backtest
    print("\n[INFO] Running backtest ...")
    trades_df, eq_curve, cap0 = run_backtest(
        df_result, initial_capital=INITIAL_CAP, risk_usd=RISK_PER_TRADE)

    # 4. Stats
    stats = calc_stats(trades_df, eq_curve, cap0)
    print("\n===== BACKTEST RESULTS (H4→M15) =====")
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")

    # 5. Save
    out_csv = f"D:/Meta5/data/MTFArrow_H4M15_trades_{YEAR}.csv"
    trades_df.to_csv(out_csv, index=False)
    print(f"\n[INFO] Trade log: {out_csv}")

    # 6. Plot
    plot_backtest(trades_df, eq_curve, cap0, stats, year=YEAR)
