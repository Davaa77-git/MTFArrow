# MTFArrow Strategy — XAUUSD M15

## Ерөнхий тойм

| Зүйл | Утга |
|---|---|
| Арилжааны хос | XAUUSD |
| Үндсэн TF | M15 |
| Сигналын TF | H1 (ASCTrend1i) |
| Нэмэлт өгөгдөл | M5 (CHoCH5 тооцоолол) |
| Нэг сигналд нээх pos | 3 (Leg 1, 2, 3) |

---

## 1. Entry Signal — ASCTrend1i (H1)

### Параметрүүд
| Параметр | Утга |
|---|---|
| `risk` | 6 |
| `wpr_period` | `3 + risk × 2 = 15` |
| `high_level` | `67 + risk = 73` |
| `low_level` | `33 - risk = 27` |
| `ALT_PERIOD` | 4 |
| `RANGE_FACTOR` | 4.6 |

### Williams %R тооцоолол
- **Стандарт:** `%R = 100 - abs(-100 × (High_max - Close) / (High_max - Low_min))` — 15-bar lookback
- **Хурдан горим:** Сүүлийн 6 bar-д `|Close[i] - Close[i-3]| ≥ avg_range × 4.6` бол → 4-bar lookback ашиглана

### Сигнал үүсгэх дүрэм
| Нөхцөл | Үйлдэл |
|---|---|
| `%R ≥ 73` | signal = +1 (bullish) |
| `%R ≤ 27` | signal = -1 (bearish) |
| signal = +1 AND `%R > 27` | signal хэвээр +1 |
| signal = -1 AND `%R < 73` | signal хэвээр -1 |
| signal = +1 → -1 | **Down Arrow** (Short дохио) |
| signal = -1 → +1 | **Up Arrow** (Long дохио) |

### H1 → M15 зураглал
- H1 дохио **shift(1)** → дараагийн H1 candle-ийн эхний bar-д харагдана
- **Entry:** 4 дэх M15 bar хаагдсаны дараа (шинэ H1 period-ийн Open)

---

## 2. Entry Filters

### 2.1 EMA50 Filter
| Чиглэл | Нөхцөл |
|---|---|
| Long | `Close ≥ EMA(50)` |
| Short | `Close ≤ EMA(50)` |

### 2.2 Pre-Move Filter (chase хийхгүй)
- Signal bar-аас **өмнөх 10 M15 bar**-ын net price change тооцно
- Long: `Close[signal] - Close[signal-10]` > `3 × ATR(22)` бол **skip**
- Short: `Close[signal-10] - Close[signal]` > `3 × ATR(22)` бол **skip**
- Зорилго: Хэт их хөдөлсний дараа chase хийхгүй байх

> **Лавлагаа:** 2025 оны ATR(22) дундаж ≈ $5.69 → 3×ATR ≈ **$17**

---

## 3. Position Management — 3 Leg

### SL тооцоолол (хуваалцсан)
- Lookback: **Signal bar-аас өмнөх 10 M15 bar**
- Long SL: `min(Low)` — 10 bar-ын хамгийн бага Low
- Short SL: `max(High)` — 10 bar-ын хамгийн өндөр High
- `risk_dist = |entry - SL|`

### TP тавих
| Leg | TP | Lot |
|---|---|---|
| Leg 1 | `entry ± 1.0 × risk_dist` (1R) | `lot_size` |
| Leg 2 | `entry ± 2.0 × risk_dist` (2R) | `lot_size` |
| Leg 3 | Тогтмол TP байхгүй | `lot_size` |

### Breakeven Trail
- **Нөхцөл:** `MFE ≥ 2.0 × risk_dist` (2R хүрэхэд)
- **Үйлдэл:** SL → Entry price руу шилжинэ
- **Хамрах хүрээ:** Нээлттэй бүх leg-т нэгэн зэрэг

### MFE (Maximum Favorable Excursion) tracking
- Long: `MFE_price = max(High)` entry-аас хойш
- Short: `MFE_price = min(Low)` entry-аас хойш
- `MFE_pts = |MFE_price - entry_price|`

---

## 4. Exit Conditions

Exit-ууд дараах **ач холбогдлын дарааллаар** шалгагдана:

### 4.1 Stop Loss — бүх leg хаана
| Чиглэл | Нөхцөл | Exit үнэ |
|---|---|---|
| Long | `Low ≤ SL` | SL үнэ |
| Short | `High ≥ SL` | SL үнэ |

---

### 4.2 CHoCH (M15, swing=10) — бүх leg хаана
`smartmoneyconcepts` сан ашиглан M15 дээр swing=10-ын CHoCH тооцно.

| Чиглэл | Нөхцөл | Exit үнэ |
|---|---|---|
| Long | `CHOCH == -1` AND swing entry-аас хойш үүссэн | Close |
| Short | `CHOCH == +1` AND swing entry-аас хойш үүссэн | Close |

> **Шалгуур:** `CHOCH_swing_idx > entry_bar_idx` — entry хийхээс өмнөх swing-ийг тооцохгүй

---

### 4.3 CHoCH5 (M5, swing=5) — бүх leg хаана
M5 өгөгдөл дээр swing=5-ын CHoCH тооцно. M5 timestamp → M15 bar index зураглал хийгдэнэ.

| Чиглэл | Нөхцөл | Exit үнэ |
|---|---|---|
| Long | `CHOCH5 == -1` AND swing entry-аас хойш AND `MFE ≥ 3R` | Close |
| Short | `CHOCH5 == +1` AND swing entry-аас хойш AND `MFE ≥ 3R` | Close |

> **Зөвхөн** MFE ≥ 3R үед идэвхждэг (илүү мэдрэмтгий exit)

---

### 4.4 H1 Engulfing — бүх leg хаана

**Bearish Engulfing (Long exit):**
1. Өмнөх H1 candle **bullish** байх
2. Одоогийн H1 candle **bearish** байх
3. Одоогийн body нь өмнөхийн **бүтэн range-ийг** хамрах: `Open ≥ High_prev` AND `Close ≤ Low_prev`
4. Өмнөх candle сүүлийн 5 H1 bar-ын **swing HIGH** байх

**Bullish Engulfing (Short exit) — тэгш хэмтэй:**
1. Өмнөх H1 candle bearish
2. Одоогийн H1 candle bullish
3. `Open ≤ Low_prev` AND `Close ≥ High_prev`
4. Өмнөх candle — swing LOW

Exit үнэ: Дараагийн M15 bar-ийн **Open**

---

### 4.5 TP1 / TP2 — зөвхөн тухайн leg хаана

| | Long | Short |
|---|---|---|
| **TP1** | `High ≥ entry + 1R` | `Low ≤ entry - 1R` |
| **TP2** | `High ≥ entry + 2R` | `Low ≤ entry - 2R` |

- TP1 хаагдахад Leg 2, 3 үргэлжилнэ
- TP2 хаагдахад Leg 3 үргэлжилнэ

---

### 4.6 DDS Exit — зөвхөн Leg 3, Leg 1+2 хаагдсаны дараа

**Long position-ийг хаах нөхцөл (бүгд нэгэн зэрэг биелэх ёстой):**
| Нөхцөл | Утга |
|---|---|
| `H4 DDS < 10` | H4 Drake Delayed Stoch oversold |
| `H1 DDS < 10` | H1 Drake Delayed Stoch oversold |
| `RSI14 rising` | M15 RSI(14) өсч байна |
| `Sto(8,3,3) rising` | M15 K line өсч байна |
| `Sto(20,10,10) rising` | M15 K line өсч байна |
| `Sto(100,10,10) rising` | M15 K line өсч байна |

**Short position-ийг хаах нөхцөл (тэгш хэмтэй):**
- H4/H1 DDS > 90 (overbought)
- RSI14, Sto(8,3,3), Sto(20,10,10), Sto(100,10,10) — бүгд доош

Exit үнэ: Close

---

## 5. Indicator Parameters

### ATR (22)
```
TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
ATR = SMA(TR, 22)
```

### EMA (50)
```
EMA50 = EWM(Close, span=50, adjust=False)
```

### RSI (14)
```
Gain/Loss = diff(Close).clip(lower/upper=0)
Avg = EWM(alpha=1/14, adjust=False)
RSI = 100 - 100 / (1 + avg_gain/avg_loss)
```

### Stochastic K line
```
raw_K = (Close - Low_min(N)) / (High_max(N) - Low_min(N)) × 100
K = SMA(raw_K, smooth)
```
| Indicator | N (K period) | Smooth |
|---|---|---|
| Sto(8,3,3) | 8 | 3 |
| Sto(20,10,10) | 20 | 10 |
| Sto(100,10,10) | 100 | 10 |

### Drake Delayed Stochastic (H4, H1)
```
raw_K = (Close.shift(13) - Low_min(8)) / (High_max(8) - Low_min(8)) × 100
DDS = SMA(raw_K, 9)
```
| Параметр | Утга |
|---|---|
| Period (H/L range) | 8 |
| Delay (close shift) | 13 bars |
| Smooth | 9 |
| Oversold threshold | < 10 |
| Overbought threshold | > 90 |

---

## 6. Backtest Parameters

| Параметр | Утга |
|---|---|
| `INITIAL_CAP` | $10,000 |
| `LOT_SIZE` | 1.0 (нэг leg) |
| `SL_LOOKBACK` | 10 bars |
| `BE_R` | 2.0 (2R-д SL → entry) |
| `YEAR` | 2025 |
| `HTF` | 60min (H1) |
| `RISK` | 6 |
| `WINDOW` | 120 bars |

---

## 7. Өгөгдлийн эх сурвалж

| Файл | Тайлбар |
|---|---|
| `D:/Meta5/data/XAUUSD_M15.csv` | Үндсэн M15 өгөгдөл |
| `D:/Meta5/data/XAUUSD_M5.csv` | CHoCH5 тооцоолоход ашиглах M5 өгөгдөл |

---

## 8. Exit Result Кодууд

| Код | Тайлбар | Өнгө |
|---|---|---|
| `SL` | Stop Loss | Улаан |
| `TP1` | 1R Take Profit (Leg 1) | Ногоон |
| `TP2` | 2R Take Profit (Leg 2) | Цайвар ногоон |
| `CHoCH` | M15 Change of Character (swing=10) | Нил ягаан |
| `CHoCH5` | M5 Change of Character (swing=5, 3R+) | Цайвар нил |
| `ENGULF` | H1 Engulfing candle | Улбар шар |
| `DDS` | Drake DDS + M15 oscillators | Цэнхэр |
| `OPEN` | Сүүлийн bar дээр нээлттэй хаагдсан | Саарал |
