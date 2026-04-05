import json

cells = []

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": (
        "# LumenY 10 \u2014 Continuation Analysis\n\n"
        "**Question:** On bars where the magnitude model predicts a big move (Q50 > threshold),\n"
        "does the market show directional continuation after the first bar?\n\n"
        "**Approach:**\n"
        "- Load the magnitude model (models_9/mfe_q50) and run it on the test set\n"
        "- Filter to bars where Q50 > 40 pips\n"
        "- For each such bar T, observe sign of bar T+1 return (market tells us direction)\n"
        "- Measure return from T+1 close over horizons T+2h, T+4h, T+8h, T+12h\n"
        "- If continuation exists: enter at T+1 close, ride to T+Nh\n\n"
        "**Key metric:** Does sign(ret_T+1) predict sign(ret_T+1_to_T+N) better than 50%?\n"
        "And what is the EV after spread?"
    )
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "import pandas as pd\n"
        "import numpy as np\n"
        "import joblib\n"
        "import matplotlib.pyplot as plt\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "from pathlib import Path\n\n"
        "FEATURES_9_DIR = Path('../backend/data/features_9')\n"
        "FEATURES_8_DIR = Path('../backend/data/features_8')\n"
        "PROCESSED_DIR  = Path('../backend/data/processed')\n"
        "MODEL_PATH     = Path('../backend/models_9/mfe_q50/model_1H_Q50.joblib')\n\n"
        "TRAIN_END  = '2024-06-30'\n"
        "AVG_SPREAD_PIPS = 2.8\n\n"
        "PIP_SIZE = {\n"
        "    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,\n"
        "    'USDCAD': 0.0001, 'USDCHF': 0.0001, 'USDJPY': 0.01,\n"
        "    'EURJPY': 0.01,   'GBPJPY': 0.01,   'AUDJPY': 0.01,   'CADJPY': 0.01,\n"
        "    'CHFJPY': 0.01,   'EURAUD': 0.0001, 'EURGBP': 0.0001, 'AUDNZD': 0.0001,\n"
        "}\n\n"
        "bundle = joblib.load(MODEL_PATH)\n"
        "model  = bundle['model']\n"
        "feature_cols = bundle['feature_cols']\n"
        "print(f'Model loaded. Train end: {bundle[\"train_end\"]}, features: {len(feature_cols)}')\n"
    )
})

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 1. Load features_9 + raw 1H closes"
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "PAIRS = [\n"
        "    'AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY',\n"
        "    'EURAUD', 'EURGBP', 'EURJPY', 'EURUSD',\n"
        "    'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'\n"
        "]\n\n"
        "# Load features_9\n"
        "dfs = []\n"
        "for f in sorted(FEATURES_9_DIR.glob('*_features.parquet')):\n"
        "    tmp = pd.read_parquet(f)\n"
        "    dfs.append(tmp)\n"
        "df_feat = pd.concat(dfs).sort_index()\n"
        "del dfs\n"
        "print(f'features_9: {df_feat.shape}')\n\n"
        "# Load 1H closes for each pair to compute forward returns\n"
        "closes = {}\n"
        "for pair in PAIRS:\n"
        "    df_1h = pd.read_parquet(PROCESSED_DIR / f'{pair}_1H.parquet')\n"
        "    if 'datetime' in df_1h.columns:\n"
        "        df_1h = df_1h.set_index('datetime')\n"
        "    df_1h.index = pd.to_datetime(df_1h.index)\n"
        "    closes[pair] = df_1h['close']\n"
        "print(f'Loaded 1H closes for {len(closes)} pairs')\n"
    )
})

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 2. Run magnitude model, filter test set"
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "# Test set only (unseen)\n"
        "df_test = df_feat[df_feat.index > TRAIN_END].copy()\n"
        "print(f'Test set: {len(df_test):,} rows ({df_test.index.min().date()} -> {df_test.index.max().date()})')\n\n"
        "# Predict Q50 magnitude\n"
        "X_test = df_test[[c for c in feature_cols if c in df_test.columns]].ffill().fillna(0)\n"
        "missing_cols = [c for c in feature_cols if c not in df_test.columns]\n"
        "if missing_cols:\n"
        "    print(f'WARNING: {len(missing_cols)} feature cols missing from features_9, filling with 0')\n"
        "    for c in missing_cols:\n"
        "        X_test[c] = 0\n"
        "X_test = X_test[feature_cols]\n\n"
        "df_test = df_test.copy()\n"
        "df_test['q50_pred'] = model.predict(X_test)\n"
        "print(f'Q50 predictions: mean={df_test[\"q50_pred\"].mean():.1f}, std={df_test[\"q50_pred\"].std():.1f}')\n\n"
        "# Distribution of predictions\n"
        "for thresh in [20, 30, 40, 50, 60, 70]:\n"
        "    n = (df_test['q50_pred'] > thresh).sum()\n"
        "    print(f'  Q50 > {thresh}: {n:,} bars ({100*n/len(df_test):.1f}%)')\n"
    )
})

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 3. Compute forward returns (T+1, T+2, T+4, T+8, T+12)"
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "# For each bar T:\n"
        "#   ret_T1   = log return of bar T+1 (the 'direction signal' bar)\n"
        "#   ret_T1_Nh = log return from T+1 close to T+N close (the 'ride' return)\n"
        "\n"
        "horizons = [2, 4, 8, 12]\n\n"
        "rows = []\n"
        "for pair in PAIRS:\n"
        "    pip = PIP_SIZE[pair]\n"
        "    spread_price = AVG_SPREAD_PIPS * pip\n\n"
        "    pair_mask = df_test['pair'] == pair\n"
        "    df_p = df_test[pair_mask].copy()\n"
        "    close = closes[pair].reindex(df_p.index)\n\n"
        "    # ret of bar T+1: from close[T] to close[T+1]\n"
        "    df_p['ret_T1'] = np.log(close.shift(-1) / close)  # return of next bar\n"
        "    df_p['ret_T1_pips'] = df_p['ret_T1'] / pip\n\n"
        "    # forward returns from T+1 to T+N (riding after observing T+1 direction)\n"
        "    # entry = close[T+1], exit = close[T+N]\n"
        "    for h in horizons:\n"
        "        df_p[f'ret_ride_{h}h'] = np.log(close.shift(-h) / close.shift(-1))\n"
        "        df_p[f'ret_ride_{h}h_pips'] = df_p[f'ret_ride_{h}h'] / pip\n\n"
        "    rows.append(df_p)\n\n"
        "df_all = pd.concat(rows).sort_index()\n"
        "print(f'Dataset with forward returns: {df_all.shape}')\n"
        "print(f'ret_T1 stats: mean={df_all[\"ret_T1_pips\"].mean():.2f}, std={df_all[\"ret_T1_pips\"].std():.2f}')\n"
    )
})

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 4. Continuation analysis by Q50 threshold"
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "# Core question: does sign(ret_T1) predict the direction of the ride return?\n"
        "# Strategy: at T+1 close, enter in direction of ret_T1, hold N bars\n\n"
        "thresholds = [0, 20, 30, 40, 50, 60]\n\n"
        "print('CONTINUATION ANALYSIS')\n"
        "print('Strategy: observe bar T+1 direction, enter at T+1 close, hold N bars')\n"
        "print(f'Spread cost per side: {AVG_SPREAD_PIPS:.1f} pips')\n"
        "print()\n\n"
        "for h in horizons:\n"
        "    print(f'--- Hold {h}H after entry ---')\n"
        "    print(f'{\"Q50>\":>8} {\"Bars\":>8} {\"WinRate\":>9} {\"Avg ride\":>10} {\"EV/trade\":>10} {\"Sharpe\":>8}')\n"
        "    print('-' * 60)\n\n"
        "    for thresh in thresholds:\n"
        "        mask = (\n"
        "            (df_all['q50_pred'] > thresh) &\n"
        "            df_all['ret_T1'].notna() &\n"
        "            df_all[f'ret_ride_{h}h'].notna()\n"
        "        )\n"
        "        s = df_all[mask].copy()\n"
        "        if len(s) < 100:\n"
        "            continue\n\n"
        "        direction = np.sign(s['ret_T1'])  # direction signal from bar T+1\n"
        "        ride_pips = s[f'ret_ride_{h}h_pips'] * direction  # signed ride return\n"
        "        pnl_pips  = ride_pips - AVG_SPREAD_PIPS  # cost to enter + exit\n\n"
        "        win_rate = (ride_pips > 0).mean()\n"
        "        ev       = pnl_pips.mean()\n"
        "        sharpe   = (pnl_pips.mean() / pnl_pips.std()) * np.sqrt(252 * 24 / h) if pnl_pips.std() > 0 else 0\n"
        "        flag     = ' <<<' if ev > 0 else ''\n\n"
        "        print(f'{thresh:>8} {len(s):>8,} {win_rate:>8.1%} {ride_pips.mean():>10.2f} {ev:>10.2f} {sharpe:>8.2f}{flag}')\n"
        "    print()\n"
    )
})

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 5. Hour distribution of signals"
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "Q50_THRESH = 40  # adjust based on results above\n"
        "HOLD_H = 4       # adjust based on results above\n\n"
        "mask = (\n"
        "    (df_all['q50_pred'] > Q50_THRESH) &\n"
        "    df_all['ret_T1'].notna() &\n"
        "    df_all[f'ret_ride_{HOLD_H}h'].notna()\n"
        ")\n"
        "s = df_all[mask].copy()\n"
        "s['direction'] = np.sign(s['ret_T1'])\n"
        "s['pnl_pips']  = s[f'ret_ride_{HOLD_H}h_pips'] * s['direction'] - AVG_SPREAD_PIPS\n"
        "s['hour']      = s.index.hour\n\n"
        "hourly = s.groupby('hour').agg(\n"
        "    trades=('pnl_pips', 'count'),\n"
        "    win_rate=('pnl_pips', lambda x: (x > 0).mean()),\n"
        "    ev=('pnl_pips', 'mean'),\n"
        "    total=('pnl_pips', 'sum')\n"
        ").reset_index()\n\n"
        "print(f'Q50 > {Q50_THRESH}, hold {HOLD_H}H — trade distribution by hour:')\n"
        "print(f'{\"Hour\":>6} {\"Trades\":>8} {\"WinRate\":>9} {\"EV/trade\":>10} {\"Total\":>10}')\n"
        "print('-' * 50)\n"
        "for _, row in hourly.iterrows():\n"
        "    flag = ' <<<' if row['ev'] > 0 else ''\n"
        "    print(f'{int(row[\"hour\"]):>6} {int(row[\"trades\"]):>8,} {row[\"win_rate\"]:>8.1%} {row[\"ev\"]:>10.2f} {row[\"total\"]:>10.2f}{flag}')\n\n"
        "# Uniformity score: std of trade count per hour (lower = more uniform)\n"
        "print(f'\\nTrade count std across hours: {hourly[\"trades\"].std():.1f} (lower = more uniform)')\n"
        "print(f'Total trades: {len(s):,}')\n"
    )
})

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 6. Visual — equity curve + hour distribution"
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "fig, axes = plt.subplots(2, 2, figsize=(16, 10))\n"
        "fig.patch.set_facecolor('#080c14')\n"
        "for ax in axes.flatten():\n"
        "    ax.set_facecolor('#080c14')\n"
        "    ax.tick_params(colors='white')\n"
        "    for spine in ax.spines.values(): spine.set_edgecolor('#1a2332')\n\n"
        "# 1. Equity curve overall\n"
        "cum_pnl = s.sort_index()['pnl_pips'].cumsum()\n"
        "axes[0,0].plot(cum_pnl.index, cum_pnl.values, color='#4fc3f7', linewidth=1.5)\n"
        "axes[0,0].axhline(0, color='white', alpha=0.2, linewidth=1)\n"
        "axes[0,0].set_title(f'Equity curve (Q50>{Q50_THRESH}, hold {HOLD_H}H) — {len(s):,} trades', color='white')\n"
        "axes[0,0].set_ylabel('Cumulative pips', color='white')\n\n"
        "# 2. Per-pair equity\n"
        "for pair in sorted(s['pair'].unique()):\n"
        "    pp = s[s['pair']==pair].sort_index()['pnl_pips'].cumsum()\n"
        "    axes[0,1].plot(pp.index, pp.values, linewidth=0.8, alpha=0.6, label=pair)\n"
        "axes[0,1].axhline(0, color='white', alpha=0.2, linewidth=1)\n"
        "axes[0,1].set_title('Per-pair equity curves', color='white')\n"
        "axes[0,1].legend(fontsize=6, facecolor='#1a2332', labelcolor='white', ncol=3)\n\n"
        "# 3. Trade count by hour\n"
        "axes[1,0].bar(hourly['hour'], hourly['trades'], color='#4fc3f7', alpha=0.8)\n"
        "axes[1,0].set_title('Trade count by hour', color='white')\n"
        "axes[1,0].set_xlabel('Hour (UTC)', color='white')\n"
        "axes[1,0].set_ylabel('# trades', color='white')\n\n"
        "# 4. EV by hour\n"
        "colors = ['#2ecc71' if v > 0 else '#ff4757' for v in hourly['ev']]\n"
        "axes[1,1].bar(hourly['hour'], hourly['ev'], color=colors, alpha=0.8)\n"
        "axes[1,1].axhline(0, color='white', alpha=0.3, linewidth=1)\n"
        "axes[1,1].set_title('EV/trade by hour (pips)', color='white')\n"
        "axes[1,1].set_xlabel('Hour (UTC)', color='white')\n"
        "axes[1,1].set_ylabel('EV (pips)', color='white')\n\n"
        "plt.suptitle(f'Continuation Strategy — Q50>{Q50_THRESH} gate, hold {HOLD_H}H after T+1 bar', color='white', fontsize=13)\n"
        "plt.tight_layout()\n"
        "plt.show()\n"
    )
})

cells.append({
    "cell_type": "markdown", "metadata": {},
    "source": "## 7. Sensitivity — Q50 threshold vs hold horizon grid"
})

cells.append({
    "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
    "source": (
        "print('EV/trade (pips) grid — rows=Q50 threshold, cols=hold horizon')\n"
        "print(f'Spread deducted: {AVG_SPREAD_PIPS} pips')\n"
        "print()\n\n"
        "header = f'{\"Q50>\":>8}' + ''.join([f'{h:>10}H' for h in horizons])\n"
        "print(header)\n"
        "print('-' * (8 + 11*len(horizons)))\n\n"
        "for thresh in [0, 20, 30, 40, 50, 60, 70]:\n"
        "    row_str = f'{thresh:>8}'\n"
        "    for h in horizons:\n"
        "        mask = (\n"
        "            (df_all['q50_pred'] > thresh) &\n"
        "            df_all['ret_T1'].notna() &\n"
        "            df_all[f'ret_ride_{h}h'].notna()\n"
        "        )\n"
        "        s_g = df_all[mask]\n"
        "        if len(s_g) < 50:\n"
        "            row_str += f'{\"N/A\":>10}'\n"
        "            continue\n"
        "        direction = np.sign(s_g['ret_T1'])\n"
        "        pnl = s_g[f'ret_ride_{h}h_pips'] * direction - AVG_SPREAD_PIPS\n"
        "        ev = pnl.mean()\n"
        "        flag = '*' if ev > 0 else ' '\n"
        "        row_str += f'{ev:>9.2f}{flag}'\n"
        "    print(row_str)\n"
    )
})

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"}
    },
    "cells": cells
}

with open('notebooks_10/01_continuation_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Written. {len(cells)} cells.')
