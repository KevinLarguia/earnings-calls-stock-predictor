import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / 'data'   / 'features.csv'
MODEL_PATH = BASE_DIR / 'models' / 'xgboost_model.pkl'

st.set_page_config(page_title='Earnings Call Predictor', layout='wide')

# ── Data & predictions ────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=['fecha_earnings'])

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_predictions():
    df = load_data()
    modelo_info = load_model()
    feature_cols = modelo_info['feature_cols']

    X = df[feature_cols].values
    y = df['target'].values

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    xgb = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42, eval_metric='logloss', verbosity=0
    )
    pred_clase = cross_val_predict(xgb, X, y, cv=cv, method='predict')
    pred_proba = cross_val_predict(xgb, X, y, cv=cv, method='predict_proba')

    df = df.copy()
    df['pred_clase'] = pred_clase
    df['prob_sube']  = pred_proba[:, 1].round(3)
    df['correcto']   = df['pred_clase'] == df['target']
    return df.sort_values('fecha_earnings').reset_index(drop=True)


def simular_estrategia(df, umbral, capital_inicial=10_000):
    cap_bnh = cap_lo = cap_ls = capital_inicial
    hist_bnh = hist_lo = hist_ls = None
    hist_bnh, hist_lo, hist_ls = [capital_inicial], [capital_inicial], [capital_inicial]

    for _, row in df.iterrows():
        ret = row['retorno_pct'] / 100
        cap_bnh *= (1 + ret)
        if row['prob_sube'] >= umbral:
            cap_lo *= (1 + ret)
            cap_ls *= (1 + ret)
        elif row['prob_sube'] <= (1 - umbral):
            cap_ls *= (1 - ret)
        hist_bnh.append(round(cap_bnh, 2))
        hist_lo.append(round(cap_lo, 2))
        hist_ls.append(round(cap_ls, 2))

    return cap_bnh, cap_lo, cap_ls, hist_bnh, hist_lo, hist_ls


# ── Layout ────────────────────────────────────────────────────────────────

st.title('Earnings Call Sentiment Predictor')
st.caption('XGBoost + FinBERT · Predice movimientos >2.5% post-earnings · 86 calls, 12 empresas, 2019-2024')

modelo_info = load_model()

with st.spinner('Generando predicciones out-of-sample...'):
    df = get_predictions()

capital_inicial = 10_000

# ── Sidebar ───────────────────────────────────────────────────────────────

with st.sidebar:
    st.header('Parámetros')
    umbral = st.slider('Umbral de confianza', min_value=0.50, max_value=0.80,
                       value=0.70, step=0.05,
                       help='Solo invertir cuando prob_sube >= umbral')
    st.caption(f'Apuestas activas: {(df["prob_sube"] >= umbral).sum()} de {len(df)}')
    st.divider()
    st.metric('CV Accuracy', f'{modelo_info["cv_accuracy_mean"]:.3f}',
              delta=f'±{modelo_info["cv_accuracy_std"]:.3f}')
    st.caption('Validación cruzada 5-fold · Baseline random: 0.500')

cap_bnh, cap_lo, cap_ls, hist_bnh, hist_lo, hist_ls = simular_estrategia(df, umbral)

# ── KPI cards ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)
c1.metric('Buy & Hold',    f'${cap_bnh:,.0f}', f'{(cap_bnh/capital_inicial-1)*100:+.1f}%')
c2.metric('Long-only ML',  f'${cap_lo:,.0f}',  f'{(cap_lo/capital_inicial-1)*100:+.1f}%',
          delta_color='normal')
c3.metric('Long+Short ML', f'${cap_ls:,.0f}',  f'{(cap_ls/capital_inicial-1)*100:+.1f}%',
          delta_color='normal')
c4.metric('Alpha (Long vs B&H)', f'${cap_lo - cap_bnh:+,.0f}',
          f'{(cap_lo/capital_inicial - cap_bnh/capital_inicial)*100:+.1f} pp',
          delta_color='normal')

st.divider()

# ── Capital curve ─────────────────────────────────────────────────────────

col_chart, col_thresh = st.columns([2, 1])

with col_chart:
    st.subheader('Retorno acumulado')
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist_bnh, label=f'Buy & Hold  ${cap_bnh:,.0f}',  color='#90a4ae', linewidth=2)
    ax.plot(hist_lo,  label=f'Long-only   ${cap_lo:,.0f}',   color='#26a69a', linewidth=2)
    ax.plot(hist_ls,  label=f'Long+Short  ${cap_ls:,.0f}',   color='#5c6bc0', linewidth=1.5, linestyle='--')
    ax.axhline(capital_inicial, color='black', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.set_xlabel('Earnings call #')
    ax.set_ylabel('Capital ($)')
    ax.legend(fontsize=9)
    ax.set_title(f'Umbral de confianza: {umbral}', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_thresh:
    st.subheader('Sweep de umbrales')
    rows = []
    for u in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        apuestas_u = df[df['prob_sube'] >= u]
        if len(apuestas_u) == 0:
            continue
        _, cap, _, _, _, _ = simular_estrategia(df, u)
        rows.append({
            'Umbral': u,
            'Apuestas': len(apuestas_u),
            'Win rate': f'{apuestas_u["correcto"].mean():.0%}',
            'Capital': f'${cap:,.0f}',
            'Retorno': f'{(cap/capital_inicial-1)*100:+.1f}%',
        })
    df_thresh = pd.DataFrame(rows).set_index('Umbral')
    st.dataframe(df_thresh, use_container_width=True)

st.divider()

# ── Per-ticker ────────────────────────────────────────────────────────────

st.subheader('Performance por empresa')

ticker_rows = []
for ticker, grp in df.groupby('ticker'):
    apuestas_g = grp[grp['prob_sube'] >= umbral]
    ticker_rows.append({
        'Ticker': ticker,
        'Calls totales': len(grp),
        'Apuestas': len(apuestas_g),
        'Win rate': f'{apuestas_g["correcto"].mean():.0%}' if len(apuestas_g) > 0 else '-',
        'Retorno prom. apuesta': f'{apuestas_g["retorno_pct"].mean():+.1f}%' if len(apuestas_g) > 0 else '-',
        'Accuracy total': f'{grp["correcto"].mean():.0%}',
    })

df_tickers = pd.DataFrame(ticker_rows).set_index('Ticker')
st.dataframe(df_tickers, use_container_width=True)

st.divider()

# ── Predictions table ─────────────────────────────────────────────────────

st.subheader('Todas las predicciones')

df_display = df[['ticker', 'fecha_earnings', 'retorno_pct', 'prob_sube', 'pred_clase', 'target', 'correcto']].copy()
df_display['fecha_earnings'] = df_display['fecha_earnings'].dt.strftime('%Y-%m-%d')
df_display['retorno_pct']    = df_display['retorno_pct'].round(2)
df_display['pred_clase']     = df_display['pred_clase'].map({1: 'SUBE >2.5%', 0: 'NO'})
df_display['target']         = df_display['target'].map({1: 'SUBE >2.5%', 0: 'NO'})
df_display['correcto']       = df_display['correcto'].map({True: '✓', False: '✗'})
df_display = df_display.rename(columns={
    'ticker': 'Ticker', 'fecha_earnings': 'Fecha', 'retorno_pct': 'Retorno %',
    'prob_sube': 'Prob. sube', 'pred_clase': 'Predicción', 'target': 'Real', 'correcto': 'OK'
})

apuesta_mask = df['prob_sube'] >= umbral
df_display_sorted = pd.concat([
    df_display[apuesta_mask],
    df_display[~apuesta_mask]
])

st.dataframe(
    df_display_sorted.style.apply(
        lambda row: ['background-color: #e8f5e9' if row['OK'] == '✓' and row['Predicción'] == 'SUBE >2.5%'
                     else 'background-color: #ffebee' if row['OK'] == '✗' and row['Predicción'] == 'SUBE >2.5%'
                     else '' for _ in row],
        axis=1
    ),
    use_container_width=True,
    height=400
)
