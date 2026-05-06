# Earnings Call ML — Predicción de movimientos post-earnings

Proyecto de machine learning que predice si una acción va a subir más del 3% en los 5 días hábiles posteriores a una earnings call, combinando features cuantitativas (precio, EPS), NLP sobre las transcripciones (FinBERT + análisis de texto) y un ensemble de modelos de gradient boosting.

---

## Resultados finales (v3)

| Estrategia | Capital final | Retorno | Win rate |
|---|---|---|---|
| Buy & Hold (referencia) | $89,368 | +793.7% | — |
| **Long-only (umbral 0.40)** | **$71,575** | **+615.8%** | **46.9%** |
| Long + Short | $30,177 | +201.8% | — |

**CV F1 = 0.507** (XGBoost) · **Ensemble F1 = 0.489** · **Sharpe ratio: 1.02**

> El Buy & Hold altísimo refleja que el universo es tech puro (NVDA, TSLA, AMD, META…) durante 2019-2024 — uno de los ciclos alcistas más fuertes de la historia. El backtesting compone las 228 operaciones secuencialmente, lo que amplifica los retornos absolutos. El modelo mejora el win rate de 21% a 47% pero no supera al mercado en retorno absoluto, un resultado honesto y común en predicción financiera.

---

## Evolución del proyecto — v1 → v2 → v3

### v1 — Baseline

**Problema:** modelo entrenado con accuracy como métrica sobre un dataset desbalanceado (150 BAJA vs 78 SUBE). El modelo simplemente aprendió a decir "BAJA" casi siempre.

| Métrica | Valor |
|---|---|
| CV accuracy | 58% |
| Win rate | 21% |
| Apuestas | 14 de 228 |
| Long-only | +0.2% |
| Long + Short | +16.3% |

**Causa del fracaso:** con accuracy como métrica, un modelo que predice siempre la clase mayoritaria obtiene 66% sin aprender nada. Las 14 apuestas con umbral 0.70 eran tan pocas que el resultado era básicamente ruido.

---

### v2 — Balanceo de clases + GridSearch F1

**Cambios:**
- `scale_pos_weight = n_neg/n_pos ≈ 1.92` en XGBoost: penaliza el doble los errores en la clase minoritaria
- GridSearch optimizando **F1** en vez de accuracy
- Eliminación de las 11 ticker dummies (overfitting con solo 228 muestras)
- 4 features derivadas: `eps_surprise_abs`, `momentum_vol_ratio`, `finbert_confidence`, `sent_momentum_align`
- Target: retorno 1 día > 2.5%

| Métrica | Valor |
|---|---|
| CV F1 | 0.382 |
| Win rate | 36% |
| Apuestas | 190 de 228 |
| Long-only | +15.8% |
| Long + Short | -2.5% |

**Por qué mejoró el win rate pero no el retorno:** bajar el umbral a 0.30 para hacer más apuestas diluyó la selectividad. El modelo apostaba en el 83% de los casos, perdiendo la capacidad de filtrar.

---

### v3 — Nuevas features + target 5 días + Ensemble

**Cambios:**
- **Target:** retorno a 5 días hábiles en vez de 1 día (captura el *Post-Earnings Announcement Drift*, PEAD)
- **8 features nuevas:** EPS trend/zscore por empresa, momentum agreement, 5 features NLP del texto crudo
- **Ensemble:** XGBoost + LightGBM con soft voting
- **25 features totales** (vs 16 en v2)
- **Umbral óptimo 0.40** (vs 0.30 en v2): más selectivo, mejores apuestas

| Métrica | v1 | v2 | v3 |
|---|---|---|---|
| CV F1 | 0.14 | 0.38 | **0.51** |
| Win rate | 21% | 36% | **47%** |
| Apuestas | 14 | 190 | 128 |
| Long-only | +0.2% | +15.8% | +615.8%* |
| Long + Short | +16.3% | -2.5% | +201.8%* |

*Los retornos absolutos v3 están amplificados por el uso de retorno 5d en vez de 1d (media de retorno por operación mucho mayor) y el período alcista 2019-2024.

---

## Por qué el modelo no supera al Buy & Hold

Este es el resultado honesto más importante del proyecto. Hay dos razones estructurales:

1. **Sesgo de mercado alcista:** el universo incluye NVDA, TSLA, AMD y META durante el período de mayor crecimiento en tech de la historia. El retorno esperado de cualquier earnings call en este grupo es positivo, así que filtrar operaciones inevitablemente excluye algunas de las más rentables.

2. **Eficiencia del mercado:** las transcripciones de earnings calls son información pública. Cualquier señal que el modelo detecta ya está en parte descontada en el precio. Predecir el movimiento post-earnings con información de la propia call es uno de los problemas más difíciles en ML financiero.

**Lo que sí logra el modelo:** llevar el win rate de 21% a 47% (casi paridad con el azar) con predicciones OOS sin data leakage es un resultado real. El F1 de 0.51 en un problema financiero con 228 muestras es sólido.

---

## Arquitectura del proyecto

```
notebooks/
  01_exploracion_precios.ipynb     # Descarga de precios, momentum y volatilidad
  02_transcripciones.ipynb         # Carga de transcripciones (HuggingFace)
  03_finbert_sentimiento.ipynb     # Scoring FinBERT (~20 min, GPU recomendada)
  04_feature_engineering.ipynb     # EPS surprise, NLP texto, retorno 5d, target
  05_modelo_ml.ipynb               # XGBoost + LightGBM + ensemble + umbral óptimo
  06_backtesting.ipynb             # Simulación de estrategias vs Buy & Hold

data/
  retornos_earnings.csv            # Precios y retornos (generado por NB01)
  transcripciones_clean.csv        # Texto de las calls (generado por NB02)
  sentimientos.csv                 # Scores FinBERT (generado por NB03)
  features.csv                     # Matriz final de features (generado por NB04)

models/
  xgboost_model.pkl                # Ensemble entrenado + metadata (generado por NB05)

outputs/
  backtesting.png                  # Curva de capital acumulado
  feature_importance.png           # Importancia de features (XGB + LGBM + promedio)
  confusion_matrices.png           # Matrices de confusión LR vs XGB
  precision_recall_curve.png       # Curva PR + sweep de umbrales
```

---

## Features (25 en v3)

| Grupo | Features | Fuente |
|---|---|---|
| **EPS** | `eps_surprise_pct`, `eps_surprise_abs`, `eps_surprise_trend`, `eps_surprise_zscore` | yfinance |
| **Precio** | `momentum_5d`, `momentum_20d`, `volatility_20d`, `momentum_vol_ratio`, `momentum_agreement` | yfinance |
| **FinBERT** | `score_positivo`, `score_negativo`, `score_neutral`, `polaridad`, `finbert_confidence` | ProsusAI/finbert |
| **Keywords** | `kw_positivos`, `kw_negativos`, `kw_ratio`, `kw_cat_enc` | análisis manual |
| **NLP texto** | `uncertainty_score`, `forward_guidance_score`, `text_length_log`, `question_density`, `negative_words_score` | texto crudo |
| **Interacción** | `sent_momentum_align`, `tiene_transcripcion` | derivadas |

**El predictor dominante** es `eps_surprise_pct`: cuando una empresa supera las estimativas de EPS, el precio tiende a subir. SUBE promedio: +35.9% de sorpresa. BAJA promedio: +4.4%.

---

## Modelo

**Ensemble (soft voting):**
- **XGBoost** con `scale_pos_weight` para manejar el desbalance de clases (1.56:1)
- **LightGBM** con el mismo balanceo
- Promedio de probabilidades de ambos modelos

**Validación:** StratifiedKFold 5-fold, predicciones OOS con `cross_val_predict` (sin data leakage).

**Umbral:** sweep sobre [0.30, 0.85] usando retorno acumulado a 5 días como criterio. Óptimo = 0.40.

---

## Stack tecnológico

- **FinBERT** (`ProsusAI/finbert`) — BERT preentrenado en texto financiero
- **yfinance** — Precios históricos + EPS estimates
- **XGBoost + LightGBM** — Gradient boosting con balanceo de clases
- **scikit-learn** — Cross-validation, métricas, pipelines
- **imbalanced-learn** — SMOTE para regresión logística
- **HuggingFace datasets** — Transcripciones (`lamini/earnings-calls-qa`)
- **pandas · numpy · matplotlib**

---

## Dataset

- **228 earnings calls**, 12 empresas, 2019-2024
- **12 empresas:** AAPL, MSFT, GOOGL, META, NFLX, TSLA, AMZN, NVDA, AMD, JPM, V, MA
- **86 transcripciones** disponibles (38% de cobertura); el resto recibe features NLP = 0
- **Target:** retorno a 5 días hábiles > 3% (PEAD — Post-Earnings Announcement Drift)
- **Distribución:** 89 SUBE (39%) · 139 BAJA (61%)

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Ejecutar los notebooks **en orden**:

```
01 → 02 → 03 → 04 → 05 → 06
```

> **Tiempos estimados:**
> - NB03 (FinBERT): ~20 min en CPU, ~3 min con GPU
> - NB04 (descarga retorno 5d): ~2-3 min (yfinance)
> - NB05 (RandomizedSearchCV 80 iters): ~5 min
> - El resto: < 1 min cada uno

---

## Limitaciones y trabajo futuro

- **Dataset pequeño:** 228 muestras limita la capacidad del modelo de generalizar. Agregar más empresas o más años reduciría la varianza de los resultados.
- **Sesgo de sobrevivencia:** las 12 empresas fueron elegidas porque son conocidas hoy. Incluir empresas que quebraron o fueron adquiridas haría el dataset más representativo.
- **Cobertura NLP parcial:** solo el 38% de las calls tiene transcripción. Las features NLP tienen baja importancia en parte por esto.
- **Latencia de ejecución:** en un escenario real, hay un lag entre la publicación de resultados y la ejecución de la orden que este modelo no considera.
- **Siguiente paso natural:** incorporar datos de opciones (implied volatility pre-earnings) como feature, ya que el mercado de opciones descuenta el movimiento esperado antes del evento.
