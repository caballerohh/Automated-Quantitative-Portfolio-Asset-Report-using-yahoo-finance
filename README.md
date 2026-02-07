# 📈 Impact-of-macro-financial-variables-on-SCCO

This repository features a Python-based analytical engine designed to merge macroeconomic indicators with financial market data. The project quantifies the relationship between **"Real Economy" variables**—such as inflation and interest rates—and the performance of equities and commodities, using **$SCCO (Southern Copper Corp)** as a proxy for the copper market.

🎯 **Objective:** To automate the extraction of multi-source financial data and apply advanced statistical techniques (Rolling Correlations and Annualized Volatility) to identify macroeconomic regimes.

---

## 📖 Extended Overview
The system integrates high-frequency market data from **Yahoo Finance** with low-frequency macroeconomic series from the **Federal Reserve Economic Data (FRED)**. By synchronizing these datasets into a unified "Master Table," the project enables the study of cross-asset dynamics, such as the effectiveness of Gold as an inflation hedge and the impact of Fed interest rate cycles on equity market risk.

[Image of a macroeconomic correlation heatmap showing CPI, Fed Funds Rate, and Equity returns]

### 🎯 Key Objectives of the Analysis
* **Macro-Market Synchronization:** Automated data pipeline that aligns disparate frequencies (daily market prices vs. monthly macro data) using resampling and time-offset techniques.
* **Rolling Correlation Dynamics:** Implementation of **24-month rolling windows** to track the evolution of historical hedges (e.g., Inflation vs. Gold) and sensitivity to capital costs (Fed Funds Rate vs. SPY).
* **Multi-Asset Risk Assessment:** Calculation of **63-day (quarterly) rolling annualized volatility** to compare systematic market risk against idiosyncratic commodity risk.
* **Statistical Visualization:** Generation of comprehensive heatmaps and multi-subplot time series to isolate trends in unemployment, CPI, and interest rates.

---

## 🔍 Assets & Indicators Analyzed
The engine processes a diverse universe of data to capture the full economic cycle:

### 🏛️ Macro Indicators (FRED)
* **CPIAUCSL:** Consumer Price Index (Inflation).
* **UNRATE:** Unemployment Rate.
* **FEDFUNDS:** Effective Federal Funds Rate (Monetary Policy).

### 💰 Financial Assets (Yahoo Finance)
* **Equities:** **SPY** (S&P 500) and **SCCO** (Southern Copper Corp) for industrial/commodity exposure.
* **Commodities:** **GLD** (Gold) as a defensive/inflation-linked asset.

---

## 📊 Key Portfolio & Macro Results
* **Risk Regimes:** The engine identified a significant volatility gap; **SCCO (38.49%)** exhibited more than triple the annualized risk of the **SPY benchmark (11.47%)** by early 2026.
* **Macro Correlations:**
    * **Inflation vs. Rates:** A strong positive correlation (**0.88**) between CPI and Fed Funds Rate, reflecting active monetary policy response.
    * **Monetary Policy Impact:** An inverse relationship (**-0.63**) between Unemployment and Rates, validating **Phillips Curve** dynamics.
* **Market Efficiency:** Gold (GLD) demonstrated a moderate positive correlation (**0.21**) with Inflation, acting as a partial hedge during the 2020-2025 cycle.

---

## 🛠️ Code Structure & Pipeline

### 1. Data Extraction 📥
* Utilizes `pandas_datareader` for **FRED API** and `yfinance` for equity markets.

### 2. Feature Engineering ⚙️
* Implementation of monthly percentage changes and data cleaning via `resample()` to maintain statistical integrity across different timeframes.

### 3. Statistical Core 🧬
* **Dynamic Correlations:** `rolling().corr()` over a 24-month window.
* **Annualized Risk:** `rolling().std() * np.sqrt(252)` to standardize daily returns into risk metrics.

### 4. Visualization Stack 🎨
* **Seaborn:** For professional correlation heatmaps.
* **Matplotlib:** For synchronized macro-trend plotting and dual-axis time series.

---

## 🚀 Technologies & Concepts Used
* **Macroeconometrics:** Monetary Policy Analysis, Inflation Hedging, and Business Cycle tracking.
* **
