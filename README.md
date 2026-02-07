# 📊 Quant-Report Pro: Automated Portfolio Risk & Performance Analytics

This repository features an exhaustive **Quantitative Risk Framework** designed to evaluate the stability and efficiency of a diversified investment portfolio. Based on the **Quarterly Financial Report (August – November 2025)**, this engine automates the validation of risk models and performance attribution across a multi-asset universe.

🎯 **Objective:** To implement a robust risk management framework using conditional volatility and tail risk metrics to optimize capital preservation and quantify Alpha generation.

---

## 📖 Extended Overview
The system provides a comprehensive quantitative evaluation of a multi-asset portfolio composed of high-growth tech equities, systemic financial entities, and fixed-income stabilizers. The analysis synchronizes log-daily returns with **Jarque-Bera normality tests** to accurately calibrate risk thresholds and identify regime changes in market volatility.



### 🎯 Key Objectives of the Analysis
* **Alpha & Beta Attribution:** Evaluation of excess returns via the CAPM model, highlighting a **Jensen's Alpha of 0.13** for the consolidated portfolio.
* **Tail Risk Modeling:** Calculation of **Value at Risk (VaR)** and **Conditional VaR (CVaR)** at 95% and 99% levels to quantify extreme loss scenarios.
* **Risk-Adjusted Efficiency:** Comparative analysis using **Sharpe (2.62)** and **Sortino (4.51)** ratios to ensure superior return per unit of downside risk.
* **Volatility Dynamics:** Implementation of **21-day Rolling Volatility** to detect regime shifts and validate strategic diversification effectiveness.

---

## 🔍 Assets & Benchmarks Analyzed
The engine processes a strategic mix of assets to isolate active management results:

* **🚀 Alpha Drivers (Tech/Growth):** **AAPL** (Profit: 18.66%) and **AMZN** (Profit: 7.25%).
* **🏦 Financial Core:** **JPM** (Beta: 0.76) and **GS** (Alpha: 0.17).
* **🛡️ Stabilizer:** **BND** (Beta: 0.01), acting as the primary hedge against systematic risk.
* **📊 Benchmarks:** **SPY** (S&P 500) and **DIA** (Dow Jones) for relative performance validation.

---

## 📈 Key Portfolio Results (Nov 2025)
* **Capital Efficiency:** The portfolio achieved a **Sharpe Ratio of 2.62**, significantly outperforming the **1.78 benchmark (SPY)**.
* **Diversification Benefit:** Portfolio volatility was contained at **0.12**, effectively neutralizing individual asset shocks through the "BND buffer".
* **Cumulative Performance:** Total **Profit of 8.04%** achieved over a 55-day trading period.
* **Risk Integrity:** Successful management of tail risk evidenced by a **VaR 99 of -1.67%**, lower than individual high-beta assets like AMZN (-4.18%).

---

## 🛠️ Code Structure & Pipeline

### 1. Data Processing Layer 📥
* Automated retrieval of tickers via **yfinance** with statistical cleaning of outliers and skewness handling.

### 2. Statistical Core 🧬
* **Normality Testing:** Application of **Jarque-Bera tests** to assess the viability of Gaussian vs. non-normal risk models.
* **Performance Attribution:** Breakdown of Jensen's Alpha and Beta to isolate value creation from market exposure.

### 3. Visualization Suite 🎨
* **Return Distribution:** Histograms featuring **Historical VaR** thresholds for tail risk visualization.
* **Rolling Metrics:** Dynamic 21-day charts to monitor the evolution of risk regimes.

---

## 🚀 Technologies & Concepts Used
* **Quantitative Finance:** Capital Asset Pricing Model (CAPM), Jensen's Alpha, Sortino Ratio.
* **Risk Management:** Value at Risk (VaR), Volatility Clustering, and Diversification Analysis.
* **Python Stack:** Pandas & NumPy (Matrix operations), Matplotlib & Seaborn (High-fidelity financial plotting).

---

## ⚙️ Installation & Requirements

### 1. Clone the repository
```bash
git clone [https://github.com/tu-usuario/Automated-Portfolio-Risk-Analysis.git](https://github.com/tu-usuario/Automated-Portfolio-Risk-Analysis.git)
cd Automated-Portfolio-Risk-Analysis
