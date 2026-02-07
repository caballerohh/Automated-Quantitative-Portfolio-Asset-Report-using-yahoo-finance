# 📊 Multi-Asset Portfolio Performance & Quantitative Risk Analytics

[cite_start]This repository features an exhaustive **Quantitative Risk Report** designed to evaluate the stability and efficiency of a diversified investment portfolio during the **August – November 2025** quarter[cite: 5, 20, 28]. [cite_start]The framework utilizes high-frequency risk metrics and statistical tests to validate the portfolio's resilience against market shocks and systematic risk[cite: 385, 386].

[cite_start]🎯 **Objective:** To quantify Alpha generation and validate portfolio resilience through the use of statistical return distribution models and risk backtesting[cite: 387, 425].

---

## 📖 Extended Overview
[cite_start]The system provides a comprehensive quantitative evaluation of a multi-asset portfolio composed of high-growth tech equities, systemic financial entities, and fixed-income stabilizers[cite: 387]. [cite_start]The analysis synchronizes log-daily returns with **Jarque-Bera normality tests** to accurately calibrate risk thresholds and identify regime changes in market volatility[cite: 419, 386].



### 🎯 Key Objectives of the Analysis
* [cite_start]**Alpha & Beta Attribution:** Evaluation of excess returns via the CAPM model, highlighting a **Jensen's Alpha of 0.13** for the consolidated portfolio[cite: 387, 405].
* [cite_start]**Tail Risk Modeling:** Calculation of **Value at Risk (VaR)** and **Conditional VaR (CVaR)** at 95% and 99% levels to quantify extreme loss scenarios[cite: 405, 426].
* [cite_start]**Risk-Adjusted Efficiency:** Comparative analysis using **Sharpe (2.62)** and **Sortino (4.51)** ratios to ensure superior return per unit of downside risk[cite: 405, 427].
* [cite_start]**Volatility Dynamics:** Implementation of **21-day Rolling Volatility** to detect regime shifts and validate strategic diversification effectiveness[cite: 386, 414].

---

## 🔍 Assets & Benchmarks Analyzed
The engine processes a strategic mix of assets to isolate active management results:

* [cite_start]**🚀 Alpha Drivers (Tech/Growth):** **AAPL** (Profit: 18.66%) [cite: 29] [cite_start]and **AMZN** (Profit: 7.25%)[cite: 92].
* [cite_start]**🏦 Financial Core:** **JPM** (Beta: 0.76) [cite: 136] [cite_start]and **GS** (Alpha: 0.17)[cite: 186].
* [cite_start]**🛡️ Stabilizer:** **BND** (Beta: 0.01), acting as the primary hedge against systematic risk[cite: 351, 375].
* [cite_start]**📊 Benchmarks:** **SPY** (S&P 500) [cite: 215] [cite_start]and **DIA** (Dow Jones) [cite: 265] for relative performance validation.

---

## 📈 Key Portfolio Results
* [cite_start]**Capital Efficiency:** The portfolio achieved a **Sharpe Ratio of 2.62**, significantly outperforming the **1.78 benchmark (SPY)**[cite: 405, 427].
* [cite_start]**Diversification Benefit:** Portfolio volatility was contained at **0.12** [cite: 405][cite_start], effectively neutralizing individual asset shocks through the "BND buffer"[cite: 386, 426].
* [cite_start]**Cumulative Performance:** Total **Profit of 8.04%** achieved over a 55-day trading period[cite: 405].
* [cite_start]**Risk Integrity:** Successful management of tail risk evidenced by a **VaR 99 of -1.67%**, lower than individual high-beta assets[cite: 405, 426].

---

## 🛠️ Code Structure & Pipeline

### 1. Data Processing Layer 📥
* [cite_start]Automated retrieval of tickers via **yfinance** with statistical cleaning of outliers and skewness handling[cite: 47, 419].

### 2. Statistical Core 🧬
* [cite_start]**Normality Testing:** Application of **Jarque-Bera tests** to assess the viability of Gaussian vs. non-normal risk models[cite: 419].
* [cite_start]**Performance Attribution:** Breakdown of Jensen's Alpha and Beta to isolate value creation from market exposure[cite: 387, 425].

### 3. Visualization Suite 🎨
* [cite_start]**Return Distribution:** Histograms featuring **Historical VaR** thresholds for tail risk visualization[cite: 420, 421].
* [cite_start]**Rolling Metrics:** Dynamic 21-day charts to monitor the evolution of risk regimes[cite: 414].

---

## 🚀 Technologies & Concepts Used
* [cite_start]**Quantitative Finance:** Capital Asset Pricing Model (CAPM), Jensen's Alpha, Sortino Ratio[cite: 387, 425, 427].
* [cite_start]**Risk Management:** Value at Risk (VaR), Volatility Clustering, and Diversification Analysis[cite: 386, 405].
* **Python Stack:** Pandas & NumPy (Matrix operations), Matplotlib & Seaborn (High-fidelity financial plotting).

---

## ⚙️ Installation & Requirements

### 1. Clone the repository
```bash
git clone [https://github.com/tu-usuario/Automated-Portfolio-Risk-Analysis.git](https://github.com/tu-usuario/Automated-Portfolio-Risk-Analysis.git)
cd Automated-Portfolio-Risk-Analysis
