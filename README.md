# Automated-quantitative-analysis-portfolio-asset-report-using.yahoofinance
This repository offers a Quantitative Financial Report focused on evaluating the risk and return of an investment portfolio. The analysis uses key metrics such as Volatility, Sharpe Ratio, and Jensen's Alpha to determine the portfolio's efficiency relative to the benchmark (SPY). Objective: To apply quantitative finance to optimize decision-making.

Extended version:
  This repository contains a comprehensive analysis of the performance of an investment portfolio over a quarter (August 20, 2025 to November 13, 2025). The main objective is to apply quantitative finance techniques to    evaluate the risk/return trade-off of individual assets and the portfolio as a whole.

Key Objectives of the Analysis
  Risk Measurement: Calculation of Volatility (21-day rolling volatility) and historical Value at Risk (VaR) at 95% and 99% confidence levels.
  Risk-Adjusted Return Evaluation: Use of key ratios such as the Sharpe Ratio and the Sortino Ratio to measure capital efficiency.
  Relative Performance Analysis: Calculation of Jensen's Alpha and Beta to determine risk-adjusted return relative to the benchmark (SPY).
  Statistical Analysis: Study of the return distribution (mean, median, skewness, kurtosis) to identify the presence of fat tails (tail risk) and normality.

Assets Analyzed
  The portfolio includes a mix of high-momentum, value, and stabilizing assets:
  Individual Equities: Apple (AAPL), Amazon (AMZN), JPMorgan (JPM), Goldman Sachs (GS).
  Equity ETFs (Benchmarks): SPDR S&P 500 (SPY) and Dow Jones Industrial Average (DIA).
  Fixed Income (Stabilizer): Vanguard Total Bond Market ETF (BND).

Key Portfolio Results
  The quantitative analysis of the portfolio demonstrated the effectiveness of diversification:
  Portfolio Beta: 0.92, indicating a slightly lower systematic risk than the market.
  Portfolio Volatility: 0.12, lower than SPY (0.11), confirming effective active risk management.
  Alpha Jensen: 0.15, confirming value creation by portfolio management.
  Sharpe Ratio: 3.28, higher than the SPY benchmark (2.37), demonstrating improved capital efficiency.

Code Structure
  The conceptual code file (comment_description and comment_metrics) contains the textual analysis structure for a financial reporting backend, where comments are generated based on calculated numerical and graphical      data.
  comment_description: Contains the analysis of price evolution, daily returns (skew, outliers), and rolling volatility, linking price action to macroeconomic or asset-specific events.
  Metrics comment: Summarizes the evaluation of risk metrics (Alpha, Beta, Sharpe, VaR), interpreting the asset's efficiency and risk for decision-making.

Technologies/Concepts Used
  Quantitative Finance
  Portfolio Management (Modern Portfolio Theory - MPT)
  Python/Pandas (Implicit for metrics)
  Return Distribution Analysis
  Risk Metrics (VaR, Volatility)
  Performance Metrics (Sharpe, Jensen's Alpha)
