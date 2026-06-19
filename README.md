# MONTE-CARLO-SIMULATION-AS-A-TOOL-FOR-PREDICTING-INVESTMENT-PORTFOLIO-PERFORMANCE
Hello and welcome, this repository  includes all the codes used for my master thesis oriented around predicting moves of portfolio constructed with usage of MPT by Markowitz.
Simulation was conducted with usage of simple Monte Carlo Simulation basing on Geometric Brownian Motion

# Portfolio Performance Prediction via Monte Carlo Simulation

## Overview
The purpose of this study is to assess the effectiveness of **Monte Carlo simulation** as a tool for predicting investment portfolio performance. This method is particularly suitable due to its ability to incorporate randomness and market uncertainty. 

The analysis focuses on companies listed on the **Warsaw Stock Exchange**, specifically twelve entities from the **WIG30 index** selected based on low return correlation. Three portfolios were constructed and subjected to a Monte Carlo simulation of **10,000 iterations** to forecast future performance, with the predicted results ultimately compared to actual historical outcomes.

---

## Methodology

### 1. Asset Selection & Correlation
* **Target Index:** WIG30 (Warsaw Stock Exchange)
* **Selection Criteria:** 12 entities chosen based on low return correlation using **Pearson’s coefficient** and **heatmap visualization**.
* **Objective:** Ensure a balance between diversification and computational complexity, enabling effective implementation of the **Markowitz mean-variance model** in Python.

### 2. Portfolio Construction
Three distinct portfolios were constructed based on Modern Portfolio Theory:
* **Maximum Return Portfolio**
* **Optimal Portfolio**
* **Minimum Risk Portfolio**

### 3. Evaluation Metric
Portfolio performance is evaluated using the **Mean Absolute Percentage Error (MAPE)**. A **10% threshold** was set, representing a balance between prediction accuracy and the volatility of the Polish stock market.

---

## Key Findings

* **Simulation Scope:** 10,000 iterations were executed to forecast future performance.
* **Accuracy:** While the forecasts did not meet the desired MAPE threshold of below 10%, they still fell within a **"reasonable" to "good" accuracy range**.
* **Conclusion:** The findings suggest that Monte Carlo simulation, when combined with Modern Portfolio Theory, offers a viable framework for approximating portfolio behavior under uncertain market conditions.

---

## Future Improvements
To improve forecasting performance, further refinement of the simulation model could include:
* Implementing more advanced statistical methods.
* Increasing data granularity.

---

## Technologies Used
* **Python** (Core analysis and simulation)
* **Markowitz Mean-Variance Model**
* **Monte Carlo Simulation**

```
