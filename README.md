# GARCH Dynamic Portfolio Hedging

## 1. Project Overview

This project implements a dynamic equity portfolio hedging system using GARCH-based volatility forecasting. Given a long equity portfolio, the backend estimates next-day volatility and compute the optimal hedge ratio to reach a target beta ($\beta^*$). Ticker SPY is used as a proxy for SP500 futures.

The methodology follows the minimum-variance hedge ratio method:

$$h^* = \rho_{P,S} \cdot \frac{\sigma_P}{\sigma_S}$$

The required SPY short position is then derived from the portfolio's rolling beta and the target beta level. The project is exposed through a Streamlit dashboard.

<br>

<img width="1470" height="871" alt="Screenshot 2026-03-26 at 10 17 13" src="https://github.com/user-attachments/assets/abf97a68-73c3-4a2b-a990-0888a9f7b4ad" />


---

## 2. Installation & Usage

After downloading the repository in your local environment, open a terminal at the project's root:

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run app.py
```

Streamlit will then launch and load the dashboard using local hosting.

---

## 3. Content & Methodology

#### 3.1. GARCH Volatility Forecasting
The portfolio's return series is fitted with a GARCH(1,1) model. The one-step-ahead conditional variance forecast is extracted daily over a 2 year rolling data window. This captures volatility clustering and produces more responsive risk estimates than simple historical volatility.

#### 3.2. Dynamic MV Hedge Ratio
The minimum variance hedge ratio is used to minimize the portfolio variance while approaching target beta. The hedge ratio is recomputed at each timestep using GARCH-forecasted volatilities and a rolling portfolio-SPY correlation.

#### 3.3. Backtesting
The hedge strategy is backtested over the historical period, producing P&L series, NAV comparison (hedged vs. unhedged), variance reduction magnitude, and volatility forecast estimates.

---

## 4. Structure

```
garch-dynamic-portfolio-hedging/
│
├── app.py                  
├── requirements.txt
│
└── src/
    ├── config.py           
    ├── data_loader.py      
    ├── portfolio.py        
    ├── garch_model.py      
    ├── risk.py             
    ├── hedge.py            
    ├── backtest.py         
    └── plots.py            
```
