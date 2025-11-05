# Heston Stochastic Volatility Model - SPY Options Calibration

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/lucas-lankry/Heston-Model-Calibration-to-SPY-Options?style=social)](https://github.com/lucas-lankry/Heston-Model-Calibration-to-SPY-Options/stargazers)


##  Project Overview

This project implements and calibrates the **Heston (1993) stochastic volatility model** to price SPY call options, demonstrating superior performance over the traditional Black-Scholes-Merton model in capturing market volatility dynamics.

### Key Features

-  **Real-time market data** collection via yfinance API
   **Heston model** implementation with rectangular integration
-  **Black-Scholes-Merton** baseline comparison
-  **Nelson-Siegel-Svensson** yield curve calibration for risk-free rates
-  **3D interactive visualizations** using Plotly
-  **Implied volatility** surface analysis
-  **Comprehensive performance metrics** and validation

---

##  Key Results 
(*Last updated: November 2025 | SPY Price: $682.06 | Options analyzed: 600*)

### Model Comparison Summary

| Metric | Heston | BSM | Heston Improvement |
|--------|--------|-----|-------------------|
| **RMSE** | $5.52 | $7.07 | **21.9% better**  |
| **MAE** | $3.34 | $5.53 | **39.7% better**  |
| **MAPE** | 4.03% | 7.25% | **44.4% better**  |
| **R²** | 0.9759 | 0.9605 | **1.6% better**  |
| **Parameters** | 6 | 1 | More flexible |

### Performance Breakdown

#### By Moneyness
- **Deep OTM**: Heston wins by 31.7%
- **OTM**: Heston wins by 52.4%
- **ATM**: Heston wins by 70.7%
- **ITM**: Heston wins by 87.6%
- **Deep ITM**: Heston wins by 95.2%

#### By Maturity
- **Short-term (<3m)**: Heston wins by 21.9%
- **Medium-term (3-6m)**: Heston wins by 30.3%
- **Long-term (6m-1y)**: Heston wins by 35.9%
- **Very long-term (>1y)**: Heston wins by 83.0%

### Calibrated Heston Parameters
```
SPY Price: $682.06

v0 (Initial variance):           0.0529 (σ₀ = 23.01%)
κ (Mean reversion speed):        2.7557
θ (Long-term variance):          0.0421 (σ_∞ = 20.51%)
σ (Volatility of volatility):   0.4871
ρ (Correlation):                -1.0000 
λ (Market price of vol risk):   0.2153
```

**Note**: Feller condition (2κθ > σ²) slightly violated (0.2318 vs 0.2373), indicating potential for variance to reach zero in extreme scenarios.

---

##  Visual Results

### Option Prices: 3D Surface Comparison

<img width="860" height="505" alt="image" src="https://github.com/user-attachments/assets/98228dc3-1f3d-490e-81b0-21ed524d02db" />


**Key Observation**: The blue mesh represents market prices. Red dots (Heston) closely follow the market surface, while green diamonds (BSM) show systematic deviations, especially for ITM options and longer maturities.

---

### Pricing Errors: Spatial Distribution

<img width="791" height="347" alt="image" src="https://github.com/user-attachments/assets/627ffd40-8bad-4c32-bfd7-ed61f0bd6cb4" />


**Key Observation**: 
- **Heston** (left): Errors concentrated near zero (yellow/orange), with minimal extreme values
- **BSM** (right): Larger systematic errors (more red regions), particularly for longer maturities

---

### Error Distributions: Statistical Comparison

<img width="843" height="335" alt="image" src="https://github.com/user-attachments/assets/e85d5742-8722-4b08-8f46-4c53ead1c848" />


**Key Observation**:
- **Heston** (red): Tight distribution centered near zero (σ = $5.50)
- **BSM** (green): Wider spread with systematic bias (σ = $6.93)
- Heston's tighter distribution demonstrates superior calibration accuracy

---

### Volatility Smile:

<img width="884" height="555" alt="image" src="https://github.com/user-attachments/assets/70c880bc-8193-48dc-afc6-46331f80ff6f" />

Across all maturities (0.05y to 1.37y):
- **Market** (blue dots): Clear downward-sloping smile pattern
- **Heston** (red line): Accurately captures the smile curvature
- **BSM** (green dashed): Flat line - **cannot model the smile by design**

The smile effect becomes more pronounced for:
- Shorter maturities (60% IV for deep OTM at 0.05y)
- Longer maturities (smoother but persistent 16-28% range at 1.37y)

---

###  Want Interactive Plots?

The static images above are screenshots. To explore **fully interactive 3D visualizations** where you can:
-  Rotate plots in real-time
-  Zoom into specific regions  
-  Hover for exact values
-  Export custom views

Simply run:
```bash
python Heston_Comparison_to_BSM.py
```

The interactive Plotly charts will open automatically in your browser!

---

##  Why These Visuals Matter

### 1. **3D Surface Plot** → Model Fit Quality
The mesh visualization shows how well models approximate the true market surface. Heston's tight clustering around the market mesh demonstrates superior fit across the entire strike-maturity space.

### 2. **Error Plots** → Systematic Bias Detection  
The spatial error distribution reveals:
- **Heston**: Random, unbiased errors (good calibration)
- **BSM**: Systematic patterns indicating model misspecification

### 3. **Histograms** → Statistical Validity
The error distributions prove Heston's superiority statistically:
- 20% tighter standard deviation
- More symmetric distribution (less bias)
- Fewer extreme outliers

### 4. **Volatility Smile** → Economic Realism 

- Real markets exhibit volatility smiles due to crash fears, skewness, and kurtosis
- BSM's flat line violates this empirical reality
- Heston's curved fit reflects true market dynamics

**Bottom Line**: BSM systematically misprices options because it cannot capture volatility smile dynamics that are fundamental to options markets.

---

##  Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lucas-lankry/Heston-Model-Calibration-to-SPY-Options.git
cd Heston-Model-Calibration-to-SPY-Options
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the comparison:
```bash
python Heston_Comparison_to_BSM.py
```

**Expected runtime**: ~5-10 minutes

---

##  Model Architecture

### Heston Model

The Heston model assumes stochastic volatility following:
```
dS_t = μS_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t)dt + σ√v_t dW_t^v
dW_t^S dW_t^v = ρdt
```

**Advantages over BSM**:
1.  Captures **volatility smile** (varying IV across strikes)
2.  Models **term structure** of volatility
3.  Accounts for **leverage effect** (ρ < 0)
4.  Handles **volatility clustering**

### Implementation Details

- **Pricing method**: Rectangular integration (midpoint rule, N=10,000)
- **Optimization**: SLSQP (Sequential Least Squares Programming)
- **Data scope**: 600 options across 20 maturities (0.05-1.37 years)
- **Strike range**: 80%-120% of spot price ($546-$780)

---

##  Visualizations

The code generates **4 interactive 3D Plotly visualizations** that can be rotated and zoomed:

| Visualization | Purpose | Key Insight |
|---------------|---------|-------------|
| **Option Prices Surface** | Model fit comparison | Heston tracks market surface; BSM deviates systematically |
| **Pricing Errors (3D)** | Error spatial distribution | Heston errors random; BSM errors show patterns |
| **Error Histograms** | Statistical validation | Heston: σ=$5.50; BSM: σ=$6.93 (20% tighter) |
| **Volatility Smile** | Economic realism test | Heston captures smile; BSM produces flat line |

###  The Volatility Smile

The volatility smile visualization is **the most important chart** because it demonstrates:

1. **Market Reality**: Options traders price OTM puts higher (implied vol ~60% at 0.05y maturity) due to crash risk
2. **Heston Success**: Red line follows the blue market dots, capturing the smile curvature
3. **BSM Failure**: Green dashed line is flat (~23% constant) - physically incorrect

This explains why Heston achieves:
- **95.2% better pricing** for deep ITM options
- **83.0% better pricing** for long-term options
- **Overall 39.7% lower MAE**

>  **Intuition**: The smile exists because market participants know volatility isn't constant (BSM assumption). Heston models volatility as stochastic, matching reality.

---

##  Technical Highlights

### Numerical Stability
- Exponential argument clipping to prevent overflow
- Near-zero denominator handling
- Non-finite value replacement

### Parameter Validation
- Feller condition checking
- Economic reasonableness tests
- Boundary constraint enforcement

### Risk-Free Rate Calibration
Nelson-Siegel-Svensson yield curve fitted to US Treasury rates:
```
Maturities: 1m, 2m, 3m, 6m, 1y, 2y, 3y, 5y, 7y, 10y, 20y, 30y
```

---

##  Project Structure
```
├── Heston_Comparison_to_BSM.py     #  Performance Comparison to BSM
├── heston_calibration.py           # Main implementation
├── Heston_Calibration_Demo.ipynb   # Notebook Jupyter
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

##  Why Heston Outperforms BSM

### Theoretical Advantages
1. **Stochastic Volatility**: Captures real-world volatility dynamics
2. **Leverage Effect**: Negative correlation (ρ = -1) models stock-volatility relationship
3. **Fat Tails**: Better fits extreme moves than log-normal distribution
4. **Smile Dynamics**: Term structure of volatility smile

### Empirical Evidence (This Project)
- **39.7% lower MAE** across all options
- **Consistently superior** across all moneyness levels
- **Especially strong** for ITM/Deep ITM options (87.6-95.2% improvement)
- **Excellent long-term** performance (83% improvement for >1y maturity)

---

##  Limitations & Considerations

1. **Feller Condition**: Current calibration slightly violates Feller condition 
2. **Computational Cost**: Heston ~50x slower than BSM (but still < 1 second per option)
3. **Calibration Stability**: 6 parameters require more data than BSM's 1 parameter
4. **Perfect Correlation**: ρ = -1.0 at boundary may indicate over-fitting

---

##  References & Resources

### Technical Resources

- [Nelson-Siegel-Svensson Yield Curve](https://nelson-siegel-svensson.readthedocs.io/)
- [QuantLib Documentation](https://www.quantlib.org/)
- [Options Pricing Theory - Hull](http://www-2.rotman.utoronto.ca/~hull/)

##  Academic Papers

1. **Heston, S. L.** (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327-343.

2. **Black, F., & Scholes, M.** (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

3. **Nelson, C. R., & Siegel, A. F.** (1987). "Parsimonious Modeling of Yield Curves." *Journal of Business*, 60(4), 473-489.

---

##  Contact

**Author**
-  Email: llankry@ncsu.edu
-  LinkedIn: http://linkedin.com/in/lucaslankry/
- GitHub: [@lucas-lankry](https://github.com/lucas-lankry)
  
---

##  Acknowledgments

- Market data provided by **Yahoo Finance** via `yfinance`
- Treasury yields from **U.S. Department of the Treasury**
- Inspired by implementations from **QuantLib** and **QuantPy**

*Last updated: November 2025 | SPY Price: $682.06 | Options analyzed: 600*
