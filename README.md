# Heston Stochastic Volatility Model - SPY Options Calibration

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

## Project Overview

This project implements the **Heston (1993) stochastic volatility model** from scratch, calibrating it to live S&P 500 ETF (SPY) options data. Unlike the Black-Scholes model which assumes constant volatility, Heston captures the dynamics of volatility smiles and term structures observed in real markets.

### Key Results

- **R² = 0.969** 
- **RMSE = $6.25 (6.3%)** 
- **89.4%** of options priced within $5 of market
- **510 options** across 17 maturities calibrated

---

## Features

### Core Implementation
-  **Characteristic function** with numerical stability safeguards
-  **Dual pricing methods**: Fast rectangular integration & adaptive quadrature
-  **Parameter validation** including Feller condition checks
-  **Risk-free curve calibration** using Nelson-Siegel-Svensson methodology
-  **Implied volatility surface** extraction and comparison

### Visualizations
-  3D option price surface comparison (Market vs Model)
-  Implied volatility surface with RMSE metrics
-  Volatility smile across multiple maturities

### Data Pipeline
-  Automated data collection via `yfinance`
-  Filtering (moneyness, maturity, bid-ask spreads)
-  Treasury yield curve integration for risk-free rates

---

##  Project Structure

```
heston-model-calibration/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── heston_calibration.py              # Main implementation
│
├── results/
│   ├── option_prices_surface.html     # 3D visualization
│   ├── implied_volatility_surface.html
│   └── volatility_smiles.html
│
├── docs/
│   └── methodology.md                 # Technical documentation
│
└── tests/
    └── test_heston.py                 # Unit tests
```

---

##  Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/heston-model-calibration.git
cd heston-model-calibration

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
matplotlib>=3.4.0
plotly>=5.0.0
yfinance>=0.1.70
nelson-siegel-svensson>=0.4.0
```

### Run Calibration

```bash
python heston_calibration.py
```

**Expected runtime:** 2-5 minutes (depending on market data availability)

---

##  The Heston Model

### Model Dynamics

The Heston model describes the evolution of an asset price S(t) and its variance v(t):

```
dS(t) = μS(t)dt + √v(t)S(t)dW₁(t)
dv(t) = κ(θ - v(t))dt + σ√v(t)dW₂(t)

where: dW₁(t)dW₂(t) = ρdt
```

### Parameters

| Parameter | Description | Calibrated Value |
|-----------|-------------|------------------|
| **v₀** | Initial variance | 0.0502 (22.4% vol) |
| **κ** | Mean reversion speed | 2.80 year⁻¹ |
| **θ** | Long-term variance | 0.0505 (22.5% vol) |
| **σ** | Volatility of volatility | 0.586 |
| **ρ** | Correlation | -0.630 |
| **λ** | Market price of vol risk | 0.109 |

### Key Insights

1. **Negative correlation (ρ = -0.63)**: Captures the leverage effect - falling prices increase volatility
2. **Fast mean reversion (κ = 2.80)**: Volatility returns to long-term level in ~3 months
3. **Positive risk premium (λ = 0.11)**: Market pays premium for volatility protection

---

##  Results & Performance

### Calibration Metrics

```
Pricing Accuracy:
├─ RMSE:          $6.25
├─ MAE:           $3.87
├─ MAPE:          4.53%
├─ R²:            0.9685
└─ Relative RMSE: 6.30%

Implied Volatility:
├─ RMSE:          7.78%
├─ MAE:           5.11%
└─ Max Error:     31.82%

Coverage:
├─ Within $1:     14.5%
├─ Within $2:     26.9%
└─ Within $5:     89.4%
```

### Visual Results

#### 1. Option Price Surface
<img width="527" height="469" alt="image" src="https://github.com/user-attachments/assets/48205afb-d6bf-4556-952e-7b5854074ac7" />
*Market prices (blue surface) vs Heston model predictions (red points)*

#### 2. Implied Volatility Surface
<img width="511" height="433" alt="image" src="https://github.com/user-attachments/assets/85cd021d-e797-44ce-bc84-0f49f659cf20" />
*Comparison of market and model-implied volatilities*

#### 3. Volatility Smile Evolution
<img width="1352" height="654" alt="image" src="https://github.com/user-attachments/assets/603a245b-795b-4920-96c6-1fc768a76a63" />
*Model captures volatility smile across different maturities*

---

##  Technical Details

### Numerical Methods

**Characteristic Function Integration:**
```python
# Fast pricing using rectangular rule
# Accurate pricing using adaptive quadrature
```

**Optimization:**
- Method: SLSQP (Sequential Least Squares Programming)
- Objective: Mean Squared Error (MSE)
- Constraints: Parameter bounds + physical feasibility
- Convergence: 79 iterations, MSE = 39.0

### Stability Features

1. **Exponential clipping** to prevent overflow
2. **Division-by-zero protection** in denominators
3. **Non-finite value handling** with fallback to zero
4. **Feller condition validation** for parameter stability

---

##  Known Limitations

### Feller Condition Status
```
2κθ = 0.283 < σ² = 0.343   VIOLATED
```

**Impact:** Theoretical possibility of variance reaching zero.

**Mitigation:**
- Violation is modest (ratio = 82.5%)
- Initial variance close to long-term level reduces risk
- Numerical safeguards prevent computational issues
- Suitable for short-to-medium term pricing (< 2 years)

### Model Limitations

1. **Short-term smile underestimation**: Heston struggles with steep short-dated skews
2. **Deep OTM options**: Higher errors for extreme strikes (< 80% or > 120% moneyness)
3. **Jump component missing**: No discontinuous price movements 

---

##  Advanced Usage

### Custom Calibration

```python
from heston_calibration import heston_price_rec, SqErr
from scipy.optimize import minimize

# Define custom parameter bounds
custom_bounds = [
    (0.01, 0.15),   # v0
    (0.5, 10.0),    # kappa
    (0.01, 0.15),   # theta
    (0.1, 2.0),     # sigma
    (-0.99, -0.3),  # rho (force negative)
    (0.0, 0.5)      # lambda
]

# Run optimization with custom settings
result = minimize(
    SqErr, 
    x0=[0.05, 3.0, 0.05, 0.3, -0.7, 0.1],
    method='SLSQP',
    bounds=custom_bounds,
    options={'maxiter': 10000, 'ftol': 1e-6}
)
```

### Price Individual Options

```python
# Price a single ATM call option
S0 = 683.44      # Current SPY price
K = 685.0        # Strike
tau = 0.25       # 3 months to expiry
r = 0.0452       # Risk-free rate

call_price = heston_price_quad(
    S0, K, 
    v0=0.0502, kappa=2.80, theta=0.0505,
    sigma=0.586, rho=-0.63, lambd=0.109,
    tau=tau, r=r
)

print(f"Call price: ${call_price:.2f}")
```

---

##  References

### Academic Papers

1. **Heston, S. L.** (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." *Review of Financial Studies*, 6(2), 327-343.

2. **Gatheral, J.** (2006). *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.

3. **Rouah, F. D.** (2013). *The Heston Model and its Extensions in Matlab and C#*. Wiley.

### Technical Resources

- [Nelson-Siegel-Svensson Yield Curve](https://nelson-siegel-svensson.readthedocs.io/)
- [QuantLib Documentation](https://www.quantlib.org/)
- [Options Pricing Theory - Hull](http://www-2.rotman.utoronto.ca/~hull/)

---

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Check code quality
flake8 heston_calibration.py
black heston_calibration.py --check
```

---

##  Contact

**Your Name**
-  Email: llankry@ncsu.edu
-  LinkedIn: http://linkedin.com/in/lucaslankry/

---

##  Acknowledgments

- Market data provided by **Yahoo Finance** via `yfinance`
- Treasury yields from **U.S. Department of the Treasury**
- Inspired by implementations from **QuantLib** and **QuantPy**
