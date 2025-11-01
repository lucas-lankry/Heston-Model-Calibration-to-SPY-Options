import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.optimize import minimize
from datetime import datetime as dt

from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

import yfinance as yf


def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Calculate the Heston characteristic function.
    
    This is the fundamental building block for pricing options under the Heston model.
    The characteristic function captures the distribution of log-returns under 
    stochastic volatility.
    
    Parameters
    ----------
    phi : float or ndarray
        Characteristic function argument (frequency domain variable)
    S0 : float
        Current underlying asset price
    v0 : float
        Initial variance (volatility squared)
    kappa : float
        Speed of mean reversion for variance process
    theta : float
        Long-term variance level (mean reversion level)
    sigma : float
        Volatility of volatility (vol of vol)
    rho : float
        Correlation between asset returns and variance, range [-1, 1]
    lambd : float
        Market price of volatility risk (risk premium parameter)
    tau : float
        Time to maturity in years
    r : float
        Risk-free interest rate
    
    Returns
    -------
    complex ndarray
        Characteristic function value(s)
        
    Notes
    -----
    Uses numerical stability techniques:
    - Clips exponential arguments to prevent overflow
    - Handles near-zero denominators
    - Returns 0 for non-finite values
    
    The characteristic function is derived from the Feynman-Kac theorem applied
    to the Heston PDE. See Heston (1993) for mathematical derivation.
    """
    # Constants from the Heston model
    a = kappa * theta
    b = kappa + lambd

    # Common terms with respect to phi
    rspi = rho * sigma * phi * 1j

    # Define d parameter (discriminant-like term)
    d = np.sqrt((rho * sigma * phi * 1j - b)**2 + (phi * 1j + phi**2) * sigma**2)

    # Define g parameter (ratio for solution structure)
    g = (b - rspi + d) / (b - rspi - d)
    
    # Calculate d * tau with clipping for numerical stability
    d_tau = d * tau
    
    # Calculate characteristic function by components
    exp1 = np.exp(r * phi * 1j * tau)
    
    # Calculate with handling of edge cases for exponential terms
    g_exp_d_tau = g * np.exp(np.clip(d_tau, -100, 100))
    
    # Prevent division by zero in denominators
    denom_term2 = 1 - g
    denom_term2 = np.where(np.abs(denom_term2) < 1e-10, 1e-10, denom_term2)
    
    num_term2 = 1 - g_exp_d_tau
    num_term2 = np.where(np.abs(num_term2) < 1e-10, 1e-10, num_term2)
    
    term2 = S0**(phi * 1j) * (num_term2 / denom_term2)**(-2 * a / sigma**2)
    
    # Calculate the exponential term for variance process
    numerator = 1 - np.exp(np.clip(d_tau, -100, 100))
    denominator = 1 - g_exp_d_tau
    denominator = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    
    ratio = numerator / denominator
    
    # Exponent argument with variance components
    exp_arg = (a * tau * (b - rspi + d) / sigma**2 + 
               v0 * (b - rspi + d) * ratio / sigma**2)
    
    # Clip the argument of the exponential to prevent overflow
    exp_arg_real = np.real(exp_arg)
    exp_arg = np.where(exp_arg_real > 100, 100 + 1j * np.imag(exp_arg), exp_arg)
    exp_arg = np.where(exp_arg_real < -100, -100 + 1j * np.imag(exp_arg), exp_arg)
    
    exp2 = np.exp(exp_arg)

    result = exp1 * term2 * exp2
    
    # Replace non-finite values with 0
    result = np.where(np.isfinite(result), result, 0 + 0j)
    
    return result


def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """
    Price European call options using Heston model with rectangular integration.
    
    This function uses the midpoint rule for numerical integration, which is
    faster than adaptive methods but less accurate. Suitable for calibration
    where many price evaluations are needed.
    
    Parameters
    ----------
    S0 : float or ndarray
        Current underlying asset price
    K : float or ndarray
        Strike price(s)
    v0 : float
        Initial variance
    kappa : float
        Speed of mean reversion
    theta : float
        Long-term variance
    sigma : float
        Volatility of volatility
    rho : float
        Correlation between returns and variance
    lambd : float
        Market price of volatility risk
    tau : float or ndarray
        Time to maturity in years
    r : float or ndarray
        Risk-free rate
    
    Returns
    -------
    float or ndarray
        European call option price(s)
        
    Notes
    -----
    Integration method: Midpoint rectangular rule over [0, 100]
    - N=10000 integration points
    - Generally achieves price accuracy within $0.01 for liquid options
    
    The pricing formula uses the characteristic function to evaluate:
    C = (S0 - K*e^(-rT))/2 + (1/π) * ∫[0,∞] Re[integrand] dφ
    """
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)

    P, umax, N = 0, 100, 10000
    dphi = umax / N  # Width of each rectangle

    for i in range(1, N):
        # Rectangular integration using midpoint rule
        phi = dphi * (2*i + 1) / 2  # Midpoint to calculate height
        numerator = (np.exp(r*tau) * heston_charfunc(phi-1j, *args) - 
                     K * heston_charfunc(phi, *args))
        denominator = 1j * phi * K**(1j * phi)

        P += dphi * numerator / denominator

    return np.real((S0 - K*np.exp(-r*tau)) / 2 + P / np.pi)

# In the model, we don't use this method to gain time, however you can still use it check the accuracy of your results

# def heston_price_quad(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
#     """
#     Price a European call option using Heston model with adaptive integration.
    
#     This function uses scipy's adaptive quadrature for higher accuracy but
#     is slower than rectangular integration. Best used for final pricing
#     or verification, not for calibration loops.
    
#     Parameters
#     ----------
#     S0 : float
#         Current underlying asset price
#     K : float
#         Strike price (must be scalar for quad integration)
#     v0 : float
#         Initial variance
#     kappa : float
#         Speed of mean reversion
#     theta : float
#         Long-term variance
#     sigma : float
#         Volatility of volatility
#     rho : float
#         Correlation between returns and variance
#     lambd : float
#         Market price of volatility risk
#     tau : float
#         Time to maturity in years
#     r : float
#         Risk-free rate
    
#     Returns
#     -------
#     float
#         European call option price
        
#     Notes
#     -----
#     Uses scipy.integrate.quad with adaptive step size.
#     More accurate than rectangular integration but ~10-50x slower.
#     Cannot handle array inputs - use only for scalar pricing.
#     """
#     def integrand_local(phi):
#         """Local integrand function for this specific option."""
#         args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
#         numerator = (np.exp(r*tau) * heston_charfunc(phi-1j, *args) - 
#                      K * heston_charfunc(phi, *args))
#         denominator = 1j * phi * K**(1j * phi)
#         return numerator / denominator

#     real_integral, err = quad(lambda phi: np.real(integrand_local(phi)), 0, 100)

#     return (S0 - K*np.exp(-r*tau)) / 2 + real_integral / np.pi


def validate_heston_parameters(v0, kappa, theta, sigma, rho, lambd):
    """
    Validate Heston model parameters and check theoretical conditions.
    
    Parameters
    ----------
    v0 : float
        Initial variance
    kappa : float
        Mean reversion speed
    theta : float
        Long-term variance
    sigma : float
        Vol of vol
    rho : float
        Correlation
    lambd : float
        Risk premium
        
    Returns
    -------
    dict
        Dictionary containing validation results and warnings
    """
    results = {
        'valid': True,
        'warnings': [],
        'feller_condition': None,
        'feller_satisfied': None
    }
    
    # Check Feller condition: 2*kappa*theta > sigma^2
    # This ensures the variance process stays positive
    feller_lhs = 2 * kappa * theta
    feller_rhs = sigma**2
    results['feller_condition'] = f"2κθ = {feller_lhs:.4f} vs σ² = {feller_rhs:.4f}"
    results['feller_satisfied'] = feller_lhs > feller_rhs
    
    if not results['feller_satisfied']:
        results['warnings'].append(
            "Feller condition violated! Variance process may reach zero, "
            "leading to potential numerical issues."
        )
    
    # Check parameter bounds
    if v0 <= 0 or theta <= 0:
        results['warnings'].append("Variance parameters must be positive")
        results['valid'] = False
        
    if kappa <= 0:
        results['warnings'].append("Mean reversion speed must be positive")
        results['valid'] = False
        
    if sigma <= 0:
        results['warnings'].append("Vol of vol must be positive")
        results['valid'] = False
        
    if not -1 <= rho <= 1:
        results['warnings'].append("Correlation must be in [-1, 1]")
        results['valid'] = False
    
    # Economic reasonableness checks
    if abs(v0 - theta) > 0.5 * theta:
        results['warnings'].append(
            f"Initial variance v0={v0:.4f} differs significantly from "
            f"long-term level θ={theta:.4f}. This may indicate unstable calibration."
        )
    
    if kappa > 10:
        results['warnings'].append(
            f"Very high mean reversion speed κ={kappa:.2f} suggests "
            "unrealistic volatility dynamics"
        )
    
    return results


# ============================================================================
# MARKET DATA COLLECTION
# ============================================================================

print("="*80)
print("HESTON MODEL CALIBRATION TO SPY OPTIONS")
print("="*80 + "\n")

# Risk-free rate from US Daily Treasury Par Yield Curve Rates
print("Step 1: Calibrating risk-free rate curve...")
yield_maturities = np.array([1/12, 2/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yields_data = np.array([0.15, 0.27, 0.50, 0.93, 1.52, 2.13, 2.32, 2.34, 
                        2.37, 2.32, 2.65, 2.52]).astype(float) / 100

# Fit Nelson-Siegel-Svensson curve
curve_fit, status = calibrate_nss_ols(yield_maturities, yields_data)
print(f"   ✓ Risk-free curve calibrated successfully\n")

# Download S&P 500 data
print("Step 2: Fetching SPY market data...")
ticker = yf.Ticker("SPY")
S0 = ticker.history(period="1d")['Close'].iloc[-1]
print(f"   Current SPY price: ${S0:.2f}")

# Get available expiration dates
expirations = ticker.options
print(f"   Available expiration dates: {len(expirations)}\n")

# Collect market prices
print("Step 3: Collecting options data...")
market_prices = {}

for expiration_date in expirations:
    try:
        opt_chain = ticker.option_chain(expiration_date)
        calls = opt_chain.calls
        
        # Filter options with valid bid/ask prices
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        
        if len(calls) > 0:
            market_prices[expiration_date] = {}
            market_prices[expiration_date]['strike'] = calls['strike'].tolist()
            market_prices[expiration_date]['price'] = ((calls['bid'] + calls['ask']) / 2).tolist()
    except Exception as e:
        print(f"   Warning: Error for {expiration_date}: {e}")
        continue

# Filter and prepare data
print("\nStep 4: Filtering options data...")
prices_list = []
maturities_list = []
strikes_list = []

min_strike = S0 * 0.80  # 80% to 120% of current price
max_strike = S0 * 1.20
options_per_expiry_limit = 30  # Limit to avoid over-representation of specific maturities

for date, v in market_prices.items():
    maturity = (dt.strptime(date, '%Y-%m-%d') - dt.today()).days / 365.25
    
    # Filter only maturities between ~2 weeks and 1.5 years
    if 0.04 < maturity < 1.5:
        options_added_for_this_expiry = 0

        for strike, price in zip(v['strike'], v['price']):
            # Filter strikes around current price (±20%)
            if min_strike <= strike <= max_strike:
                maturities_list.append(maturity)
                strikes_list.append(strike)
                prices_list.append(price)

                options_added_for_this_expiry += 1

                if options_added_for_this_expiry >= options_per_expiry_limit:
                    break

# Create DataFrame
volSurfaceLong = pd.DataFrame({
    'maturity': maturities_list,
    'strike': strikes_list,
    'price': prices_list
})

# Calculate risk-free rate for each option
volSurfaceLong['rate'] = volSurfaceLong['maturity'].apply(curve_fit)

print(f"   Total options collected: {len(volSurfaceLong)}")
print(f"   Strike range: ${volSurfaceLong['strike'].min():.2f} - ${volSurfaceLong['strike'].max():.2f}")
print(f"   Maturity range: {volSurfaceLong['maturity'].min():.2f} - {volSurfaceLong['maturity'].max():.2f} years")
print(f"   Unique strikes: {volSurfaceLong['strike'].nunique()}")
print(f"   Unique maturities: {volSurfaceLong['maturity'].nunique()}\n")


# ============================================================================
# CALIBRATION
# ============================================================================

print("="*80)
print("CALIBRATION PROCESS")
print("="*80 + "\n")

# Extract variables for calibration
r = volSurfaceLong['rate'].to_numpy('float')
K = volSurfaceLong['strike'].to_numpy('float')
tau = volSurfaceLong['maturity'].to_numpy('float')
P = volSurfaceLong['price'].to_numpy('float')

# Parameter configuration with bounds
# Note on lambda: The market price of volatility risk (lambda) adjusts for
# the difference between physical and risk-neutral measures. Setting lambda=0
# assumes no risk premium for volatility risk, which is unrealistic in practice.
# Including lambda allows the model to capture the volatility risk premium
# observed in options markets (typically negative, meaning investors pay premium
# for volatility protection).
params = {
    "v0": {"x0": 0.1, "lbub": [1e-3, 0.1], "desc": "Initial variance"},
    "kappa": {"x0": 3, "lbub": [1e-3, 5], "desc": "Mean reversion speed"},
    "theta": {"x0": 0.05, "lbub": [1e-3, 0.1], "desc": "Long-term variance"},
    "sigma": {"x0": 0.3, "lbub": [1e-2, 1], "desc": "Volatility of volatility"},
    "rho": {"x0": -0.8, "lbub": [-1, 0], "desc": "Correlation (typically negative)"},
    "lambd": {"x0": 0.03, "lbub": [-1, 1], "desc": "Market price of vol risk"},
}

x0 = [param["x0"] for key, param in params.items()]
bnds = [param["lbub"] for key, param in params.items()]

print("Initial parameter guesses:")
for key, param in params.items():
    print(f"   {key:6s} = {param['x0']:7.4f}  [{param['lbub'][0]:6.3f}, {param['lbub'][1]:6.3f}]  # {param['desc']}")

print("\nNote on lambda parameter:")
print("   Lambda captures the market price of volatility risk - the compensation")
print("   investors require for bearing volatility risk. Setting lambda=0 would")
print("   assume no risk premium, which contradicts empirical evidence showing")
print("   investors pay premiums for volatility protection (e.g., VIX options).")
print("   Including lambda improves model fit and economic realism.\n")

def SqErr(x):
    """
    Calculate mean squared error between market and model prices.
    
    This is the objective function for calibration. We use MSE rather than
    relative errors to avoid overweighting cheap OTM options.
    """
    v0, kappa, theta, sigma, rho, lambd = x
    
    # Use rectangular integration for speed during optimization
    model_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
    
    # Mean squared error
    mse = np.sum((P - model_prices)**2) / len(P)
    
    return mse

print("Starting optimization (this may take several minutes)...")
print("Method: SLSQP (Sequential Least Squares Programming)")
print("Pricing method: Rectangular integration (fast)\n")

result = minimize(SqErr, x0, tol=1e-3, method='SLSQP', 
                  options={'maxiter': 1e4}, bounds=bnds)

if result.success:
    print("✓ Optimization converged successfully!")
else:
    print(f"⚠ Optimization finished with status: {result.message}")

print(f"   Iterations: {result.nit}")
print(f"   Function evaluations: {result.nfev}")
print(f"   Final MSE: {result.fun:.6f}\n")

# Extract calibrated parameters
v0, kappa, theta, sigma, rho, lambd = result.x

print("="*80)
print("CALIBRATED PARAMETERS")
print("="*80)
print(f"   v0    = {v0:.6f}  (initial variance, annualized)")
print(f"   kappa = {kappa:.6f}  (mean reversion speed, 1/year)")
print(f"   theta = {theta:.6f}  (long-term variance, annualized)")
print(f"   sigma = {sigma:.6f}  (volatility of volatility)")
print(f"   rho   = {rho:.6f}  (correlation between returns and variance)")
print(f"   lambd = {lambd:.6f}  (market price of volatility risk)")
print()
print(f"   Implied initial volatility: {np.sqrt(v0):.2%}")
print(f"   Implied long-term volatility: {np.sqrt(theta):.2%}")
print("="*80 + "\n")

# Validate parameters
validation = validate_heston_parameters(v0, kappa, theta, sigma, rho, lambd)
print("PARAMETER VALIDATION")
print("="*80)
print(f"Feller Condition: {validation['feller_condition']}")
print(f"   Status: {'✓ SATISFIED' if validation['feller_satisfied'] else '✗ VIOLATED'}")
if validation['feller_satisfied']:
    print("   The variance process will remain positive with probability 1.")
else:
    print("   WARNING: Variance may reach zero, causing numerical instability.")

if validation['warnings']:
    print("\nWarnings:")
    for warning in validation['warnings']:
        print(f"   ⚠ {warning}")
print("="*80 + "\n")


# ============================================================================
# PRICING WITH CALIBRATED PARAMETERS
# ============================================================================

print("Calculating model prices with calibrated parameters...")
heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)
volSurfaceLong['heston_price'] = heston_prices
print("✓ Pricing complete\n")


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

print("="*80)
print("PERFORMANCE METRICS")
print("="*80)

# Calculate errors
errors = P - heston_prices
abs_errors = np.abs(errors)

# RMSE and related metrics
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(abs_errors)
mape = np.mean(abs_errors / P) * 100

print(f"\nPricing Error Metrics:")
print(f"   RMSE:         ${rmse:.4f}")
print(f"   MAE:          ${mae:.4f}")
print(f"   MAPE:         {mape:.2f}%")

# Relative RMSE
rmse_pct = (rmse / P.mean()) * 100
print(f"   Relative RMSE: {rmse_pct:.2f}% (of mean price ${P.mean():.2f})")

# R² score
ss_res = np.sum(errors**2)
ss_tot = np.sum((P - P.mean())**2)
r2 = 1 - (ss_res / ss_tot)
print(f"   R²:           {r2:.4f}")

# Error distribution
print(f"\nError Distribution:")
print(f"   Min:     ${abs_errors.min():.4f}")
print(f"   25%:     ${np.percentile(abs_errors, 25):.4f}")
print(f"   Median:  ${np.percentile(abs_errors, 50):.4f}")
print(f"   75%:     ${np.percentile(abs_errors, 75):.4f}")
print(f"   Max:     ${abs_errors.max():.4f}")

# Additional statistics
print(f"\nAdditional Statistics:")
print(f"   Mean error:          ${np.mean(errors):.4f} (should be near 0)")
print(f"   Std dev of errors:   ${np.std(errors):.4f}")
print(f"   Options within $1:    {np.sum(abs_errors < 1)/len(abs_errors)*100:.1f}%")
print(f"   Options within $2:    {np.sum(abs_errors < 2)/len(abs_errors)*100:.1f}%")
print(f"   Options within $5:    {np.sum(abs_errors < 5)/len(abs_errors)*100:.1f}%")

print("\n" + "="*80 + "\n")


# ============================================================================
# VISUALIZATION
# ============================================================================

print("Generating 3D visualization...")
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ============================================================================
# IMPLIED VOLATILITY CALCULATION
# ============================================================================

print("Calculating implied volatilities...")

def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate Black-Scholes call option price.
    
    Parameters
    ----------
    S : float or ndarray
        Current stock price
    K : float or ndarray
        Strike price
    T : float or ndarray
        Time to maturity
    r : float or ndarray
        Risk-free rate
    sigma : float or ndarray
        Volatility
        
    Returns
    -------
    float or ndarray
        Call option price
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_volatility(price, S, K, T, r, initial_guess=0.3, tol=1e-6, max_iter=100):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters
    ----------
    price : float or ndarray
        Observed option price
    S : float or ndarray
        Current stock price
    K : float or ndarray
        Strike price
    T : float or ndarray
        Time to maturity
    r : float or ndarray
        Risk-free rate
    initial_guess : float
        Initial volatility guess
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    float or ndarray
        Implied volatility (returns np.nan if fails to converge)
    """
    # Handle array inputs
    if isinstance(price, np.ndarray):
        # Ensure S is broadcast correctly if it's a scalar
        S_array = np.full_like(price, S) if np.isscalar(S) else S
        
        return np.array([implied_volatility(p, s, k, t, r_val, initial_guess, tol, max_iter) 
                        for p, s, k, t, r_val in zip(price, S_array, K, T, r)])
    
    # Check for valid inputs
    intrinsic = max(S - K, 0)
    if price <= intrinsic or price >= S:
        return np.nan
    
    sigma = initial_guess
    
    for _ in range(max_iter):
        bs_price = black_scholes_call(S, K, T, r, sigma)
        diff = bs_price - price
        
        if abs(diff) < tol:
            return sigma
        
        # Vega (derivative of BS price w.r.t. sigma)
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if vega < 1e-10:
            return np.nan
        
        # Newton-Raphson update
        sigma = sigma - diff / vega
        
        # Keep sigma positive and reasonable
        sigma = max(0.001, min(sigma, 5.0))
    
    return np.nan


# Calculate implied volatilities
print("   Computing market implied volatilities...")
market_iv = implied_volatility(P, S0, K, tau, r)

print("   Computing Heston implied volatilities...")
heston_iv = implied_volatility(heston_prices, S0, K, tau, r)

# Add to dataframe
volSurfaceLong['market_iv'] = market_iv
volSurfaceLong['heston_iv'] = heston_iv

# Calculate moneyness for analysis
volSurfaceLong['moneyness'] = volSurfaceLong['strike'] / S0

# Filter out failed conversions
valid_iv = ~np.isnan(market_iv) & ~np.isnan(heston_iv)
n_valid = np.sum(valid_iv)
n_failed = len(market_iv) - n_valid

print(f"   ✓ IV calculation complete")
print(f"   Valid conversions: {n_valid}/{len(market_iv)} ({n_valid/len(market_iv)*100:.1f}%)")
if n_failed > 0:
    print(f"   Failed conversions: {n_failed} (deep ITM/OTM options)")

# IV error metrics
iv_errors = market_iv[valid_iv] - heston_iv[valid_iv]
iv_rmse = np.sqrt(np.mean(iv_errors**2))
iv_mae = np.mean(np.abs(iv_errors))

print(f"\nImplied Volatility Metrics:")
print(f"   RMSE:     {iv_rmse:.4f} ({iv_rmse*100:.2f}%)")
print(f"   MAE:      {iv_mae:.4f} ({iv_mae*100:.2f}%)")
print(f"   Max error: {np.max(np.abs(iv_errors)):.4f}")

print("\n")

# ============================================================================
# VISUALIZATION - OPTION PRICES
# ============================================================================

print("Generating visualizations...")

fig = go.Figure(data=[
    go.Mesh3d(
        x=volSurfaceLong.maturity, 
        y=volSurfaceLong.strike, 
        z=volSurfaceLong.price, 
        color='mediumblue', 
        opacity=0.55,
        name='Market Prices'
    )
])

fig.add_scatter3d(
    x=volSurfaceLong.maturity, 
    y=volSurfaceLong.strike, 
    z=volSurfaceLong.heston_price, 
    mode='markers',
    marker=dict(size=3, color='red'),
    name='Heston Prices'
)

fig.update_layout(
    title_text=f'Option Prices: Market vs Heston | N={len(volSurfaceLong)} | RMSE=${rmse:.2f} ({rmse_pct:.1f}%) | R²={r2:.3f}',
    scene=dict(
        xaxis_title='Time to Maturity (Years)',
        yaxis_title='Strike Price ($)',
        zaxis_title='Call Option Price ($)'
    ),
    height=800,
    width=1000
)

fig.show()

# ============================================================================
# VISUALIZATION - IMPLIED VOLATILITY SURFACE
# ============================================================================

# Filter valid IVs for plotting
vol_surface_valid = volSurfaceLong[valid_iv].copy()

fig2 = go.Figure(data=[
    go.Mesh3d(
        x=vol_surface_valid.maturity, 
        y=vol_surface_valid.strike, 
        z=vol_surface_valid.market_iv * 100,  # Convert to percentage
        color='green', 
        opacity=0.55,
        name='Market IV'
    )
])

fig2.add_scatter3d(
    x=vol_surface_valid.maturity, 
    y=vol_surface_valid.strike, 
    z=vol_surface_valid.heston_iv * 100, 
    mode='markers',
    marker=dict(size=3, color='orange'),
    name='Heston IV'
)

fig2.update_layout(
    title_text=f'Implied Volatility Surface: Market vs Heston | RMSE={iv_rmse*100:.2f}%',
    scene=dict(
        xaxis_title='Time to Maturity (Years)',
        yaxis_title='Strike Price ($)',
        zaxis_title='Implied Volatility (%)'
    ),
    height=800,
    width=1000
)

fig2.show()

# ============================================================================
# VISUALIZATION - VOLATILITY SMILE BY MATURITY
# ============================================================================

# Select a few representative maturities
unique_maturities = sorted(vol_surface_valid.maturity.unique())
num_plots = min(6, len(unique_maturities))
maturity_indices = np.linspace(0, len(unique_maturities)-1, num_plots, dtype=int)
selected_maturities = [unique_maturities[i] for i in maturity_indices]

fig3 = make_subplots(
    rows=2, cols=3,
    subplot_titles=[f'T = {mat:.2f}y' for mat in selected_maturities],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

for idx, maturity in enumerate(selected_maturities):
    row = idx // 3 + 1
    col = idx % 3 + 1
    
    # Filter data for this maturity (with some tolerance)
    mat_data = vol_surface_valid[np.abs(vol_surface_valid.maturity - maturity) < 0.01].copy()
    mat_data = mat_data.sort_values('moneyness')
    
    if len(mat_data) > 0:
        # Market IV
        fig3.add_trace(
            go.Scatter(
                x=mat_data.moneyness,
                y=mat_data.market_iv * 100,
                mode='markers',
                name='Market',
                marker=dict(color='green', size=8),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Heston IV
        fig3.add_trace(
            go.Scatter(
                x=mat_data.moneyness,
                y=mat_data.heston_iv * 100,
                mode='lines+markers',
                name='Heston',
                line=dict(color='orange', width=2),
                marker=dict(size=5),
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Add ATM line
        fig3.add_vline(
            x=1.0, 
            line_dash="dash", 
            line_color="gray", 
            opacity=0.5,
            row=row, col=col
        )

# Update axes
for i in range(1, 7):
    row = (i-1) // 3 + 1
    col = (i-1) % 3 + 1
    fig3.update_xaxes(title_text="Moneyness (K/S)", row=row, col=col)
    fig3.update_yaxes(title_text="Implied Vol (%)", row=row, col=col)

fig3.update_layout(
    height=700,
    width=1400,
    title_text="Volatility Smile Across Maturities: Market vs Heston Model",
    showlegend=True
)

fig3.show()

print("✓ All visualizations complete\n")

print("\n" + "="*80)
print("CALIBRATION COMPLETE")

print("="*80)
