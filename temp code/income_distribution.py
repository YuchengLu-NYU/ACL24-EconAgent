#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, probplot, pareto
from scipy.optimize import minimize
#%%
# 1. Load your cleaned data
df = pd.read_csv("acs_national_processed.csv")  # must have HINCP_adj & WGTP

# 2. Extract incomes and weights
x = df["HINCP_adj"].values
w = df["WGTP"].values

# 3. Define negative log-likelihood function for weighted MLE
def neg_log_likelihood(params, x, w):
    mu, sigma = params
    # Convert to log-normal parameters
    scale = np.exp(mu)
    shape = sigma
    # Calculate weighted log-likelihood
    log_pdf = lognorm.logpdf(x, shape, loc=0, scale=scale)
    return -np.sum(w * log_pdf)

# 4. Initial guess and bounds
initial_guess = [np.mean(np.log(x)), np.std(np.log(x))]
bounds = [(None, None), (0.001, None)]  # sigma must be positive

# 5. Perform weighted MLE
result = minimize(neg_log_likelihood, initial_guess, args=(x, w), bounds=bounds)
mu, sigma = result.x
scale = np.exp(mu)
print(f"Estimated log-mean μ = {mu:.4f},  log-sd σ = {sigma:.4f}")

#%%
# 6. Quick diagnostics

# 6a. Histogram + fitted PDF overlay (on linear axes)
linear_bins = np.linspace(x.min(), x.max(), 100)
counts, bins = np.histogram(x, bins=linear_bins, weights=w, density=True)
centers = (bins[:-1] + bins[1:]) / 2
pdf_fit = lognorm.pdf(centers, sigma, loc=0, scale=scale)

plt.figure()
plt.bar(centers, counts, width=np.diff(bins), alpha=0.4, label="Empirical", align='center', edgecolor='none')
plt.plot(centers, pdf_fit, linewidth=2, label="Fitted log-normal")
plt.xlabel("Household income (USD)")
plt.ylabel("Density")
plt.title("Income Distribution: Linear Scale")
plt.legend()
plt.tight_layout()
plt.show()

# 6b. QQ-plot of log-incomes
plt.figure()
probplot(np.log(x), dist="norm", sparams=(mu, sigma), plot=plt)
plt.title("QQ-plot: log(income) vs. Normal")
plt.xlabel("Theoretical quantiles")
plt.ylabel("Sample quantiles")
plt.tight_layout()
plt.show()

# %%
# Plot raw income distribution histogram
plt.figure()
plt.hist(x, bins=100, weights=w, density=True, alpha=0.7)
plt.xlabel("Household income (USD)")
plt.ylabel("Density")
plt.title("Raw Income Distribution")
plt.tight_layout()
plt.show()

# %%
# Filter out zero incomes and replot
x_no_zeros = x[x > 0]
w_no_zeros = w[x > 0]

plt.figure()
plt.hist(x_no_zeros, bins=100, weights=w_no_zeros, density=True, alpha=0.7)
plt.xlabel("Household income (USD)")
plt.ylabel("Density") 
plt.title("Income Distribution (Excluding Zero Incomes)")
plt.tight_layout()
plt.show()

# %%
# Create a table of summary statistics
summary_stats = {
    'Statistic': [
        'Mean Income',
        'Median Income', 
        'Standard Deviation',
        'Minimum',
        'Maximum',
        'Sample Size'
    ],
    'Value': [
        f"${x_no_zeros.mean():,.2f}",
        f"${np.median(x_no_zeros):,.2f}",
        f"${x_no_zeros.std():,.2f}", 
        f"${x_no_zeros.min():,.2f}",
        f"${x_no_zeros.max():,.2f}",
        f"{len(x_no_zeros):,}"
    ]
}

# Display as formatted table
print("\nIncome Distribution Summary Statistics")
print("-" * 50)
for stat, val in zip(summary_stats['Statistic'], summary_stats['Value']):
    print(f"{stat:<20} {val:>25}")

# %%
# Fit Pareto distribution to income data (excluding zeros)

# Fit only to positive incomes
x_pareto = x_no_zeros
w_pareto = w_no_zeros

xm=0

# Only use data above the threshold for fitting
x_pareto_tail = x_pareto[x_pareto > xm]
w_pareto_tail = w_pareto[x_pareto > xm]

# Fit only the shape parameter, fixing loc=0 and scale=xm
pareto_shape, pareto_loc, pareto_scale = pareto.fit(x_pareto_tail, floc=0, fscale=xm)
print(f"Pareto fit (xm=0): shape (alpha) = {pareto_shape:.4f}, scale (xm) = {pareto_scale:.2f}")

# Plot histogram and fitted Pareto PDF (linear scale)
linear_bins = np.linspace(x_pareto.min(), x_pareto.max(), 100)
counts, bins = np.histogram(x_pareto, bins=linear_bins, weights=w_pareto, density=True)
centers = (bins[:-1] + bins[1:]) / 2
pdf_pareto = pareto.pdf(centers, pareto_shape, loc=pareto_loc, scale=pareto_scale)

plt.figure()
plt.bar(centers, counts, width=np.diff(bins), alpha=0.4, label="Empirical", align='center', edgecolor='none')
plt.plot(centers, pdf_pareto, linewidth=2, label="Fitted Pareto (xm=0)")
plt.xlabel("Household income (USD)")
plt.ylabel("Density")
plt.title("Pareto Fit to Income Distribution (Linear Scale, xm=0)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
