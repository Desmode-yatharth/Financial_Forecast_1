SPY Forecasting: Direction Classification & Volatility Regression
Overview

This project builds a leakage-safe forecasting pipeline on SPY (S&P 500 ETF) daily price data, and uses it to test two distinct hypotheses:

Can next-day price direction (up/down) be predicted from the last 9 days of lagged returns?
Can next-day volatility magnitude (|return| or return²) be predicted from the same 9-day lagged window?

Both questions are tested with an escalating set of models — a naive baseline, a linear model, and an LSTM — so that any deep-learning result is judged against how much a much simpler model already explains, not in isolation.

Objective

The original goal was not "hit a target accuracy number." It was to build a pipeline rigorous enough that its results — positive or negative — are actually trustworthy: correct train/val/test separation with no temporal leakage, a proven-stationary input representation, and an honest baseline comparison for every model, so a result like "54% accuracy" can be read correctly as "no better than guessing the majority class" rather than presented as a win.

Methodology
Data
Source: yfinance, SPY, full history (period='max'), OHLCV columns.
Chronological train/val/test split throughout — never shuffled, since shuffling a time series before splitting leaks near-identical, overlapping windows across the split boundary.
Stationarity proof (why returns, not raw price)

Raw price is a near-unit-root process: its mean and variance both grow with time (Var(P_t) ≈ t·σ²), which makes it a poor regression input — models end up fitting the shared trend rather than genuine structure. This was proven two ways, not assumed:

Numerically: a hand-written mean_var() function computing mean and variance of Close vs. Return across multiple non-overlapping date windows — Close's mean climbs steadily window to window; Return's mean and variance stay roughly flat.
Visually: a two-panel plot (Close vs. Return over the full 1993–2026 history) showing the same pattern directly.

Return = (Close[t] - Close[t-1]) / Close[t-1] is used as the base quantity for everything downstream.

Direction classification pipeline
Windowing: stride-1 sliding window, 9 lagged returns → binary label (1 if Return[t] > 0, else 0).
Baseline: majority-class rate, measured directly from the data (54.1% up / 45.9% down) — not assumed.
Models: logistic regression (penalty=None, to rule out regularization as a confound), then an LSTM.
Volatility regression pipeline
Targets: |Return| and Return², both standard realized-volatility proxies (Var(r) ≈ E[r²] when E[r] ≈ 0, which holds here).
Baseline: persistence (v̂_t = v_{t-1}) — a much harder baseline to beat than direction's majority class, since volatility is genuinely autocorrelated (confirmed via .autocorr() — all 9 lags positively correlated, 0.25–0.35).
Models: persistence → linear regression → GARCH(1,1) with Student's-t errors (fit via the arch package, chosen over a Normal-error specification because the return series shows real fat-tail behavior) → LSTM.
Results Summary
Task	Model	Metric	Result	vs. Baseline
Direction	Majority-class baseline	Accuracy	54.1%	—
Direction	Logistic Regression	Accuracy / AUC	~53.4% / 0.51	No improvement
Direction	LSTM	Accuracy / AUC	~54% / 0.51	No improvement
Volatility (ABS(r))	Persistence	RMSE
Volatility (ABS(r))		Linear Regression	RMSE / R²
Volatility (ABS(r))	LSTM	RMSE / R²
Volatility (r²)	Persistence	RMSE	0.000786	—
Volatility (r²)	Linear Regression	RMSE / R²	0.000631 / 0.330	Beats persistence
Volatility (r²)	LSTM (log1p target)	RMSE / R²	0.000687 / 0.206	Beats persistence, below linear regression

GARCH(1,1) rolling one-step-ahead comparison for |r|, and the fully consolidated cross-model plot, are still pending — see Limitations.

Key Findings
No exploitable linear or nonlinear signal was found for next-day direction. Both models converge to the majority-class baseline with ROC-AUC at chance (~0.51). This was diagnosed, not just observed: fitting with penalty=None ruled out regularization as the cause; a correlation matrix and a hand-computed gradient trace confirmed the model's large, unstable coefficients were a multicollinearity artifact, netting to near zero on real input rows. This result is consistent with efficient-markets theory.
Volatility magnitude is genuinely predictable from its own recent history — a real, replicated finding, confirmed independently via autocorrelation analysis before any model was fit. Linear regression explains ~32–33% of validation variance on both |r| and r².
The LSTM did not outperform linear regression on either volatility target, despite its added nonlinear capacity — a controlled, repeated result suggesting the exploitable structure here is predominantly linear, and the LSTM's extra flexibility is being spent fitting training noise rather than finding real additional structure.
Residual analysis of the volatility LSTM showed its largest errors concentrated almost entirely in March 2020 (the COVID crash) — a specific, dated finding, not a vague "sometimes struggles" caveat. This points to a real, interpretable limitation: short lag windows can track ongoing volatility clustering but cannot anticipate a sudden regime shift.
