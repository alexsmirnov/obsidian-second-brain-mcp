**MetricGate** May 13, 2026
Run this analysis with our [Honest DiD (Rambachan-Roth) Calculator](/calculator/relaxed-parallel-trends-rambachan#src=doc-cta). 
# Honest DiD Inference under Relaxed Parallel Trends (Rambachan-Roth 2023)
## Overview
**Rambachan and Roth (2023)** deliver formally valid inference for [difference-in-differences](/docs/difference-in-differences/) and event-study estimates when the parallel-trends assumption fails by a bounded, restricted amount. Their `HonestDiD` package converts hypotheses about post-treatment trend deviations into confidence sets for the causal target. The procedure pairs naturally with an [event-study DiD](/docs/event-study-did/), the [Sun-Abraham estimator](/docs/event-study-sun-abraham/), or the [Callaway-Sant'Anna](/docs/did-staggered-callaway-santanna/) staggered DiD framework. 
staggered_panel  mtcars  iris  airquality  ToothGrowth  PlantGrowth  ChickWeight  CO2  sleep  faithful  women 
[ Use your own data → ](/calculator/relaxed-parallel-trends-rambachan#src=doc-embed-data)
Unit ID Variable*
Pick a variable  Unit  Time  FirstTreatTime  Treated  Outcome 
Time Variable*
Pick a variable  Unit  Time  FirstTreatTime  Treated  Outcome 
Outcome Variable*
Pick a variable  Unit  Time  FirstTreatTime  Treated  Outcome 
Treatment Onset*
Pick a variable  Unit  Time  FirstTreatTime  Treated  Outcome 
Significance Level (alpha)
Restriction Family
Relative Magnitudes (Mbar) 
Relative Magnitudes (Mbar)  Smoothness (M) 
Additional Outputs
Select All 
Event-Study Coefficients 
Sensitivity Table 
Breakdown Analysis 
Sensitivity Plot 
Related analyses
[Staggered DiD: Callaway-Sant'Anna Estimator](/docs/did-staggered-callaway-santanna/)[Synthetic Control (Abadie)](/docs/synthetic-control-abadie/)[Honest DiD Parallel-Trends Sensitivity](/docs/honest-did-sensitivity/)[Difference-in-Differences](/docs/difference-in-differences/)[Event Study DiD](/docs/event-study-did/)[Two-Way Fixed Effects Panel](/docs/fixed-effects-two-way-panel/)[Causal Mediation Sensitivity Analysis](/docs/causal-mediation-sensitivity/)
### Key Takeaways
  * **Honest inference:** Confidence sets remain valid even when parallel trends is violated within a researcher-specified amount.
  * **Two restriction families:** Smoothness restrictions bound the magnitude of second differences; relative-magnitude restrictions cap post-trend deviations by a multiple of pre-trend deviations.
  * **Sensitivity, not testing:** The output is a continuous sensitivity curve, not a binary reject/fail decision.


## When to Use
  * You have an event-study with concerning pre-trend coefficients.
  * You want to communicate the size of parallel-trends violations that would overturn your conclusion.
  * You face a referee asking "how robust is this to small parallel-trends violations?"
  * You need to combine the headline ATT with a defensible robustness analysis.
  * Your DiD design has at least three pre-treatment periods (more is better).


## Methodology
Let =(,)\widehat{\beta} = (\widehat{\beta}_{\mathrm{pre}}, \widehat{\beta}_{\mathrm{post}}) be the event-study coefficient vector and its covariance. Define the causal target as a linear functional \theta = \ell^\top \beta_{\mathrm{post}} (for example, the average post-treatment ATT). 
Parallel trends imposes where is the differential trend. Rambachan and Roth let live in a researcher-chosen set : 
(M)={δ:∣−2+∣≤M}\Delta^{\mathrm{SD}}(M) = \\{ \delta : |\delta_{t+1} - 2\delta_t + \delta_{t-1}| \leq M \\}
bounds the smoothness of the underlying trend, while: 
()={δ:∣∣≤⋅∣∣}\Delta^{\mathrm{RM}}(\bar M) = \\{ \delta : |\delta_{\mathrm{post},s}| \leq \bar M \cdot \max_{r < 0} |\delta_{\mathrm{pre},r}| \\}
caps post-period violations as a multiple of the worst pre-period violation. The honest confidence set is C={θ:rejection condition}CS_\alpha = \\{\theta : \inf_{\delta \in \Delta} \text{rejection condition}\\}rejection condition}; it has uniform coverage at level across the restriction class. `HonestDiD` implements this with fixed-length and conditional-FLCI procedures, returning the breakdown value of or at which the post-treatment effect becomes inconclusive. 
## Steps
  1. **Fit event study:** Estimate and via `fixest::feols` or a comparable estimator.
  2. **Define target:** Choose the linear functional \theta = \ell^\top \beta_{\mathrm{post}}.
  3. **Pick restriction:** Smoothness \Delta^{\mathrm{SD}}(M) or relative magnitude \Delta^{\mathrm{RM}}(\bar M).
  4. **Sweep over M values:** `createSensitivityResults*()` returns confidence sets at each .
  5. **Find breakdown:** Report the at which the CS first contains zero.
  6. **Plot sensitivity:** `createSensitivityPlot*()` visualises the trade-off.


## Key Concepts
  * **Honest CS:** A confidence set with valid coverage uniformly over the restriction class.
  * **Breakdown value:** Smallest restriction parameter at which the headline conclusion overturns.
  * **Smoothness restriction:** Caps how curved the underlying trend can be; rules out sharp post-period accelerations.
  * **Relative-magnitude restriction:** Caps post-period trend deviations by a multiple of pre-period deviations.
  * **FLCI:** Fixed-Length [Confidence Interval](/docs/confidence-interval/); a specific honest CS construction.


## R Code Example
# Rambachan & Roth (2023) honest DiD inference # Field-standard: HonestDiD package library(HonestDiD) library(fixest) # Fit event study first set.seed(1) n <- 40; T <- 7 df <- expand.grid(id = 1:n, t = 1:T) df$g <- rep(sample(c(Inf, 5, 6), n, replace = TRUE), each = T)[seq_len(nrow(df))] df$treat <- as.integer(is.finite(df$g) & df$t >= df$g) df$y <- 0.1*df$t + rnorm(nrow(df)) + 2*df$treat fit <- feols(y ~ i(t, as.integer(is.finite(g)), ref = 4) | id + t, data = df, cluster = ~id) beta <- coef(
# Rambachan & Roth (2023) honest DiD inference
# Field-standard: HonestDiD package
# Fit event study first
df <- expand.grid(id = 1:n, t = 1:T)
df$g <- rep(sample(c(Inf,5,6),, replace = TRUE), each = T)[seq_len(nrow(df))]
df$treat <- as.integer(is.finite(df$g) & df$t >= df$g)
df$y <- 0.1*df$t + rnorm(nrow(df))2*df$treat
fit <- feols(y ~ i(t, as.integer(is.finite(g)), ref = 4) | id + t, data = df, cluster = ~id)
# HonestDiD calls isSymmetric() on sigma, which has no method for fixest_vcov;
# unclass to a plain matrix.
V <- as.matrix(vcov(fit))
sens <- createSensitivityResults_relativeMagnitudes(
betahat = beta, sigma = V,
numPrePeriods = 3, numPostPeriods = length(beta)3,
Mbarvec = c(0,0.5,1), gridPoints = 30
createSensitivityPlot_relativeMagnitudes(sens, originalResults = NULL)
R
## Interpreting Results
  * **Original CI vs honest CI:** Honest CIs are wider; the gap quantifies the price of robustness.
  * **Breakdown M:** Larger breakdown M means greater robustness; report this number in the headline.
  * **Sensitivity plot:** Read off the M at which CIs start to include zero.
  * **Smoothness vs relative magnitude:** Smoothness is interpretable in outcome units; relative magnitude is unit-free.
  * **Choice of M:** Pre-register or report a range; M=1 (relative magnitudes) is a standard benchmark.


## Limitations
  * Requires multiple pre-treatment periods; with one pre-period the restrictions are uninformative.
  * Assumes the chosen restriction class plausibly contains the true trend.
  * Honest CIs can be very wide when M is large, sometimes uninformative.
  * Software currently supports balanced designs better than highly unbalanced staggered designs.
  * Reporting requires judgement about which M to highlight.


## Alternative Methods
  * Visual diagnosis of pre-trends but no formal correction.
  * Fixes heterogeneity bias but still assumes parallel trends.
  * Standard staggered DiD; complementary, not a substitute.
  * Different identification strategy; useful when parallel trends is implausible.


## How to Run Honest DiD (Rambachan-Roth) in MetricGate
MetricGate runs the full honest-DiD sensitivity analysis in your browser — it fits the event study with `fixest`, sweeps the trend-violation bound, and reports the breakdown value — computed in [R](/docs/r-script-editor/). No install required.
### Step 1 — Open the calculator
Open the [Honest DiD (Rambachan-Roth) Calculator](/calculator/relaxed-parallel-trends-rambachan#src=doc-step). The input panel lets you set:
  * **Dataset** — the panel (long format: one row per unit-time)
  * **Unit ID Variable** — the panel identifier (units that adopt treatment at different times are fine)
  * **Time Variable** — the calendar period
  * **Outcome Variable** — the numeric response
  * **Treatment Onset (first-treatment time) Variable** — the period each unit is first treated (0 or a large value marks never-treated)
  * **Significance Level (alpha)** — CI coverage, default `0.05`
  * **Restriction Family** — `Relative Magnitudes (Mbar)` or `Smoothness (M)`
  * **Additional Outputs** — Event-Study Coefficients, Sensitivity Table, Breakdown Analysis, Sensitivity Plot


### Step 2 — Load your data and run it
Load your data, map the columns, and run:
  * Import a CSV/Excel file with **Import data** , or click **Load Example** for the built-in `staggered_panel` dataset
  * Set **Unit ID Variable** → `Unit`, **Time Variable** → `Time`
  * Set **Outcome Variable** → `Outcome`, and **Treatment Onset (first-treatment time) Variable** → the column giving each unit's first treated period (a treated 0/1 flag is no longer what the panel asks for)
  * Set **Treatment Onset** → `FirstTreatTime`, keep **Restriction Family** = `Relative Magnitudes (Mbar)`
  * Click **Run** — or use the embedded calculator above


### Step 3 — Read the output
For the example, MetricGate returns:
  * **Event-study estimation:** 17 event-time bins (8 pre, 9 post), fit with `fixest::feols` and unit-clustered standard errors.
  * **Target coefficient (event-time 0):** TWFE point estimate **1.4232** (SE 0.2310), original 95% CI **[0.9704, 1.8760]**.
  * **Sensitivity table (Relative Magnitudes):** at `Mbar = 1.000` the honest CI is **[0.2190, 2.6273]** (max bias 0.7514) — still excludes zero, so the effect survives a violation as large as the worst pre-trend.
  * **Breakdown analysis:** pre-period max |coef| = 0.7514; **Breakdown Mbar* = 1.2915** — the honest CI first includes zero only once the post-trend bound exceeds 1.29× the largest pre-trend deviation.


Switch the restriction to [Smoothness (M)](/docs/honest-did-sensitivity/) with no re-entry to bound the trend in outcome units instead.
### How your data should look
MetricGate expects **long (panel) format** : one row per unit-time, with an outcome, a treated flag, and the first-treatment time for each unit. For the example:  
| Unit  | Time  | Outcome  | Treated  | FirstTreatTime  |  
| --- | --- | --- | --- | --- |  
Each unit's `FirstTreatTime` repeats across its rows; a value of `0` (or a very large value) marks a never-treated unit. Event-time is derived internally as `Time − FirstTreatTime`.
### How to report Honest DiD (Rambachan-Roth) (APA style)
A complete write-up reports:
  * The event-study point estimate and original CI
  * The restriction family (Relative Magnitudes or Smoothness) and the benchmark bound
  * The honest CI at that bound
  * The breakdown value (Mbar* or M*) at which the conclusion overturns


For the example:
> The first post-treatment effect was estimated at 1.42 (95% CI [0.97, 1.86]). Under the Rambachan-Roth (2023) relative-magnitudes restriction, the effect remained significant at Mbar = 1 (honest 95% CI [0.22, 2.63]); the breakdown value was Mbar* = 1.29, so the post-treatment trend would have to deviate by more than 1.29 times the largest pre-treatment deviation to overturn the conclusion.
MetricGate prints every one of these values, so you can copy them into your write-up.
## Conclusion
The Rambachan-Roth honest DiD framework lets researchers report event-study results with confidence sets that remain valid under bounded parallel-trends violations. The `HonestDiD` R package implements the sensitivity sweeps and breakdown analyses that have become standard in modern applied econometrics. Run a [sensitivity analysis](/docs/sensitivity-analysis/) on your own event study with our [Honest DiD Calculator](/calculator/relaxed-parallel-trends-rambachan#src=doc-step). 
## Cite this tool
Using this calculator in research or coursework? Cite it like this:
APAVancouverBibTeXRISMLA

```
MetricGate. (2026). Rambachan-Roth (2023) Honest DiD Inference Explained Calculator [Web application]. https://metricgate.com/docs/relaxed-parallel-trends-rambachan/
```
Copy 
## Frequently Asked Questions
What is Rambachan-Roth (2023) Honest DiD inference?
Honest DiD inference is a sensitivity-analysis method from Rambachan and Roth (2023) that produces valid confidence sets for difference-in-differences and event-study estimates when the parallel-trends assumption is violated by a bounded amount. Instead of assuming trends are exactly parallel, it lets the differential trend live in a researcher-specified set and reports how large a violation your conclusion can survive. It is implemented in the `HonestDiD` R package.
When should I use Honest DiD sensitivity analysis?
Use it when your event study shows concerning pre-trend coefficients, or when a referee asks how robust your DiD result is to small parallel-trends violations. It is most informative when you have at least three pre-treatment periods (more is better); with a single pre-period the restrictions carry no information. It also helps you communicate the size of a violation that would overturn your headline effect.
How do I interpret the breakdown value (Mbar* or M*)?
The breakdown value is the smallest restriction parameter at which the honest confidence set first includes zero, meaning the effect becomes inconclusive. A larger breakdown value means greater robustness: for relative magnitudes, Mbar* = 1.29 means the post-treatment trend would have to deviate by more than 1.29 times the largest pre-trend deviation to overturn the conclusion. Report this number alongside your point estimate and original CI.
What are the assumptions and limitations of Honest DiD?
The method assumes the chosen restriction class (smoothness or relative magnitudes) plausibly contains the true differential trend, and it needs multiple pre-treatment periods to be informative. Honest confidence intervals are always wider than the original CI, and can become very wide or uninformative when the bound M is large. Software currently handles balanced designs better than highly unbalanced staggered designs, and you must exercise judgement about which M to highlight.
What is the difference between smoothness and relative-magnitude restrictions?
Both bound the differential trend but in different ways. A smoothness restriction caps the magnitude of second differences, limiting how curved the underlying trend can be, and is interpretable in outcome units. A relative-magnitude restriction caps each post-period deviation at a multiple (Mbar) of the worst pre-period deviation, making it unit-free; Mbar = 1 is a common benchmark. You can switch between the two families to see whether your conclusion is robust under each.
How do I run Honest DiD (Rambachan-Roth) in MetricGate?
Open the Honest DiD calculator, load a long-format panel (or click Load Example), and map the Unit ID, Time, Outcome, Treatment Indicator, and Treatment Onset columns. Pick a restriction family (Relative Magnitudes or Smoothness), set alpha, and click Run. MetricGate fits the event study with `fixest`, sweeps the trend-violation bound, and reports the sensitivity table and breakdown value, so you get the full honest-inference workflow without hand-coding the HonestDiD package.
## Ready to run this analysis?
Stop struggling with complex R code or clunky software. Launch our blazing-fast, AI-powered calculator directly in your browser. 
