[Skip to main content](https://arxiv.org/html/2312.05985#infobox)
[ License: arXiv.org perpetual non-exclusive license ](https://info.arxiv.org/help/license/index.html#licenses-available)
arXiv:2312.05985v4 [econ.EM] 07 Apr 2025
# Fused Extended Two-Way Fixed Effects for Difference-in-Differences With Staggered Adoptions
Gregory Faletto ††thanks: gregory.faletto@marshall.usc.edu Affiliation: University of Southern California 
August 24, 2026
###### Abstract
To address the bias of the canonical two-way fixed effects estimator for difference-in-differences under staggered adoptions, proposed the extended two-way fixed effects estimator, which adds many parameters. However, this reduces efficiency. Restricting some of these parameters to be equal (for example, subsequent treatment effects within a cohort) helps, but ad hoc restrictions may reintroduce bias. We propose a machine learning estimator with a single tuning parameter, fused extended two-way fixed effects (FETWFE), that enables automatic data-driven selection of these restrictions. We prove that under an appropriate sparsity assumption FETWFE identifies the correct restrictions with probability tending to one, which improves efficiency. We also prove the consistency, oracle property, and asymptotic normality of FETWFE for several classes of heterogeneous marginal treatment effect estimators under either conditional or marginal parallel trends, and we prove the same results for conditional average treatment effects under conditional parallel trends. We provide an R package implementing fused extended two-way fixed effects, and we demonstrate FETWFE in simulation studies and an empirical application.
Keywords: Difference-in-differences, staggered adoptions, treatment effect heterogeneity, bridge regression, fused lasso.
## Introduction
We investigate difference-in-differences estimation under staggered adoptions. Suppose units are observed at times, where our theoretical results will assume grows asymptotically while is fixed. For convenience we assume a balanced panel where all units are observed at all times, so we have observations.
Suppose R\leq T-1 cohorts start receiving a treatment at times r∈ℛ=⊆r\in\mathcal{R}=\\{r_{1},r_{2},\ldots,r_{R}\\}\subseteq\\{2,\ldots,T\\}. For example, we might want to investigate the effect of a law or policy that is implemented state-by-state across time ([Callaway and Sant’Anna, 2021](https://arxiv.org/html/2312.05985#bib.bib29); [Goodman-Bacon, 2021](https://arxiv.org/html/2312.05985#bib.bib60); [Borusyak et al., 2024](https://arxiv.org/html/2312.05985#bib.bib17)), or the effect of a product or service that is released across different regions over time ([De Chaisemartin and d’Haultfoeuille, 2020](https://arxiv.org/html/2312.05985#bib.bib36)). We consider settings where the treatment is an absorbing state, so that once a unit starts being treated, they continue receiving treatment thereafter, but this can be relaxed with little complication. Since no cohort enters at time and we assume the probability of never selecting into treatment is strictly positive for all units (see Assumption F2 in Section ), for sufficiently large we have untreated units available as a baseline comparison for all treated cohorts at all times.
For i\in\\{1,\ldots,N\\}=:[N] and , we encode the treatment status for each unit in a random variable W_{i}\in\\{0\\}\cup\mathcal{R}, where if unit is in cohort r\in\mathcal{R} or if unit is never treated. Let \tilde{y}_{(it)}(r) be the potential outcome for unit at time if they were in cohort . We will be interested in estimating treatment effects of the form  
| :=𝔼⁡[−∣],,,\displaystyle\tau_{\text{ATT}}(r,t):=\mathbb{E}\left[\tilde{y}_{(it)}(r)-\tilde{y}_{(it)}(0)\mid W_{i}=r\right],\qquad r\in\mathcal{R},t\geq r,  |  
| --- |  
as well as conditional treatment effects that depend on the value of unit covariates; see (), to come. It is well-known that in the , \mathcal{R}=\\{2\\} case, under no anticipation and parallel trends assumptions  
| =𝔼⁡[−∣]−𝔼⁡[−∣],\tau_{\text{ATT}}(2,2)=\mathbb{E}\left[\tilde{y}_{i2}(2)-\tilde{y}_{i1}(2)\mid W_{i}=2\right]-\mathbb{E}\left[\tilde{y}_{i2}(0)-\tilde{y}_{i1}(0)\mid W_{i}=0\right],  |  
| --- |  
and this treatment effect can be estimated consistently by the canonical two-way fixed effects regression  
| =++⋅𝟙𝟙+,\tilde{y}_{it}=\alpha_{i}^{*}+\gamma_{t}^{*}+\tau^{*}\cdot\sum_{r\in\mathcal{R}}\mathbbm{1}\\{W_{i}=r\\}\mathbbm{1}\\{t\geq r\\}+\epsilon_{it},  |  
| --- |  
where \alpha_{i}^{*} are fixed effects corresponding to units, \gamma_{t}^{*} are fixed effects corresponding to time, \mathbbm{1}\\{\cdot\\} is an indicator variable that equals 1 on the event and 0 otherwise, \epsilon_{it} is a noise term, and for conciseness we denote :=\tilde{y}_{it}:=\tilde{y}_{it}(W_{i}), the observed response for unit at time .
In the more general and arbitrary setting, one might hope that regression () would aggregate the treatment effects in a sensible way to estimate some form of an average treatment effect, but it is now well-known that this isn’t the case, and many alternative estimators have been proposed ([Borusyak and Jaravel, 2018](https://arxiv.org/html/2312.05985#bib.bib16); [De Chaisemartin and d’Haultfoeuille, 2020](https://arxiv.org/html/2312.05985#bib.bib36); [Sun and Abraham, 2021](https://arxiv.org/html/2312.05985#bib.bib111); [Goodman-Bacon, 2021](https://arxiv.org/html/2312.05985#bib.bib60); [Callaway and Sant’Anna, 2021](https://arxiv.org/html/2312.05985#bib.bib29)).
### 1.1 The Extended Two-Way Fixed Effects Estimator
Most of the alternative estimators proposed in these and other works depart from ordinary least squares (OLS) estimation, but 11 1 Along with . shows that linear regression can still be unbiased provided that enough parameters are estimated. He first shows that () is also the estimand for in the regression  
| =+⋅𝟙+⋅𝟙+⋅𝟙𝟙+,\tilde{y}_{it}=\eta^{*}+\nu_{2}^{*}\cdot\mathbbm{1}\\{W_{i}=2\\}+\gamma_{2}^{*}\cdot\mathbbm{1}\\{t=2\\}+\tau^{*}\cdot\mathbbm{1}\\{W_{i}=2\\}\mathbbm{1}\\{t=2\\}+\epsilon_{it},  |  
| --- |  
where can be interpreted as the expected outcome for the never-treated units at , is the selection bias for the treated units at , and is the trend in the untreated potential outcomes. proposes generalizing this to the arbitrary case and allowing for each parameter to be linear in covariates rather than constant. This leads to a linear model on cohort dummy variables, time dummies, covariates, treatment indicators, and interactions of those terms: for all , , and r\in\mathcal{R},  
| =+++𝟙(++𝟙(+))+.\tilde{y}_{it}=\eta^{*}+\gamma_{t}^{*}+\boldsymbol{X}_{i}^{\top}(\boldsymbol{\kappa}^{*}+\boldsymbol{\xi}_{t}^{*})+\sum_{r\in\mathcal{R}}\mathbbm{1}\\{W_{i}=r\\}\left(\nu_{r}^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\zeta}_{r}^{*}+\mathbbm{1}\\{t\geq r\\}(\tau_{rt}^{*}+\boldsymbol{\dot{X}}_{(ir)}^{\top}\boldsymbol{\rho}_{rt}^{*})\right)+\epsilon_{it}.  |  
| --- |  
Here \boldsymbol{X}_{i}\in\mathbb{R}^{d} is a set of time-invariant (pre-treatment) covariates, \eta^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\kappa}^{*} is the conditionally expected outcome of the never-treated units at , the \gamma_{t}^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\xi}_{t}^{*} terms are the conditional trends in the untreated potential outcomes, \nu_{r}^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\zeta}_{r}^{*} is the conditional selection bias for cohort , and +\tau_{rt}^{*}+\boldsymbol{\dot{X}}_{(ir)}^{\top}\boldsymbol{\rho}_{rt}^{*} is the conditional treatment effect for cohort at time . (See Assumption (LINS) in Section for more details.)
calls () the extended two-way fixed effects (ETWFE) estimator ([Wooldridge, 2021](https://arxiv.org/html/2312.05985#bib.bib125), Equation 6.33) and shows that in this regression \tau_{rt}^{*} is unbiased for \tau_{\text{ATT}}(r,t) under conditional no-anticipation and parallel trends assumptions; see Assumptions (CNAS) and (CCTS) in Section . The ETWFE estimator can also be used to estimate conditional average treatment effects on the treated units  
| :=𝔼[−∣=r,=𝒙],i∈,r∈ℛ,t∈.\tau_{\text{CATT}}(r,t,\boldsymbol{x}):=\mathbb{E}[\tilde{y}_{(it)}(r)-\tilde{y}_{(it)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}],\qquad i\in[N],r\in\mathcal{R},t\in\\{r,\ldots,T\\}.  |  
| --- |  
### 1.2 Restrictions in the ETWFE Model
As points out, the ETWFE estimator () may contain a large number of parameters to estimate if , , or is moderately large relative to . In this case, ordinary least squares estimation of () may lead to very imprecise estimates of at least some of the parameters, trading the bias problem that model () has for a low-efficiency (or high-variance) problem.
suggests some possible ways to mitigate this problem by imposing restrictions in order to reduce the number of parameters to be estimated. For example, one might assume that the treatment effects only vary across cohorts and not with time, so that \tau_{rt}=\tau_{r} for all r\in\mathcal{R}, . (Other works that consider similar restrictions include and .) If many of these restrictions hold, then imposing them results in more efficient estimation of the treatment effects than regression (). However, imposing these restrictions in an ad hoc way could reintroduce bias if the wrong restrictions are imposed, or leave unnecessary inefficiency if not enough restrictions are imposed.
The first goal of this paper is to propose a novel estimator that imposes these restrictions in an automatic, data-driven way. We propose the fused extended two-way fixed effects (FETWFE) estimator, a novel machine learning method for difference-in-differences estimation. Before we describe FETWFE, we discuss sparse fusion penalties in linear regression, which play a key role in our estimator.
### 1.3 Sparse Fusion Penalties
Given a real-valued design matrix 𝑿\boldsymbol{X} and a response \boldsymbol{y}\in\mathbb{R}^{N}, the well-known lasso estimator is defined by the optimization problem  
| \displaystyle\boldsymbol{\hat{\beta}}(\lambda_{N})\in\penalty\   | ​{+}\displaystyle\underset{\boldsymbol{\beta}\in\mathbb{R}^{p}}{\arg\min}\left\\{\sum_{i=1}^{N}\left(y_{i}-\boldsymbol{\beta}^{\top}\boldsymbol{X}_{i}\right)^{2}+\lambda_{N}\sum_{j=1}^{p}|\beta_{j}|\right\\}  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​{+},\displaystyle\underset{\boldsymbol{\beta}\in\mathbb{R}^{p}}{\arg\min}\left\\{\left\lVert\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\beta}\right\rVert_{2}^{2}+\lambda_{N}\lVert\boldsymbol{\beta}\rVert_{1}\right\\},  |  
where \lambda_{N} is a tuning parameter that can be chosen from a set of candidate values by cross-validation or a criterion like BIC, and recall that the norm22 2 It is a norm for finite , though we also consider q\in(0,1), in which case \lVert\cdot\rVert_{q} does not satisfy the triangle inequality. of an -vector \boldsymbol{v}=(v_{1},v_{2},\ldots,v_{n})^{\top} is defined as :=\lVert\boldsymbol{v}\rVert_{q}:=\left(\sum_{i=1}^{n}|v_{i}|^{q}\right)^{1/q} . The penalty term leads to sparse estimated coefficients \boldsymbol{\hat{\beta}}(\lambda_{N}) with entries set exactly equal to 0.
The fused lasso adds a second tuning parameter and penalty term:  
| ∈\displaystyle\boldsymbol{\hat{\beta}}(\lambda_{N}^{(1)},\lambda_{N}^{(2)})\in\penalty\   | ​{++​}.\displaystyle\underset{\boldsymbol{\beta}\in\mathbb{R}^{p}}{\arg\min}\left\\{\sum_{i=1}^{N}\left(y_{i}-\boldsymbol{\beta}^{\top}\boldsymbol{X}_{i}\right)^{2}+\lambda_{N}^{(1)}\sum_{j=1}^{p}|\beta_{j}|+\lambda_{N}^{(2)}\sum_{j=2}^{p}|\beta_{j}-\beta_{j-1}|\right\\}.  |  
| --- | --- |  
The fused lasso tends to yield solutions that both set some individual coefficients exactly equal to 0 (due to the first penalty term) and also set the differences between some adjacent pairs of coefficients exactly equal to 0 (due to the second penalty term). That is, some adjacent pairs of coefficients are set exactly equal to each other, or fused together.
FETWFE uses fusion penalties for q\in(0,2] (bridge regularization) to shrink certain sets of parameters in () towards each other in a way that aligns with intuition and previously proposed restrictions. See Figure for a visual depiction of which of the marginal average treatment effects \hat{\tau}_{rt} we propose penalizing towards each other, and see further details about the FETWFE penalty term () in Section . A lasso estimator is within our framework if one chooses . Choices of q\in(0,1] will result in sparse solutions, and choices of q\in(0,1) will allow for convergence at rates and asymptotic normality of the estimated coefficients. These fusion penalties allow the restrictions to be selected in a hands-off, automatic way that comes with theoretical guarantees.
Visualization of which of the estimated marginal average treatment effect terms \hat{\tau}_{rt} from regression () (which estimate the average treatment effects \tau_{\text{ATT}}(r,t) from Equation ) we penalize towards each other in the FETWFE penalty (). In this setting, and \mathcal{R}=\\{2,\ldots,5\\}. The horizontal axis depicts time and the vertical axis depicts cohorts. FETWFE works well under an assumption that the linked treatment effects tend to be close together, and at least some of them are exactly equal. See further details in Section .
### 1.4 Demonstrating the Efficacy of FETWFE
The second goal of this paper is to provide theoretical guarantees for the performance of FETWFE. We prove that under a suitable sparsity assumption on the true coefficients and regularity conditions, FETWFE learns the correct restrictions with probability tending to one (Theorem ). FETWFE then estimates only the needed parameters, improving efficiency. We also prove that for q\in(0,2] our proposed estimators of two classes of heterogeneous marginal average treatment effects are consistent under either conditional or marginal parallel trends, and two classes of conditional average treatment effect estimators are consistent under conditional parallel trends (Theorem ). To the best of our knowledge, this is the first work establishing the consistency of a difference-in-differences estimator for marginal average treatment effects under either form of parallel trends. For estimating propensity-weighted conditional average treatment effects, the generalized propensity scores can be estimated by an arbitrary nonparametric estimator with theoretical convergence guarantees, so our theory applies to propensity score estimators as simple as multinomial logit or as flexible as deep neural networks or random forests .
We also show that our treatment effect estimators are asymptotically normal for q\in(0,1) and we provide feasible, provably consistent finite-sample variance estimators (Theorems and ). For probability-weighted treatment effect estimates, we propose both an asymptotically normal test statistic if the generalized propensity scores are estimated on a separate data set from FETWFE (Theorem b) and an asymptotically subgaussian test statistic if both models are estimated on the same data set (Theorem c). The data-driven restrictions FETWFE learns improve parameter estimation while preserving asymptotic unbiasedness and consistency, allowing for the construction of asymptotically valid confidence intervals.
Under the assumptions of Theorem we show that FETWFE has an oracle property : the asymptotic variance of its estimates depend only on the parameters of the smaller restricted model, and it converges at the same rate as an OLS-estimated model on the true restricted model even if the number of covariates grows to infinity with . Our theoretical guarantees leverage extensions of the bridge regression theory of (Theorems and in the appendix) that may be of independent interest.
Finally, a third goal of this paper is to show demonstrate and facilitate the practical use of FETWFE. An open-source R package implementing FEWFE is available on the Comprehensive R Archive Network . We demonstrate that FETWFE is useful in practice through simulation studies that we conduct in Section comparing the FETWFE estimator to competitor estimators. We also illustrate FETWFE in an empirical application in Section .
### 1.5 Related Literature
There has been an abundance of recent research on difference-in-differences and it is not possible to cite all of the work in this stream; see [De Chaisemartin and d’Haultfoeuille (2023)](https://arxiv.org/html/2312.05985#bib.bib38); [Sun and Shapiro (2022)](https://arxiv.org/html/2312.05985#bib.bib112); [Roth et al. (2023)](https://arxiv.org/html/2312.05985#bib.bib104), and for recent reviews. reviews earlier literature. review the literature from a political science perspective. [de Chaisemartin and D’Haultfœuille (2023)](https://arxiv.org/html/2312.05985#bib.bib37) and review causal inference in panel data more broadly. Beyond those reviews and the papers we have already cited, throughout the paper we cite papers in the difference-in-differences literature and relate them to FETWFE; see Sections and and Appendix , in particular.
Besides , there are many other estimators that target similar estimands to ours, and we discuss several of these in Section , though none of these are machine learning estimators. It is also natural to compare FETWFE to other machine learning and non- and semi-parametric difference-in-differences estimators. Generally these methods are more flexible in the kinds of models than they can estimate than FETWFE (which is a regularized linear model, though our theoretical results allow for nonparametric estimation of the generalized propensity scores), but much less flexible in the estimands they target. Further, these methods tend to require assumptions that an estimator is available that achieves consistency at particular rates, so applying these results would involve further checking the required assumptions for a particular estimator. In contrast, we provide “end-to-end" assumptions for the consistency, asymptotic normality, and oracle property of three classes of estimators (), (), and ().
The parametric and semi-parametric estimators proposed by , , and yield theoretical results for a wide variety of machine learning estimators with theoretical guarantees under conditional parallel trends assumptions, but only in the setting and only for a single average treatment effect estimand (specifically, our estimand in Equation ). accomplish similar goals with different estimands under different assumptions, focusing mainly on a setting with and a single mediator of interest. propose related non-parametric estimators that achieve convergence and asymptotic normality in the repeated cross-sections setting only for an average treatment effect, and consistency for conditional average treatment effects at much slower rates, in contrast to the rate for conditional average treatment effects estimated by FETWFE (see Theorem ). provide similar semi-parametric estimators for both marginal and conditional average treatment effects in the staggered adoptions setting with arbitrary , but they do not provide any theoretical guarantees.
propose an approach using random forests that is somewhat more flexible in its estimands. Their approach allows for arbitrary , though they focus on the case with only one treated cohort that starts an absorbing treatment at time , providing a brief sketch as to how their method might be extended to allow for staggered adoptions. Further, their method requires splitting the data and estimating completely separate models to estimate the treatment effect of the treated cohort at each time 2,3,\ldots,T, while FETWFE allows for the full sample to be used to jointly estimate all treatment effects simultaneously. Indeed, a central idea of our method is that the fusion penalties we use allow FETWFE to borrow strength across cohorts and time in order to improve estimation efficiency. (None of the above works mentions fusion penalties.)
Our work also contributes to the much broader topic of estimating conditional average treatment effects. In Section we mention some prominent works in this stream of literature.
Outside of causal inference, our approach is clearly closely related to both the fused lasso and the generalized lasso ; see (). Our work also draws much from the bridge regression theory of , and, in turn, . propose a linear panel data model with a fused lasso-like penalty, though not for causal inference. propose a related Bayesian approach. also propose a penalized panel data model that combines categories, but not in the causal inference setting. Similarly, proposes a shrinkage estimator for linear panel data models (though not for causal inference) and cites more related work in this stream.
FETWFE also relates more specifically to the generalized fused lasso . Like FETWFE, the generalized fused lasso involves applying penalties to the absolute differences between a variety of pairs of coefficients, though the generalized fused lasso only uses the case. Some works that prove theoretical results for the generalized fused lasso include and . prove consistency for the generalized fused lasso under different assumptions from ours, including a restriction to the fixed case. They prove selection consistency and asymptotic normality only for an adaptive version of their method. prove similar results (with similar limitations) for a broader class of generalized linear models.
### 1.6 Organization of the Paper
After we introduce some notation, we establish our setup and basic assumptions in Section and present our difference-in-differences assumptions in Section . In particular, we provide a testable sufficient condition (CIUN) for several kinds of estimators, including FETWFE, to be consistent under either marginal or conditional parallel trends. In Section we introduce causal estimands that FETWFE can be used to estimate, and in Section we propose the fused extended two-way fixed effects model, as well as our estimators for our proposed causal estimands. We present our theoretical guarantees for FETWFE in Section . In Section we demonstrate the superior performance of FETWFE in our setting against extended two-way fixed effects (ETWFE), bridge-penalized ETWFE, and a slightly more flexible model than () in simulation studies. We illustrate the usefulness of FETWFE in an empirical application using data from in Section . Finally, Section concludes.
### 1.7 Notation
Throughout we will use subscripts like \tilde{y}_{(it)} to refer to the i^{\text{th}} unit at the t^{\text{th}} time, \boldsymbol{\tilde{y}}_{(i\cdot)}\in\mathbb{R}^{T} to refer to all response observations corresponding to unit , and so on. For matrices, \boldsymbol{Z}_{(it)\cdot} refers to the row corresponding to unit at time , \boldsymbol{Z}_{(i\cdot)j} refers to the observations of the j^{\text{th}} feature in unit , and so on. We will assume that unit-specific covariates are time invariant, and it will be more convenient to refer to the covariates for unit with the notation \boldsymbol{X}_{i}.
Recall that for a sequence of random variables and a sequence of positive real numbers , we say that X_{N}=\mathcal{O}_{\mathbb{P}}(a_{N}) if for every \epsilon>0 there exists a finite M_{\epsilon}>0 and N_{\epsilon}>0 such that for all N>N_{\epsilon},  
| ϵ.\mathbb{P}\left(\left|\frac{X_{N}}{a_{N}}\right|>M_{\epsilon}\right)<\epsilon.  |  
| --- |  
If a_{N}\to 0, X_{N}=\mathcal{O}_{\mathbb{P}}(a_{N}) is sufficient for X_{N}\xrightarrow{p}0, and can be interpreted as the rate of convergence ([DasGupta, 2008](https://arxiv.org/html/2312.05985#bib.bib35), Chapter 1).
## Basic Setup and Assumptions
We will expand on the setup from the introduction. For each , assume  
| =𝔼++,\boldsymbol{\tilde{y}}_{(i\cdot)}=\mathbb{E}[\boldsymbol{\tilde{y}}_{(i\cdot)}\mid W_{i},\boldsymbol{X}_{i}]+c_{i}\boldsymbol{1}_{T}+\boldsymbol{u}_{(i\cdot)},  |  
| --- |  
where c_{i}\in\mathbb{R} and \boldsymbol{u}_{(i\cdot)}\in\mathbb{R}^{T} are both random variables and \boldsymbol{1}_{T} is a -vector with 1 in every entry.
Assumption (F1): \mathbb{E}[c_{i}\mid W_{i},\boldsymbol{X}_{i}]=0, 𝔼[∣,,]=𝟎,\mathbb{E}[\boldsymbol{u}_{(i\cdot)}\mid c_{i},W_{i},\boldsymbol{X}_{i}]=\boldsymbol{0}, =\mathrm{Var}(c_{i}\mid W_{i},\boldsymbol{X}_{i})=\sigma_{c}^{2}, and Var(∣,,)=\mathrm{Var}(\boldsymbol{u}_{(i\cdot)}\mid c_{i},W_{i},\boldsymbol{X}_{i})=\sigma^{2}\boldsymbol{I}_{T} almost surely, where \sigma_{c}^{2} and are assumed to be known and finite and \boldsymbol{I}_{T}\in\mathbb{R}^{T\times T} is the identity matrix.
With Assumption (F1) we can use a generalized least squares transform to create a data set with conditionally uncorrelated rows. Let ϵ:=c𝟏+𝒖\boldsymbol{\epsilon}_{(i\cdot)}:=c_{i}\boldsymbol{1}_{T}+\boldsymbol{u}_{(i\cdot)}, and observe that for every almost surely Var=+=:𝛀∈\mathrm{Var}(\boldsymbol{\epsilon}_{(i\cdot)}\mid W_{i},\boldsymbol{X}_{i})=\sigma^{2}\boldsymbol{I}_{T}+\sigma_{c}^{2}\boldsymbol{1}_{T}\boldsymbol{1}_{T}^{\top}=:\boldsymbol{\Omega}\in\mathbb{R}^{T\times T}. Under (F1), for every almost surely 𝔼[σ∣,]=𝟎\mathbb{E}[\sigma\boldsymbol{\Omega}^{-1/2}\boldsymbol{\epsilon}_{(i\cdot)}\mid W_{i},\boldsymbol{X}_{i}]=\boldsymbol{0} and  
| Var(σ∣,)=Var=.\mathrm{Var}(\sigma\boldsymbol{\Omega}^{-1/2}\boldsymbol{\epsilon}_{(i\cdot)}\mid W_{i},\boldsymbol{X}_{i})=\sigma^{2}\boldsymbol{\Omega}^{-1/2}\mathrm{Var}(\boldsymbol{\epsilon}_{(i\cdot)}\mid W_{i},\boldsymbol{X}_{i})\boldsymbol{\Omega}^{-1/2}=\sigma^{2}\boldsymbol{I}_{T}.  |  
| --- |  
We can carry out this transformation for any known finite and symmetric positive definite 𝛀=Var\boldsymbol{\Omega}=\mathrm{Var}(\boldsymbol{\epsilon}_{(i\cdot)}\mid W_{i},\boldsymbol{X}_{i}). All of our results can be easily adapted to apply for any such 𝛀\boldsymbol{\Omega}, though we focus on the specified setting for concreteness.
Next we make an assumption on the joint distribution of the potential outcomes, treatment, and covariates. Let  
| :=,\pi_{r}(\boldsymbol{x}):=\mathbb{P}(W_{i}=r\mid\boldsymbol{X}_{i}=\boldsymbol{x}),\qquad r\in\\{0\\}\cup\mathcal{R}  |  
| --- |  
be the conditional probability of a unit’s treatment status, a propensity score.
Assumption (F2): The joint random variables (,,,)N(W_{i},\boldsymbol{X}_{i},c_{i},\boldsymbol{u}_{(i\cdot)})_{i=1}^{N} are independent and identically distributed (iid). Further, assume that for all r\in\\{0\\}\cup\mathcal{R}, almost surely 0<\pi_{r}(\boldsymbol{X}_{i})<1.
Notice that for each \boldsymbol{X}_{i} all of the potential outcomes \tilde{y}_{it}(r) are well-defined under Assumption (F2) since treatment probabilities are always strictly between 0 and 1. So far we have made no assumptions on the relationship between the treatment assignments and the potential outcomes for a given unit, but we do so in the next section.
## Difference-in-Differences Assumptions
We will use the generalizations of the no anticipation and common trends assumptions described in the introduction—Assumptions (CNAS) and (CCTS), respectively—that were proposed by , in addition to some variations and other assumptions.
Assumption (CNAS) (“conditional no-anticipation with staggered interventions"): For treatment cohorts r\in\mathcal{R}, almost surely  
| 𝔼[−∣=r,]=0∀i∈,tr.\mathbb{E}[\tilde{y}_{(it)}(r)-\tilde{y}_{(it)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]=0\qquad\forall i\in[N],t<r.  |  
| --- |  
This requires that before treatment, in conditional expectation each individual’s expected outcome is identical to what it would have been if the unit were never treated. Some works (two recent examples are and ) define the potential outcomes of the units at each time in terms of their treatment status at that time rather than in terms of their cohort membership. In our notation, this amounts to assuming that =\tilde{y}_{(it)}(r)=\tilde{y}_{(it)}(0) almost surely for . As points out, this stronger assumption is sufficient for (CNAS). Such a framework may not be troubling, but we only need the weaker assumption (CNAS). See further discussion in [Wooldridge (2021, Sections 5.3, 6.1, 6.3)](https://arxiv.org/html/2312.05985#bib.bib125).
### 3.1 Parallel Trends Assumptions
We will discuss and use a variety of parallel or common trends assumptions.
Assumption (CCTS) (“conditional common trends with staggered interventions"): For all r\in\mathcal{R}, i\in[N],t\in\\{2,\ldots,T\\}, almost surely  
| 𝔼[−∣=r,]=𝔼[−∣=0,].\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]=\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}].  |  
| --- |  
Assumption (CCTS) requires that each unit’s untreated potential outcomes \tilde{y}_{(it)}(0) follow the same trend over time in conditional expectation regardless of when or if they are treated. Assumption (CCTS) is used by , and also by [Gavrilova et al. (2023, Assumption 1)](https://arxiv.org/html/2312.05985#bib.bib57) in the \mathcal{R}=\\{2\\} case. Assumption (CCTS) has become widely adopted in the setting; early examples include [Heckman et al. (1997, Assumption A-5)](https://arxiv.org/html/2312.05985#bib.bib65), [Heckman et al. (1998, Equation 12)](https://arxiv.org/html/2312.05985#bib.bib66), and , and some more recent examples include [Sant’Anna and Zhao (2020, Assumption 2)](https://arxiv.org/html/2312.05985#bib.bib107), [Zimmert (2020, Assumption 6(iv))](https://arxiv.org/html/2312.05985#bib.bib130), , [Roth et al. (2023, Assumption 6)](https://arxiv.org/html/2312.05985#bib.bib104), and [Ghanem et al. (2024, Assumption PT-X)](https://arxiv.org/html/2312.05985#bib.bib59).
We also consider versions of (CCTS) that only apply before or after treatment begins for each cohort.
Assumption (CCTSB) (“conditional common trends before treatment"): () holds almost surely for all r\in\mathcal{R}\setminus\\{2\\}, , and t\in\\{2,\ldots,r-1\\}.
Assumption (CCTSA) (“conditional common trends after treatment"): () holds almost surely for all r\in\mathcal{R}, , and t\in\\{r,\ldots,T\\}.
Assumption (CCTSB) requires that (CCTS) holds only until the period before each unit is treated, while (CCTSA) requires (CCTS) to hold after treatment begins. Many previous works have distinguished between the parallel trends before treatment versus after treatment, noting that (CCTSB) can be directly tested under assumption (CNAS) while (CCTSA) cannot, since untreated potential outcomes are not observed for units after treatment, and noting that parallel trends holding before treatment is no guarantee of holding after treatment. See [Bilinski and Hatfield (2018)](https://arxiv.org/html/2312.05985#bib.bib13); [Kahn-Lang and Lang (2020)](https://arxiv.org/html/2312.05985#bib.bib78); [Dette (2020)](https://arxiv.org/html/2312.05985#bib.bib39); [Sun and Abraham (2021)](https://arxiv.org/html/2312.05985#bib.bib111); [Ban and Kedagni (2022)](https://arxiv.org/html/2312.05985#bib.bib10); [Callaway and Sant’Anna (2022)](https://arxiv.org/html/2312.05985#bib.bib28); [Roth (2022)](https://arxiv.org/html/2312.05985#bib.bib102), [Henderson and Sperlich (2022, Equation 22)](https://arxiv.org/html/2312.05985#bib.bib68), [Borusyak et al. (2024)](https://arxiv.org/html/2312.05985#bib.bib17); [Rambachan and Roth (2023)](https://arxiv.org/html/2312.05985#bib.bib101), and . See [Wooldridge (2021, Section 7.1, Equation 7.4)](https://arxiv.org/html/2312.05985#bib.bib125) for an example of how (CCTSB) can be tested under (CNAS).
We can contrast (CCTS) with an unconditional or marginal version, which calls Assumption (CTS).
Assumption (CTS) (“common trends with staggered interventions"): Almost surely for all r\in\mathcal{R}  
| 𝔼⁡[−∣]=𝔼⁡[−∣],,.\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r]=\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0],\qquad i\in[N],t\in\\{2,\ldots,T\\}.  |  
| --- |  
We will also refer to a version of (CTS) that holds only after treatment:
Assumption (CTSA) (“common trends with staggered interventions after treatment"): () holds almost surely for all , r\in\mathcal{R}, and t\in\\{r,\ldots,T\\}.
Because (CTS) does not account for covariates, (CCTS) is often thought of as more plausible than (CTS). We generally agree, though in Appendix we add some nuance by pointing out that (CTS) can hold when (CCTS) does not hold (Theorem ). In such a setting it is not possible to estimate conditional average treatment effects using regression (); we show this formally in Theorem (a) in the appendix. Further, it turns out that even marginal average treatment effect estimates from models like FETWFE that rely on (CCTS) will in general be inconsistent.
However, (CTS) holding is enough to estimate marginal average treatment effects consistently using many estimators, so it is reasonable to wonder whether regression () might be able to estimate treatment effects consistently under (CTS) and some additional assumptions. The following assumption will turn out to allow FETWFE to consistently estimate marginal average treatment effects if (CTS) holds, even if (CCTS) does not.
Assumption (CIUN) (“conditional independence of trends of the untreated potential outcomes for never-treated units"): Almost surely for all and any t\in\\{2,\ldots,T\\},  
| 𝔼[−∣=0]=𝔼[−∣=0,].\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0\right]=\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}\right].  |  
| --- |  
Assumption (CIUN) requires that the trends in the untreated potential outcomes are mean-independent of the covariates conditional on treatment status for units that are untreated. Since (CIUN) depends only on the observed potential outcomes, it can be tested, as we discuss in the next section.
### 3.2 Linearity Assumption (Assumption LINS)
We present an assumption that is slightly different than one introduced by on the functional forms of the conditional distributions of the untreated potential outcomes, the trends −\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0), and the difference-in-differences estimands.
Assumption (LINS): For each N\in\mathbb{N}, assume we observe covariates, where may be fixed or increasing in . There exist fixed parameters \eta^{*}\in\mathbb{R},\\{\nu_{r}^{*}\\}_{r\in\mathcal{R}}\in\mathbb{R}, ∈ℝ\\{\gamma_{t}^{*}\\}_{t\in\\{2,\ldots,T\\}}\in\mathbb{R}, and ∈ℝ\\{\tau_{rt}^{*}\\}_{r\in\mathcal{R},t\geq r}\in\mathbb{R}, along with fixed sequences of parameters ,∈,∈\\{\boldsymbol{\kappa}_{N}^{*}\\}_{N\in\mathbb{N}}\in\mathbb{R}^{d_{N}},\\{\boldsymbol{\zeta}_{rN}^{*}\\}_{r\in\mathcal{R},N\in\mathbb{N}}\in\mathbb{R}^{d_{N}},\\{\boldsymbol{\xi}_{tN}^{*}\\}_{t\in\\{2,\ldots,T\\},N\in\mathbb{N}}\in\mathbb{R}^{d_{N}}, and ∈\\{\boldsymbol{\rho}_{rtN}^{*}\\}_{r\in\mathcal{R},t\geq r,N\in\mathbb{N}}\in\mathbb{R}^{d_{N}} such that for each N\in\mathbb{N}, for all and any fixed \boldsymbol{x}\in\mathbb{R}^{d_{N}} in the support of the random vector \boldsymbol{X}_{i} it holds that  
| 𝔼[∣=0,=𝒙]=\displaystyle\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]=\penalty\   | \displaystyle\eta^{*}+\boldsymbol{x}^{\top}\boldsymbol{\kappa}_{N}^{*},  |  
| --- | --- |  
| 𝔼[∣=r,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]\qquad  |  
| −𝔼[∣=0,=𝒙]=\displaystyle-\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]=\penalty\   | ,\displaystyle\nu_{r}^{*}+\boldsymbol{x}^{\top}\boldsymbol{\zeta}_{rN}^{*}\qquad\forall r\in\mathcal{R},  |  
| 𝔼[−∣=0,=𝒙]=\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]=\penalty\   | ,,and\displaystyle\gamma_{t}^{*}+\boldsymbol{x}^{\top}\boldsymbol{\xi}_{tN}^{*},\qquad t\in\\{2,\ldots,T\\},\qquad\text{and}  |  
| 𝔼[−∣=r,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(r)-\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]\qquad  |  
| −𝔼[−∣=0,=𝒙]=\displaystyle-\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]=\penalty\   | +,r∈ℛ,,\displaystyle\tau_{rt}^{*}+\boldsymbol{\dot{x}}_{r}^{\top}\boldsymbol{\rho}_{rtN}^{*},\qquad r\in\mathcal{R},t\geq r,  |  
where  
| :=\boldsymbol{\dot{x}}_{r}:=\boldsymbol{x}-\mathbb{E}\left[\boldsymbol{X}_{i}\mid W_{i}=r\right]  |  
| --- |  
are covariate vectors that are centered with respect to their cohort means. (For conciseness, hereafter we omit from the subscripts of the parameters defined here that may vary with .)
The version of this assumption presented by uses a slightly different expression for () and requires to be fixed, since ETWFE is inconsistent in the high-dimensional setting we consider. Notice the similarity of the quantity on the left side of () to the canonical two-period difference-in-differences estimand (). shows that Assumption (LINS), along with Assumptions (CNAS) and (CCTS), make () correctly specified for the potential outcomes and the the left side of () equal to the conditional average treatment effects. See also our derivation () and Theorem (a) in the appendix.
###### .
We can gain insight on (CCTS) by re-arranging and using the conditional independence from to arrive at the following equivalent condition:  
| 𝔼[∣=r,]−𝔼[∣=0,]\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]-\mathbb{E}[\tilde{y}_{(it)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}]  |  
| --- |  
| =\displaystyle=\penalty\   | 𝔼[∣=r,]−𝔼[∣=0,],r∈ℛ,i∈,t∈.\displaystyle\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]-\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}],\qquad r\in\mathcal{R},i\in[N],t\in\\{2,\ldots,T\\}.  |  
(This is similar to a framing used by, for example, [Heckman et al. (1998, Equations 10 and 12)](https://arxiv.org/html/2312.05985#bib.bib66), [Lechner et al. (2011, Equation 3.2)](https://arxiv.org/html/2312.05985#bib.bib81), [Henderson and Sperlich (2022, Assumption 1)](https://arxiv.org/html/2312.05985#bib.bib68), [Ban and Kedagni (2022, Section 2.1)](https://arxiv.org/html/2312.05985#bib.bib10), and [Park and Tchetgen (2023, Section 2.2)](https://arxiv.org/html/2312.05985#bib.bib98) in the setting.) Under (CCTS) and (LINS), () quantifies the time-invariant difference in conditionally expected potential outcomes between units in each cohort and units in the never-treated cohort that appears on each side of (). The inclusion of the terms in () provides robustness against time-invariant confounding (see Proposition (a) in the appendix).
On the other hand, under unconfoundedness of the untreated potential outcomes (which we define formally and discuss in more detail in () in the appendix), () (and each side of Equation ) is identically 0. Although including these covariates in a regression does not harm unbiasedness under unconfoundedness, it reduces efficiency. Fortunately, we will see later that under the assumptions of Theorem , if the estimated model contains any coefficients that equal 0 exactly, FETWFE successfully screens out these covariates and achieves the same asymptotic efficiency as a model that omitted them. In this way FETWFE provides robustness without compromising asymptotic efficiency under unconfoundedness, avoiding the kind of trade-off between robustness and asymptotic efficiency that discusses.
Many previous works have also noted that difference-in-differences estimation allows for selection bias and can effectively control for some kinds of unobserved time-invariant confounding. explore this idea in depth, and also raise interesting issues surrounding the plausibility of an assumption that time-varying covariates do not affect the observed response and the implications when this assumption is violated. propose a matching estimator in a setting with time-varying covariates. For more in-depth investigations into the parallel trends assumption, see , , and . See also our further discussion in Appendix .
Under Assumption (CIUN) and (LINS), the covariates \boldsymbol{\xi}_{t}^{*} all equal 𝟎\boldsymbol{0}. So a test of this null hypothesis (for example, a Wald test) serves as a test for the null hypothesis that both (CIUN) and (LINS) hold.
## Causal Estimands
Next we will discuss several classes of causal estimands we hope to estimate using FETWFE. The building block of our estimands will be the conditional average treatment effects on the treated units \tau_{\text{CATT}}(r,t,\boldsymbol{x}) from (), which matches definition (6.23) in . Estimating \tau_{\text{CATT}}(r,t,\boldsymbol{x}) is useful if we are interested in identifying which units are most affected by the treatment at a specific time. Besides economics ([Florens et al., 2008](https://arxiv.org/html/2312.05985#bib.bib52); [Djebbari and Smith, 2008](https://arxiv.org/html/2312.05985#bib.bib42)), domains where conditional average treatment effect estimation have been found useful include personalized medicine ([Feinstein et al., 1972](https://arxiv.org/html/2312.05985#bib.bib50); [Lesko, 2007](https://arxiv.org/html/2312.05985#bib.bib83); [Frankovich et al., 2011](https://arxiv.org/html/2312.05985#bib.bib54); [Foster et al., 2011](https://arxiv.org/html/2312.05985#bib.bib53)), education ([Morgan, 2001](https://arxiv.org/html/2312.05985#bib.bib93); [Brand and Xie, 2010](https://arxiv.org/html/2312.05985#bib.bib19); [Murphy et al., 2016](https://arxiv.org/html/2312.05985#bib.bib94)), targeted marketing ([Bottou et al., 2013](https://arxiv.org/html/2312.05985#bib.bib18); [Athey et al., 2018](https://arxiv.org/html/2312.05985#bib.bib9); [Hitsch and Misra, 2018](https://arxiv.org/html/2312.05985#bib.bib69); [Ascarza, 2018](https://arxiv.org/html/2312.05985#bib.bib5); [Yang et al., 2020](https://arxiv.org/html/2312.05985#bib.bib128); [Ellickson et al., 2022](https://arxiv.org/html/2312.05985#bib.bib43)), psychology ([Bolger et al., 2019](https://arxiv.org/html/2312.05985#bib.bib15); [Winkelbeiner et al., 2019](https://arxiv.org/html/2312.05985#bib.bib124)), sociology , and political science ([Imai and Strauss, 2011](https://arxiv.org/html/2312.05985#bib.bib73); [Grimmer et al., 2017](https://arxiv.org/html/2312.05985#bib.bib61)).
Many methods for estimating conditional average (or heterogeneous33 Within the differences-in-differences literature, the descriptor “heterogeneous” has often been used in reference to treatment effects that vary across cohorts or across time; e.g., [De Chaisemartin and d’Haultfoeuille (2020)](https://arxiv.org/html/2312.05985#bib.bib36); [Sun and Abraham (2021)](https://arxiv.org/html/2312.05985#bib.bib111); [Wooldridge (2021)](https://arxiv.org/html/2312.05985#bib.bib125). Here we refer to heterogeneity in treatment effects across the distribution of \boldsymbol{X}_{i}, though our model also accommodates heterogeneity in treatment effects across cohorts and time. To avoid ambiguity, we will generally use the term “conditional average treatment effects.”) treatment effects have been developed within econometrics over the last decade; see [Abrevaya et al. (2015)](https://arxiv.org/html/2312.05985#bib.bib3); [Athey and Imbens (2016)](https://arxiv.org/html/2312.05985#bib.bib6); [Wager and Athey (2018)](https://arxiv.org/html/2312.05985#bib.bib121); [Nie and Wager (2021)](https://arxiv.org/html/2312.05985#bib.bib96); [Semenova and Chernozhukov (2021)](https://arxiv.org/html/2312.05985#bib.bib108), and for a few examples.
### 4.1 Conditional Average Treatment Effect Estimands
We can use () to construct a variety of causal estimands. For any set of finite constants \\{\psi_{rt}\\}_{r\in\mathcal{R},t\in\\{r,\ldots,T\\}}, consider the estimand  
| ​​,\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\tau_{\text{CATT}}(r,t,\boldsymbol{x}),  |  
| --- |  
a weighted average of the cohort- and time-specific conditional average treatment effects. This is similar to the framework of estimation targets of and [Callaway and Sant’Anna (2021, Equation 3.1)](https://arxiv.org/html/2312.05985#bib.bib29), though they study average treatment effects (like our estimand (), to come) rather than conditional average treatment effects. Refer to these papers for many examples of choices of \\{\psi_{rt}\\}. Similar aggregations are also discussed by [Brown and Butts (2022, Remark 3.2)](https://arxiv.org/html/2312.05985#bib.bib23) and .
To give one example among many possible estimands of interest using specific \\{\psi_{rt}\\}, we could choose for some r^{\prime}\in\mathcal{R}  
| ={\psi_{rt}=\begin{cases}\frac{1}{T-r^{\prime}+1},&r=r^{\prime},\\\ 0,&\text{otherwise}\end{cases}  |  
| --- |  
to define the conditional average treatment effect across time for a unit in cohort with covariates 𝒙\boldsymbol{x}:  
| \displaystyle\tau_{\text{CATT}}(r^{\prime},\boldsymbol{x}):=\penalty\   | ​\displaystyle\frac{1}{T-r^{\prime}+1}\sum_{t=r^{\prime}}^{T}\tau_{\text{CATT}}(r^{\prime},\boldsymbol{x},t)  |  
| --- | --- |  
| =\displaystyle=\penalty\   | 𝔼[−∣=,=𝒙].\displaystyle\frac{1}{T-r^{\prime}+1}\sum_{t=r^{\prime}}^{T}\mathbb{E}[\tilde{y}_{(it)}(r^{\prime})-\tilde{y}_{(it)}(0)\mid W_{i}=r^{\prime},\boldsymbol{X}_{i}=\boldsymbol{x}].  |  
A version of () for average (rather than conditional average) treatment effects was considered by [Wooldridge (2021, Equation 6.41)](https://arxiv.org/html/2312.05985#bib.bib125), and a slightly more general form (also accommodated by our framework) was considered by [Goodman-Bacon (2021, Equation 12)](https://arxiv.org/html/2312.05985#bib.bib60).
#### 4.1.1 Propensity-Weighted Conditional Average Treatment Effects
A second class of causal estimands can be generated by weighting each \tau_{\text{CATT}}(r,t,\boldsymbol{x}) by the probability a unit is in cohort given that they are treated conditional on 𝒙\boldsymbol{x},  
| :=ℙ⁡(∣)=​​.\tilde{\pi}_{r}(\boldsymbol{x}):=\mathbb{P}(W_{i}=r\mid W_{i}\neq 0,\boldsymbol{X}_{i}=\boldsymbol{x})=\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{-1}\pi_{r}(\boldsymbol{x})\qquad\forall r\in\mathcal{R}.  |  
| --- |  
In particular, for any set of finite constants \\{\psi_{rt}\\}, consider the estimands  
| ​​​​.\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\tilde{\pi}_{r}(\boldsymbol{x})\tau_{\text{CATT}}(r,t,\boldsymbol{x}).  |  
| --- |  
One example of an estimand we can form in this class arises from choosing  
| :={otherwise,\psi_{rt}:=\begin{cases}0,&r=0,\\\ \frac{1}{T-r+1},&\text{otherwise,}\end{cases}  |  
| --- |  
which yields a conditional average treatment effect by marginalizing () across treatment status:  
| \displaystyle\tau_{\text{CATT}}(\boldsymbol{x}):=\penalty\   | ​​​​\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\frac{1}{T-r+1}\tilde{\pi}_{r}(\boldsymbol{x})\tau_{\text{CATT}}(r,t,\boldsymbol{x})  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​​​\displaystyle\sum_{r\in\mathcal{R}}\tilde{\pi}_{r}(\boldsymbol{x})\tau_{\text{CATT}}(r,\boldsymbol{x})  |  
| =\displaystyle=\penalty\   | 𝔼[∣≠0,=𝒙].\displaystyle\mathbb{E}\left[\tau_{\text{CATT}}(W_{i},\boldsymbol{X}_{i})\mid W_{i}\neq 0,\boldsymbol{X}_{i}=\boldsymbol{x}\right].  |  
This estimand is closely related to many previously considered heterogenous treatment effects; one of many works that considers a similar estimand (outside of the specific context of difference-in-differences) is [Wager and Athey (2018, Equation 1)](https://arxiv.org/html/2312.05985#bib.bib121).
### 4.2 Marginal Average Treatment Effects
Outside of conditional average treatment effects, we can also marginalize these conditional treatment effects over the distribution of \boldsymbol{X}_{i} to get average treatment effects. Since these estimands are "coarser," it may be possible to get more precise estimates of these estimands given a fixed sample size.
For =𝔼⁡[∣]\tau_{\text{ATT}}(r,t)=\mathbb{E}\left[\tau_{\text{CATT}}(r,t,\boldsymbol{X}_{i})\mid W_{i}=r\right] from (), consider the classes of estimands  
| OPEN)\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\tau_{\text{ATT}}(r,t))  |  
| --- |  
and  
| ​​​,\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\tilde{\pi}_{r}\tau_{\text{ATT}}(r,t),  |  
| --- |  
where :==𝔼⁡\tilde{\pi}_{r}:=\mathbb{P}(W_{i}=r\mid W_{i}\neq 0)=\mathbb{E}[\tilde{\pi}_{r}(\boldsymbol{X}_{i})\mid W_{i}\neq 0] is the marginal probability of a unit being in cohort given that it is treated.
Estimand () was used by and [Hatamyar et al. (2023, Equation 1)](https://arxiv.org/html/2312.05985#bib.bib64), and it matches the estimand from and [Brown and Butts (2022, Equation 3)](https://arxiv.org/html/2312.05985#bib.bib23). It is also closely related to the cohort-specific average treatment effect on the treated (3) from (though we use the acronym CATT to refer to conditional, not cohort, effects) and estimands from .
As we mentioned earlier, the class of estimands () is the same framework of estimation targets considered by and is similar to aggregations considered by [Callaway and Sant’Anna (2021, Equation 3.1)](https://arxiv.org/html/2312.05985#bib.bib29), , [Brown and Butts (2022, Remark 3.2)](https://arxiv.org/html/2312.05985#bib.bib23), and [Hatamyar et al. (2023, Equation 18)](https://arxiv.org/html/2312.05985#bib.bib64).
The probability-weighted class of estimands () bears a resemblance to the probability-weighted estimands proposed by . Theorem in the appendix extends our consistency result to accommodate classes of estimands with much broader forms of estimated probability weights than () and (), like estimand (26) of or the broad class of estimands used by . Theorems and extend our asymptotic normality results to these more general estimators as well.
As a particular choice of \\{\psi_{rt}\\}, we will again highlight () for any r^{\prime}\in\mathcal{R}, which for estimand () marginalizes \tau_{\text{CATT}}(r^{\prime},\boldsymbol{X}_{i}) over the distribution of \boldsymbol{X}_{i}:  
| \displaystyle\tau_{\text{ATT}}(r^{\prime}):=\penalty\   | ​\displaystyle\frac{1}{T-r^{\prime}+1}\sum_{t=r^{\prime}}^{T}\tau_{\text{ATT}}(r^{\prime},t)  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​𝔼⁡[−∣]\displaystyle\frac{1}{T-r^{\prime}+1}\sum_{t=r^{\prime}}^{T}\mathbb{E}\left[\tilde{y}_{(it)}(r^{\prime})-\tilde{y}_{(it)}(0)\mid W_{i}=r^{\prime}\right]  |  
| =\displaystyle=\penalty\   | 𝔼⁡[∣].\displaystyle\mathbb{E}\left[\tau_{\text{CATT}}(r^{\prime},\boldsymbol{X}_{i})\mid W_{i}=r^{\prime}\right].  |  
Estimand () is identical to estimand (3.7) in , and is also the estimand of estimator (6.41) in .
#### 4.2.1 The Average Effect of the Treatment on Treated Units
Lastly, one of the most basic causal parameters of interest is the average effect of treatment on the treated group (ATT). By using the from () with (), we can express this estimand as  
| \displaystyle\tau_{\text{ATT}}:=\penalty\   | 𝔼⁡[]=𝔼⁡[​​​∣].\displaystyle\mathbb{E}\left[\tau_{\text{CATT}}(\boldsymbol{X}_{i})\mid W_{i}\neq 0\right]=\mathbb{E}\left[\sum_{r\in\mathcal{R}}\tilde{\pi}_{r}(\boldsymbol{X}_{i})\tau_{\text{CATT}}(r,\boldsymbol{X}_{i})\mid W_{i}\neq 0\right].  |  
| --- | --- |  
Estimand () is similar to estimand (6.40) proposed by (a similar estimand is also mentioned by ), though we weight the cohort average treatment effects \tau_{\text{CATT}}(r,\boldsymbol{X}_{i}) by the probabilities \tilde{\pi}_{r}(\boldsymbol{X}_{i}) rather than taking an average where each average treatment effect in each time period is weighted equally (we could also take an equal average with our estimand from Equation ). This probability-weighted average is closely related to estimand (26) from .
The virtues of estimand \tau_{\text{ATT}} include summarizing the effect of the treatment in a single number and potentially being a less noisy quantity to estimate. Estimating \tau_{\text{ATT}} will involve taking a weighted average over many estimated coefficients rather than relying on a single, less precise, estimated coefficient. In contrast, more specific estimands like \tau_{\text{CATT}}(r,t,\boldsymbol{x}) will typically be difficult to estimate with precision because their estimators will depend heavily on a small subset of the observed data (though this problem will be mitigated by our fusion penalty, discussed in the next section, which allows us to borrow information from nearby estimators).
Notice that \tau_{\text{ATT}} can be obtained not just by marginalizing \tau_{\text{CATT}}(\boldsymbol{X}_{i}) over the distribution of \boldsymbol{X}_{i}, but also by marginalizing \tau_{\text{ATT}}(W_{i}) over the distribution of cohort assignments:  
| \displaystyle\tau_{\text{ATT}}=\penalty\   | 𝔼[∣≠0]=𝔼[𝔼[∣,≠0]∣≠0]\displaystyle\mathbb{E}\left[\tau_{\text{CATT}}(W_{i},\boldsymbol{X}_{i})\mid W_{i}\neq 0\right]=\mathbb{E}\left[\mathbb{E}\left[\tau_{\text{CATT}}(W_{i},\boldsymbol{X}_{i})\mid W_{i},W_{i}\neq 0\right]\mid W_{i}\neq 0\right]  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ⋅𝔼⁡[∣]=\displaystyle\sum_{r\in\mathcal{R}}\tilde{\pi}_{r}\cdot\mathbb{E}[\tau_{\text{CATT}}(W_{i},\boldsymbol{X}_{i})\mid W_{i}=r]=\sum_{r\in\mathcal{R}}\tilde{\pi}_{r}\tau_{\text{ATT}}(r)  |  
| =\displaystyle=\penalty\   | ​​,\displaystyle\sum_{r\in\mathcal{R}}\tilde{\pi}_{r}\frac{1}{T-r+1}\sum_{t=r}^{T}\tau_{\text{ATT}}(r,t),  |  
which shows that estimand () is in the class defined by () and matches estimand (3.11) from . This formulation will also be useful for estimation.
## Fused Extended Two-Way Fixed Effects
We will carry out regression on the same design matrix \boldsymbol{\tilde{Z}} proposed by to estimate regression () by OLS, but we will add fusion penalties to leverage our belief that some of the parameters are equal to each other. This design matrix contains columns of cohort dummies, columns of time dummies for times \\{2,\ldots,T\\}, covariates, \mathfrak{W}:=\sum_{r\in\mathcal{R}}(T-r+1) treatment dummies for each possible base treatment effect , and d_{N}(R+T-1+\mathfrak{W}) interactions between covariates and the cohort dummies, time dummies, and treatment dummies. As is typical in penalized regression, we will center \boldsymbol{\tilde{y}} and the columns of \boldsymbol{\tilde{Z}} before estimating our regression. It is not necessary to estimate the intercept term in order to estimate the treatment effects.
### 5.1 Restrictions in the ETWFE Estimator
Let be the number of columns of \boldsymbol{\tilde{Z}}; that is, \boldsymbol{\tilde{Z}}\in\mathbb{R}^{NT\times p_{N}}. We will assume p_{N}\leq NT (see Assumption R2 later), but as we discussed in the introduction, it is clear that =+𝔚+p_{N}=R+T-1+\mathfrak{W}+d_{N}(1+R+T-1+\mathfrak{W}) may be fairly large compared to , particularly if is not very small. This could lead to imprecise treatment effect estimates if () is estimated by OLS. suggests alleviating this problem by assuming restrictions in the model—assuming some of these parameters equal each other. Some of the possibilities suggests include assuming that the treatment-covariate interactions \boldsymbol{\rho}_{rt} are fixed across time or assuming the within-cohort treatment effects are fixed across time. also considers the possibility of treatment effects that are fixed across time. Another weaker assumption proposes is assuming that different treatment effects do not need to be estimated for every individual time period since treatment starts; instead, these times could be condensed into “early treated" and “late treated" effects, for example.
These suggestions seem sensible, but making these simplifying assumptions without strong justification risks re-introducing the bias that was removed by adding these parameters to begin with. In settings where a practitioner believes some of these restrictions hold but has no clear way to choose which particular parameters may be equal, we instead propose a data-driven approach to estimate which, if any, of these parameters may be equal.
### 5.2 Fusion Penalization in FETWFE
As described in the introduction, we will assign fusion penalties to the differences between estimated parameters which may be equal, and then carry out a penalized linear regression. For example, for the treatment effects, we propose penalizing the absolute differences between treatment effects within the same cohort at adjacent times, |\tau_{r_{k},t}-\tau_{r_{k},t-1}|, and we propose the same penalty structure for interaction effects, |−||\rho_{r_{k},t,j}-\rho_{r_{k},t-1,j}| separately for each j\in[d_{N}]. This leverages our belief that some of these effects may be fixed across time while allowing the optimization problem to learn from the data which restrictions to choose.
We also propose penalizing the absolute differences between initial treatment effects for each cohort, |−||\tau_{r_{k},r_{k}}-\tau_{r_{k-1},r_{k-1}}|, and likewise we penalize the absolute differences between adjacent initial interaction effects |−||\rho_{r_{k},r_{k},j}-\rho_{r_{k-1},r_{k-1},j}| for each . This is motivated by the idea that perhaps time since beginning treatment could be the only dimension in which treatment effect changes. It is also natural to penalize the adjacent cohort fixed effects absolute differences |\nu_{r_{k}}-\nu_{r_{k-1}}| and time fixed effects absolute differences |\gamma_{t}-\gamma_{t-1}| towards each other, with analogous penalties for the interaction terms |\zeta_{r_{k}j}-\zeta_{r_{k-1},j}| and |\xi_{tj}-\xi_{t-1,j}|.
Finally, we may also want to regularize the coefficients directly, as is common in regressions with large numbers of parameters to estimate relative to the number of observations. Because of the fusion penalties we employ, regularizing only a handful of base terms allows for this regularization. We can also penalize all of the base covariate coefficients directly, allowing us to screen out irrelevant covariates as grows.
In summary, we propose estimating a model like () but with a penalty term  
| \displaystyle p(\boldsymbol{\beta};q):=\penalty\   | +++\displaystyle\sum_{k=2}^{R}|\nu_{r_{k}}-\nu_{r_{k-1}}|^{q}+|\nu_{r_{R}}|^{q}+\sum_{t=3}^{T}|\gamma_{t}-\gamma_{t-1}|^{q}+|\gamma_{T}|^{q}  |  
| --- | --- |  
| +(|+|−+|+|−+|)\displaystyle+\sum_{j=1}^{d_{N}}\left(|\kappa_{j}|^{q}+\sum_{k=2}^{R}|\zeta_{r_{k},j}-\zeta_{r_{k-1},j}|^{q}+|\zeta_{r_{R},j}|^{q}+\sum_{t=3}^{T}|\xi_{tj}-\xi_{t-1,j}|^{q}+|\xi_{Tj}|^{q}\right)  |  
| +(|−|q+q)\displaystyle+|\tau_{r_{1},r_{1}}|^{q}+\sum_{k=2}^{R}\left(|\tau_{r_{k},r_{k}}-\tau_{r_{k-1},r_{k-1}}|^{q}+\sum_{t\geq r_{k}+1}|\tau_{r_{k},t}-\tau_{r_{k},t-1}|^{q}\right)  |  
| +[|+(|−+|−)],\displaystyle+\sum_{j=1}^{d_{N}}\left[|\rho_{r_{1},r_{1},j}|^{q}+\sum_{k=2}^{R}\left(|\rho_{r_{k},r_{k},j}-\rho_{r_{k-1},r_{k-1},j}|^{q}+\sum_{t\geq r_{k}+1}|\rho_{r_{k},t,j}-\rho_{r_{k},t-1,j}|^{q}\right)\right],  |  
where \boldsymbol{\beta}\in\mathbb{R}^{p_{N}} collects all of the coefficients. (Refer again to Figure for a graphical depiction of the penalty terms on the estimated treatment effects .) We can more compactly express this as  
| ,p(\boldsymbol{\beta};q)=\lVert\boldsymbol{D}_{N}\boldsymbol{\beta}\rVert_{q}^{q},  |  
| --- |  
where \boldsymbol{D}_{N}\in\mathbb{R}^{p_{N}\times p_{N}} is a suitably defined differences matrix which we later define explicitly in ().
###### .
Our theory could easily be extended to allow a much broader class of penalties than the one penalty we specify in (). Any choice of \boldsymbol{D}_{N} that has all finite entries, is invertible, and is block diagonal with blocks of sizes not increasing in would work for our theoretical results. Such modifications would lead to a change only in constant factors in our theory. For example, one could choose to only use fusion penalties for the treatment parameters and and directly penalize all of the remaining parameters. For another example, if it is believed that treatment effects are more likely to be equal in time since treatment rather than within cohorts, one could replace our structure of treatment effect penalties with fusion penalties across treatment effects at the same time since treatment; see Figure for a graphical depiction of such a structure. It is important that the chosen \boldsymbol{D}_{N} makes Assumption , which we describe later in Section , plausible.
Another example of a possible valid penalty structure that mainly penalizes treatment effects across cohorts towards each other based on time since the start of treatment, as mentioned in Remark . Compare to Figure . All of our theoretical results for FETWFE would work with such a penalty structure up to a change in constants in the results.
It is much easier to see that these candidate restrictions are plausible due to the time-structured nature of the parameters than it is to know exactly which restrictions to choose. FETWFE allows practitioners to leverage this structure to improve estimation efficiency without requiring them to choose restrictions by hand, which could introduce bias if too many restrictions are chosen or compromise efficiency if too few restrictions are chosen.
### 5.3 Constructing the FETWFE Optimization Problem
Finally, we take care of some details. First we account for the random effects discussed in Section . As suggested by (), in practice we will arrange the rows of \boldsymbol{\tilde{Z}} to contain consecutive blocks of rows for each observation and carry out a regression on \boldsymbol{\tilde{y}} and \boldsymbol{\tilde{Z}} transformed by  
| 𝑮:=σ𝑰⊗𝛀∈ℝ,\boldsymbol{G}_{N}:=\sigma\boldsymbol{I}_{N}\otimes\boldsymbol{\Omega}^{-1/2}\in\mathbb{R}^{NT\times NT},  |  
| --- |  
where denotes the Kronecker product (recall that 𝛀\boldsymbol{\Omega} was defined in Section ). We will also center both the response \boldsymbol{G}_{N}\boldsymbol{\tilde{y}} and each column of the covariates \boldsymbol{G}_{N}\boldsymbol{\tilde{Z}}, as is common in penalized regression. Denote these centered objects by 𝒚\boldsymbol{y} and 𝒁\boldsymbol{Z}. Then our optimization problem is  
| :=​{+},\boldsymbol{\hat{\beta}}_{N}^{(q)}(\lambda_{N}):=\underset{\boldsymbol{\beta}\in\mathbb{R}^{p_{N}}}{\arg\min}\left\\{\lVert\boldsymbol{y}-\boldsymbol{Z}\boldsymbol{\beta}\rVert_{2}^{2}+\lambda_{N}\lVert\boldsymbol{D}_{N}\boldsymbol{\beta}\rVert_{q}^{q}\right\\},  |  
| --- |  
where \lambda_{N} is a tuning parameter. For brevity, we will hereafter write =\boldsymbol{\hat{\beta}}_{N}^{(q)}(\lambda_{N})=\boldsymbol{\hat{\beta}}^{(q)}.
###### .
Optimization problem () is convex if and only if , but we will be interested in solving this problem with q\in(0,1) as well. In this case, finding the global optimum can in principle be difficult, but many heuristics exist to solve this optimization problem that seem to work well in practice; see, for example [Huang et al. (2008, Section 4.1)](https://arxiv.org/html/2312.05985#bib.bib71) and . In Sections and we implement FETWFE by solving () with the grpreg R package . Our simulation results seem to align with our theory, including the asymptotic normality of our estimators, suggesting that these heuristics seem to work well for FETWFE.
### 5.4 Causal Estimators
As shown in Theorem (a), under (CNAS), (LINS), and (CCTS)  
| =+⊤​\tau_{\text{CATT}}(r,t,\boldsymbol{x})=\tau_{rt}^{*}+(\boldsymbol{x}-\mathbb{E}\left[\boldsymbol{X}_{i}\mid W_{i}=r\right])^{\top}\boldsymbol{\rho}_{rt}^{*}  |  
| --- |  
for any fixed 𝒙\boldsymbol{x}. Since under these assumptions (and other regularity conditions) FETWFE consistently estimates \tau_{rt}^{*} and \boldsymbol{\rho}_{rt}^{*}, we can estimate this quantity by  
| :=+​,,,\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x}):=\hat{\tau}_{rt}^{(q)}+\left(\boldsymbol{x}-\boldsymbol{\overline{X}}_{r}\right)^{\top}\hat{\rho}_{rt}^{(q)},\qquad r\in\mathcal{R},t\geq r,  |  
| --- |  
where \hat{\tau}_{rt}^{(q)} and \hat{\rho}_{rt}^{(q)} are the estimators of and from \boldsymbol{\hat{\beta}}^{(q)} as defined in () and  
| =∑𝑿\boldsymbol{\overline{X}}_{r}=\frac{1}{N_{r}}\sum_{\\{i:W_{i}=r\\}}\boldsymbol{X}_{i}  |  
| --- |  
is the sample mean of the covariates for units observed in cohort , with :=𝟙N_{r}:=\sum_{i=1}^{N}\mathbbm{1}\\{W_{i}=r\\} the total number of units in cohort . The forms of () and () then naturally suggest the estimators  
| OPEN​​)\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x}))  |  
| --- |  
and  
| ​()​​,\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{\hat{\pi}_{r}(\boldsymbol{x})}{\sum_{r^{\prime}\in\mathcal{R}}\hat{\pi}_{r^{\prime}}(\boldsymbol{x})}\right)\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x}),  |  
| --- |  
where \hat{\pi}_{r}(\boldsymbol{x}) is an estimator of the propensity scores \pi_{r}(\boldsymbol{x}). We will discuss what properties this estimator \hat{\pi}_{r}(\boldsymbol{x}) should have in more detail in Section .
Finally we discuss estimating the average treatment effects marginalized over \boldsymbol{X}_{i}. As we showed in Theorem (a) and (b), because the treatment effects are interacted with the covariates centered with respect to their cohort means, under either (CCTS) or (CTS) and (CIUN) we can estimate the average treatment effects using  
| :=\hat{\tau}_{\text{ATT}}(r,t):=\hat{\tau}_{rt}^{(q)}  |  
| --- |  
as proposed by [Wooldridge (2021, Sections 5.3 and 6.3)](https://arxiv.org/html/2312.05985#bib.bib125). In particular, we propose the estimators  
| ​​\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\hat{\tau}_{\text{ATT}}(r,t)  |  
| --- |  
and  
| ​​​\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\frac{N_{r}}{N_{\tau}}\hat{\tau}_{\text{ATT}}(r,t)  |  
| --- |  
for estimands () and (), respectively. Notice that estimator () avoids requiring the specification of an estimator \boldsymbol{\hat{\pi}}(\boldsymbol{x}) of the generalized propensity scores by instead using the observed cohort counts to estimate the marginal cohort assignment probabilities, as we alluded to at the end of Section .
## Theoretical Results and Discussion
Before we present theoretical guarantees for FETWFE, we will state a sparsity assumption that we will require along with some additional regularity conditions. We will require fewer assumptions to prove consistency alone than we will to prove restriction selection consistency and asymptotic normality. We emphasize that the conceptually most important assumptions are (CCTSB), (CCTSA), and (CTSA), which validate the difference-in-differences approach; (LINS), which justifies a model that is linear in the covariates \boldsymbol{X}_{i}; and Assumption S(), defined below, which assumes sparsity in a sense that matches our penalty (). (See Remark for some comments on how this can be made more flexible.)
### 6.1 Consistency
Assumption S(): Assume the vector \boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*} (where the differences matrix \boldsymbol{D}_{N} associated with our penalty () is defined in Equation ) is sparse, with indices \mathcal{S}\subseteq[p_{N}] nonzero and the remaining indices in [p_{N}]\setminus\mathcal{S} equal to 0, where |\mathcal{S}|=s_{N}.
Assumption S() allows us to have sparsity in the appropriate sense: the differences between parameters that we penalized in () are sparse, and irrelevant covariates have true coefficient equal to 0. That is, all but of the restrictions that we expect to hold will in fact hold. Crucially, Assumption S() does not require us to know which restrictions to select, or even how many restrictions to select—the value of does not need to be known. Our consistency and selection consistency theorems will allow to grow with subject to certain constraints in later assumptions.
The remaining regularity conditions for Theorem are more routine. Before we describe them in detail, we briefly characterize them informally.
  * •
Assumption (R1) requires the tails of the covariates to not be too heavy, but accommodates a wide variety of covariate distributions.
  * •
Assumption (R2) requires the eigenvalues of the empirical Gram matrix to be reasonably well-behaved. In particular, it requires 𝒁\boldsymbol{Z} to have full column rank, so p_{N}\leq NT. Otherwise this is a mild assumption as long as (1) none of the covariates are very highly correlated and (2) none of the cohort propensity scores tend too closely to 0. For example, an overlap assumption that \pi_{r}(\boldsymbol{x})\geq\pi_{\text{low}} for some \pi_{\text{low}}>0 for all N\in\mathbb{N}, r\in\\{0\\}\cup\mathcal{R}, and 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i} would help the plausibility of this assumption.
  * •
In Assumption (R2), the assumptions on the penalty parameter \lambda_{N} are not of practical importance because in practice \lambda_{N} can be selected from a set of candidate values by a criterion like BIC. (We show in simulation studies in Section that the theoretical guarantees from this section seem to hold in practice when \lambda_{N} is selected in this way.)
  * •
Assumption (R3) is essentially a technical assumption. It does suggest that FETWFE may not work well if \boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*} has many very small entries, or a small number of very large entries.


In particular, these assumptions allow the number of covariates and the number of relevant coefficients to grow to infinity with and allow for unbounded distributions of the covariates.
We now state and describe the assumptions.
Assumption (R1): For each , \mathbb{E}[X_{ij}^{4}]<\infty for all j\in[d_{N}] and \mathbb{E}[u_{(it)}^{4}]<\infty for all . Further, for some ,  
| {​𝔼⁡[]}≤K.\sup_{1\leq N\leq\infty}\left\\{\frac{1}{p_{N}}\sum_{t=1}^{T}\sum_{j=1}^{p_{N}}\mathbb{E}\left[(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(1t)j}^{2}\right]\right\\}\leq K.  |  
| --- |  
We show later (Lemma ) that \boldsymbol{D}_{N} is invertible. The matrix \boldsymbol{Z}\boldsymbol{D}_{N}^{-1} is of interest because it turns out this is the design matrix we will actually use for estimation; see the proof of Theorem for details. Because the singular values of \boldsymbol{D}_{N}^{-1} are bounded away from 0 and from above for all even if d_{N}\to\infty (see Lemma ), this is qualitatively similar to making the same statement about 𝒁\boldsymbol{Z} itself. We discuss a closely related point in more detail below.
Before we state Assumption (R2), we will define some needed notation. For any matrix \boldsymbol{A}\in\mathbb{R}^{N\times p_{N}}, denote the empirical Gram matrix by \boldsymbol{\hat{\Sigma}}(\boldsymbol{A}):  
| :=∈.\boldsymbol{\hat{\Sigma}}(\boldsymbol{A}):=\frac{1}{N}\boldsymbol{A}^{\top}\boldsymbol{A}\in\mathbb{R}^{p_{N}\times p_{N}}.  |  
| --- |  
Notice that if 𝑨\boldsymbol{A} has centered columns then \boldsymbol{\hat{\Sigma}}(\boldsymbol{A}) is the estimated covariance matrix. Let and be the smallest and largest eigenvalues of \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z}).(Since \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z}) is random, and are as well.)
Assumption (R2): Assume is almost surely upper-bounded by a constant for all and h_{N}^{\prime}\xrightarrow{a.s.}0, where  
| :=h_{N}^{\prime}:=\sqrt{\frac{p_{N}+\lambda_{N}s_{N}}{Ne_{1N}}}  |  
| --- |  
and \xrightarrow{a.s.} denotes almost sure convergence. Also, assume \lim_{N\to\infty}\lambda_{N}\sqrt{\frac{s_{N}}{N}}=0.
points out that () requires the design matrix to be full rank, so this assumption requires that grows slower than , \lambda_{N} does not grow too quickly, and does not vanish too quickly. (In practice, in finite samples \lambda_{N} can be selected by cross-validation or a criterion like BIC, similarly to the lasso.) Under the qualitative assumption that the eigenvalues of \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z}) and \boldsymbol{\hat{\Sigma}}(\boldsymbol{G}_{N}\boldsymbol{\tilde{Z}}) are ‘‘close,"44 4 This can be made formal by noting that centering the columns of \boldsymbol{G}_{N}\boldsymbol{\tilde{Z}} can be characterized as multiplication on the left by a centering matrix 𝑪\boldsymbol{C}, then bounding the singular values of 𝑪\boldsymbol{C}. See, for example, Section 8.1 of for further details. (R2) roughly requires that \boldsymbol{G}_{N}\boldsymbol{\tilde{Z}} is full rank, and we can provide sufficient conditions for this to hold. First, the singular values of \boldsymbol{G}_{N} are bounded from above and below by positive constants for all (see Lemma in the appendix), so \boldsymbol{\tilde{Z}} being full rank is enough for \boldsymbol{G}_{N}\boldsymbol{\tilde{Z}} to be full rank. For \boldsymbol{\tilde{Z}} to be full rank, we require at least observed units in each cohort (see Lemma in the appendix), all of the marginal cohort probabilities must be large enough relative to the growth rate of as , and the distribution of \boldsymbol{X}_{i} must be well-behaved as .
Assumption (R3): There exist constants 0<b_{0}<b_{1}<\infty such that ≤⁡≤⁡≤b_{0}\leq\min_{j\in\mathcal{S}}\left\\{\left|(\boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*})_{j}\right|\right\\}\leq\max_{j\in\mathcal{S}}\left\\{\left|(\boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*})_{j}\right|\right\\}\leq b_{1}.
The lower bound in Assumption (R3) is essentially a signal strength assumption.
Note that we allow s_{N}\to\infty and p_{N}\to\infty as long as the above assumptions are satisfied.
We are now prepared to state our consistency theorem.
###### (Consistency of FETWFE).
Assume that Assumptions (CNAS), (CCTSB), and (LINS) hold, as well as Assumptions (F1), (F2), S(), and (R1) - (R3). Let .
  1. (a)
Suppose that either Assumptions (CTSA) and (CIUN) hold or Assumption (CCTSA) holds. Then for any set of finite constants \\{\psi_{rt}\\},  
| |​(−)|=​\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right)\right|=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right)  |  
| --- |  
| |​(​​−)|=​,\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{N_{r}}{N_{\tau}}\hat{\tau}_{\text{ATT}}(r,t)-\tilde{\pi}_{r}\tau_{\text{ATT}}(r,t)\right)\right|=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right),  |  
| --- |  
where  
| h_{N}:=\frac{1}{e_{1N}}\sqrt{\frac{p_{N}}{N}}  |  
| --- |  
and h_{N}^{\prime} is as defined in ().
  2. (b)
Suppose Assumption (CCTSA) holds and  is fixed. For all  and all r\in\\{0\\}\cup\mathcal{R}, assume all of the eigenvalues of \mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r) are bounded between \lambda_{\text{min}}>0 and \lambda_{\text{max}}<\infty. Then for any set of finite constants \\{\psi_{rt}\\} and any fixed 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i},  
| |​(−)|=​.\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right)\right|=\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}).  |  
| --- |  
Further, if \hat{\pi}_{r}(\boldsymbol{x}) is an estimator of \pi_{r}(\boldsymbol{x}) that satisfies =|\hat{\pi}_{r}(\boldsymbol{x})-\pi_{r}(\boldsymbol{x})|=\mathcal{O}_{\mathbb{P}}(a_{N}) for a decreasing sequence  for each r\in\\{0\\}\cup\mathcal{R} and all 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i}, then  
| |​(​​−​​​)|\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{\hat{\pi}_{r}(\boldsymbol{x})}{\sum_{r^{\prime}\in\mathcal{R}}\hat{\pi}_{r^{\prime}}(\boldsymbol{x})}\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tilde{\pi}_{r}(\boldsymbol{x})\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right)\right|  |  
| --- |  
| =\displaystyle=\penalty\   | ​(∨).\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}\vee a_{N}).  |  


###### Proof.
Provided in Section of the appendix. ∎
Theorem shows that all of our causal estimators are consistent and characterizes their rates of consistency. In particular, it also shows that the classes of marginal parallel trends estimators () and () are consistent under either (CCTSA) or (CIUN) and (CTSA) for the marginal average treatment effects () and (). Although we require (CCTSB) to hold regardless of whether (CIUN) holds, (CCTSB) also depends only on the observed outcomes and is testable. In the next section, we show that under mild assumptions FETWFE automatically provides a valid estimate of whether (CIUN) holds (by estimating whether the \boldsymbol{\xi}_{t}^{*} coefficients all equal 𝟎\boldsymbol{0}).
To interpret the rate of convergence \min\\{h_{N},h_{N}^{\prime}\\}, consider the upper bound from (). Suppose the minimum eigenvalue is bounded away from 0 with high probability, which happens under relatively mild assumptions in our p_{N}\leq NT setting (see our more detailed discussion later in Section about Assumption R6). Then ​=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right)=\mathcal{O}_{\mathbb{P}}\left(\sqrt{p_{N}/N}\right); see Lemma in the appendix.
In general we will focus more on q\in(0,1) as required by Theorems , , and , but we note that Theorem holds for any . This is useful because the convex lasso () and ridge () optimization problems, for example, may be more tractable on very large data sets than the nonconvex bridge estimator with .
To give a concrete example of an estimator \boldsymbol{\hat{\pi}}(\boldsymbol{x}), in the setting of Theorem (b) where is fixed, the multinomial logit model is -consistent for the generalized propensity scores by standard maximum likelihood theory under correct specification. If this model is overly simplistic or is large relative to , regularized machine learning methods with theoretical convergence guarantees—like deep neural networks ([Farrell et al., 2021](https://arxiv.org/html/2312.05985#bib.bib49), see Lemma 9c), random forests , the -penalized classifiers of , or the group lasso multinomial logit model of —would also work. On the other hand, since the cohorts have a natural ordering, we could add more structure by using an ordinal response estimator like the proportional odds model .
It may be possible to relax Assumption (LINS) so that linearity only has to hold approximately, along the lines of e.g. Condition ASTE in . (At a high level, we might expect that if approximate linearity holds so that the error due to misspecification is on the same order as the estimation error of the population least squares model, we can achieve convergence at the same rate as when the linearity assumption holds exactly.)
It may also be possible to relax the fixed assumption in part by using high-dimensional central limit theorems like those in and , to name two recent examples among many papers containing such results.
Lastly, we point out that under Theorem the generalized propensity scores and treatment effects can be estimated on the same data sets; that is, the full data set can be used for both. Intuitively, this is because these estimators are used for the distinct tasks of estimating the effects \tau_{\text{CATT}}(r,t,\boldsymbol{x}) and estimating the weights used to combine them in a weighted average. For technical details, see the proof of Theorem in Appendix .
In the appendix, we present an extension of Theorem , Theorem , which proves consistency for a set of estimators that allows for much broader functions of the estimated cohort membership probabilities than the classes () and (). This result includes, for example, all of the estimands from as special cases.
### 6.2 Restriction Selection Consistency
Again, we first state some additional needed regularity conditions and then present our selection consistency result, Theorem . Theorem shows that with probability tending towards one FETWFE identifies the correct restrictions, resulting in fusing together parameters that equal each other, screening out irrelevant covariates, and improving estimation efficiency. At a high level, Assumptions (R4) and (R5) add the following requirements:
  * •
Assumption (R4) is most plausible if the covariates have bounded support and if is not increasing in . Intuitively, this means that in finite samples results relying on Assumption (R4) will be most likely to hold when is reasonably large compared to , though we show in the simulation studies in Section that in practice the conclusion of Theorem seems to hold even when is not all that large and is fairly close to .
  * •
Assumption (R5) is a technical assumption that does not require much more than (R2). (R5) is easier to satisfy if is increasing only slowly in and if the sparsity is fixed. (In finite samples, we would expect better results if and the sparsity are small relative to .)


Assumption (R4): There exists a positive constant e_{\text{max}}<\infty such that e_{2N}\leq e_{\text{max}} almost surely (recall from Section that is the largest eigenvalue of \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z})).
A sufficient condition for Assumption (R4) is that the distribution of \boldsymbol{X}_{i} has bounded support in the sense that \lVert\boldsymbol{X}_{i}\rVert_{2}\leq B almost surely for some .
Assumption (R5):  
| ​∞.\lambda_{N}\frac{e_{1N}^{2-q}}{\sqrt{N^{q}p_{N}^{2-q}}}\xrightarrow{a.s.}\infty.  |  
| --- |  
We will briefly examine how Assumption (R5) interacts with Assumption (R2). For simplicity, assume the minimum eigenvalue e_{1N}\geq c for some with probability tending towards 1, so that we can ignore asymptotically (using a similar argument to the one in the proof of Lemma in the appendix). Then Assumption (R5) is equivalent to the deterministic condition  
| ​→∞.\lambda_{N}\frac{1}{N^{q/2}p_{N}^{1-q/2}}\to\infty.  |  
| --- |  
If we also assume is fixed, we have from Assumption (R2) that  
| and.\displaystyle\frac{\lambda_{N}}{N}\to 0\qquad\text{and}\qquad\lim_{N\to\infty}\frac{\lambda_{N}}{N^{1/2}}=0.  |  
| --- |  
We will require q\in(0,1), and we see that under these assumptions for (R5) and (R2) to hold simultaneously we require N^{q/2}p_{N}^{1-q/2} to grow slower than ; that is, must grow slower than  
| =.\left(\frac{N^{1/2}}{N^{q/2}}\right)^{2/(2-q)}=N^{(1-q)/(2-q)}.  |  
| --- |  
Even if is not bounded away from 0 with high probability, Assumption (R5) requires that does not vanish too quickly. As we discussed in reference to Assumption (R2), this requires that the probability of being in any one cohort does not vanish too quickly as . It also requires that \lambda_{N} not decrease too quickly.
Notice that other than these relatively mild regularity conditions on the eigenvalues of \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z}), we do not require a condition like the irrepresentable condition or neighborhood stability condition ([Meinshausen and Bühlmann, 2006](https://arxiv.org/html/2312.05985#bib.bib91)) that are required for selection consistency of the lasso. However, our restriction selection consistency result will only work for ; we do not prove selection consistency of FETWFE with a lasso () penalty.
###### (Selection consistency).
For the FETWFE estimator \boldsymbol{\hat{\beta}}^{(q)} defined in (), define  
| :={:}\hat{\mathcal{S}}:=\left\\{j\in[p_{N}]:(\boldsymbol{D}_{N}\boldsymbol{\hat{\beta}}^{(q)})_{j}\neq 0\right\\}  |  
| --- |  
to be the set of nonzero values in \boldsymbol{D}_{N}\boldsymbol{\hat{\beta}}^{(q)}. Recall from Assumption S() that  
| 𝒮={:}.\mathcal{S}=\left\\{j\in[p_{N}]:(\boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*})_{j}\neq 0\right\\}.  |  
| --- |  
Assume that Assumptions (CNAS), (CCTSB), and (LINS) hold, as well as Assumptions (F1), (F2), S(), and (R1) - (R5). Assume q\in(0,1). Then =1\lim_{N\to\infty}\mathbb{P}\left(\hat{\mathcal{S}}=\mathcal{S}\right)=1.
###### Proof.
This is immediate from Theorem (c) in the appendix. ∎
Theorem shows that FETWFE fulfills its promise to identify the correct restrictions in accordance with our penalty structure, as discussed in Section . This serves our goal of improving estimation by avoiding unnecessarily estimating parameters that are in fact equal.
FETWFE also screens out irrelevant covariates and interactions. As mentioned in Section , if Assumptions (LINS) and (CIUN) both hold than the \boldsymbol{\xi}_{t}^{*} coefficients must all equal 0. If this is true, in large enough samples FETWFE will estimate \boldsymbol{\xi}_{t}^{*} to equal exactly 𝟎\boldsymbol{0} with high probability.
The proof of Theorem relies on an extension of Theorem 2 of (Theorem in the appendix) to show that not only does the bridge estimator exclude false selections, it also selects all of the relevant components of \boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*} with probability tending to 1. Theorem requires no added assumptions from Theorem 2 of .
One consequence of Theorem is that if \tau_{\text{ATT}}(r,t)=0 for any , then  
| ℙ⁡=1.\lim_{N\to\infty}\mathbb{P}\left(\hat{\tau}_{\text{ATT}}(r,t)=0\right)=1.  |  
| --- |  
The same is true of all of the other causal estimators. This is stronger than convergence in probability to 0, and this stronger notion of convergence turns out to be a key ingredient in the proof of Theorems and , to come.
### 6.3 Asymptotic Normality and Oracle Property
Our asymptotic convergence results (Theorems and below) require one additional regularity condition, an eigenvalue condition which does not require much more than Assumptions (R2) and (R5).
Assumption (R6): The minimum eigenvalue of  
| \mathbb{E}\left[\boldsymbol{\hat{\Sigma}}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)\right]  |  
| --- |  
is greater than or equal to a fixed for all sufficiently large, and  
| \frac{\lambda_{N}}{e_{1N}\sqrt{N}}\in o_{\mathbb{P}}\left(1\right).  |  
| --- |  
Further, for all and all r\in\\{0\\}\cup\mathcal{R}, all of the eigenvalues of \mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r) are bounded between \lambda_{\text{min}}>0 and \lambda_{\text{max}}<\infty.
The assumption that the minimum eigenvalue of () is bounded away from 0 is fairly mild since we already require \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z}) to be full rank and for its minimum eigenvalue to not vanish too quickly, and the minimum singular value of \boldsymbol{D}_{N} is bounded from below by a positive constant; see Lemma in the appendix. Further, this assumption is stronger than necessary for the sake of interpretability; we only require a bound on the minimum eigenvalue of a different matrix () for all . This condition holding for () is sufficient but not necessary.
Condition () is another assumption that requires to not vanish too quickly, which is qualitatively similar to Assumptions (R2) and (R5). Holding fixed (as is required in Theorems and ) and ignoring the distinction between almost sure convergence and convergence in probability, () is more restrictive on the growth rate of \lambda_{N} than (R2) since (R2) requires  
| \frac{\lambda_{N}}{Ne_{1N}}\to 0  |  
| --- |  
while () requires  
| \frac{\lambda_{N}}{\sqrt{N}e_{1N}}\to 0.  |  
| --- |  
However, if is bounded away from 0 with high probability, using a similar argument to the one in the proof of Lemma in the appendix, () is essentially equivalent to the other requirement in (R2) that \lim_{N\to\infty}\lambda_{N}\sqrt{s_{N}/N}=0 (in the setting of Theorem where is fixed). If the smallest eigenvalue of \mathbb{E}\left[\boldsymbol{\hat{\Sigma}}(\boldsymbol{Z})\right] is bounded away from 0, one can show that the asymptotic distribution for the minimum eigenvalue of \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z}) has support bounded away from 0 in our setting where \boldsymbol{X}_{i} has finite fourth moments and p_{N}\leq NT . If we also assume that \boldsymbol{X}_{i} is subgaussian (a weaker assumption than requiring \boldsymbol{X}_{i} to have bounded support), under Theorem 5.39 from is bounded away from 0 with high probability for finite large enough relative to .
#### 6.3.1 Oracle Property
Next we show that under Assumptions (R1) - (R6) the estimators () and () are both oracle estimators , converge at a rate (faster than the guarantee from Theorem ), and are asymptotically Gaussian.
###### (Oracle Property of FETWFE).
Assume that Assumptions (CNAS), (CCTSB), and (LINS) hold, as well as Assumptions (F1), (F2), S() for a fixed , and (R1) - (R6). Assume q\in(0,1). Suppose that either Assumptions (CTSA) and (CIUN) hold or Assumption (CCTSA) holds.
For an arbitrary set of finite constants \\{\psi_{rt}\\}, if at least one of the \tau_{\text{ATT}}(r,t) with a nonzero is nonzero, then the sequence of random variables  
| ​​(−)\displaystyle\sqrt{NT}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t))  |  
| --- |  
converges in distribution to a mean zero Gaussian random variables with variance that depends only on the parameters of the model with all restrictions correctly identified.
Further, if \hat{\tau}_{\text{ATT}}(r,t) and the probability ratios N_{r}/N_{\tau} are estimated on two independent data sets of size , then the sequence of random variables  
| ​​(−)\displaystyle\sqrt{NT}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{N_{r}}{N_{\tau}}\hat{\tau}_{\text{ATT}}(r,t)-\tilde{\pi}_{r}\tau_{\text{ATT}}(r,t)\right)  |  
| --- |  
also converges in distribution to a mean zero Gaussian random variable with variance that depends only on the parameters of the model with all restrictions correctly identified.
(If all of the \tau_{\text{ATT}}(r,t) with nonzero terms equal 0, both sequences of random variables converge in probability to 0.)
###### Proof.
Provided in Section of the appendix. ∎
Theorems and leverage novel (to the best of our knowledge) extensions of Theorem 2 from that establish finite-sample variance estimators, including when the weights in a linear combination of the estimated coefficients are themselves estimated; see Theorem in the appendix.
Theorem (and, more directly, Theorem (d) and (f) in the appendix) show that even if p_{N}\to\infty FETWFE converges at a rate and its asymptotic covariance matrix depends only on the parameters of the correct model. That is, FETWFE estimates the model with the same asymptotic efficiency as an OLS-estimated ETWFE estimator that knows the correct restrictions. FETWFE leverages the possible restrictions in the ETWFE estimator to improve efficiency without requiring the practitioner to make assumptions on which restrictions to choose, which could induce asymptotic bias if the wrong restrictions are chosen or compromise asymptotic efficiency if not enough restrictions are chosen. This addresses the need to consider both bias and variance when estimating treatment effects .
In Remark we mentioned that the inclusion of the parameters () in our regression allows for unbiased estimation of treatment effects under both (CCTS) and unconfoundedness of the untreated potential outcomes, but under unconfoundedness the parameters equal exactly 0 and are inefficient to include. However, Theorem shows that when coefficients that equal 0 are included, FETWFE sets these coefficients equal to exactly 0 with high probability and estimates the nonzero coefficients with the same asymptotic efficiency as if those coefficients were omitted from the model. FETWFE is therefore a good model to choose if one wants to maximize robustness without compromising asymptotic efficiency. The same is true if (CIUN) holds and the regression estimands \boldsymbol{\xi}_{t}^{*} all equal 𝟎\boldsymbol{0}, or if any irrelevant covariates are included in the regression. So practitioners do not need to hesitate to include a possibly relevant covariate that could be helpful for the plausibility of Assumption (CCTSB) or (CCTS) as long as 𝒁\boldsymbol{Z} has full column rank.
#### 6.3.2 Asymptotically Valid Confidence Intervals
Theorem is conceptual and yields insight on the kind of behavior we can expect from FETWFE in large samples. The following result, which has the same assumptions as Theorem , is more practical and provides explicit formulas for variance estimators for asymptotically valid confidence intervals of treatment effect estimates.
###### (Asymptotic Confidence Intervals for FETWFE).
Assume that Assumptions (CNAS), (CCTSB), and (LINS) hold, as well as Assumptions (F1), (F2), S() for a fixed , and (R1) - (R6). Assume q\in(0,1). Suppose that either Assumptions (CTSA) and (CIUN) hold or Assumption (CCTSA) holds. Let \\{\psi_{rt}\\} be an arbitrary set of finite constants, and for all of the below results, assume at least one of the \tau_{\text{ATT}}(r,t) corresponding to a nonzero is nonzero (otherwise, all of the below sequences of random variables converge in probability to 0).
  1. (a)
For the estimator () it holds that  
| ​​(−)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C})}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t))\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
| --- | --- |  
where \hat{v}_{N}^{(\text{C})} is a finite-sample variance estimator defined in (). (All variance estimators depend only on the known  and the observed data.)
  2. (b)
Suppose one set of data of size  is used to estimate the cohort probabilities N_{r}/N_{\tau} and another independent data set of size  is used to estimate each \hat{\tau}_{\text{ATT}}(r,t). Then for the estimator () it holds that  
| ​​(−)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{\text{C};\hat{\pi}}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{N_{r}}{N_{\tau}}\hat{\tau}_{\text{ATT}}(r,t)-\tilde{\pi}_{r}\tau_{\text{ATT}}(r,t)\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
| --- | --- |  
where \hat{v}_{N}^{\text{C};\hat{\pi}} is a finite-sample variance estimator defined in ().
  3. (c)
Suppose a single set of data of size  is used to estimate both the cohort probabilities N_{r}/N_{\tau} and each \hat{\tau}_{\text{ATT}}(r,t). Then the estimator () is asymptotically subgaussian: the sequence of random variables  
| N​​(−)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{\text{C, (cons)};\hat{\pi}}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{N_{r}}{N_{\tau}}\hat{\tau}_{\text{ATT}}(r,t)-\tilde{\pi}_{r}\tau_{\text{ATT}}(r,t)\right)  |  
| --- |  
converges in distribution to a mean-zero subgaussian random variable with variance at most 1, where N\hat{v}_{N}^{\text{C, (cons)};\hat{\pi}} is a finite-sample conservative variance estimator defined in ().


###### Proof.
Provided in Section of the appendix. ∎
Theorem shows that we can form asymptotically valid confidence intervals with estimator \hat{T}_{N} and variance estimator \hat{v}_{N} with  
| :=[−​​,+​​],\text{CI}_{N}(\hat{v}_{N};\alpha):=\left[\hat{T}_{N}-\Phi^{-1}\left(1-\frac{\alpha}{2}\right)\sqrt{\frac{\hat{v}_{N}}{NT}},\hat{T}_{N}+\Phi^{-1}\left(1-\frac{\alpha}{2}\right)\sqrt{\frac{\hat{v}_{N}}{NT}}\right],  |  
| --- |  
where \Phi(\cdot) is the distribution function of the standard normal distribution.
Theorem (b) requires data splitting for asymptotic normality of the estimator (). However, the data used for estimating the marginal cohort probabilities do not need to include observed responses. In some domains like public health and medicine where difference-in-differences has many potential uses ([Rothbard et al., 2023](https://arxiv.org/html/2312.05985#bib.bib105); [Wing et al., 2018](https://arxiv.org/html/2312.05985#bib.bib123)), such unlabeled data may be more widely available than labeled data ([Candès et al., 2018](https://arxiv.org/html/2312.05985#bib.bib30), Section 1.3.2). In a setting like this, asymptotic normality can be achieved without reducing the amount of data used to estimate the model. (It is straightforward to adjust the variance estimator appropriately if the sample sizes of the two data sets are not equal.)
Theorem (c) does not require sample splitting, at the price of sacrificing asymptotic normality for mere asymptotic subgaussianity and necessitating a conservative variance estimator. Because a subgaussian random variable can be defined as a variable satisfying ℙ≤2exp\mathbb{P}(|X|\geq x)\leq 2\exp\\{-x^{2}/K^{2}\\} for some for all ([Vershynin, 2018](https://arxiv.org/html/2312.05985#bib.bib118), Proposition 2.5.2(a)), the tails of such a variable decay at least as quickly as a Gaussian random variable, so we might expect confidence intervals to have similar asymptotic validity to the confidence intervals from part (b). In practice, we show in the simulation studies in Section that these confidence intervals do seem to retain validity.
We cannot test the null hypothesis of treatment effects equalling 0 under Theorem , since these test statistics are not asymptotically normal if the estimated coefficients all equal 0; instead they converge in probability to 0 (even after scaling by ).
Theorem in the appendix is a generalization of both Theorems and for the classes of conditional average treatment effects estimators () and () in the fixed setting under the assumption that an asymptotically Gaussian estimator of the generalized propensity scores (like the multinomial logit or proportional odds models) is available and an extra split of the data is available to estimate the cohort covariate means.
## Simulation Studies
To test the efficacy of FETWFE under our assumptions, we conduct simulation studies in R using the fetwfe package and the simulator package . We choose parameters that bear resemblance to the empirical application from Section III of (,,,N=51,T=33,R=12,d_{N}=3), which we explore in Section . We use fewer cohorts for simplicity of presentation, more covariates to increase in compensation for the smaller , and more units to avoid realizations where no units are assigned to a cohort by random chance. We generate data with units, time periods, cohorts entering at times \\{2,\ldots,6\\}, and features, which results in a total of p_{N}=2209 covariates and NT=3600 observations. To generate a \boldsymbol{\beta}_{N}^{*} such that \boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*} is sparse, we generate a single random sparse \boldsymbol{\theta}_{N}^{*}\in\mathbb{R}^{p_{N}}, then transform this into a coefficient vector \boldsymbol{\beta}_{N}^{*}=\boldsymbol{D}_{N}^{-1}\boldsymbol{\theta}_{N}^{*} to use across all simulations. We generate \boldsymbol{\theta}_{N}^{*} by taking a -vector of all 0 entries and setting each entry equal to 2 randomly with probability 0.1. We set the sign of each term randomly, but since each entry of \boldsymbol{\beta}_{N}^{*} is a sum of terms in \boldsymbol{\theta}_{N}^{*}, we bias the signs to avoid treatment effects that are too close to 0: individual nonzero terms in \boldsymbol{\theta}_{N}^{*} are positive with probability 0.6 and negative otherwise.
On each of 700 simulations, we randomly generate independent realizations of \boldsymbol{X}_{(i)}\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I}_{d_{N}}) for the time-invariant covariates. We randomly assign treatments with probabilities =(,…,)(\pi_{0},\pi_{1},\ldots,\pi_{R})=(1/(R+1),\ldots,1/(R+1)), where  
| :={\pi_{r}:=\begin{cases}\mathbb{P}(W_{i}=0),&r=0,\\\ \mathbb{P}(W_{i}=r),&r\in\mathcal{R},\end{cases}  |  
| --- |  
are the marginal probabilities of treatment assignments. To ensure the model is estimable, in the rare instances where there are no untreated units or one cohort has no observed units, we draw another set of random assignments.
After drawing covariates and treatment assignments, we construct \boldsymbol{\tilde{Z}}. Then we draw independent random effects c_{i}\sim\mathcal{N}(0,5) as well as independent noise terms u_{(it)}\sim\mathcal{N}(0,5) to generate \boldsymbol{\epsilon}_{(i\cdot)}=c_{i}\boldsymbol{1}_{T}+\boldsymbol{u}_{(i\cdot)} and finally generate \boldsymbol{\tilde{y}}=\boldsymbol{\tilde{Z}}\boldsymbol{\beta}_{N}^{*}+\boldsymbol{\epsilon}_{(\cdot\cdot)}.
Besides FETWFE, we also consider three competitor methods. The first is ETWFE, estimated by OLS on \boldsymbol{\tilde{Z}}. We also consider bridge regression on 𝒁\boldsymbol{Z} directly penalizing 𝜷\boldsymbol{\beta} rather than penalizing \boldsymbol{D}_{N}\boldsymbol{\beta} (BETWFE). For both FETWFE and BETWFE, we follow the experiments of and use , selecting the penalty \lambda_{N} over a grid of 100 values (equally spaced on a logarithmic scale) using BIC. We implement bridge regression for both BETWFE and FETWFE using the R grpreg package . Finally, we also consider a slightly more flexible version of (), where we estimate the model  
| =++𝑿𝜿+⋅𝟙𝟙+\tilde{y}_{it}=\nu_{r}+\gamma_{t}+\boldsymbol{X}\boldsymbol{\kappa}+\sum_{r\in\mathcal{R}}\tau_{r}\cdot\mathbbm{1}\\{W_{i}=r\\}\mathbbm{1}\\{t\geq r\\}+\epsilon_{it}  |  
| --- |  
by OLS (TWFE_COVS). (Notice that adding unit-specific covariates to Model () induces collinearity. Fitting the model instead on cohort fixed effects, like ETWFE, also matches, e.g., Equation 3.2 in . See also the discussion around Equation 2.5 in .) The inclusion of covariates might make it seem more plausible that Assumptions (CNAS) and (CCTS) could hold in this model, though points out that adding time-invariant covariates does not change the estimates of the treatment effects. Separate treatment effects for each cohort, while insufficient to avoid bias and capture the variation in treatment effects, should improve estimation relative to () and allows for the estimation of cohort-specific treatment effects to compare to the other methods.
### 7.1 Estimation Error
We start by evaluating the error of each method in estimating the average treatment effect (). The assumptions of Theorem are satisfied, so we expect FETWFE to estimate the treatment effects more accurately than ETWFE, which will suffer from high variance; BETWFE, which does not assume the correct form of sparsity; and TWFE_COVS, which is asymptotically biased. We estimate the average treatment effect using the FETWFE estimator () with from (), as well as analogous estimators using the competitor estimated regression coefficients and observed cohort counts.
We calculate the squared error of each average treatment effect estimate for each method on each iteration. Boxplots of the results are displayed in Figure . We also provide the means and standard errors for the squared errors of each method in Table . Table contains -values for paired one-tailed -tests of the alternative hypothesis that the squared error for FETWFE is less than the squared error for each competitor method; all results show significantly better performance for FETWFE at the 0.05 significance level.
These results show that even if most of the restrictions we would naturally consider hold and we are only interested in overall average treatment effect (), a simple model like TWFE_COVS is an unsuitable estimator. This aligns with the conclusions of previous works, as we discussed in the introduction. We see that ETWFE has better estimation error, though in practice it still does not perform as well as FETWFE because it estimates about ten times as many parameters as are needed, leading to imprecise estimates.
Boxplots of squared errors for each treatment effect estimate across all 700 simulations. Vertical axis is on a log scale. Means and standard errors for the squared error of the estimator of the ATT () for each method across all 700 simulations in the first simulation study from Section .  
| FETWFE  | ETWFE  | BETWFE  | TWFE_COVS |  
| --- | --- | --- | --- |  
| 0.0399 (0.00263)  | 1.19 (0.633)  | 0.339 (0.00336)  | 0.938 (0.0395) |  
-values from paired one-tailed -tests of the alternative hypothesis that the FETWFE ATT estimator () has a lower squared estimation error for the ATT () than each competitor method calculated across all 700 simulations in the first simulation study from Section . (Low -values indicate better predictive performance for FETWFE.)  
| ETWFE  | BETWFE  | TWFE_COVS |  
| --- | --- | --- |  
| 0.0343  | 1.24e-320  | 1.68e-86 |  
In Appendix , we also present estimation error results for the cohort-specific average treatment effects \tau_{\text{ATT}}(r) as defined in () for each of the five cohorts, and estimation error for the coefficients of \boldsymbol{\beta}_{N}^{*} corresponding to the interaction effects, \boldsymbol{\rho}^{*}. These results show that FETWFE outperforms the competitor methods for these estimands as well.
### 7.2 Restriction Selection Consistency
Next, we examine the extent to which we can trust FETWFE to select the correct restrictions, as we know it does with asymptotically high probability under Theorem . In the same experiment as before, on each simulation we calculate the percentage of restrictions FETWFE successfully identifies among the treatment parameters. In particular, we calculate the percentage of entries of \boldsymbol{D}\boldsymbol{\beta}_{N}^{*} that FETWFE successfully estimates as either zero or nonzero.
On average across the 700 simulations, FETWFE successfully decides whether to use a restriction or not in of the possible cases, with a standard error of 0.025\%. Of the true restrictions (that is, the entries of \boldsymbol{D}\boldsymbol{\beta}_{N}^{*} that equal 0), on average (0.021\%) are correctly identified. This suggests that FETWFE is effective at identifying restrictions in practice, which contributes to FETWFE’s success in estimating treatment effects that we saw in the previous section. (Figure in Appendix shows a boxplot of the proportion of restrictions successfully identified across all 700 simulations.)
To get a better sense of the asymptotics, we conduct an additional simulation study in the same way as the first one, but with ,55 5 Notice that increasing increases the number of observations while also increasing the number of treatment effects and interactions to estimate. So we might expect larger to have a roughly neutral effect on the convergence properties, while greatly increasing the computational cost of simulations with large . R=3,d_{N}=2 (resulting in ), and observations (resulting in 300 observations per cohort in expectation). The sparse \boldsymbol{\theta}_{N}^{*} is generated as before, but with entries each equalling 0 with probability 0.5. Again, we conduct 700 simulations. In this second simulation study, FETWFE successfully decides whether to use a restriction or not in (0.042\%) of the possible cases across the 700 simulations, and on average FETWFE identifies (0.096\%) of the true restrictions.
### 7.3 Asymptotic Distribution
Finally we investigate the asymptotic distribution of the FETWFE estimators. On each simulation, we estimate nominal asymptotic confidence intervals for the cohort average treatment effects using () and the variance estimator from Theorem (a). In the first simulation study, we find coverage rates of 0.791 for the cohort (standard error 0.015), 0.884 for the cohort (standard error 0.012), (0.013) for the cohort, (0.015) for the cohort, and (0.014) for the cohort. We see that the coverage is unsatisfactory for FETWFE in this relatively small sample size, where in expectation 20 units are observed in each cohort.
We also estimate conservative confidence intervals for the average treatment effect with the conservative variance estimator from Theorem (c), yielding a coverage rate of 0.993 (0.001). That is, the conservative confidence intervals seem to achieve the desired coverage level in practice, even though the asymptotic distribution of the test statistic is not known to be Gaussian (merely subgaussian).
Lastly, we examine the non-conservative estimator from Theorem (b): on each simulation study, we generate independent unlabeled samples where only cohort assignments (drawn from the same distribution as the labeled units) are observed, and we use these to estimate asymptotically exact confidence intervals for the average treatment effect. The resulting coverage is 0.986 (0.001). The coverage may be better for the average treatment effect than the cohort average treatment effects because the number of observations that are relevant for estimating the overall average treatment effect is much larger than that for each of the cohort average treatment effects.
In the second simulation study, described previously in Section , we find coverage rates of 0.944 (0.009) for the cohort, 0.941 (0.009) for the cohort, and 0.930 (0.010) for the third cohort. The conservative confidence intervals for the average treatment effects have coverage 0.997 (0.001), and the asymptotically exact confidence intervals have coverage 0.973 (0.002). Overall, the coverage rates are close to the nominal level in this finite setting. This is especially promising because we are using heuristics to solve the FETWFE optimization problem, as discussed in Remark .
## Empirical Application
We use the R fetwfe package to analyze the data from , as collected in the divorce data set from . estimate the causal effect of unilateral or “no-fault" divorce laws on suicide rates among women. The data set consists of observations from states (including the District of Columbia) for years, along with demographic covariates. The response variable is the aggregate state suicide rate among women by year (as in the first column of Table 1 of ). See and [Goodman-Bacon (2021, Section III)](https://arxiv.org/html/2312.05985#bib.bib60) for more details on the data.
We remove states treated at time from the data, leaving observations. 5 states are never treated, and cohorts are treated at various times. Similarly to the analysis in [Goodman-Bacon (2021, Section IV(B))](https://arxiv.org/html/2312.05985#bib.bib60), we consider the time-varying covariates murderrate (state female homicide rate), lnpersinc (natural log of state personal income per capita), and afdcrolls (state welfare participation rate), including the values from time as time-invariant controls. However, murderrate is missing for some observations at so is omitted, leaving covariates. In total, we have NT=1386 observations and p_{N}=908 coefficients to estimate.
We estimate the variances \sigma_{c}^{2} (the variance of the unit random effects) and (the variance of the added noise) using the estimators from with ridge regression. We use FETWFE with to estimate the marginal average treatment effects, again leveraging the R grpreg package as in the simulation studies and selecting \lambda_{N} by BIC. We estimate that the overall average percent change to the suicide rate for women is -3.76\%, which is similar to the estimate of -2.52\% (standard error ) that calculates when using similar controls. Applying Theorem (c), we estimate a conservative standard error of , yielding a conservative confidence interval of (,)(-12.73\%,5.21\%) for the average treatment effect.
In Table we present point estimates for the estimated average treatment effects for each of the cohorts. (We omit standard errors since the individual cohorts have very small sample sizes.) We estimate that four cohorts have non-zero treatment effects, with the 1970 cohort (consisting of California and Iowa) having by far the largest treatment effect estimate. We emphasize that these individual estimates are likely less reliable than the average treatment effect estimate since most cohorts have a small number of units.
Cohort average treatment effect estimates from the empirical application in Section , targeting estimand \tau_{\text{ATT}}(r) as defined in ().  
| Cohort  | \hat{\tau}_{\text{ATT}}(r) |  
| --- | --- |  
| 1970  | -40.142%40.142\% |  
| 1973  |  
| 1976  |  
| 1977  |  
## Conclusion
In this work, we have proposed the fused extended two-way fixed effects estimator for difference-in-differences with staggered adoptions. Our estimator leverages the fact that the extended two-way fixed effects estimator includes coefficients that are likely to be similar (for example, because of their proximity in time), but it selects appropriate restrictions automatically rather than relying on hand-selected restrictions, which could be biased. FETWFE retains the asymptotic unbiasedness (and normality) of ETWFE but leverages sparsity for large efficiency gains. In particular, we showed that our estimator is consistent, selection consistent, asymptotically Gaussian, and has an oracle property for several classes of heterogeneous marginal and conditional average treatment effects. Finally, we demonstrated the usefulness of FETWFE in practice in both simulation studies and an empirical application.
### 9.1 Future Work
Throughout the paper we have mentioned places where our work might be relatively straightforward to extend. Here we specify some other possible extensions. It would be useful to develop theory allowing for FETWFE to be estimated when the design matrix 𝒁\boldsymbol{Z} is deficient in column rank; for example, when . One could extend our method to this setting by suitably adapting the marginal bridge estimator of , which involves a screening step to remove irrelevant covariates before estimating a regression on the reduced design matrix in a second step. This allows for consistency and asymptotic normality of the second-step estimator under similar assumptions to those that we have used in this work. Alternatively, consistency (but not asymptotic normality) could be established under weaker assumptions in the case by suitably adapting theory for lasso estimators like Corollary 2 from .
Another potentially useful extension would be relaxing the parallel trends assumption to allow for heterogeneous trends as described in . This requires adding extra parameters to the ETWFE estimator, but FETWFE is well-poised to adapt to this setting because it can automatically restrict to 0 any parameters that are added unnecessarily.
## Acknowledgement
I thank Jacob Bien for very helpful discussions that shaped the direction of this work and edits to an early draft.
## Funding
This work was supported by the University of Southern California.
## References
  * Abadie [2005] A. Abadie.  Semiparametric difference-in-differences estimators.  _The review of economic studies_ , 72(1):1–19, 2005. 
  * Abadie and Imbens [2006] A. Abadie and G. W. Imbens.  Large sample properties of matching estimators for average treatment effects.  _econometrica_ , 74(1):235–267, 2006. 
  * Abrevaya et al. [2015] J. Abrevaya, Y.-C. Hsu, and R. P. Lieli.  Estimating conditional average treatment effects.  _Journal of Business & Economic Statistics_, 33(4):485–505, 2015. 
  * Arkhangelsky and Imbens [2023] D. Arkhangelsky and G. Imbens.  Causal models for longitudinal and panel data: A survey.  Technical report, National Bureau of Economic Research, 2023. 
  * Ascarza [2018] E. Ascarza.  Retention futility: Targeting high-risk customers might be ineffective.  _Journal of Marketing Research_ , 55(1):80–98, 2018. 
  * Athey and Imbens [2016] S. Athey and G. Imbens.  Recursive partitioning for heterogeneous causal effects: Table 1.  _Proceedings of the National Academy of Sciences_ , 113(27):7353–7360, May 2016.  doi: 10.1073/pnas.1510489113. 
  * Athey and Imbens [2006] S. Athey and G. W. Imbens.  Identification and inference in nonlinear difference-in-differences models.  _Econometrica_ , 74(2):431–497, 2006. 
  * Athey and Imbens [2022] S. Athey and G. W. Imbens.  Design-based analysis in difference-in-differences settings with staggered adoption.  _Journal of Econometrics_ , 226(1):62–79, 2022. 
  * Athey et al. [2018] S. Athey, D. Blei, R. Donnelly, F. Ruiz, and T. Schmidt.  Estimating Heterogeneous Consumer Preferences for Restaurants and Travel Time Using Mobile Location Data.  _ArXiv e-prints_ , Jan. 2018. 
  * Ban and Kedagni [2022] K. Ban and D. Kedagni.  Generalized difference-in-differences models: Robust bounds.  _arXiv preprint arXiv:2211.06710_ , 2022. 
  * Belloni et al. [2014] A. Belloni, V. Chernozhukov, and C. Hansen.  Inference on treatment effects after selection among high-dimensional controls.  _The Review of Economic Studies_ , 81(2):608–650, 2014. 
  * Bien [2016] J. Bien.  The simulator: an engine to streamline simulations.  _arXiv preprint arXiv:1607.00021_ , 2016. 
  * Bilinski and Hatfield [2018] A. Bilinski and L. A. Hatfield.  Seeking evidence of absence: Reconsidering tests of model assumptions.  _arXiv preprint arXiv:1805.03273_ , 2018. 
  * Blackwell et al. [2024] M. Blackwell, A. Glynn, H. Hilbig, and C. H. Phillips.  Estimating controlled direct effects with panel data: An application to reducing support for discriminatory policies.  2024. 
  * Bolger et al. [2019] N. Bolger, K. S. Zee, M. Rossignac-Milon, and R. R. Hassin.  Causal processes in psychology are heterogeneous.  _Journal of Experimental Psychology: General_ , 148(4):601, 2019. 
  * Borusyak and Jaravel [2018] K. Borusyak and X. Jaravel.  _Revisiting event study designs_.  SSRN, 2018. 
  * Borusyak et al. [2024] K. Borusyak, X. Jaravel, and J. Spiess.  Revisiting event study designs: Robust and efficient estimation.  _Review of Economic Studies_ , page rdae007, 2024. 
  * Bottou et al. [2013] L. Bottou, J. Peters, J. Quiñonero-Candela, D. X. Charles, D. M. Chickering, E. Portugaly, D. Ray, P. Simard, and E. Snelson.  Counterfactual reasoning and learning systems: The example of computational advertising.  _Journal of Machine Learning Research_ , 14(11), 2013. 
  * Brand and Xie [2010] J. E. Brand and Y. Xie.  Who benefits most from college? evidence for negative selection in heterogeneous economic returns to higher education.  _American sociological review_ , 75(2):273–302, 2010. 
  * Breen et al. [2015] R. Breen, S. Choi, and A. Holm.  Heterogeneous causal effects and sample selection bias.  _Sociological Science_ , 2:351–369, 2015. 
  * Breheny and Huang [2009] P. Breheny and J. Huang.  Penalized methods for bi-level variable selection.  _Statistics and its interface_ , 2(3):369, 2009. 
  * Breheny and Huang [2015] P. Breheny and J. Huang.  Group descent algorithms for nonconvex penalized linear and logistic regression models with grouped predictors.  _Statistics and Computing_ , 25:173–187, 2015. 
  * Brown and Butts [2022] N. Brown and K. Butts.  A Unified Framework for Dynamic Treatment Effect Estimation in Interactive Fixed Effect Models.  Working Paper 1495, Economics Department, Queen’s University, Nov. 2022. 
  * Caetano et al. [2022] C. Caetano, B. Callaway, S. Payne, and H. S. Rodrigues.  Difference in differences with time-varying covariates.  _arXiv preprint arXiv:2202.02903_ , 2022. 
  * Callaway [2023] B. Callaway.  _Handbook of Labor, Human Resources and Population Economics_ , chapter Difference-in-Differences for Policy Evaluation, pages 1–61.  Springer International Publishing, 2023. 
  * Callaway and Karami [2023] B. Callaway and S. Karami.  Treatment effects in interactive fixed effects models with a small number of time periods.  _Journal of Econometrics_ , 233(1):184–208, 2023. 
  * Callaway and Li [2023] B. Callaway and T. Li.  Policy evaluation during a pandemic.  _Journal of Econometrics_ , 236(1):105454, 2023. 
  * Callaway and Sant’Anna [2022] B. Callaway and P. Sant’Anna.  Pre–testing in a did setup using the did package.  _Vignette URL https://bcallaway11. github. io/did/articles/pre-testing. html_ , 2022. 
  * Callaway and Sant’Anna [2021] B. Callaway and P. H. Sant’Anna.  Difference-in-differences with multiple time periods.  _Journal of Econometrics_ , 225(2):200–230, 2021. 
  * Candès et al. [2018] E. Candès, Y. Fan, L. Janson, and J. Lv.  Panning for gold: ‘model-X’ knockoffs for high dimensional controlled variable selection.  _Journal of the Royal Statistical Society. Series B: Statistical Methodology_ , 80(3):551–577, 2018.  ISSN 14679868.  doi: 10.1111/rssb.12265.  URL <https://rss-onlinelibrary-wiley-com.libproxy1.usc.edu/doi/pdf/10.1111/rssb.12265>. 
  * Chang [2020] N.-C. Chang.  Double/debiased machine learning for difference-in-differences models.  _The Econometrics Journal_ , 23(2):177–191, 2020. 
  * Chernozhukov et al. [2023] V. Chernozhukov, D. Chetverikov, and Y. Koike.  Nearly optimal central limit theorem and bootstrap approximations in high dimensions.  _The Annals of Applied Probability_ , 33(3):2374–2425, 2023. 
  * Chiu et al. [2023] A. Chiu, X. Lan, Z. Liu, and Y. Xu.  What to do (and not to do) with causal panel analysis under parallel trends: Lessons from a large reanalysis study.  _Available at SSRN 4490035_ , 2023. 
  * Cinelli et al. [2022] C. Cinelli, A. Forney, and J. Pearl.  A crash course in good and bad controls.  _Sociological Methods & Research_, page 00491241221099552, 2022. 
  * DasGupta [2008] A. DasGupta.  _Asymptotic theory of statistics and probability_ , volume 180.  Springer, 2008. 
  * De Chaisemartin and d’Haultfoeuille [2020] C. De Chaisemartin and X. d’Haultfoeuille.  Two-way fixed effects estimators with heterogeneous treatment effects.  _American Economic Review_ , 110(9):2964–2996, 2020. 
  * de Chaisemartin and D’Haultfœuille [2023] C. de Chaisemartin and X. D’Haultfœuille.  Difference-in-differences for simple and complex natural experiments.  _Available at SSRN_ , November 2023. 
  * De Chaisemartin and d’Haultfoeuille [2023] C. De Chaisemartin and X. d’Haultfoeuille.  Two-way fixed effects and differences-in-differences with heterogeneous treatment effects: A survey.  _The Econometrics Journal_ , 26(3):C1–C30, 2023. 
  * Dette [2020] H. Dette.  Difference-in-differences estimation under non-parallel trends.  2020. 
  * Ding [2024] P. Ding.  Linear model and extensions.  _arXiv preprint arXiv:2401.00649_ , 2024. 
  * Ding and Li [2019] P. Ding and F. Li.  A bracketing relationship between difference-in-differences and lagged-dependent-variable adjustment.  _Political Analysis_ , 27(4):605–615, 2019. 
  * Djebbari and Smith [2008] H. Djebbari and J. Smith.  Heterogeneous impacts in progresa.  _Journal of Econometrics_ , 145(1-2):64–80, 2008. 
  * Ellickson et al. [2022] P. B. Ellickson, W. Kar, and J. C. Reeder III.  Estimating marketing component effects: Double machine learning from targeted digital promotions.  _Marketing Science_ , 2022. 
  * Faletto [2025] G. Faletto.  _fetwfe: Fused Extended Two-Way Fixed Effects_ , 2025.  R package version 0.4.4. 
  * Fan and Li [2001] J. Fan and R. Li.  Variable selection via nonconcave penalized likelihood and its oracle properties.  _Journal of the American statistical Association_ , 96(456):1348–1360, 2001. 
  * Fan and Peng [2004] J. Fan and H. Peng.  Nonconcave penalized likelihood with a diverging number of parameters.  2004. 
  * Fan et al. [2022] Q. Fan, Y.-C. Hsu, R. P. Lieli, and Y. Zhang.  Estimation of conditional average treatment effects with high-dimensional data.  _Journal of Business & Economic Statistics_, 40(1):313–327, 2022. 
  * Farrell [2015] M. H. Farrell.  Robust inference on average treatment effects with possibly more covariates than observations.  _Journal of Econometrics_ , 189(1):1–23, 2015. 
  * Farrell et al. [2021] M. H. Farrell, T. Liang, and S. Misra.  Deep neural networks for estimation and inference.  _Econometrica_ , 89(1):181–213, 2021. 
  * Feinstein et al. [1972] A. R. Feinstein, J. F. Rubinstein, and W. A. Ramshaw.  Estimating prognosis with the aid of a conversational-mode computer program.  _Annals of Internal Medicine_ , 76(6):911–921, 1972. 
  * Flack and Jee [2020] E. Flack and E. Jee.  _bacondecomp: Goodman-Bacon Decomposition_ , 2020.  R package version 0.1.1. 
  * Florens et al. [2008] J.-P. Florens, J. J. Heckman, C. Meghir, and E. Vytlacil.  Identification of treatment effects using control functions in models with continuous, endogenous treatment and heterogeneous effects.  _Econometrica_ , 76(5):1191–1206, 2008. 
  * Foster et al. [2011] J. C. Foster, J. M. Taylor, and S. J. Ruberg.  Subgroup identification from randomized clinical trial data.  _Statistics in medicine_ , 30(24):2867–2880, 2011. 
  * Frankovich et al. [2011] J. Frankovich, C. A. Longhurst, and S. M. Sutherland.  Evidence-based medicine in the emr era.  _N Engl J Med_ , 365(19):1758–1759, 2011. 
  * Gao et al. [2022] W. Gao, F. Xu, and Z.-H. Zhou.  Towards convergence rate analysis of random forests for classification.  _Artificial Intelligence_ , 313:103788, 2022. 
  * García-Portugués [2023] E. García-Portugués.  _Notes for Nonparametric Statistics_.  2023.  Version 6.9.0. ISBN 978-84-09-29537-1. 
  * Gavrilova et al. [2023] E. Gavrilova, A. Langørgen, and F. Zoutman.  Dynamic causal forests, with an application to payroll tax incidence in norway, 2023. 
  * Gelman and Vákár [2021] A. Gelman and M. Vákár.  Slamming the sham: A bayesian model for adaptive adjustment with noisy control data.  _Statistics in medicine_ , 40(15):3403–3424, 2021. 
  * Ghanem et al. [2024] D. Ghanem, P. H. C. Sant’Anna, and K. Wüthrich.  Selection and parallel trends, 2024. 
  * Goodman-Bacon [2021] A. Goodman-Bacon.  Difference-in-differences with variation in treatment timing.  _Journal of Econometrics_ , 225(2):254–277, 2021. 
  * Grimmer et al. [2017] J. Grimmer, S. Messing, and S. J. Westwood.  Estimating heterogeneous treatment effects and the effects of heterogeneous treatments with ensemble methods.  _Political Analysis_ , 25(4):413–434, 2017. 
  * Ham and Miratrix [2022] D. W. Ham and L. Miratrix.  Benefits and costs of matching prior to a difference in differences analysis when parallel trends does not hold.  _arXiv preprint arXiv:2205.08644_ , 2022. 
  * Hastie et al. [2015] T. Hastie, R. Tibshirani, and M. Wainwright.  Statistical learning with sparsity.  _Monographs on statistics and applied probability_ , 143:143, 2015. 
  * Hatamyar et al. [2023] J. Hatamyar, N. Kreif, R. Rocha, and M. Huber.  Machine learning for staggered difference-in-differences and dynamic treatment effect heterogeneity, 2023. 
  * Heckman et al. [1997] J. J. Heckman, H. Ichimura, and P. E. Todd.  Matching as an econometric evaluation estimator: Evidence from evaluating a job training programme.  _The review of economic studies_ , 64(4):605–654, 1997. 
  * Heckman et al. [1998] J. J. Heckman, H. Ichimura, J. A. Smith, and P. E. Todd.  Characterizing selection bias using experimental data, 1998. 
  * Heiler and Mareckova [2021] P. Heiler and J. Mareckova.  Shrinkage for categorical regressors.  _Journal of Econometrics_ , 223(1):161–189, 2021. 
  * Henderson and Sperlich [2022] D. J. Henderson and S. Sperlich.  A complete framework for model-free difference-in-differences estimation.  _Available at SSRN: https://ssrn.com/abstract=4300808_ , (No. 15799), 2022. 
  * Hitsch and Misra [2018] G. J. Hitsch and S. Misra.  Heterogeneous treatment effects and optimal targeting policy evaluation.  _Available at SSRN 3111957_ , 2018. 
  * Höfling et al. [2010] H. Höfling, H. Binder, and M. Schumacher.  A coordinate-wise optimization algorithm for the fused lasso.  _arXiv preprint arXiv:1011.6409_ , 2010. 
  * Huang et al. [2008] J. Huang, J. L. Horowitz, and S. Ma.  Asymptotic properties of bridge estimators in sparse high-dimensional regression models.  _Annals of Statistics_ , 36(2):587–613, 2008.  ISSN 00905364.  doi: 10.1214/009053607000000875. 
  * Imai and Kim [2019] K. Imai and I. S. Kim.  When should we use unit fixed effects regression models for causal inference with longitudinal data?  _American Journal of Political Science_ , 63(2):467–490, 2019. 
  * Imai and Strauss [2011] K. Imai and A. Strauss.  Estimation of heterogeneous treatment effects from randomized experiments, with application to the optimal planning of the get-out-the-vote campaign.  _Political Analysis_ , 19(1):1–19, 2011. 
  * Imai et al. [2023] K. Imai, I. S. Kim, and E. H. Wang.  Matching methods for causal inference with time-series cross-sectional data.  _American Journal of Political Science_ , 67(3):587–605, 2023. 
  * Imbens and Rubin [2015] G. Imbens and D. B. Rubin.  _Causal inference for statistics, social, and biomedical sciences: an introduction_.  Cambridge University Press, 2015. 
  * Imbens and Wooldridge [2009] G. W. Imbens and J. M. Wooldridge.  Recent developments in the econometrics of program evaluation.  _Journal of economic literature_ , 47(1):5–86, 2009. 
  * Kaddoura and Westerlund [2022] Y. Kaddoura and J. Westerlund.  Estimation of panel data models with random interactive effects and multiple structural breaks when t is fixed.  _Journal of Business & Economic Statistics_, pages 1–13, 2022. 
  * Kahn-Lang and Lang [2020] A. Kahn-Lang and K. Lang.  The promise and pitfalls of differences-in-differences: Reflections on 16 and pregnant and other applications.  _Journal of Business & Economic Statistics_, 38(3):613–620, 2020. 
  * Kock [2013] A. B. Kock.  Oracle efficient variable selection in random and fixed effects panel data models.  _Econometric Theory_ , 29(1):115–152, 2013. 
  * Kwon [2021] S. Kwon.  Optimal shrinkage estimation of fixed effects in linear panel data models.  _EliScholar–A Digital Platform for Scholarly Publishing at Yal e_ , page 1, 2021. 
  * Lechner et al. [2011] M. Lechner et al.  The estimation of causal effects by difference-in-difference methods.  _Foundations and Trends® in Econometrics_ , 4(3):165–224, 2011. 
  * Lehmann [1999] E. L. Lehmann.  _Elements of large-sample theory_.  Springer, 1999. 
  * Lesko [2007] L. Lesko.  Personalized medicine: elusive dream or imminent reality?  _Clinical Pharmacology & Therapeutics_, 81(6):807–816, 2007. 
  * Levy and Abramovich [2023] T. Levy and F. Abramovich.  Generalization error bounds for multiclass sparse linear classifiers.  _Journal of Machine Learning Research_ , 24(151):1–35, 2023. 
  * Lindner and McConnell [2019] S. Lindner and K. J. McConnell.  Difference-in-differences and matching on outcomes: a tale of two unobservables.  _Health Services and Outcomes Research Methodology_ , 19:127–144, 2019. 
  * Lopes [2022] M. E. Lopes.  Central limit theorem and bootstrap approximation in high dimensions: Near 1/n rates via implicit smoothing.  _The Annals of Statistics_ , 50(5):2492–2513, 2022. 
  * Manski and Pepper [2018] C. F. Manski and J. V. Pepper.  How do right-to-carry laws affect crime rates? coping with ambiguity using bounded-variation assumptions.  _Review of Economics and Statistics_ , 100(2):232–244, 2018. 
  * Marchenko and Pastur [1967] V. A. Marchenko and L. A. Pastur.  Distribution of eigenvalues for some sets of random matrices.  _Matematicheskii Sbornik_ , 114(4):507–536, 1967. 
  * Marx et al. [2024] P. Marx, E. Tamer, and X. Tang.  Parallel trends and dynamic choices.  _Journal of Political Economy Microeconomics_ , 2(1):129–171, 2024. 
  * McCullagh [1980] P. McCullagh.  Regression models for ordinal data.  _Journal of the Royal Statistical Society: Series B (Methodological)_ , 42(2):109–127, 1980. 
  * Meinshausen and Bühlmann [2006] N. Meinshausen and P. Bühlmann.  High-dimensional graphs and variable selection with the lasso.  2006. 
  * Meyer [1995] B. D. Meyer.  Natural and quasi-experiments in economics.  _Journal of business & economic statistics_, 13(2):151–161, 1995. 
  * Morgan [2001] S. L. Morgan.  Counterfactuals, causal effect heterogeneity, and the catholic school effect on learning.  _Sociology of education_ , pages 341–374, 2001. 
  * Murphy et al. [2016] M. Murphy, S. Redding, and J. Twyman.  _Handbook on personalized learning for states, districts, and schools_.  IAP, 2016. 
  * Negahban et al. [2012] S. N. Negahban, P. Ravikumar, M. J. Wainwright, and B. Yu.  A unified framework for high-dimensional analysis of m-estimators with decomposable regularizers.  2012. 
  * Nie and Wager [2021] X. Nie and S. Wager.  Quasi-oracle estimation of heterogeneous treatment effects.  _Biometrika_ , 108(2):299–319, 2021. 
  * Nie et al. [2021] X. Nie, C. Lu, and S. Wager.  Nonparametric heterogeneous treatment effect estimation in repeated cross sectional designs, 2021. 
  * Park and Tchetgen [2023] C. Park and E. T. Tchetgen.  A universal difference-in-differences approach for causal inference.  2023. 
  * Park and Yamauchi [2023] J. H. Park and S. Yamauchi.  Change-point detection and regularization in time series cross-sectional data analysis.  _Political Analysis_ , 31(2):257–277, 2023. 
  * Pesaran [2015] M. H. Pesaran.  _Time Series and Panel Data Econometrics_.  Number 9780198759980 in OUP Catalogue. Oxford University Press, 2015.  URL <https://ideas.repec.org/b/oxp/obooks/9780198759980.html>. 
  * Rambachan and Roth [2023] A. Rambachan and J. Roth.  A more credible approach to parallel trends.  _Review of Economic Studies_ , page rdad018, 2023. 
  * Roth [2022] J. Roth.  Pretest with caution: Event-study estimates after testing for parallel trends.  _American Economic Review: Insights_ , 4(3):305–322, 2022. 
  * Roth and Sant’Anna [2023] J. Roth and P. H. Sant’Anna.  When is parallel trends sensitive to functional form?  _Econometrica_ , 91(2):737–747, 2023. 
  * Roth et al. [2023] J. Roth, P. H. Sant’Anna, A. Bilinski, and J. Poe.  What’s trending in difference-in-differences? a synthesis of the recent econometrics literature.  _Journal of Econometrics_ , 2023. 
  * Rothbard et al. [2023] S. Rothbard, J. C. Etheridge, and E. J. Murray.  A tutorial on applying the difference-in-differences method to health data.  _Current Epidemiology Reports_ , pages 1–11, 2023. 
  * Rutanen [2021] K. Rutanen.  The smallest non-zero singular value of ab.  Mathematics Stack Exchange, 2021.  URL:https://math.stackexchange.com/q/4018462 (version: 2021-02-10). 
  * Sant’Anna and Zhao [2020] P. H. Sant’Anna and J. Zhao.  Doubly robust difference-in-differences estimators.  _Journal of Econometrics_ , 219(1):101–122, 2020. 
  * Semenova and Chernozhukov [2021] V. Semenova and V. Chernozhukov.  Debiased machine learning of conditional average treatment effects and other causal functions.  _The Econometrics Journal_ , 24(2):264–289, 2021. 
  * Shaikh and Toulis [2021] A. M. Shaikh and P. Toulis.  Randomization tests in observational studies with staggered adoption of treatment.  _Journal of the American Statistical Association_ , 116(536):1835–1848, 2021. 
  * Stevenson and Wolfers [2006] B. Stevenson and J. Wolfers.  Bargaining in the shadow of the law: Divorce laws and family distress.  _The Quarterly Journal of Economics_ , 121(1):267–288, 2006. 
  * Sun and Abraham [2021] L. Sun and S. Abraham.  Estimating dynamic treatment effects in event studies with heterogeneous treatment effects.  _Journal of Econometrics_ , 225(2):175–199, 2021. 
  * Sun and Shapiro [2022] L. Sun and J. M. Shapiro.  A linear panel model with heterogeneous coefficients and variation in exposure.  _Journal of Economic Perspectives_ , 36(4):193–204, 2022. 
  * Tibshirani [1996] R. Tibshirani.  Regression Shrinkage and Selection via the Lasso.  _Journal of the Royal Statistical Society. Series B: Statistical Methodology_ , 58(1):267–288, 1996. 
  * Tibshirani et al. [2005] R. Tibshirani, M. Saunders, S. Rosset, J. Zhu, and K. Knight.  Sparsity and smoothness via the fused lasso.  _Journal of the Royal Statistical Society: Series B (Statistical Methodology)_ , 67(1):91–108, 2005. 
  * Tibshirani and Taylor [2011] R. J. Tibshirani and J. Taylor.  The solution path of the generalized lasso.  _The annals of statistics_ , 39(3):1335–1371, 2011. 
  * van der Vaart [2000] A. van der Vaart.  _Asymptotic Statistics_.  Asymptotic Statistics. Cambridge University Press, 2000.  ISBN 9780521784504. 
  * Vershynin [2012] R. Vershynin.  _Introduction to the non-asymptotic analysis of random matrices_ , pages 210–268.  Cambridge University Press, 2012.  doi: 10.1017/CBO9780511794308.006. 
  * Vershynin [2018] R. Vershynin.  _High-Dimensional Probability: An Introduction with Applications in Data Science_.  Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, 2018.  ISBN 9781108415194. 
  * Viallon et al. [2013] V. Viallon, S. Lambert-Lacroix, H. Höfling, and F. Picard.  Adaptive generalized fused-lasso: Asymptotic properties and applications.  2013. 
  * Viallon et al. [2016] V. Viallon, S. Lambert-Lacroix, H. Hoefling, and F. Picard.  On the robustness of the generalized fused lasso to prior specifications.  _Statistics and Computing_ , 26(1-2):285–301, 2016. 
  * Wager and Athey [2018] S. Wager and S. Athey.  Estimation and Inference of Heterogeneous Treatment Effects using Random Forests.  _Journal of the American Statistical Association_ , 113(523):1228–1242, jul 2018.  ISSN 0162-1459.  doi: 10.1080/01621459.2017.1319839.  URL <https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839>. 
  * Wasserman [2004] L. Wasserman.  _All of statistics: a concise course in statistical inference_ , volume 26.  Springer, 2004. 
  * Wing et al. [2018] C. Wing, K. Simon, and R. A. Bello-Gomez.  Designing difference in difference studies: Best practices for public health policy research.  _Annual Review of Public Health_ , 39(1):453–469, 2018.  doi: 10.1146/annurev-publhealth-040617-013507.  URL <https://doi.org/10.1146/annurev-publhealth-040617-013507>.  PMID: 29328877. 
  * Winkelbeiner et al. [2019] S. Winkelbeiner, S. Leucht, J. M. Kane, and P. Homan.  Evaluation of differences in individual treatment response in schizophrenia spectrum disorders: a meta-analysis.  _JAMA psychiatry_ , 76(10):1063–1073, 2019. 
  * Wooldridge [2021] J. M. Wooldridge.  Two-way fixed effects, the two-way mundlak regression, and difference-in-differences estimators.  _Available at SSRN 3906345_ , 2021. 
  * Xie et al. [2012] Y. Xie, J. E. Brand, and B. Jann.  Estimating heterogeneous treatment effects with observational data.  _Sociological methodology_ , 42(1):314–347, 2012. 
  * Xin et al. [2014] B. Xin, Y. Kawahara, Y. Wang, and W. Gao.  Efficient generalized fused lasso and its application to the diagnosis of alzheimer’s disease.  In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 28, 2014. 
  * Yang et al. [2020] J. Yang, D. Eckles, P. Dhillon, and S. Aral.  Targeting for long-term outcomes.  _arXiv preprint arXiv:2010.15835_ , 2020. 
  * Zhao and Yu [2006] P. Zhao and B. Yu.  On Model Selection Consistency of Lasso.  _Journal of Machine Learning Research_ , 7:2541–2563, 2006. 
  * Zimmert [2020] M. Zimmert.  Efficient difference-in-differences estimation with high-dimensional common trend confounding, 2020. 
  * Zou [2006] H. Zou.  The adaptive lasso and its oracle properties.  _J. Amer. Statist. Assoc._ , 101(476):1418–1429, 2006.  ISSN 0162-1459. 
  * Zou and Hastie [2005] H. Zou and T. Hastie.  Regularization and variable selection via the elastic net.  _Journal of the royal statistical society: series B (statistical methodology)_ , 67(2):301–320, 2005. 


In Appendix we investigate in more detail the relatively plausibility of conditional and marginal parallel trends, how Assumption (CIUN) interacts with these assumptions, and how we can achieve consistency of FETWFE for marginal average treatment effects under either form of parallel trends if (CIUN) holds. We present additional results from the simulation studies from Section in Appendix . We state Theorem , which Theorems , , , and all depend on, in Appendix , and we prove it in Appendix . In Appendix we state Theorem , which shows the consistency of FETWFE for broader classes of estimators than those considered in Theorem . Theorem similarly extends Theorem , and Theorem shows that two classes of conditional treatment effect estimators are asymptotically Gaussian and provides consistent variance estimators for constructing asymptotically valid confidence intervals, analogously to Theorem for marginal treatment effect estimators. We prove the results from Appendix , along with Theorems , , and , in Appendix . In Appendix we provide proofs for results that were stated but not proven in Appendix and supporting results used in Appendix . We prove supporting results for Theorem in Appendix . Finally, in Appendix we state and prove some technical lemmas.
## Appendix A More on Parallel Trends and Assumption (CIUN)
In Section we discuss the plausibility of marginal and conditional parallel trends in relation to an assumption of marginal or conditional (respectively) independence of the treatment and the potential outcomes. The latter assumptions imply the former, which motivates difference-in-differences estimators like FETWFE as more plausible than identification strategies that rely on stronger untestable assumptions.
We point out that no such strict hierarchy exists to relate marginal and conditional parallel trends in Section —there exist settings where either assumption can hold without the other holding. Investigating this sheds light on when conditional or marginal parallel trends may be more plausible. We later show (Theorem ) that conditional parallel trends is necessary to estimate conditional average treatment effects via FETWFE, but there are estimators that are consistent for marginal average treatment effects under either conditional or marginal parallel trends. So it is natural to wonder what conditions allow an estimator to be consistent for marginal average treatment effects under either assumption, and whether this might be possible for FETWFE. In Section we explore when this might hold, ultimately investigating Assumption (CIUN) in more detail and showing how it affects the consistency of regression (). These ideas underlie the proof of Theorem .
### A.1 Relationships Between Unconfounded Treatment Assignment, Marginally Randomized Treatment Assignment, and Parallel Trends Assumptions
In this section we present Proposition , which relates the plausibility of several assumptions on the potential outcomes. Ultimately, this motivates difference-in-differences estimators like FETWFE as having more plausible untestable assumptions than methods that rely on unconfoundedness (like matching) or randomized assignment.
We consider an assumption of unconfoundedness of the untreated potential outcomes, which we define in this setting as  
| |,.\tilde{y}_{it}(0)\mathchoice{\mathrel{\hbox to0.0pt{$\displaystyle\perp$\hss}\mkern 2.0mu{\displaystyle\perp}}}{\mathrel{\hbox to0.0pt{$\textstyle\perp$\hss}\mkern 2.0mu{\textstyle\perp}}}{\mathrel{\hbox to0.0pt{$\scriptstyle\perp$\hss}\mkern 2.0mu{\scriptstyle\perp}}}{\mathrel{\hbox to0.0pt{$\scriptscriptstyle\perp$\hss}\mkern 2.0mu{\scriptscriptstyle\perp}}}W_{i}\ \mid\ \boldsymbol{X}_{i}\qquad i\in[N],t\in\\{1,\ldots,T\\}.  |  
| --- |  
This requires that treatment assignment is independent of the untreated potential outcomes conditional on the time-invariant (pre-treatment) covariates \boldsymbol{X}_{i}. considered a similar assumption in the case. As we discuss in more detail later in this section, it has previously been noted that the assumption of unconfoundedness [[Imbens and Rubin, 2015](https://arxiv.org/html/2312.05985#bib.bib75), Def. 3.6] that is common in causal inference is a stronger assumption than (CCTS).
Sometimes unconfoundedness is taken to be a stronger assumption that all of the potential outcomes are independent of treatment status conditional on covariates; such a condition would imply (). [Shaikh and Toulis [2021, Assumption 2.1]](https://arxiv.org/html/2312.05985#bib.bib109) is one example of a work that uses such a stronger assumption in a setting with continuous treatment status.
We note that different notions of unconfoundedness have been considered in the difference-in-differences setting. For example, [Imbens and Wooldridge [2009, Section 6.5.4]](https://arxiv.org/html/2312.05985#bib.bib76) consider unconfoundedness in a two-period difference-in-differences setting with no covariates, where the lagged outcome is conditioned on rather than covariates. Similarly, consider unconfoundedness conditional on several lagged time-varying variables. and consider an ignorability condition where the untreated potential outcome is independent of treatment assignment status conditional on covariates and the lagged outcome.
We also refer to an assumption of marginal independence of the untreated potential outcomes and treatment status,  
| ,.\tilde{y}_{it}(0)\mathchoice{\mathrel{\hbox to0.0pt{$\displaystyle\perp$\hss}\mkern 2.0mu{\displaystyle\perp}}}{\mathrel{\hbox to0.0pt{$\textstyle\perp$\hss}\mkern 2.0mu{\textstyle\perp}}}{\mathrel{\hbox to0.0pt{$\scriptstyle\perp$\hss}\mkern 2.0mu{\scriptstyle\perp}}}{\mathrel{\hbox to0.0pt{$\scriptscriptstyle\perp$\hss}\mkern 2.0mu{\scriptscriptstyle\perp}}}W_{i}\qquad i\in[N],t\in\\{1,\ldots,T\\}.  |  
| --- |  
Completely randomized assignment, an assumption considered by, for example, [Athey and Imbens [2022, Assumption 1]](https://arxiv.org/html/2312.05985#bib.bib8) and [Roth and Sant’Anna [2023, Case 1]](https://arxiv.org/html/2312.05985#bib.bib103), is sufficient for ().
Lastly, we say that (CCTS) or (CTS) holds in a transformation-invariant sense if () or () (respectively) hold under any transformation h(\tilde{y}_{it}(r)) of the potential outcomes such that the expectations exist. Some works have expressed concern that parallel trends assumptions are fragile because they may hold only if the response is not transformed, or only under a specific transformation [[Meyer, 1995](https://arxiv.org/html/2312.05985#bib.bib92), [Athey and Imbens, 2006](https://arxiv.org/html/2312.05985#bib.bib7), [Kahn-Lang and Lang, 2020](https://arxiv.org/html/2312.05985#bib.bib78)]. investigate in detail when marginal parallel trends assumptions are transformation-invariant in the setting, and they point out that their arguments extend straightforwardly to the arbitrary setting as well as to conditional parallel trends assumptions. Proposition explicitly draws attention to an implication of their Propositions 3.1 and 3.2 when extended to these more general settings.
###### Proposition A.1.
Assume that the conditional expectations in () and () exist and are finite.
  1. (a)
Unconfoundedness of the untreated potential outcomes is a strictly stronger assumption than conditional parallel trends: () is sufficient for Assumption CCTS to hold in a transformation-invariant sense, but (CCTS) does not imply unconfoundedness of the untreated potential outcomes.
  2. (b)
Marginal independence of treatment status and the untreated potential outcomes is a strictly stronger assumption than marginal parallel trends: () is sufficient for Assumption CTS to hold in a transformation-invariant sense, but (CTS) does not imply ().


###### Proof.
  1. (a)
Under unconfoundedness (), for any , t\in\\{1,\ldots,T\\},r\in\mathcal{R} almost surely it holds that  
| 𝔼[h∣,]=𝔼[h∣]\mathbb{E}[h(\tilde{y}_{(it)}(0))\mid W_{i},\boldsymbol{X}_{i}]=\mathbb{E}[h(\tilde{y}_{(it)}(0))\mid\boldsymbol{X}_{i}]  |  
| --- |  
for any arbitrary such that the above expectation exists. For (CCTS) to hold with transformational invariance, we need that for any such that the expectations exist and any , t\in\\{2,\ldots,T\\},r\in\mathcal{R}, almost surely  
| 𝔼[h−h∣,]=𝔼[h−h∣].\mathbb{E}[h(\tilde{y}_{(it)}(0))-h(\tilde{y}_{(i1)}(0))\mid W_{i},\boldsymbol{X}_{i}]=\mathbb{E}[h(\tilde{y}_{(it)}(0))-h(\tilde{y}_{(i1)}(0))\mid\boldsymbol{X}_{i}].  |  
| --- |  
But this is immediate from (). We show that the reverse implication does not hold by constructing a counterexample. (CCTS) can be rearranged as  
| 𝔼[∣,]−𝔼=𝔼[∣,]−𝔼∀t∈.\mathbb{E}[\tilde{y}_{(it)}(0)\mid W_{i},\boldsymbol{X}_{i}]-\mathbb{E}[\tilde{y}_{(it)}(0)\mid\boldsymbol{X}_{i}]=\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i},\boldsymbol{X}_{i}]-\mathbb{E}[\tilde{y}_{(i1)}(0)\mid\boldsymbol{X}_{i}]\penalty\ \forall t\in[T].  |  
| --- |  
Suppose that for all t\in\\{1,2,\ldots,T\\}, it holds that 𝔼[∣,]−𝔼=f\mathbb{E}[\tilde{y}_{(it)}(0)\mid W_{i},\boldsymbol{X}_{i}]-\mathbb{E}[\tilde{y}_{(it)}(0)\mid\boldsymbol{X}_{i}]=f(W_{i},\boldsymbol{X}_{i}) almost surely for an arbitrary f(\cdot,\cdot) that is not identically 0. Then from () we see that (CCTS) will hold. However, unconfoundedness does not hold since () with as the identity function is violated.
  2. (b)
This proof is similar. Under marginal independence of treatment status and the untreated potential outcomes (), for any , t\in\\{1,\ldots,T\\},r\in\mathcal{R} almost surely it holds that  
| 𝔼⁡[∣]=𝔼⁡[]\mathbb{E}[h(\tilde{y}_{(it)}(0))\mid W_{i}]=\mathbb{E}[h(\tilde{y}_{(it)}(0))]  |  
| --- |  
for any such that the expectation holds. It is immediate that for any such and any , t\in\\{2,\ldots,T\\},r\in\mathcal{R}, almost surely 𝔼⁡[−∣]=𝔼⁡[−]\mathbb{E}[h(\tilde{y}_{(it)}(0))-h(\tilde{y}_{(i1)}(0))\mid W_{i}]=\mathbb{E}[h(\tilde{y}_{(it)}(0))-h(\tilde{y}_{(i1)}(0))], so (CTS) holds with transformational invariance. On the other hand, (CTS) can be rearranged as  
| 𝔼⁡−=𝔼⁡−.\mathbb{E}[\tilde{y}_{(it)}(0)\mid W_{i}]-\mathbb{E}[\tilde{y}_{(it)}(0)]=\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}]-\mathbb{E}[\tilde{y}_{(i1)}(0)]\qquad\forall t\in[T].  |  
| --- |  
If for all t\in\\{1,2,\ldots,T\\} we have 𝔼⁡−=\mathbb{E}[\tilde{y}_{(it)}(0)\mid W_{i}]-\mathbb{E}[\tilde{y}_{(it)}(0)]=g(W_{i}) almost surely for an arbitrary that is not identically 0, then () holds but () with as the identity function is violated. So (CTS) holds but marginal independence of treatment status and the untreated potential outcomes does not.


Since unconfoundedness often seems more plausible than marginal independence of the potential outcomes and treatment status, Proposition (a) and (b) considered together suggest an intuition that conditional parallel trends may often be more plausible than marginal parallel trends. Proposition also points out that difference-in-differences estimators are strictly more robust than identification strategies that rely on assumptions of completely randomized assignment or unconfoundedness.
Proposition (a) verifies that (CCTS) is a strictly weaker assumption than unconfoundedness. As we mentioned earlier, this is a fairly straightforward observation that has been noted before; for example, , and mention that this fact holds in the case. Besides the previously mentioned intuition that Proposition (a) and (b) provide that conditional parallel trends may be more plausible in many applications than marginal parallel trends, we also draw attention to the facts in Proposition (a) to point out explicitly that difference-in-differences identification strategies like FETWFE that rely on (CCTS) are more robust than methods like matching on time-invariant covariates that rely on an unconfoundedness assumption [[Abadie and Imbens, 2006](https://arxiv.org/html/2312.05985#bib.bib2), [Imbens and Rubin, 2015](https://arxiv.org/html/2312.05985#bib.bib75)].
The implications of Proposition (b) are analogous, and previous works have also shown that random or “as good as random" treatment assignment is sufficient for (CTS) [[Athey and Imbens, 2022](https://arxiv.org/html/2312.05985#bib.bib8), [Roth and Sant’Anna, 2023](https://arxiv.org/html/2312.05985#bib.bib103), [Ghanem et al., 2024](https://arxiv.org/html/2312.05985#bib.bib59)].
### A.2 Neither Conditional Nor Marginal Parallel Trends is Strictly More Plausible
Proposition showed that (CTS) and (CCTS) are strictly more plausible than (respectively) marginal independence of the potential outcomes and treatment and unconfoundedness. Theorem shows that (CTS) and (CCTS) cannot be related in this way—neither is strictly more plausible than the other.
###### .
Assume that the conditional expectations in () and () exist and are finite. Neither of conditional parallel trends or marginal parallel trends is a relaxation of the other: Assumption CCTS () (conditional parallel trends) does not imply Assumption CTS () (marginal parallel trends), nor does (CTS) imply (CCTS).
###### Proof.
We prove Theorem by providing counterexamples where one of the parallel trends assumptions holds but the other does not. We briefly provide some intuition behind the counterexamples here in order to highlight some of the relative strengths and weaknesses of each assumption.
(CCTS) may hold when the time-invariant covariates have a time-varying effect on the potential outcomes, while (CTS) will not hold. This seems plausible if external conditions change that change the relevance of some features for the response. and touch on closely related issues, investigating settings where parallel trends does not hold because time-invariant confounders have time-varying effects on the potential outcomes. We also expand on related ideas in Section .
On the other hand, (CCTS) requires parallel trends to hold on every subgroup of \boldsymbol{X}_{i}. But (CTS) can hold if this is violated and parallel trends holds merely on average across the distribution of the covariates. That is, for any r\in\\{0\\}\cup\mathcal{R}, the quantity  
| 𝔼[−∣=r,]−𝔼[−∣]\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]-\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid\boldsymbol{X}_{i}]  |  
| --- |  
may be positive on some subgroups of \boldsymbol{X}_{i} if this is canceled out by () being negative on other subgroups such that () holds after marginalizing the left side of () across the distribution of \boldsymbol{X}_{i} conditional on and the right side across the distribution of \boldsymbol{X}_{i} conditional on . In this particular sense, (CTS) allows for more variation in the time trends across the distribution of \boldsymbol{X}_{i} than (CCTS). We also emphasize that both (CCTS) and (CTS) allow for heterogeneity in the sense that the values of 𝔼[−∣,]\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i},\boldsymbol{X}_{i}] and 𝔼⁡[−∣]\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid\boldsymbol{X}_{i}] may vary with \boldsymbol{X}_{i}.
However, if () does not hold, () holding exactly after marginalizing () appropriately may seem somewhat implausible. Nonetheless, it could be the case that () is not close to holding, but () approximately holds. Relaxing (CTS) by allowing small violations has been considered in recent works on sensitivity analyses for difference-in-differences estimation [[Manski and Pepper, 2018](https://arxiv.org/html/2312.05985#bib.bib87), [Ban and Kedagni, 2022](https://arxiv.org/html/2312.05985#bib.bib10), [Rambachan and Roth, 2023](https://arxiv.org/html/2312.05985#bib.bib101), [Ghanem et al., 2024](https://arxiv.org/html/2312.05985#bib.bib59)]. That said, [Ghanem et al. [2024, Section 7.2]](https://arxiv.org/html/2312.05985#bib.bib59) point out that adjusting for covariates (that is, performing a sensitivity analysis under an assumption of conditional parallel trends that may not hold exactly) can result in smaller plausible ranges of marginal average treatment effects.
Overall, this suggests an intuition that (CCTS) may be somewhat more plausible in practice than (CTS), but since neither assumption nests the other, an ideal estimator would be asymptotically unbiased under either assumption. We discuss this in more detail in the next section.
### A.3 Parallel Trends, Assumption (CIUN), and Regression ()
Theorem motivates interest in estimators of marginal average treatment effects that are consistent under either conditional or marginal parallel trends. We have seen that Assumption (CIUN) is helpful for this—under the conditions of Theorem , FETWFE consistently estimates heterogeneous marginal average treatment effects under either conditional or marginal parallel trends if (CIUN) holds. In this section we fill in more detail about (CIUN), build some intuition for its role in the consistency of difference-in-differences estimators, and establish the consistency of regression () under (CIUN).
We first introduce a related assumption that, unlike (CIUN), is not directly testable.
Assumption (CIU) (“conditional independence of trends of the untreated potential outcomes"): Almost surely for all r\in\\{0\\}\cup\mathcal{R}, , and any t\in\\{2,\ldots,T\\},  
| 𝔼⁡[−∣]=\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r\right]=\penalty\   | 𝔼[−∣=r,].\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}\right].  |  
| --- | --- |  
(CIU) is similar to (CIUN), but it requires the trends in the untreated potential outcomes to be mean-independent of \boldsymbol{X}_{i} for units in all cohorts, so it cannot be tested.
It is immediate that (CCTS) and (CTS) are equivalent under (CIU).
###### Proposition A.3.
Under (CIU), conditional parallel trends (CCTS) and marginal parallel trends (CTS) are equivalent.
###### Proof.
We can apply (CIU) on each side of (CTS) to immediately yield that for any , r\in\mathcal{R}, and any ,  
| 𝔼⁡[−∣]=\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r]=\penalty\   | 𝔼⁡[−∣]\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0]  |  
| --- | --- |  
| ⇔𝔼[−∣=r,]=\displaystyle\iff\qquad\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]=\penalty\   | 𝔼[−∣=0,].\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}].  |  
One implication of Proposition is that any method that consistently estimates the marginal average treatment effects under either (CCTS) or (CTS) is consistent under both assumptions if (CIU) holds.
Although (CIUN) does not guarantee the equivalence of (CTS) and (CCTS), under (CIUN) conditional and marginal parallel trends are still connected.
###### Proposition A.4.
Suppose (CIUN) holds. Then conditional parallel trends (CCTS) implies marginal parallel trends (CTS).
###### Proof.
Applying (CIUN) to one side of (CCTS), we have that for any , r\in\mathcal{R}, and any ,  
| 𝔼[−∣=r,]=\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]=\penalty\   | 𝔼[−∣=0,]\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}]  |  
| --- | --- |  
| ⇔𝔼[−∣=r,]=\displaystyle\iff\qquad\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]=\penalty\   | 𝔼⁡[−∣]\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0]  |  
| ⟹𝔼[𝔼[−∣=r,]∣=r]=\displaystyle\implies\penalty\ \mathbb{E}\left[\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]\mid W_{i}=r\right]=\penalty\   | 𝔼⁡[𝔼⁡[−∣]∣]\displaystyle\mathbb{E}\left[\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0]\mid W_{i}=r\right]  |  
| ⇔𝔼[−∣=r]=\displaystyle\iff\qquad\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r\right]=\penalty\   | 𝔼⁡[−∣].\displaystyle\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0].  |  
Proposition shows that any method that consistently estimates the marginal average treatment effects is also consistent under (CIUN) and (CCTS).
Notice that Assumptions (CIU) and (CIUN) are closely related to the time-varying effects of the covariates on the potential outcomes that we mentioned in our discussion of Theorem in Section , which can cause marginal parallel trends to fail to hold even if conditional parallel trends holds. Proposition shows that (CIUN) rules out such settings.
Next we present a key result that underlies Theorem , and in turn Theorems , , and .
###### (Assumption (CIUN) and the Consistency of Regression ).
Assume that (CNAS) and () from (LINS) hold.
  1. (a)
(Consistency under conditional parallel trends.) The following are equivalent: (i) conditional parallel trends after treatment (CCTSA) holds, and (ii) for all r\in\mathcal{R} and all , for the regression estimands \tau_{rt}^{*} and \boldsymbol{\rho}_{rt}^{*} defined in () it holds almost surely that the conditional average treatment effects \tau_{\text{CATT}}(r,t,\boldsymbol{X}_{i}) defined in () equal +()⊤​\tau_{rt}^{*}+\left(\boldsymbol{X}_{i}-\mathbb{E}\left[\boldsymbol{X}_{i}\mid W_{i}=r\right]\right)^{\top}\boldsymbol{\rho}_{rt}^{*}.
  2. (b)
(Consistency under marginal parallel trends and CIUN.) If (CIUN) and marginal parallel trends after treatment (CTSA) hold, then for all r\in\mathcal{R} and all  the regression estimand \tau_{rt}^{*} is equal to the marginal average treatment effect \tau_{\text{ATT}}(r,t) from ()..
  3. (c)
(Identifying the bias-inducing covariates under CTSA if CIUN does not hold.) Suppose (CTSA) and () from (LINS) hold. Then for any r\in\mathcal{R} and t\in\\{r,\ldots,T\\},  
| =−(−)⊤​.\tau_{rt}^{*}=\tau_{\text{ATT}}(r,t)-\left(\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]-\mathbb{E}\left[\boldsymbol{X}_{i}\mid W_{i}=0\right]\right)^{\top}\boldsymbol{\xi}_{t}^{*}.  |  
| --- |  


###### Proof.
The proofs of parts (b) and (c) are provided in Appendix . Part (a) follows from [Wooldridge [2021, Sections 6.3 and 6.4]](https://arxiv.org/html/2312.05985#bib.bib125), but for illustrative purposes we provide a proof in our notation here. For all r\in\mathcal{R},t\geq r and any 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i},  
| +\displaystyle\tau_{rt}^{*}+\boldsymbol{\dot{x}}_{r}^{\top}\boldsymbol{\rho}_{rt}^{*}  |  
| --- |  
| \displaystyle\stackrel{{\scriptstyle(a)}}{{=}}\penalty\   | 𝔼[−∣=r,=𝒙]−𝔼[−∣=0,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(r)-\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]-\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(b)}}{{=}}\penalty\   | 𝔼[∣=r,=𝒙]−(𝔼[∣=r,=𝒙]\displaystyle\mathbb{E}[\tilde{y}_{(it)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}]-\bigg(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}]  |  
| +𝔼[−∣=0,=𝒙])\displaystyle+\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}]\bigg)  |  
| \displaystyle\stackrel{{\scriptstyle(c)}}{{=}}\penalty\   | 𝔼[∣=r,=𝒙]−𝔼[∣=r,=𝒙]\displaystyle\mathbb{E}[\tilde{y}_{(it)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}]-\mathbb{E}[\tilde{y}_{(it)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}]  |  
| =\displaystyle=\penalty\   | \displaystyle\tau_{\text{CATT}}(r,t,\boldsymbol{x}),  |  
where in we used () from (LINS), in we used (CNAS), and follows from rearranging (CCTSA).
#### A.3.1 Discussion of Theorem 
Theorem (a) and (b) show the consistency of regression (): if (LINS), (CNAS), and (CIUN) hold, any estimator that is consistent for the regression estimands in () is also consistent for the marginal average treatment effects under either conditional parallel trends after treatment or marginal parallel trends after treatment. Theorem (a) and (b) applies to ETWFE immediately under the conditions of Theorem 6.1(ii) in , which shows that the ETWFE estimator of \tau_{rt}^{*} is consistent. It also applies to FETWFE in light of Theorem (b), which is used to prove Theorem .
Theorem (c) shows that it is specifically the covariates \boldsymbol{X}_{i}\mathbbm{1}\\{t=t^{\prime}\\} for t^{\prime}\in\\{2,\ldots,T\\} (which correspond to regression estimands \boldsymbol{\xi}_{t}^{*}) included in regression () that induce an asymptotic bias in the estimated marginal average treatment effects \tau_{\text{ATT}}(r,t^{\prime}) if (CIUN) does not hold. Suppose we assume that () from (LINS) holds and \boldsymbol{X}_{i} and are not mean-independent; in particular, for all r\in\mathcal{R} we have ≠\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\neq\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=0]66 6 Notice from Theorem (c) that =\tau_{rt}^{*}=\tau_{\text{ATT}}(r,t) if and only if either =\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]=\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=0] or \boldsymbol{\xi}_{t}^{*}=\boldsymbol{0}. There is an interesting connection here: it is also true that (CCTS) implies (CTS) under either independence of \boldsymbol{X}_{i} and or (CIUN), which under (LINS) is equivalent to \boldsymbol{\xi}_{t}^{*}=\boldsymbol{0}. The latter fact is from Proposition . Under independence of \boldsymbol{X}_{i} and , the distribution of \boldsymbol{X}_{i} conditional on is identical to the marginal distribution of \boldsymbol{X}_{i} for all r\in\\{0\\}\cup\mathcal{R}, so we can marginalize each side of () across the marginal distribution of \boldsymbol{X}_{i} to yield ().. Then if (CCTSA) and (CIUN) both do not hold, not only does Theorem (a) show that there is no hope of estimating conditional average treatment effects, Theorem (c) shows that in general including the covariates \boldsymbol{X}_{i}\mathbbm{1}\\{t=t^{\prime}\\} in a regression induces bias in the estimated average treatment effects (since if (CIUN) does not hold and () from (LINS) does then \boldsymbol{\xi}_{t}^{*}\neq\boldsymbol{0} for at least some ). This is true even if (CTS) holds and the marginal average treatment effects are estimable (by, for example, model (6.15) in , which omits these covariates). That is, the covariates \boldsymbol{X}_{i}\mathbbm{1}\\{t=t^{\prime}\\} for t^{\prime}\in\\{2,\ldots,T\\} are “bad controls," in the language of . This observation aligns with the observation that makes that including time-invariant covariates and their interactions with cohort dummies has no effect on the estimated marginal average treatment effects—it is specifically the inclusion of the covariate/time dummy interactions that changes the estimand for each \tau_{rt}^{*}.
Notice that these covariates correspond exactly to the kind of effect of the covariates on the untreated trends that is ruled out by Assumptions (CIU) and (CIUN), as well as the time-varying effects of the covariates on the potential outcomes that we mentioned in our discussion of Theorem in Section .
However, if () holds, we see that under (CIUN) the coefficients \boldsymbol{\xi}_{t}^{*} must all equal 𝟎\boldsymbol{0}. In this case, Theorem (c) shows that no harm is done to asymptotic unbiasedness by including those coefficients in the regression, and any estimator that is consistent for \tau_{rt}^{*} from () is consistent for \tau_{\text{ATT}}(r,t). (This aligns with part (b), which shows that if (CIUN) holds then we can consistently estimate the marginal average treatment effects under (CTSA) with regression .)
As we showed in Theorem , under the required assumptions no harm is done to the asymptotic efficiency of FETWFE by including irrelevant covariates (as we mentioned earlier in Remark ), like when (CIUN) holds and all of the \boldsymbol{\xi}_{t}^{*} equal 𝟎\boldsymbol{0}.
Theorem (c) has similarities to Theorems 4.1 and 5.1 from , which analyze the , \mathcal{R}=\\{2\\} and arbitrary , \mathcal{R}=\\{T\\} cases, respectively. In particular, their estimand \mathbb{E}[\hat{\tau}_{\text{DiD}}^{X}] matches our estimand \tau_{rt}^{*} under our assumptions, and they also identify the bias of this estimand in order to investigate when conditional parallel trends might be more plausible than marginal parallel trends. Besides the fact that we consider a more general staggered adoptions (arbitrary ) setting, they also consider different structural equations for the potential outcomes (for example, they omit interactions between the covariates and treatment status or cohort membership, though they do allow for time-varying coefficients on the covariates) and they consider a setting with a latent confounder.
## Appendix B Additional Results For Simulation Studies
Table contains results for the cohort-specific squared estimation errors for the cohort-specific average treatment effects \tau_{\text{ATT}}(r) as defined in (). We see that on average FETWFE outperforms every method for most cohorts, except that BETWFE outperforms FETWFE on average for cohort 2. This is because the true treatment effect for Cohort 2 is close to 0 (0.069), so BETWFE’s direct penalization of the treatment coefficients directly towards 0 happens to be useful. In addition, this difference in the mean outcomes seems to be driven by outliers—we see from the boxplots in Figure that in the median case FETWFE still outperforms BETWFE.
Table contains -values for paired one-tailed -tests of the alternative hypothesis that FETWFE has lower squared estimation error than the competitor methods, showing that the differences are statistically significant at the level in all settings except for the Cohort 2 BETWFE estimate and the last two ETWFE estimates, which are not significant because the ETWFE estimator has very high variance (see Table ).
Table contains results for the sum of squared errors of all methods’ estimates of the treatment interaction coefficients . We see that FETWFE outperforms every method. Table contains -values for paired one-tailed -tests of the alternative hypothesis that FETWFE has lower squared estimation error than the competitor methods, showing that the differences are statistically significant at the level.
Figure depicts a boxplot of the proportion of the possible restrictions that FETWFE selects correctly in the first simulation study. Figures , , , , and display boxplots of these squared errors, similar to Figure for the average treatment effect. These plots also indicate that FETWFE typically outperforms the other methods.
Means and standard errors for the squared errors of estimates of \tau_{\text{ATT}}(r) by each method for each cohort in the simulation study from Section .  
| FETWFE  | ETWFE  | BETWFE  | TWFE_COVS |  
| --- | --- | --- | --- |  
| Cohort 2  | 0.035 (0.006)  | 1.502 (0.312)  | 0.008 (0)  | 7.265 (0.163) |  
| Cohort 3  | 0.037 (0.004)  | 0.943 (0.19)  | 0.425 (0.011)  | 1.881 (0.091) |  
| Cohort 4  | 0.129 (0.009)  | 0.856 (0.179)  | 3.492 (0.016)  | 1.269 (0.062) |  
| Cohort 5  | 0.093 (0.007)  | 4.12 (3.079)  | 4.794 (0.011)  | 1.685 (0.082) |  
| Cohort 6  | 0.068 (0.005)  | 3.212 (2.016)  | 3.759 (0.016)  | 0.949 (0.049) |  
-values for paired one-tailed -tests of the alternative hypothesis that the FETWFE estimate of \tau_{\text{ATT}}(r) has lower squared error than the competitor methods for each cohort in the simulation study from Section .  
| ETWFE  | BETWFE  | TWFE_COVS |  
| --- | --- | --- |  
| Cohort 2  | 1.14e-05  | 1  | 8.2e-206 |  
| Cohort 3  | 8.06e-06  | 2.35e-145  | 1.85e-72 |  
| Cohort 4  | 0.000119  | 0  | 3.89e-63 |  
| Cohort 5  | 0.118  | 0  | 1.12e-68 |  
| Cohort 6  | 0.0795  | 0  | 8.71e-60 |  
Means and standard errors for the summed squared errors of estimates of the treatment interaction coefficients \boldsymbol{\rho}^{*} by each method for each cohort in the simulation study from Section .  
| FETWFE  | ETWFE  | BETWFE |  
| --- | --- | --- |  
| 0.529 (0.00827)  | 38.1 (18.3)  | 4.82 (0.0119) |  
-values for paired one-tailed -tests of the alternative hypothesis that the FETWFE estimates of the treatment interaction coefficients \boldsymbol{\rho}^{*} have lower summed squared error than the competitor methods for each cohort in the simulation study from Section . Boxplot displaying proportions of correct treatment effect restriction decisions correctly made by FETWFE across each of the 700 simulations from the first simulation study in Section . Boxplots of squared errors for each method’s estimate of \tau_{\text{ATT}}(2) across all 700 simulations. Vertical axis is on a log scale. Boxplots of squared errors for each method’s estimate of \tau_{\text{ATT}}(3) across all 700 simulations. Vertical axis is on a log scale. Boxplots of squared errors for each method’s estimate of \tau_{\text{ATT}}(4) across all 700 simulations. Vertical axis is on a log scale. Boxplots of squared errors for each method’s estimate of \tau_{\text{ATT}}(5) across all 700 simulations. Vertical axis is on a log scale. Boxplots of squared errors for each method’s estimate of \tau_{\text{ATT}}(6) across all 700 simulations. Vertical axis is on a log scale.
## Appendix C Theorem 
Theorem connects the bridge regression theory of to the difference-in-differences theory of . It also makes use of some extensions of the theory of (Theorem , Theorem ). Equipped with Theorem , Theorems , , , and are relatively straightforward to prove.
###### .
  1. (a)
(Correct specification.) Suppose Assumptions (CNAS), (CCTSB), and (LINS) hold. Then for all r\in\mathcal{R},  
| 𝔼[∣=0,]=\displaystyle\mathbb{E}[\tilde{y}_{(it)}\mid W_{i}=0,\boldsymbol{X}_{i}]=\penalty\   | ++and\displaystyle\eta^{*}+\gamma_{t}^{*}+\boldsymbol{X}_{i}^{\top}(\boldsymbol{\kappa}^{*}+\boldsymbol{\xi}_{t}^{*})\qquad\text{and}  |  
| --- | --- |  
| 𝔼[∣=r,]=\displaystyle\mathbb{E}[\tilde{y}_{(it)}\mid W_{i}=r,\boldsymbol{X}_{i}]=\penalty\   | ++++𝟙(+),\displaystyle\eta^{*}+\gamma_{t}^{*}+\nu_{r}^{*}+\boldsymbol{X}_{i}^{\top}(\boldsymbol{\kappa}^{*}+\boldsymbol{\xi}_{t}^{*}+\boldsymbol{\zeta}_{r}^{*})+\mathbbm{1}\\{t\geq r\\}(\tau_{rt}^{*}+\boldsymbol{\dot{X}}_{(ir)}^{\top}\boldsymbol{\rho}_{rt}^{*}),  |  
where =\boldsymbol{\dot{X}}_{(ir)}=\boldsymbol{X}_{i}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r], for identifiability we define the time-dependent parameters to equal 0 at , and recall that \eta^{*},\gamma_{t}^{*}, etc. are the regression estimands defined in (LINS).
  2. (b)
(Estimation consistency.) In addition to the assumptions from part , suppose Assumptions (F1), (F2), S(), and (R1) - (R3) hold. Then for any =​\lVert\boldsymbol{\hat{\beta}}^{(q)}-\boldsymbol{\beta}_{N}^{*}\rVert_{2}=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h^{\prime}_{N}\\}\right), where  is defined in () and h_{N}^{\prime} is defined in ().
  3. (c)
(Selection consistency.) Define  
| \boldsymbol{\theta}_{N}^{*}:=\boldsymbol{D}_{N}\boldsymbol{\beta}_{N}^{*}  |  
| --- |  
and \boldsymbol{\hat{\theta}}^{(q)}:=\boldsymbol{D}_{N}\boldsymbol{\hat{\beta}}^{(q)}. In addition to the previous assumptions, assume that q\in(0,1) and Assumptions (R4) - (R5) hold. Then =1\lim_{N\to\infty}\mathbb{P}\left(\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=\boldsymbol{0}\right)=1. (This also implies that \boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}\xrightarrow{p}\boldsymbol{0}ℙ⁡=1\lim_{N\to\infty}\mathbb{P}\left(\sqrt{NT}\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=\boldsymbol{0}\right)=1, and ℙ⁡(=0)=1\lim_{N\to\infty}\mathbb{P}\left(\sqrt{NT}\boldsymbol{v}_{N}^{\top}\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=0\right)=1 for any sequence of finite vectors \boldsymbol{v}_{N}\ \in\mathbb{R}^{p_{N}}.)
Further, let :=\hat{\mathcal{S}}_{N}:=\\{j:\hat{\theta}^{(q)}_{j}\neq 0\\}, and similarly let \mathcal{S}_{N}\subseteq[p_{N}] be the set of components of \boldsymbol{\theta}_{N}^{*} that are nonzero. Then  
| =1.\lim_{N\to\infty}\mathbb{P}\left(\hat{\mathcal{S}}_{N}=\mathcal{S}_{N}\right)=1.  |  
| --- |  
  4. (d)
(Asymptotic normality and oracle property.) In addition to the previous assumptions, assume Assumption (R6) holds and Assumption S() holds for a fixed , so \mathcal{S}_{N}=\mathcal{S} is fixed as well. Let \\{\boldsymbol{\psi}_{N}\\}_{N=1}^{\infty} be a sequence of real-valued vectors where for each \boldsymbol{\psi}_{N}\in\mathbb{R}^{p_{N}}(\boldsymbol{\psi}_{N})_{\mathcal{S}}=\boldsymbol{\alpha}\in\mathbb{R}^{s} is fixed and finite and (\boldsymbol{\psi}_{N})_{[p_{N}]\setminus\mathcal{S}}=\boldsymbol{b}_{N}, where \\{\boldsymbol{b}_{N}\\}_{N=1}^{\infty} is any sequence of constants where each \boldsymbol{b}_{N}\in\mathbb{R}^{p_{N}-s} contains all finite entries. Then if \boldsymbol{\alpha}\neq\boldsymbol{0},  
| 𝒩(0,(Cov)𝜶),\sqrt{NT}\boldsymbol{\psi}_{N}^{\top}(\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\theta}_{N}^{*})\xrightarrow{d}\mathcal{N}\left(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\mathrm{Cov}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)_{(\cdot\cdot)\mathcal{S}}\right)^{-1}\boldsymbol{\alpha}\right),  |  
| --- |  
where \mathrm{Cov}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)_{(\cdot\cdot)\mathcal{S}} is the columns and rows of the covariance matrix corresponding to the nonzero components of \boldsymbol{\theta}_{N}^{*}.
  5. (e)
(Asymptotic normality with feasible variance estimator.) Maintain the assumptions from part . For a random matrix \boldsymbol{A}\in\mathbb{R}^{NT\times p_{N}} with centered columns, define  
| :=\widehat{\mathrm{Cov}}(\boldsymbol{A}):=\frac{1}{NT}\boldsymbol{A}^{\top}\boldsymbol{A}  |  
| --- |  
to be the sample covariance matrix of 𝑨\boldsymbol{A}. For any \mathcal{A}_{N}\subseteq[p_{N}] such that \mathcal{A}_{N}\neq\emptyset, consider \widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right), the estimated covariance matrix of the columns of \boldsymbol{Z}\boldsymbol{D}_{N}^{-1} corresponding to \mathcal{A}_{N}. For sets (\mathcal{A}_{N})_{N=1}^{\infty}, define \boldsymbol{\alpha}(\mathcal{A}_{N})\in\mathbb{R}^{|\mathcal{A}_{N}|} to be a fixed, finite vector for each unique \mathcal{A}_{N}, and define the sequence of -vectors \\{\boldsymbol{\psi}_{N}(\mathcal{A}_{N})\\}_{N=1}^{\infty} to have components in the entries corresponding to \mathcal{A}_{N} equal to \boldsymbol{\alpha}(\mathcal{A}_{N}) and remaining entries \boldsymbol{b}_{n}(\mathcal{A}_{N}) all finite (but otherwise arbitrary). Define the sequence of random variables U_{1}(\mathcal{A}_{N}) to equal  
| ​𝜶(𝒜)(((𝒁𝑫)))𝜶(𝒜)⋅(−)\frac{1}{\sigma}\sqrt{\frac{NT}{\boldsymbol{\alpha}(\mathcal{A}_{N})^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right)\right)^{-1}\boldsymbol{\alpha}(\mathcal{A}_{N})}}\cdot\left(\boldsymbol{\psi}_{N}(\mathcal{A}_{N})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\psi}_{N}(\mathcal{S})^{\top}\boldsymbol{\theta}_{N}^{*}\right)  |  
| --- |  
if \mathcal{A}_{N}\neq\emptyset and \widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right) is invertible and 0 otherwise. Then if \boldsymbol{\alpha}(\mathcal{S})\neq\boldsymbol{0},  
| .U_{1}(\hat{\mathcal{S}}_{N})\xrightarrow{d}\mathcal{N}(0,1).  |  
| --- |  
  6. (f)
(Asymptotic normality when 𝜶\boldsymbol{\alpha} is estimated independently from \boldsymbol{\hat{\theta}}^{(q)}.) Maintain the assumptions from part . For sets (\mathcal{A}_{N})_{N=1}^{\infty}, let \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N})\in\mathbb{R}^{|\mathcal{A}_{N}|} be a sequence of random variables for each \mathcal{A}_{N} with the properties that for a given data set, \mathcal{A}=\mathcal{A}^{\prime} implies \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A})=\boldsymbol{\hat{\alpha}}_{N}(\mathcal{A^{\prime}}) and  
| ​​\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\left(\boldsymbol{\hat{\alpha}}_{N}(\mathcal{S})-\boldsymbol{\alpha}\right)\xrightarrow{d}\mathcal{N}(\boldsymbol{0}_{s},v_{\psi}(\boldsymbol{\alpha}))  |  
| --- |  
for some v_{\psi}(\boldsymbol{\alpha})>0. Define the sequence of random vectors \\{\boldsymbol{\hat{\psi}}_{N}(\mathcal{A}_{N})\\}_{N=1}^{\infty} to have components in \mathcal{A}_{N} equal to \boldsymbol{\hat{\alpha}}(\mathcal{A}_{N}) and components \boldsymbol{\hat{b}}_{N}(\mathcal{A}_{N}) in the components corresponding to [p_{N}]\setminus\mathcal{A}_{N} that are all almost surely finite for all . Define the sequence of random variables  
| :=⋅(−).U_{2}(\mathcal{A}_{N}):=\sqrt{NT}\cdot\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A}_{N})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\psi}_{N}(\mathcal{S})^{\top}\boldsymbol{\theta}_{N}^{*}\right).  |  
| --- |  
Assume \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) and \boldsymbol{\hat{\theta}}^{(q)} are estimated on independent data sets of size . Then if \boldsymbol{\alpha}\neq\boldsymbol{0},  
| 𝒩(0,(Cov)𝜶+).U_{2}(\hat{\mathcal{S}}_{N})\xrightarrow{d}\mathcal{N}\left(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\mathrm{Cov}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{S}}\right)\right)^{-1}\boldsymbol{\alpha}+v_{\psi}(\boldsymbol{\alpha})\right).  |  
| --- |  
  7. (g)
(Asymptotic normality when 𝜶\boldsymbol{\alpha} is estimated independently from \boldsymbol{\hat{\theta}}^{(q)} with feasible variance estimator.) Maintain the assumptions from part . Define the sequence of random variables U_{3}(\mathcal{A}_{N}) to equal  
| ⋅(−)\sqrt{\frac{NT}{\hat{v}_{N}(\mathcal{A}_{N})}}\cdot\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A}_{N})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\psi}_{N}(\mathcal{S})^{\top}\boldsymbol{\theta}_{N}^{*}\right)  |  
| --- |  
if \mathcal{A}_{N}\neq\emptyset\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})\neq\boldsymbol{0}, and \widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right) is invertible and 0 otherwise, where  
| :=()+\hat{v}_{N}(\mathcal{A}_{N}):=\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right)\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})+\hat{v}_{\psi}(\boldsymbol{\alpha})  |  
| --- |  
and \hat{v}_{\psi}(\boldsymbol{\alpha}) is an estimator of v_{\psi}(\boldsymbol{\alpha}) that satisfies \hat{v}_{\psi}(\boldsymbol{\alpha})\xrightarrow{p}v_{\psi}(\boldsymbol{\alpha}). Assume \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) and \boldsymbol{\hat{\theta}}^{(q)} are estimated on independent data sets of size . Then if \boldsymbol{\alpha}\neq\boldsymbol{0},  
| .U_{3}(\hat{\mathcal{S}}_{N})\xrightarrow{d}\mathcal{N}(0,1).  |  
| --- |  
  8. (h)
(Asymptotic subgaussianity when 𝜶\boldsymbol{\alpha} is estimated on the same data set as \boldsymbol{\hat{\theta}}^{(q)} with feasible variance estimator.) Maintain the assumptions from part . Define the sequence of random variables U_{4}(\mathcal{A}_{N}) to equal  
| ⋅(−)\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{cons})}(\mathcal{A}_{N})}}\cdot\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A}_{N})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\psi}_{N}(\mathcal{S})^{\top}\boldsymbol{\theta}_{N}^{*}\right)  |  
| --- |  
if \mathcal{A}_{N}\neq\emptyset\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})\neq\boldsymbol{0}, and \widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right) is invertible and 0 otherwise, where \hat{v}_{N}^{(\text{cons})}(\mathcal{A}_{N}) is defined as  
| ()\displaystyle\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right)\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})  |  
| --- |  
| +2​σ(𝒜)(((𝒁𝑫)))(𝒜)(𝜶)+.\displaystyle+2\sqrt{\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{A}_{N}}\right)\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{A}_{N})\hat{v}_{\psi}(\boldsymbol{\alpha})}+\hat{v}_{\psi}(\boldsymbol{\alpha}).  |  
Assume \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) and \boldsymbol{\hat{\theta}}^{(q)} are estimated on the same data set of size . Then if \boldsymbol{\alpha}\neq\boldsymbol{0}U_{4}(\hat{\mathcal{S}}_{N}) converges in distribution to a mean-zero subgaussian random variable with variance at most 1.


###### Proof.
Provided in Section . ∎
## Appendix D Extensions of Main Results
Theorem is an immediate consequence of the following more general result that allows for estimands that include broader classes of functions of the marginal cohort membership probabilities or propensity scores.
###### (Consistency of FETWFE; extension of Theorem ).
Assume that Assumptions (CNAS), (CCTSB), and (LINS) hold, as well as Assumptions (F1), (F2), S(), and (R1) - (R3). Let .
  1. (a)
Suppose that either Assumptions (CTSA) and (CIUN) hold or Assumption (CCTSA) holds. Then for any set of finite constants \\{\psi_{rt}\\},  
| |​(−)|=​\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right)\right|=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right)  |  
| --- |  
and for any differentiable function \boldsymbol{f}:[0,1]^{R+1}\to\mathbb{R}^{R} that is pointwise finite and has a Jacobian \nabla\boldsymbol{f}\in\mathbb{R}^{(R+1)\times R}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) that exists and is nonzero and finite,  
| |​(​(,,…,)​​−​​​)|=​,\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})\hat{\tau}_{\text{ATT}}(r,t)-f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\tau_{\text{ATT}}(r,t)\right)\right|=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right),  |  
| --- |  
where \hat{\pi}_{r}:=N_{r}/N for all r\in\mathcal{R}h_{N}^{\prime} is defined in (), and  is defined in ().
  2. (b)
Suppose Assumption (CCTSA) holds and  is fixed. For all  and all r\in\\{0\\}\cup\mathcal{R}, assume all of the eigenvalues of \mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r) are bounded between \lambda_{\text{min}}>0 and \lambda_{\text{max}}<\infty. Then for any set of finite constants \\{\psi_{rt}\\} and any fixed 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i},  
| |​(−)|=​.\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right)\right|=\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}).  |  
| --- |  
Further, for any 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i}, let \boldsymbol{\hat{\pi}}(\boldsymbol{x}) be an estimator for \pi_{r}(\boldsymbol{x}) satisfying =|\hat{\pi}_{r}(\boldsymbol{x})-\pi_{r}(\boldsymbol{x})|=\mathcal{O}_{\mathbb{P}}(a_{N}) for a decreasing sequence  for all r\in\\{0\\}\cup\mathcal{R}, and let \boldsymbol{a}_{r} and \boldsymbol{b}_{r}\in\mathbb{R}^{R+1} be finite vectors satisfying \boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})>0 and \boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})>0 for any 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i}. Then  
| |​(​​−​​)|\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{{\hat{\pi}}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{{\hat{\pi}}}(\boldsymbol{x})}\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right)\right|  |  
| --- |  
| =\displaystyle=\penalty\   | ​(∨).\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}\vee a_{N}).  |  


###### Proof.
Provided in Section . ∎
Theorem is a similar generalization of Theorem .
###### (Asymptotic Confidence Intervals for FETWFE; extension of Theorem ).
Assume that Assumptions (CNAS), (CCTSB), and (LINS) hold, as well as Assumptions (F1), (F2), S() for a fixed , and (R1) - (R6). Assume q\in(0,1). Suppose that either Assumptions (CTSA) and (CIUN) hold or Assumption (CCTSA) holds. Let \\{\psi_{rt}\\} be an arbitrary set of finite constants, and for all of the below results, assume at least one of the \tau_{\text{ATT}}(r,t) corresponding to a nonzero is nonzero (otherwise, all of the below sequences of random variables converge in probability to 0).
  1. (a)
For the estimator () it holds that  
| ​​(−)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C})}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t))\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
| --- | --- |  
where \hat{v}_{N}^{(\text{C})} is a finite-sample variance estimator defined in (). (All variance estimators depend only on the known  and the observed data.)
  2. (b)
Suppose one set of data of size  is used to estimate the cohort probabilities N_{r}/N_{\tau} and another independent data set of size  is used to estimate each \hat{\tau}_{\text{ATT}}(r,t). Then for any differentiable function \boldsymbol{f}:[0,1]^{R+1}\to\mathbb{R}^{R} that is pointwise finite and has a Jacobian \nabla\boldsymbol{f}\in\mathbb{R}^{(R+1)\times R}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) that exists and is nonzero and finite, it holds that  
| ​​(​(,,…,)​​−​​​)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{\text{C};\hat{\pi}}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(f_{r}\left(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}}\right)\hat{\tau}_{\text{ATT}}(r,t)-f_{r}\left(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}\right)\tau_{\text{ATT}}(r,t)\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
| --- | --- |  
where \hat{v}_{N}^{\text{C};\hat{\pi}} is a finite-sample variance estimator defined in ().
  3. (c)
Suppose a single set of data of size  is used to estimate both the cohort probabilities N_{r}/N_{\tau} and each \hat{\tau}_{\text{ATT}}(r,t). Then for any differentiable function \boldsymbol{f}:[0,1]^{R+1}\to\mathbb{R}^{R} that is pointwise finite and has a Jacobian \nabla\boldsymbol{f}\in\mathbb{R}^{(R+1)\times R}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) that exists and is nonzero and finite, the estimator () is asymptotically subgaussian: the sequence of random variables  
| N​​(​(,,…,)​​−​​​)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{\text{C, (cons)};\hat{\pi}}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(f_{r}\left(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}}\right)\hat{\tau}_{\text{ATT}}(r,t)-f_{r}\left(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}\right)\tau_{\text{ATT}}(r,t)\right)  |  
| --- |  
converges in distribution to a mean-zero subgaussian random variable with variance at most 1, where N\hat{v}_{N}^{\text{C, (cons)};\hat{\pi}} is a finite-sample conservative variance estimator defined in ().


###### Proof.
Provided in Section . ∎
Theorem shows the asymptotic normality of two classes of conditional treatment effect estimators, analogously to Theorem , and provides consistent variance estimators for constructing asymptotically valid confidence intervals. It also provides sufficient assumptions for the FETWFE estimators of conditional average treatment effects to converge at rates.
###### (Asymptotic Confidence Intervals for conditional average treatment effects in FETWFE).
Assume that Assumptions (CNAS), (CCTS), and (LINS) hold, as well as Assumptions (F1), (F2), S() for a fixed , and (R1) - (R6). Assume q\in(0,1), and assume is fixed. Let \\{\psi_{rt}\\} be an arbitrary set of finite constants, and for both of the below results, assume at least one of the \tau_{\text{CATT}}(r,t,\boldsymbol{x}) corresponding to a nonzero is nonzero (otherwise, all of the below sequences of random variables converge in probability to 0).
  1. (a)
For \hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x}) as defined in (), suppose one set of data of size  is used to estimate each cohort conditional covariate mean \boldsymbol{\overline{X}}_{r} and another independent data set of size  is used to estimate each \hat{\tau}_{rt}^{(q)} and \hat{\rho}_{rt}^{(q)} via the FETWFE regression. Then for any set of finite constants \\{\psi_{rt}\\}_{r\in\mathcal{R},t\in\\{r,\ldots,T\\}} and any fixed 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i}, it holds that  
| ​​(−)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C;CATT})}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}(\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tau_{\text{CATT}}(r,t,\boldsymbol{x}))\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
| --- | --- |  
where \hat{v}_{N}^{(\text{C;CATT})} is a finite-sample variance estimator defined in ().
  2. (b)
Assume we have an estimator (,,…,)(\hat{\pi}_{0}(\boldsymbol{x}),\hat{\pi}_{r_{1}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})) of the generalized propensity scores (,,…,)(\pi_{0}(\boldsymbol{x}),\pi_{r_{1}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})) that satisfies for any fixed 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i}  
| ​(,,…,)\sqrt{N}\left(\hat{\pi}_{0}(\boldsymbol{x})-\pi_{0}(\boldsymbol{x}),\hat{\pi}_{r_{1}}(\boldsymbol{x})-\pi_{r_{1}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})-\pi_{r_{R}}(\boldsymbol{x})\right)\xrightarrow{d}\mathcal{N}(0,\boldsymbol{\Sigma}_{\pi})  |  
| --- |  
for some finite positive definite \boldsymbol{\Sigma}_{\pi}\in\mathbb{R}^{(R+1)\times(R+1)}. Assume we also have available a consistent estimator \boldsymbol{\hat{\Sigma}}_{\pi} that converges entrywise in probability to \boldsymbol{\Sigma}_{\pi}. Suppose one set of data of size  is used to estimate each cohort conditional covariate mean \boldsymbol{\overline{X}}_{r}, another independent data set of size  is used to estimate each \hat{\tau}_{rt}^{(q)} and \hat{\rho}_{rt}^{(q)} via the FETWFE regression, and a third independent data set is used to estimate the generalized propensity scores. Then for any differentiable function f:[0,1]^{R+1}\to\mathbb{R}^{R} that is pointwise finite and has a Jacobian \nabla\boldsymbol{f}\in\mathbb{R}^{(R+1)\times R}(,,…,)(\pi_{0}(\boldsymbol{x}),\pi_{r_{1}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})) that exists and is nonzero and finite, it holds that  
| ​​(​(,,…,)​​CLOSE\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C;CATT,}\pi)}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\big(f_{r}\left(\hat{\pi}_{0}(\boldsymbol{x}),\hat{\pi}_{r_{1}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})  |  
| --- |  
| OPEN−​(,,…,)​​),\displaystyle-f_{r}\left(\pi_{0}(\boldsymbol{x}),\pi_{r_{1}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\tau_{\text{CATT}}(r,t,\boldsymbol{x})\big)\xrightarrow{d}\mathcal{N}(0,1),  |  
where \hat{v}_{N}^{(\text{C;CATT,}\pi)} is a finite-sample variance estimator defined in ().


###### Proof.
Provided in Section . ∎
It is trivial to extend Theorem to show that these classes of conditional average treatment effect estimators enjoy oracle properties analogous to those of Theorem . Theorem applies immediately to the classes of conditional average treatment effects estimators () and () by the same argument we used to prove Theorem since the Jacobian \nabla\boldsymbol{f}\in\mathbb{R}^{(R+1)\times R}  
| :=()\nabla\boldsymbol{f}:=\begin{pmatrix}0&0&\cdots&0\\\ \frac{\sum_{r^{\prime}\in\mathcal{R}\setminus r_{1}}\pi_{r^{\prime}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}&\frac{-\pi_{r_{2}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}&\cdots&\frac{-\pi_{r_{R}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}\\\ \frac{-\pi_{r_{1}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}&\frac{\sum_{r^{\prime}\in\mathcal{R}\setminus r_{2}}\pi_{r^{\prime}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}&\cdots&\frac{-\pi_{r_{R}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}\\\ \vdots&\vdots&\ddots&\vdots\\\ \frac{-\pi_{r_{1}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}&\frac{-\pi_{r_{2}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}&\cdots&\frac{\sum_{r^{\prime}\in\mathcal{R}\setminus r_{R}}\pi_{r^{\prime}}(\boldsymbol{x})}{\left(\sum_{r^{\prime}\in\mathcal{R}}\pi_{r^{\prime}}(\boldsymbol{x})\right)^{2}}\end{pmatrix}  |  
| --- |  
at (,,…,)(\pi_{0}(\boldsymbol{x}),\pi_{r_{1}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})) and exists and is nonzero and finite by Assumption (F2).
Other extensions are possible as well. For example, when the data is not split into independent subsamples we can achieve asymptotic subgaussianity with a conservative variance estimator using a similar argument to the one used to prove Theorem (c).
## Appendix E Proofs of Main Results
We present two lemmas we will use in the following proofs.
###### .
The number of treatment effects 𝔚\mathfrak{W} is at most (T-1)^{2}. Further, after re-ordering the columns of \boldsymbol{\tilde{Z}}, we can write \boldsymbol{D}_{N} as  
| diag⁡(,,,,,,),\operatorname{diag}\big(\boldsymbol{D}^{(1)}(R),\boldsymbol{D}^{(1)}(T-1),\boldsymbol{I}_{d_{N}},\boldsymbol{I}_{d_{N}}\otimes\boldsymbol{D}^{(1)}(R),\boldsymbol{I}_{d_{N}}\otimes\boldsymbol{D}^{(1)}(T-1),\boldsymbol{D}^{(2)}(\mathcal{R}),\boldsymbol{I}_{d_{N}}\otimes\boldsymbol{D}^{(2)}(\mathcal{R})\big),  |  
| --- |  
where \boldsymbol{D}^{(1)}(t)\in\mathbb{R}^{t\times t} is defined in () and \boldsymbol{D}^{(2)}(\mathcal{R})\in\mathbb{R}^{\mathfrak{W}\times\mathfrak{W}} is defined in (). Further, \boldsymbol{D}_{N} is invertible, and \boldsymbol{D}_{N}^{-1} can be written as  
| \displaystyle\operatorname{diag}\Big(  | ,(,,,⊗(CLOSECLOSE,\displaystyle\boldsymbol{D}^{(1)}(R)^{-1},((\boldsymbol{D}^{(1)}(T-1))^{-1},\boldsymbol{I}_{d_{N}},\boldsymbol{I}_{d_{N}}\otimes\boldsymbol{D}^{(1)}(R)^{-1},\boldsymbol{I}_{d_{N}}\otimes((\boldsymbol{D}^{(1)}(T-1))^{-1},  |  
| --- | --- |  
| OPEN,).\displaystyle(\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1},\boldsymbol{I}_{d_{N}}\otimes(\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}\Big).  |  
Finally, both \boldsymbol{D}_{N} and \boldsymbol{D}_{N}^{-1} consists of blocks of size at most (T-1)^{2}\times(T-1)^{2}. For \boldsymbol{D}_{N} the entries of these blocks are either , 0, or 1; and for \boldsymbol{D}_{N}^{-1} the entries of these blocks are either 0 or 1. Both matrices equal 0 elsewhere.
###### .
Under Assumption (R3), ⁡≤\max_{j\in p_{N}}\\{|(\boldsymbol{\beta}_{N}^{*})_{j}|\\}\leq(T-1)^{2}b_{1}.
Proofs of both results are provided in Appendix .
### E.1 Proofs of Theorems and 
We start by proving Theorem , which leads almost immediately to a proof of Theorem .
###### Proof of Theorem .
  1. (a)
Observe that under our assumptions our model is correctly specified due to Theorem (a). Recall that due to Theorem , under our assumptions the regression estimands correspond to our causal estimands.
Examining our estimator \hat{\tau}_{\text{ATT}}(r,t) for the average treatment effect for a unit in cohort at time from (), we see that we can then apply Theorem (b) to immediately obtain that for every r\in\mathcal{R} and ,  
| |−|=||≤=​.\left|\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right|=\left|\hat{\tau}_{rt}^{(q)}-\tau_{rt}^{*}\right|\leq\lVert\boldsymbol{\hat{\beta}}^{(q)}-\boldsymbol{\beta}_{N}^{*}\rVert_{2}=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right).  |  
| --- |  
Similarly, for any deterministic fixed linear combination of average treatment effects, using () and Theorem we have that for our class of estimators () and estimands (),  
| |​​−|≤\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\hat{\tau}_{\text{ATT}}(r,t)-\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\tau_{\text{ATT}}(r,t)\right|\leq\penalty\   | ​|−|\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}|\psi_{rt}|\left|\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right|  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​,\displaystyle\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right),  |  
where we used the fact that is fixed (and upper bounds ).
Next we characterize the rate of convergence of each ​(,,…,)f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}}). Because by assumption \boldsymbol{f}(\cdot) has a finite Jacobian \nabla\boldsymbol{f}\in\mathbb{R}^{(R+1)\times R} at (\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) that exists and is nonzero and finite, we have by the asymptotic normality of the maximum likelihood estimator (,,,…,)\left(\frac{N_{0}}{N},\frac{N_{r_{1}}}{N},\frac{N_{r_{2}}}{N},\ldots,\frac{N_{r_{R}}}{N}\right) (Theorem 14.6 in ) and the delta method (Theorem 3.1 in ) that  
| ​(f⁡(,,…,)−f⁡)\displaystyle\sqrt{N}\left(f(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{M}\nabla\boldsymbol{f})  |  
| --- | --- |  
| ⟹​(​(,,…,)−​)\displaystyle\implies\qquad\sqrt{N}\left(f_{r^{\prime}}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f_{r^{\prime}}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\right)\xrightarrow{d}\penalty\   | 𝒩(0,∇∇𝒇),\displaystyle\mathcal{N}(0,\boldsymbol{e}_{r^{\prime}}^{\top}\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{M}\nabla\boldsymbol{f}\boldsymbol{e}_{r^{\prime}}),  |  
for any r^{\prime}\in[R], where  
| :=()∈\boldsymbol{\Sigma}_{M}:=\begin{pmatrix}\pi_{0}(1-\pi_{0})&-\pi_{0}\pi_{r_{1}}&\cdots&-\pi_{0}\pi_{r_{R}}\\\ -\pi_{0}\pi_{r_{1}}&\pi_{r_{1}}(1-\pi_{r_{1}})&\cdots&-\pi_{r_{1}}\pi_{r_{R}}\\\ \vdots&\vdots&\ddots&\vdots\\\ -\pi_{0}\pi_{r_{R}}&-\pi_{r_{1}}\pi_{r_{R}}&\cdots&\pi_{r_{R}}(1-\pi_{r_{R}})\end{pmatrix}\in\mathbb{R}^{(R+1)\times(R+1)}  |  
| --- |  
and \boldsymbol{e}_{r}\in\mathbb{R}^{R} is the selector vector with a 1 in the r^{\text{th}} entry and zeroes elsewhere. Therefore by Theorem 2.4(i) in and Proposition 1.8(iii) in , we have that for all   
| ​|​(,,…,)−​|=\displaystyle\sqrt{N}\left|f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\right|=\penalty\   | \displaystyle\mathcal{O}_{\mathbb{P}}\left(1\right)  |  
| --- | --- |  
| ⟹|​(,,…,)−​|=\displaystyle\implies\qquad\left|f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\right|=\penalty\   | \displaystyle\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right).  |  
Finally, for our class of estimators () and estimands (), we have  
| |​​(,,…,)​​−​​​​|\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})\hat{\tau}_{\text{ATT}}(r,t)-\psi_{rt}f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\tau_{\text{ATT}}(r,t)\right|  |  
| --- |  
| =\displaystyle=\penalty\   | |[(−)\displaystyle\Bigg|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\big[f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\left(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right)  |  
| +(​(,,…,)−​)​​\displaystyle+\left(f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\right)\tau_{\text{ATT}}(r,t)  |  
| +((,,…,)−)(−)]|\displaystyle+(f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}))\left(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right)\big]\Bigg|  |  
| ≤\displaystyle\leq\penalty\   | ​(​​|−|+CLOSE\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\big(f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\left|\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right|+  |  
| |​(,,…,)−​|​​\displaystyle\left|f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\right|\tau_{\text{ATT}}(r,t)  |  
| OPEN+|​(,,…,)−​|​|−|)\displaystyle+|f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})|\left|\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right|\big)  |  
| \displaystyle\stackrel{{\scriptstyle(a)}}{{=}}\penalty\   | ​++​​​\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\})+\mathcal{O}_{\mathbb{P}}(1/\sqrt{N})+\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\})\mathcal{O}_{\mathbb{P}}(1/\sqrt{N})  |  
| \displaystyle\stackrel{{\scriptstyle(b)}}{{=}}\penalty\   | ​,\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}),  |  
where in we used that all the ​f_{r}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) and \tau_{\text{ATT}}(r,t) are finite, (), and (); and in we used that 1/\sqrt{N}=\mathcal{O}(h_{N}) since is upper-bounded by a constant and is nondecreasing.
  2. (b)
Again, using Theorem ,  
| =+(𝒙−𝔼)=+,\tau_{\text{CATT}}(r,t,\boldsymbol{x})=\tau_{rt}^{*}+(\boldsymbol{\rho}_{rt}^{*})^{\top}\left(\boldsymbol{x}-\mathbb{E}\left[\boldsymbol{X}_{(1t)\cdot}\mid W_{i}=r\right]\right)=\tau_{rt}^{*}+(\boldsymbol{\rho}_{rt}^{*})^{\top}\left(\boldsymbol{x}-\boldsymbol{\mu}_{r}\right),  |  
| --- |  
where :=𝔼\boldsymbol{\mu}_{r}:=\mathbb{E}\left[\boldsymbol{X}_{(it)\cdot}\mid W_{i}=r\right]. Using this and () we have  
| |−|\displaystyle\left|\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right|  |  
| --- |  
| =\displaystyle=\penalty\   | |+−|\displaystyle\left|\hat{\tau}_{rt}^{(q)}-\tau_{rt}^{*}+(\boldsymbol{x}-\boldsymbol{\overline{X}}_{r})^{\top}\boldsymbol{\hat{\rho}}_{rt}-\left(\boldsymbol{x}-\boldsymbol{\mu}_{r}\right)^{\top}\boldsymbol{\rho}_{rt}^{*}\right|  |  
| =\displaystyle=\penalty\   | |+​+​−|\displaystyle\left|\hat{\tau}_{rt}^{(q)}-\tau_{rt}^{*}+\left(\boldsymbol{x}-\boldsymbol{\mu}_{r}\right)^{\top}\left(\boldsymbol{\hat{\rho}}_{rt}-\boldsymbol{\rho}_{rt}^{*}\right)+\left(\boldsymbol{\mu}_{r}-\boldsymbol{\overline{X}}_{r}\right)^{\top}\left(\boldsymbol{\hat{\rho}}_{rt}-\boldsymbol{\rho}_{rt}^{*}\right)-\left(\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right)^{\top}\boldsymbol{\rho}_{rt}^{*}\right|  |  
| ≤\displaystyle\leq\penalty\   | ||+|​|+|​|+||\displaystyle\left|\hat{\tau}_{rt}^{(q)}-\tau_{rt}^{*}\right|+\left|\left(\boldsymbol{x}-\boldsymbol{\mu}_{r}\right)^{\top}\left(\boldsymbol{\hat{\rho}}_{rt}-\boldsymbol{\rho}_{rt}^{*}\right)\right|+\left|\left(\boldsymbol{\mu}_{r}-\boldsymbol{\overline{X}}_{r}\right)^{\top}\left(\boldsymbol{\hat{\rho}}_{rt}-\boldsymbol{\rho}_{rt}^{*}\right)\right|+\left|\left(\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right)^{\top}\boldsymbol{\rho}_{rt}^{*}\right|  |  
| \displaystyle\stackrel{{\scriptstyle(c)}}{{\leq}}\penalty\   | ||+​+​+​\displaystyle\left|\hat{\tau}_{rt}^{(q)}-\tau_{rt}^{*}\right|+\left\lVert\boldsymbol{x}-\boldsymbol{\mu}_{r}\right\rVert_{2}\left\lVert\boldsymbol{\hat{\rho}}_{rt}-\boldsymbol{\rho}_{rt}^{*}\right\rVert_{2}+\left\lVert\boldsymbol{\mu}_{r}-\boldsymbol{\overline{X}}_{r}\right\rVert_{2}\left\lVert\boldsymbol{\hat{\rho}}_{rt}-\boldsymbol{\rho}_{rt}^{*}\right\rVert_{2}+\left\lVert\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right\rVert_{2}\left\lVert\boldsymbol{\rho}_{rt}^{*}\right\rVert_{2}  |  
| \displaystyle\stackrel{{\scriptstyle(d)}}{{=}}\penalty\   | ​+​+​​​+\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\})+\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\})+\mathcal{O}_{\mathbb{P}}\left(1/\sqrt{N}\right)\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h_{N}^{\prime}\\}\right)+\mathcal{O}_{\mathbb{P}}\left(1/\sqrt{N}\right)  |  
| =\displaystyle=\penalty\   | ​,\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}),  |  
where in we applied Cauchy-Schwarz and follows from Theorem , the fact that \left\lVert\boldsymbol{\rho}_{rt}^{*}\right\rVert_{2} is fixed and bounded since is fixed and each coefficient is finite due to Lemma , and Lemma :
###### .
For each r\in\\{0\\}\cup\mathcal{R}, let \boldsymbol{\psi}_{r}\in\mathbb{R}^{d} be constant vectors, with \boldsymbol{\psi}_{r}\neq\boldsymbol{0} for at least one . For all and all r\in\\{0\\}\cup\mathcal{R}, assume all of the eigenvalues of \mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r) are bounded between \lambda_{\text{min}}>0 and \lambda_{\text{max}}<\infty. Then  
| ​,\sqrt{N}\sum_{r\in\\{0\\}\cup\mathcal{R}}\boldsymbol{\psi}_{r}^{\top}\left(\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right)\xrightarrow{d}\mathcal{N}(0,v_{R}),  |  
| --- |  
where  
| :=​Cov​​.v_{R}:=\sum_{r\in\\{0\\}\cup\mathcal{R}}\frac{\boldsymbol{\psi}_{r}^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\psi}_{r}}{\mathbb{P}(W_{i}=r)}.  |  
| --- |  
Further, for any r\in\\{0\\}\cup\mathcal{R},  
| =.\lVert\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\rVert_{2}=\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right).  |  
| --- |  
###### Proof.
Next, for any deterministic fixed set of constants \\{\psi_{rt}\\}, the above result gives us  
| |​​−​​|\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right|  |  
| --- |  
| ≤\displaystyle\leq\penalty\   | ​|−|\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}|\psi_{rt}|\left|\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right|  |  
| =\displaystyle=\penalty\   | ​.\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}).  |  
Finally, we have  
| |​()​​−​​​|\displaystyle\left|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}\right)\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right|  |  
| --- |  
| =\displaystyle=\penalty\   | |[()(−)\displaystyle\Bigg|\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\Bigg[\left(\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{\pi}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{\pi}(\boldsymbol{x})}\right)(\hat{\tau}_{\text{CATT}}(r,\boldsymbol{x},t)-\tau_{\text{CATT}}(r,t,\boldsymbol{x}))  |  
| +[−]​​\displaystyle+\left[\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}-\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right]\tau_{\text{CATT}}(r,t,\boldsymbol{x})  |  
| +[−](−)]|\displaystyle+\left[\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}-\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right](\hat{\tau}_{\text{CATT}}(r,\boldsymbol{x},t)-\tau_{\text{CATT}}(r,t,\boldsymbol{x}))\Bigg]\Bigg|  |  
| ≤\displaystyle\leq\penalty\   | [|−|\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\Bigg[\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{\pi}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{\pi}(\boldsymbol{x})}\left|\hat{\tau}_{\text{CATT}}(r,\boldsymbol{x},t)-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right|  |  
| +|−|​​\displaystyle+\left|\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}-\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right|\tau_{\text{CATT}}(r,t,\boldsymbol{x})  |  
| +|−||−|]\displaystyle+\left|\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}-\frac{\boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right|\left|\hat{\tau}_{\text{CATT}}(r,\boldsymbol{x},t)-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right|\Bigg]  |  
| \displaystyle\stackrel{{\scriptstyle(e)}}{{\leq}}\penalty\   | ​++​​​\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\})+\mathcal{O}_{\mathbb{P}}(a_{N})+\mathcal{O}_{\mathbb{P}}(a_{N})\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\})  |  
| \displaystyle\stackrel{{\scriptstyle(f)}}{{=}}\penalty\   | ​(∨),\displaystyle\mathcal{O}_{\mathbb{P}}(\min\\{h_{N},h_{N}^{\prime}\\}\vee a_{N}),  |  
where in we used Proposition 1.8 from and in we used (), the fact that |\psi_{rt}| is bounded for any fixed 𝒙\boldsymbol{x} for all r\in\mathcal{R} due to Lemma , the fact that ||\left|\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right| and ​𝝅​\boldsymbol{a}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})/\boldsymbol{b}_{r}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x}) are bounded for any fixed 𝒙\boldsymbol{x}, and Lemma :
###### .
For any 𝐱\boldsymbol{x} in the support of \boldsymbol{X}_{i}, let \boldsymbol{\hat{\pi}}(\boldsymbol{x}) be an estimator for \pi_{r}(\boldsymbol{x}) satisfying =|\hat{\pi}_{r}(\boldsymbol{x})-\pi_{r}(\boldsymbol{x})|=\mathcal{O}_{\mathbb{P}}(a_{N}) for a decreasing sequence for all r\in\\{0\\}\cup\mathcal{R}. Let 𝐚\boldsymbol{a} and \boldsymbol{b}\in\mathbb{R}^{R+1} be finite vectors satisfying \boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})>0 and \boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})>0 for any 𝐱\boldsymbol{x} in the support of \boldsymbol{X}_{i}. Then  
| |−|=.\left|\frac{\boldsymbol{a}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}-\frac{\boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right|=\mathcal{O}_{\mathbb{P}}\left(a_{N}\right).  |  
| --- |  
###### Proof.
The proof is complete.


###### Proof of Theorem .
In light of Theorem , to prove part (a) it is enough to note that  
| ​(,,…,)=()f_{r}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi_{r_{R}}})=\begin{pmatrix}\frac{\hat{\pi}_{r_{1}}}{\sum_{r\in\mathcal{R}}\hat{\pi}_{r}}\\\ \frac{\hat{\pi}_{r_{2}}}{\sum_{r\in\mathcal{R}}\hat{\pi}_{r}}\\\ \vdots\\\ \frac{\hat{\pi}_{r_{R}}}{\sum_{r\in\mathcal{R}}\hat{\pi}_{r}}\end{pmatrix}  |  
| --- |  
(where \hat{\pi}_{r}=N_{r}/N) is pointwise (that is, almost surely) finite and show that its Jacobian at (\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) exists and is nonzero and finite. The Jacobian at (\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) is  
| ()∈;\begin{pmatrix}0&0&\cdots&0\\\ \frac{\sum_{r\in\mathcal{R}\setminus r_{1}}\hat{\pi}_{r}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}&-\frac{\hat{\pi}_{r_{2}}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}&\cdots&-\frac{\hat{\pi}_{r_{R}}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}\\\ -\frac{\hat{\pi}_{r_{1}}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}&\frac{\sum_{r\in\mathcal{R}\setminus r_{2}}\hat{\pi}_{r}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}&\cdots&-\frac{\hat{\pi}_{r_{R}}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}\\\ \vdots&\vdots&\ddots&\vdots\\\ -\frac{\hat{\pi}_{r_{1}}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}&-\frac{\hat{\pi}_{r_{2}}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}&\cdots&\frac{\sum_{r\in\mathcal{R}\setminus r_{R}}\hat{\pi}_{r}}{\left(\sum_{r\in\mathcal{R}}\hat{\pi}_{r}\right)^{2}}\end{pmatrix}\in\mathbb{R}^{(R+1)\times R};  |  
| --- |  
notice it is nonzero as required.
Part (b) is immediate from Theorem (b) using a_{r}=\boldsymbol{e}_{r}, the selector vector with a 1 in the r^{\text{th}} position and zeroes elsewhere, and \boldsymbol{b}_{r}=(1,1,\ldots,1)^{\top}. ∎
### E.2 Proofs of Theorems , , and 
Next we prove Theorem . Then we will use this result to prove Theorem , and finally Theorem .
###### Proof of Theorem .
By Theorem , the average treatment effect \tau_{\text{ATT}}(r,t) is equal to the regression estimand \tau_{rt}^{*}. We will prove Theorem by applying Theorem (e), (g), and (h) with particular choices of vectors \boldsymbol{\psi}_{N} and \boldsymbol{\hat{\psi}}_{N}.
  1. (a)
Let be the index of the row of \boldsymbol{D}_{N} corresponding to \hat{\tau}_{rt}^{(q)} (that is, the column of \boldsymbol{\tilde{Z}} corresponding to this regression coefficient), so that  
| =\hat{\tau}_{rt}^{(q)}=(\boldsymbol{D}_{N}^{-1})_{i(r,t)\cdot}\boldsymbol{\hat{\theta}}^{(q)}  |  
| --- |  
and  
| =.\tau_{rt}^{*}=(\boldsymbol{D}_{N}^{-1})_{i(r,t)\cdot}\boldsymbol{\theta}_{N}^{*}.  |  
| --- |  
We can see from Lemma that each row of \boldsymbol{D}_{N}^{-1} contains at most (T-1)^{2} nonzero entries, all of the nonzero entries equal 1, and the entries corresponding to the relevant features will stay fixed even if p_{N}\to\infty. So the vector =()⊤\boldsymbol{\psi}_{N}=\left((\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}\right)^{\top} is a valid choice for Theorem (d) - (h). In fact, any fixed, finite linear combination of rows (\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot} is a valid choice. We therefore have from Theorem (e) that if at least one of the nonzero components of (\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot} corresponds to one of the features in ,  
| ​(​−​)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{\text{ATT};r,t}}}\left((\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}\boldsymbol{\hat{\theta}}^{(q)}-(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}\boldsymbol{\theta}_{N}^{*}\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1)  |  
| --- | --- |  
| ⇔(−)\displaystyle\iff\qquad\sqrt{\frac{NT}{\hat{v}_{N}^{\text{ATT};r,t}}}\left(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t)\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
where  
| :=()()⊤,\hat{v}_{N}^{\text{ATT};r,t}:=\sigma^{2}(\boldsymbol{D}_{N}^{-1})_{i(r,t),\hat{\mathcal{S}}}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\hat{\mathcal{S}}}\right)\right)^{-1}((\boldsymbol{D}_{N}^{-1})_{i(r,t),\hat{\mathcal{S}}})^{\top},  |  
| --- |  
\widehat{\mathrm{Cov}}(\cdot) is defined in () and ∈(\boldsymbol{D}_{N}^{-1})_{i(r,t),\hat{\mathcal{S}}}\in\mathbb{R}^{|\hat{\mathcal{S}}|} is a row vector that contains the |\hat{\mathcal{S}}| components of the row of \boldsymbol{D}_{N}^{-1} corresponding to the selected features. Similarly, for any deterministic fixed set of constants \\{\psi_{rt}\\}, since  
| =​​,\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\tau_{rt}^{*}=\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}\boldsymbol{\theta}_{N}^{*},  |  
| --- |  
we can choose :=​⊤\boldsymbol{\psi}_{N}^{(\text{C})}:=\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}^{\top} to establish that  
| ​(​)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C})}}}\left((\boldsymbol{\psi}_{N}^{(\text{C})})^{\top}(\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\theta}_{N}^{*})\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1)  |  
| --- | --- |  
| ⇔(−)\displaystyle\iff\qquad\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C})}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}(\hat{\tau}_{\text{ATT}}(r,t)-\tau_{\text{ATT}}(r,t))\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
where  
| :=().\hat{v}_{N}^{(\text{C})}:=\sigma^{2}\left((\boldsymbol{\psi}_{N}^{(\text{C})})_{\hat{\mathcal{S}}}\right)^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\hat{\mathcal{S}}}\right)\right)^{-1}(\boldsymbol{\psi}_{N}^{(\text{C})})_{\hat{\mathcal{S}}}.  |  
| --- |  
  2. (b)
We will prove the convergence of the results with estimated cohort probabilities by applying Theorem (g). Consider the random vector  
| :=​(,,…,)​​∈\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})}:=\sum_{r\in\mathcal{R}}f_{r}(\hat{\pi}_{r_{1}},\hat{\pi}_{r_{2}},\ldots,\hat{\pi}_{r_{R}})\sum_{t=r}^{T}\psi_{rt}(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}\in\mathbb{R}^{p_{N}}  |  
| --- |  
and define  
| :=​​​∈,\boldsymbol{\psi}_{N}^{(\text{C; }\pi)}:=\sum_{r\in\mathcal{R}}f_{r}(\pi_{r_{1}},\pi_{r_{2}},\ldots,\pi_{r_{R}})\sum_{t=r}^{T}\psi_{rt}(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}\in\mathbb{R}^{p_{N}},  |  
| --- |  
so that  
| ​−\displaystyle(\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-(\boldsymbol{\psi}_{N}^{(\text{C; }\pi)})^{\top}\boldsymbol{\theta}_{N}^{*}  |  
| --- |  
| =\displaystyle=\penalty\   | ​(​(,,…,)​​−​​​).\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(f_{r}(\hat{\pi}_{r_{1}},\hat{\pi}_{r_{2}},\ldots,\hat{\pi}_{r_{R}})\hat{\tau}_{\text{ATT}}(r,t)-f_{r}(\pi_{r_{1}},\pi_{r_{2}},\ldots,\pi_{r_{R}})\tau_{\text{ATT}}(r,t)\right).  |  
We have assumed that the (,,…,)(\hat{\pi}_{r_{1}},\hat{\pi}_{r_{2}},\ldots,\hat{\pi}_{r_{R}}) values (and therefore \boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})}) are estimated on an independent data set from the one used to estimate \boldsymbol{\hat{\theta}}^{(q)}. It remains to show that  
| ​​(−)\sqrt{NT}(\boldsymbol{\theta}^{*})_{\mathcal{S}}^{\top}\left((\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\mathcal{S}}-(\boldsymbol{\psi}_{N}^{(\text{C; }\pi)})_{\mathcal{S}}\right)  |  
| --- |  
is asymptotically normal and to identify a consistent estimator for the asymptotic variance. First we establish asymptotic normality. Because by assumption \boldsymbol{f}(\cdot) has a finite Jacobian \nabla\boldsymbol{f}\in\mathbb{R}^{(R+1)\times R} at (\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) that exists and is nonzero and finite, using the asymptotic normality of the maximum likelihood estimator (,,,…,)\left(\frac{N_{0}}{N},\frac{N_{r_{1}}}{N},\frac{N_{r_{2}}}{N},\ldots,\frac{N_{r_{R}}}{N}\right) (Theorem 14.6 in ) and the delta method (Theorem 3.1 in ) that  
| ​(𝒇⁡(,,…,)−𝒇⁡)\displaystyle\sqrt{N}\left(\boldsymbol{f}(\hat{\pi}_{0},\hat{\pi}_{r_{1}},\ldots,\hat{\pi}_{r_{R}})-\boldsymbol{f}(\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}})\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{M}\nabla\boldsymbol{f}),  |  
| --- | --- |  
where \boldsymbol{\Sigma}_{M} was defined in (). Then  
| ​​(−)\displaystyle\sqrt{NT}(\boldsymbol{\theta}^{*})_{\mathcal{S}}^{\top}\left((\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\mathcal{S}}-(\boldsymbol{\psi}_{N}^{(\text{C; }\pi)})_{\mathcal{S}}\right)  |  
| --- |  
| =\displaystyle=\penalty\   | ​​([​(,,…,)−​]​​)\displaystyle\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})_{\mathcal{S}}^{\top}\left(\sum_{r\in\mathcal{R}}\left[f_{r}(\hat{\pi}_{r_{1}},\hat{\pi}_{r_{2}},\ldots,\hat{\pi}_{r_{R}})-f_{r}(\pi_{r_{1}},\pi_{r_{2}},\ldots,\pi_{r_{R}})\right]\sum_{t=r}^{T}\psi_{rt}(\boldsymbol{D}_{N}^{-1})_{i(r,t),\mathcal{S}}\right)  |  
| =\displaystyle=\penalty\   | ​​(​[𝒇⁡(,,…,)−𝒇⁡])\displaystyle\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})_{\mathcal{S}}^{\top}\left(\boldsymbol{M}^{\top}\left[\boldsymbol{f}(\hat{\pi}_{r_{1}},\hat{\pi}_{r_{2}},\ldots,\hat{\pi}_{r_{R}})-\boldsymbol{f}(\pi_{r_{1}},\pi_{r_{2}},\ldots,\pi_{r_{R}})\right]\right)  |  
| \displaystyle\xrightarrow{d}\penalty\   | 𝒩,\displaystyle\mathcal{N}(0,\boldsymbol{M}^{\top}\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{M}\nabla\boldsymbol{f}\boldsymbol{M}),  |  
where  
| 𝑴:=(​​​)∈.\boldsymbol{M}:=\begin{pmatrix}\sum_{t=r_{1}}^{T}\psi_{r_{1}t}(\boldsymbol{D}_{N}^{-1})_{i(r_{1},t),\mathcal{S}}\\\ \sum_{t=r_{2}}^{T}\psi_{r_{2}t}(\boldsymbol{D}_{N}^{-1})_{i(r_{2},t),\mathcal{S}}\\\ \vdots\\\ \\\ \sum_{t=r_{R}}^{T}\psi_{r_{R}t}(\boldsymbol{D}_{N}^{-1})_{i(r_{R},t),\mathcal{S}}\end{pmatrix}\in\mathbb{R}^{R\times s}.  |  
| --- |  
Further, we can estimate this covariance matrix by \boldsymbol{\hat{V}}(\mathcal{A}_{N}) defined for any \mathcal{A}_{N}\subseteq[p_{N}] by  
| =T⋅∇∇,\boldsymbol{\hat{V}}(\mathcal{A}_{N})=T\cdot\boldsymbol{\hat{M}}(\mathcal{A}_{N})^{\top}\nabla\boldsymbol{\hat{f}}^{\top}\boldsymbol{\hat{\Sigma}}_{M}\nabla\boldsymbol{\hat{f}}\boldsymbol{\hat{M}}(\mathcal{A}_{N}),  |  
| --- |  
where  
| :=(​​​)∈,\boldsymbol{\hat{M}}(\mathcal{A}_{N}):=\begin{pmatrix}\sum_{t=r_{1}}^{T}\psi_{r_{1}t}(\boldsymbol{D}_{N}^{-1})_{i(r_{1},t),\mathcal{A}_{N}}\\\ \sum_{t=r_{2}}^{T}\psi_{r_{2}t}(\boldsymbol{D}_{N}^{-1})_{i(r_{2},t),\mathcal{A}_{N}}\\\ \vdots\\\ \\\ \sum_{t=r_{R}}^{T}\psi_{r_{R}t}(\boldsymbol{D}_{N}^{-1})_{i(r_{R},t),\mathcal{A}_{N}}\end{pmatrix}\in\mathbb{R}^{R\times|\mathcal{A}_{N}|},  |  
| --- |  
| :=​()∈,\boldsymbol{\hat{\Sigma}}_{M}:=\frac{1}{N^{2}}\begin{pmatrix}N_{0}(N-N_{0})&-N_{0}N_{r_{1}}&\cdots&-N_{0}N_{r_{R}}\\\ -N_{0}N_{r_{1}}&N_{r_{1}}(N-N_{r_{1}})&\cdots&-N_{r_{1}}N_{r_{R}}\\\ \vdots&\vdots&\ddots&\vdots\\\ -N_{0}N_{r_{R}}&-N_{r_{1}}N_{r_{R}}&\cdots&N_{r_{R}}(N-N_{r_{R}})\end{pmatrix}\in\mathbb{R}^{(R+1)\times(R+1)},  |  
| --- |  
and \nabla\boldsymbol{\hat{f}} is defined by replacing each instance of in \nabla\boldsymbol{f} with its estimator \hat{\pi}_{r}. Using the consistency of \boldsymbol{\hat{\theta}}^{(q)} and the sample covariance matrix, the continuous mapping theorem yields that  
| ​​​T​​​​​,(\boldsymbol{\hat{\theta}}^{(q)})_{\mathcal{S}}^{\top}\boldsymbol{\hat{V}}(\mathcal{S})\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}}\xrightarrow{p}T(\boldsymbol{\theta}_{N}^{*})_{\mathcal{S}}^{\top}(\nabla\boldsymbol{\psi}_{N}^{(\text{C; }\pi)})_{\mathcal{S}}^{\top}\boldsymbol{\Sigma}_{M}(\nabla\boldsymbol{\psi}_{N}^{(\text{C; }\pi)})_{\mathcal{S}}(\boldsymbol{\theta}_{N}^{*})_{\mathcal{S}},  |  
| --- |  
where \boldsymbol{\Sigma}_{M} was defined in (). It follows from Theorem (g) that if at least one of the nonzero components of \boldsymbol{\psi}_{N}^{(\text{C; }\pi)} corresponds to one of the features in , we have  
| ​(​−)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{\text{C};\hat{\pi}}}}\left((\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-(\boldsymbol{\psi}_{N}^{(\text{C; }\pi)})^{\top}\boldsymbol{\theta}_{N}^{*}\right)\xrightarrow{d}\mathcal{N}(0,1)  |  
| --- |  
| \displaystyle\iff\qquad  | ​​(​(,,…,)​​−​​​)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{\text{C};\hat{\pi}}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(f_{r}(\hat{\pi}_{r_{1}},\hat{\pi}_{r_{2}},\ldots,\hat{\pi}_{r_{R}})\hat{\tau}_{\text{ATT}}(r,t)-f_{r}(\pi_{r_{1}},\pi_{r_{2}},\ldots,\pi_{r_{R}})\tau_{\text{ATT}}(r,t)\right)  |  
| \displaystyle\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1),  |  
where  
| :=()+.\hat{v}_{N}^{\text{C};\hat{\pi}}:=\sigma^{2}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\hat{\mathcal{S}}}^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\hat{\mathcal{S}}}\right)\right)^{-1}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\hat{\mathcal{S}}}+(\boldsymbol{\hat{\theta}}^{(q)})_{\hat{\mathcal{S}}}^{\top}\boldsymbol{\hat{V}}(\hat{\mathcal{S}})(\boldsymbol{\hat{\theta}}^{(q)})_{\hat{\mathcal{S}}}.  |  
| --- |  
  3. (c)
Part (b) follows from the same argument as part (a) but applying Theorem (h) to yield the stated convergence when using the conservative variance estimator  
| NC, (cons);\displaystyle\hat{v}_{N}^{\text{C, (cons)};\hat{\pi}}  |  
| --- |  
| \displaystyle:=\penalty\   | ()+\displaystyle\sigma^{2}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\hat{\mathcal{S}}}^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\hat{\mathcal{S}}}\right)\right)^{-1}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\hat{\mathcal{S}}}+(\boldsymbol{\hat{\theta}}^{(q)})_{\hat{\mathcal{S}}}^{\top}\boldsymbol{\hat{V}}(\hat{\mathcal{S}})(\boldsymbol{\hat{\theta}}^{(q)})_{\hat{\mathcal{S}}}  |  
| +2​σ()(((𝒁𝑫)))()⋅()()().\displaystyle+2\sqrt{\sigma^{2}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\hat{\mathcal{S}}}^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\hat{\mathcal{S}}}\right)\right)^{-1}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})})_{\hat{\mathcal{S}}}\cdot(\boldsymbol{\hat{\theta}}^{(q)})_{\hat{\mathcal{S}}}^{\top}\boldsymbol{\hat{V}}(\hat{\mathcal{S}})(\boldsymbol{\hat{\theta}}^{(q)})_{\hat{\mathcal{S}}}}.  |  
(Recall that the random vector \boldsymbol{\hat{\psi}}_{N}^{(\text{C; }\hat{\pi})} was defined in Equation and the covariance matrix estimator \boldsymbol{\hat{V}}(\hat{\mathcal{S}}) was defined in Equation .)


###### Proof of Theorem .
Part (a) is immediate from Theorem (a). To prove parts (b) and (c), in light of Theorem (b) and (c) it suffices to show that (,…,)(N_{r_{1}}/N_{\tau},\ldots,N_{r_{R}}/N_{\tau}) is almost surely finite and has a Jacobian at (\pi_{0},\pi_{r_{1}},\ldots,\pi_{r_{R}}) that exists and is nonzero and finite, which we already showed in the proof of Theorem .
###### Proof of Theorem .
In the case where the treatment effects are nonzero, this result follows from the same arguments that we used in the proof of Theorem , but applying Theorem (d) and (f). The oracle properties persist even in the case that the treatment effects equal 0 because, as noted in Theorem , in this case the random variables scaled by converge in probability (and distribution) to 0. ∎
### E.3 Proof of Theorem 
  1. (a)
From Theorem , under our assumptions the regression estimands correspond to our causal estimands. Similarly to the proof of Theorem , we will prove the result by applying Theorem (g) and Lemma . Let i(r,t,\boldsymbol{x}) denote the indices of rows of \boldsymbol{D}_{N} corresponding to \hat{\rho}_{rt}^{(q)} (that is, the columns of \boldsymbol{\tilde{Z}} corresponding to the indicator variables for those regression coefficients), so that  
| =​,\boldsymbol{\rho}_{rt}^{*}=(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\boldsymbol{\theta}_{N}^{*},  |  
| --- |  
and for any r\in\mathcal{R}, , and 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i},  
| \displaystyle\tau_{\text{CATT}}(r,t,\boldsymbol{x})=\penalty\   | +⊤​\displaystyle\tau_{rt}^{*}+\left(\boldsymbol{x}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)^{\top}\boldsymbol{\rho}_{rt}^{*}  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​+⊤​​.\displaystyle(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}\boldsymbol{\theta}_{N}^{*}+\left(\boldsymbol{x}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)^{\top}(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\boldsymbol{\theta}_{N}^{*}.  |  
Define  
| :=​(+⊤​)\boldsymbol{\psi}_{N}^{(\text{C;CATT})}:=\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left((\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}+\left(\boldsymbol{x}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)^{\top}(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\right)  |  
| --- |  
and  
| :=​(+​),\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})}:=\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left((\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}+(\boldsymbol{x}-\boldsymbol{\overline{X}}_{r})^{\top}(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\right),  |  
| --- |  
so  
| ​−​=​(−).\displaystyle(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-(\boldsymbol{\psi}_{N}^{(\text{C;CATT})})^{\top}\boldsymbol{\theta}_{N}^{*}=\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right).  |  
| --- |  
Since \boldsymbol{\hat{\theta}}^{(q)} and each \boldsymbol{\overline{X}}_{r} are estimated on two independent data sets, \boldsymbol{\hat{\theta}}^{(q)} is independent of \boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})}. So to apply Theorem (g) we need to establish the asymptotic normality of ​(−)⊤​\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})}-\boldsymbol{\psi}_{N}^{(\text{C;CATT})}\right)^{\top}\boldsymbol{\theta}_{N}^{*} and identify a consistent estimator for its asymptotic variance. Applying Lemma we see that  
| =\displaystyle=\penalty\   | ​(−)⊤​\displaystyle\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})}-\boldsymbol{\psi}_{N}^{(\text{C;CATT})}\right)^{\top}\boldsymbol{\theta}_{N}^{*}  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​(​(()⊤​))​\displaystyle\sqrt{NT}\left(\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left((\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]-\boldsymbol{\overline{X}}_{r})^{\top}(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\right)\right)\boldsymbol{\theta}_{N}^{*}  |  
| =\displaystyle=\penalty\   | (−()⊤(−𝔼))\displaystyle\sqrt{NT}\left(\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}-\psi_{rt}\left((\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\boldsymbol{\theta}_{N}^{*}\right)^{\top}\left(\boldsymbol{\overline{X}}_{r}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)\right)  |  
| \displaystyle\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}\left(0,v_{R}^{\text{CATT}}\right),  |  
where  
| \displaystyle v_{R}^{\text{CATT}}:=\penalty\   | T​​Cov​​\displaystyle T\sum_{r\in\mathcal{R}}\frac{(\boldsymbol{\psi}_{r}^{\text{CATT}})^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\psi}_{r}^{\text{CATT}}}{\mathbb{P}(W_{i}=r)}  |  
| --- | --- |  
for  
| \displaystyle\boldsymbol{\psi}_{r}^{\text{CATT}}:=\penalty\   | ​(​)⊤=,.\displaystyle\sum_{t=r}^{T}\psi_{rt}\left((\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\boldsymbol{\theta}_{N}^{*}\right)^{\top}=\sum_{t=r}^{T}\psi_{rt}\boldsymbol{\rho}_{rt}^{*},\qquad r\in\mathcal{R}.  |  
| --- | --- |  
Notice that each v_{R}^{\text{CATT}} is strictly positive and finite because each \mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r) is assumed to be finite and have a strictly positive minimum eigenvalue under Assumption (R6) and each \mathbb{P}(W_{i}=r) is strictly positive under Assumption (F2). Also, by the consistency of \boldsymbol{\hat{\theta}}^{(q)} and each and \widehat{\mathrm{Cov}}(\boldsymbol{X}_{i}\mid W_{i}=r), along with the continuous mapping theorem, we have that  
| T​​​​T\sum_{r\in\mathcal{R}}\frac{(\boldsymbol{\hat{\psi}}_{r}^{\text{CATT}})^{\top}\widehat{\mathrm{Cov}}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\hat{\psi}}_{r}^{\text{CATT}}}{N_{r}/N}\xrightarrow{p}v_{R}^{\text{CATT}}  |  
| --- |  
where  
| :=,.\boldsymbol{\hat{\psi}}_{r}^{\text{CATT}}:=\sum_{t=r}^{T}\psi_{rt}\boldsymbol{\hat{\rho}}_{rt}^{(q)},\qquad r\in\mathcal{R}.  |  
| --- |  
Then from Theorem (g) we have  
| ​(​−​)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C;CATT})}}}\left((\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-(\boldsymbol{\psi}_{N}^{(\text{C;CATT})})^{\top}\boldsymbol{\theta}_{N}^{*}\right)\xrightarrow{d}\mathcal{N}(0,1)  |  
| --- |  
| ⇔(−)𝒩\displaystyle\iff\qquad\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C;CATT})}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\left(\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})-\tau_{\text{CATT}}(r,t,\boldsymbol{x})\right)\xrightarrow{d}\mathcal{N}(0,1)  |  
for  
| \displaystyle\hat{v}_{N}^{(\text{C;CATT})}:=\penalty\   | ()\displaystyle\sigma^{2}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})})^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\hat{\mathcal{S}}}\right)\right)^{-1}\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT})}  |  
| --- | --- |  
| +T​​​.\displaystyle+T\sum_{r\in\mathcal{R}}\frac{(\boldsymbol{\hat{\psi}}_{r}^{\text{CATT}})^{\top}\widehat{\mathrm{Cov}}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\hat{\psi}}_{r}^{\text{CATT}}}{N_{r}/N}.  |  
  2. (b)
Observe that  
| ​​(,,…,)​​\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\tau_{\text{CATT}}(r,t,\boldsymbol{x})  |  
| --- |  
| =\displaystyle=\penalty\   | ​​(,,…,)​(⊤CLOSE\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\Big((\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}^{\top}  |  
| OPEN+⊤​)​\displaystyle+(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}^{\top}\left(\boldsymbol{x}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)\Big)\boldsymbol{\theta}_{N}^{*}  |  
| =\displaystyle=\penalty\   | ​​(,,…,)​​\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)(\boldsymbol{\psi}_{rt}^{*})^{\top}\boldsymbol{\theta}_{N}^{*}  |  
| =\displaystyle=\penalty\   | ()⊤​\displaystyle(\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)})^{\top}\boldsymbol{\theta}_{N}^{*}  |  
where  
| :=+⊤​∈\boldsymbol{\psi}_{rt}^{*}:=(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}+(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}^{\top}\left(\boldsymbol{x}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)\in\mathbb{R}^{p}  |  
| --- |  
and  
| :=​​(,,…,)​∈.\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)}:=\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\boldsymbol{\psi}_{rt}^{*}\in\mathbb{R}^{p}.  |  
| --- |  
Define the estimators  
| :=+⊤​\boldsymbol{\hat{\psi}}_{rt}^{*}:=(\boldsymbol{D}_{N}^{-1})_{i(r,t),\cdot}+(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}^{\top}\left(\boldsymbol{x}-\boldsymbol{\overline{X}}_{r}\right)  |  
| --- |  
and  
| :=​​(,,…,)​\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)}:=\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)\boldsymbol{\hat{\psi}}_{rt}^{*}  |  
| --- |  
and notice that  
| ()⊤​−()⊤​\displaystyle(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-(\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)})^{\top}\boldsymbol{\theta}_{N}^{*}  |  
| --- |  
| =\displaystyle=\penalty\   | ​(​(,,…,)​​CLOSE\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\big(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})  |  
| OPEN−​(,,…,)​​).\displaystyle-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\tau_{\text{CATT}}(r,t,\boldsymbol{x})\big).  |  
Since by assumption each \hat{\pi}_{r}(\boldsymbol{x}) and \boldsymbol{\overline{X}}_{r} is independent of \boldsymbol{\hat{\theta}}^{(q)}, \boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)} is independent of \boldsymbol{\hat{\theta}}^{(q)}, and the result follows from applying Theorem (g) if we can show that  
| ​​(−)\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\left(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)}-\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)}\right)  |  
| --- |  
is asymptotically normal and identify a consistent estimator for the asymptotic variance. Showing this is conceptually straightforward—we can decompose ​​(−)\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\left(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)}-\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)}\right) into a term that converges in probability to 0, a term that has an asymptotic distribution depending on \sqrt{N}\left(\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right) (asymptotically normal due to Lemma ), and a term that depends on the \hat{\pi}_{r}(\boldsymbol{x}) predictions (asymptotically normal by assumption and application of the delta method). Then the sum converges to a normal distribution because of the independence of each \hat{\pi}_{r}(\boldsymbol{x}) and \boldsymbol{\overline{X}}_{r} and Slutsky’s theorem.
However, the calculations are sufficiently tedious that we defer the work to the proof of Lemma .
###### .
Under the assumptions of Theorem (b),  
| ​​(−)𝒩⁡(0,)\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\left(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)}-\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)}\right)\xrightarrow{d}\mathcal{N}\left(0,v_{R}^{(\text{C;CATT,}\pi)}\right)  |  
| --- |  
| v:=∑(𝝍rCATT; prob)⊤​Cov​​𝝍rCATT; prob+T⋅𝝍∇𝒇𝚺∇𝒇𝝍,v_{R}^{(\text{C;CATT,}\pi)}:=\sum_{r\in\mathcal{R}}\frac{(\boldsymbol{\psi}_{r}^{\text{CATT; prob}})^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\psi}_{r}^{\text{CATT; prob}}}{\mathbb{P}(W_{i}=r)}+T\cdot\boldsymbol{\psi}_{\pi}^{\top}\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{\pi}\nabla\boldsymbol{f}\boldsymbol{\psi}_{\pi},  |  
| --- |  
𝝍rCATT; prob\boldsymbol{\psi}_{r}^{\text{CATT; prob}} is defined in (), and \boldsymbol{\Sigma}_{\pi} was defined in the statement of Theorem . Further, \hat{v}_{R}^{(\text{C;CATT,}\pi)}, defined in (), is a consistent estimator for v_{R}^{(\text{C;CATT,}\pi)}.
###### Proof.
Then from Lemma and Theorem (g) we have  
| ​(()⊤​−()⊤​)\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C;CATT,}\pi)}}}\left((\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)})^{\top}\boldsymbol{\hat{\theta}}^{(q)}-(\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)})^{\top}\boldsymbol{\theta}_{N}^{*}\right)\xrightarrow{d}\mathcal{N}(0,1)  |  
| --- |  
| \displaystyle\iff\qquad  | ​​(​(,,…,)​​CLOSE\displaystyle\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{C;CATT,}\pi)}}}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\big(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)\hat{\tau}_{\text{CATT}}(r,t,\boldsymbol{x})  |  
| OPEN−​(,,…,)​​).\displaystyle-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\tau_{\text{CATT}}(r,t,\boldsymbol{x})\big)\xrightarrow{d}\mathcal{N}(0,1).  |  
for  
| :=\displaystyle\hat{v}_{N}^{(\text{C;CATT,}\pi)}:=\penalty\   | ()⊤()\displaystyle\sigma^{2}(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)})^{\top}\left(\widehat{\mathrm{Cov}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\hat{\mathcal{S}}}\right)\right)^{-1}\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)}  |  
| --- | --- |  
| +.\displaystyle+\hat{v}_{R}^{(\text{C;CATT,}\pi)}.  |  


## Appendix F Proof of Theorem 
Before we prove the individual statements, we outline our general approach. Notice that the assumptions leading to equation (6.33) in are satisfied, so we have for any   
| 𝔼=,\mathbb{E}\left[\boldsymbol{\tilde{y}}_{(i\cdot)}\mid W_{i},\boldsymbol{X}_{i}\right]=\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{\beta}_{N}^{*},  |  
| --- |  
where the subscript refers to the observations corresponding to unit . (This verifies part (a) of Theorem , though we also provide a proof in our notation below.) We propose estimating the bridge regression (),  
| =​{+}.\boldsymbol{\hat{\beta}}^{(q)}=\underset{\boldsymbol{\beta}\in\mathbb{R}^{p_{N}}}{\arg\min}\left\\{\lVert\boldsymbol{y}-\boldsymbol{Z}\boldsymbol{\beta}\rVert_{2}^{2}+\lambda_{N}\lVert\boldsymbol{D}_{N}\boldsymbol{\beta}\rVert_{q}^{q}\right\\}.  |  
| --- |  
We proved in Lemma that the differences matrix \boldsymbol{D}_{N} is invertible. We can therefore solve () through the reparameterization \boldsymbol{\hat{\theta}}^{(q)}=\boldsymbol{D}_{N}\boldsymbol{\hat{\beta}}^{(q)}, solving for  
| =​{+}\boldsymbol{\hat{\theta}}^{(q)}=\underset{\boldsymbol{\theta}\in\mathbb{R}^{p_{N}}}{\arg\min}\left\\{\lVert\boldsymbol{y}-\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\boldsymbol{\theta}\rVert_{2}^{2}+\lambda_{N}\lVert\boldsymbol{\theta}\rVert_{q}^{q}\right\\}  |  
| --- |  
using standard bridge regression. Then \boldsymbol{\hat{\beta}}^{(q)}=\boldsymbol{D}_{N}^{-1}\boldsymbol{\hat{\theta}}^{(q)} is identical to the solution to () [[Tibshirani and Taylor, 2011](https://arxiv.org/html/2312.05985#bib.bib115), Section 3].
Having cast () as a standard bridge regression optimization problem, our results will follow from Theorem 1, Lemma 3, and Theorem 2(ii) from if we show the assumptions are met in our setting where  
| \displaystyle\boldsymbol{\tilde{Y}}_{iN}=\penalty\   | \displaystyle\boldsymbol{\tilde{y}}_{(i\cdot)},  |  
| --- | --- |  
| \displaystyle\boldsymbol{\tilde{X}}_{iN}=\penalty\   | \displaystyle\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1},  |  
| \displaystyle\beta_{0}=\penalty\   | \displaystyle\boldsymbol{\theta}_{N}^{*},  |  
| \displaystyle\boldsymbol{c}_{iN}=\penalty\   | \displaystyle c_{i}\boldsymbol{1}_{T},  |  
| \displaystyle\boldsymbol{\tilde{\epsilon}}_{iN}=\penalty\   | \displaystyle\boldsymbol{u}_{(i\cdot)},  |  
| \displaystyle\boldsymbol{Y}_{iN}=\penalty\   | \displaystyle\boldsymbol{y}_{(i\cdot)},  |  
| \displaystyle\boldsymbol{X}_{iN}=\penalty\   | \displaystyle\boldsymbol{Z}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1},  |  
| \displaystyle\boldsymbol{X}_{N}=\penalty\   | \displaystyle\boldsymbol{Z}\boldsymbol{D}_{N}^{-1},  |  
| \displaystyle\boldsymbol{W}_{N}=\penalty\   | \displaystyle(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{S}},  |  
| \displaystyle\boldsymbol{\Sigma}_{1N}=\penalty\   | ,\displaystyle\boldsymbol{\hat{\Sigma}}((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{S}}),  |  
| \displaystyle\boldsymbol{\epsilon}_{iN}=\penalty\   | \displaystyle\sigma\boldsymbol{\Omega}^{-1/2}\boldsymbol{\epsilon}_{(i\cdot)},  |  
| \displaystyle\gamma=\penalty\   | \displaystyle q,\qquad\text{and}  |  
| \displaystyle k_{N}=\penalty\   | \displaystyle s_{N},  |  
where the left side of each equation uses the notation from and the right side uses the notation in our setting. We focus on the random effects setting of because we cannot perform the forward orthogonal deviations transform since \boldsymbol{\tilde{Z}} (and \boldsymbol{\tilde{Z}}\boldsymbol{D}_{N}^{-1}) contain (linear combinations of) time-invariant covariates and cohort indicators.
Now we are prepared to prove each statement.
  1. (a)
As mentioned earlier, under our assumptions correct specification is immediate from Equation (6.33) in . We also provide a derivation for the correct specification here in our notation. Observe that for any ,  
| 𝔼[∣,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(i1)}\mid W_{i},\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| --- |  
| =\displaystyle=\penalty\   | 𝟙𝔼[∣=0,=𝒙]+𝟙𝔼[∣=r,=𝒙]\displaystyle\mathbbm{1}\\{W_{i}=0\\}\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]+\sum_{r\in\mathcal{R}}\mathbbm{1}\\{W_{i}=r\\}\mathbb{E}\left[\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(a)}}{{=}}\penalty\   | 𝟙𝔼[∣=0,=𝒙]+𝟙𝔼[∣=r,=𝒙]\displaystyle\mathbbm{1}\\{W_{i}=0\\}\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]+\sum_{r\in\mathcal{R}}\mathbbm{1}\\{W_{i}=r\\}\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(b)}}{{=}}\penalty\   | 𝟙+𝟙(++)\displaystyle\mathbbm{1}\\{W_{i}=0\\}\left(\eta^{*}+\boldsymbol{x}^{\top}\boldsymbol{\kappa}^{*}\right)+\sum_{r\in\mathcal{R}}\mathbbm{1}\\{W_{i}=r\\}\left(\eta^{*}+\nu_{r}^{*}+\boldsymbol{x}^{\top}\left(\boldsymbol{\kappa}^{*}+\boldsymbol{\zeta}_{r}^{*}\right)\right)  |  
| =\displaystyle=\penalty\   | ++𝟙,\displaystyle\eta^{*}+\boldsymbol{x}^{\top}\boldsymbol{\kappa}^{*}+\sum_{r\in\mathcal{R}}\mathbbm{1}\\{W_{i}=r\\}\left(\nu_{r}^{*}+\boldsymbol{x}^{\top}\boldsymbol{\zeta}_{r}^{*}\right),  |  
where in we used Assumption (CNAS) and in we used () and () from (LINS). Next, for any t\in\\{2,\ldots,T\\} and such that ,  
| 𝔼[∣=0,]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}\mid W_{i}=0,\boldsymbol{X}_{i}\right]  |  
| --- |  
| =\displaystyle=\penalty\   | 𝔼[−∣=0,=𝒙]+𝔼[∣=0,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]+\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(c)}}{{=}}\penalty\   | ++,\displaystyle\eta^{*}+\gamma_{t}^{*}+\boldsymbol{x}^{\top}(\boldsymbol{\xi}_{t}^{*}+\boldsymbol{\kappa}^{*}),  |  
where in we used () from (LINS) and (). Then for any such that W_{i}=r\in\mathcal{R}\setminus\\{2\\} and t\in\\{2,\ldots,r-1\\},  
| 𝔼[∣=r,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| --- |  
| =\displaystyle=\penalty\   | 𝔼[∣=r,=𝒙]+𝔼[−∣=r,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]+\mathbb{E}\left[\tilde{y}_{(it)}(r)-\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(e)}}{{=}}\penalty\   | 𝔼[∣=r,=𝒙]+𝔼[−∣=r,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]+\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(f)}}{{=}}\penalty\   | 𝔼[∣=r,=𝒙]+𝔼[−∣=0,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]+\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(g)}}{{=}}\penalty\   | +++,\displaystyle\eta^{*}+\nu_{r}^{*}+\boldsymbol{x}^{\top}(\boldsymbol{\kappa}^{*}+\boldsymbol{\zeta}_{r}^{*}+\boldsymbol{\xi}_{t}^{*})+\gamma_{t}^{*},  |  
where in we used (CNAS), in we used (CCTSB), and in we used () and () from (LINS). Finally, for any such that W_{i}=r\in\mathcal{R} and t\in\\{r,\ldots,T\\},  
| 𝔼[∣=r,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| --- |  
| =\displaystyle=\penalty\   | 𝔼[∣=r,=𝒙]+𝔼[−∣=0,=𝒙]\displaystyle\mathbb{E}\left[\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]+\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| +𝔼[−∣=r,=𝒙]−𝔼[−∣=0,=𝒙]\displaystyle+\mathbb{E}\left[\tilde{y}_{(it)}(r)-\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}\right]-\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}\right]  |  
| \displaystyle\stackrel{{\scriptstyle(h)}}{{=}}\penalty\   | +++++,\displaystyle\eta^{*}+\nu_{r}^{*}+\boldsymbol{x}^{\top}(\boldsymbol{\kappa}^{*}+\boldsymbol{\zeta}_{r}^{*}+\boldsymbol{\xi}_{t}^{*})+\gamma_{t}^{*}+\tau_{rt}^{*}+\boldsymbol{\dot{x}}_{r}\boldsymbol{\rho}_{rt}^{*},  |  
where in we used () and () and () from (LINS). Therefore regression () is correctly specified and () holds.
  2. (b)
First we will show that Assumptions (FE1) - (FE3), (RE4), (A1), (A3), (A5), and (A6) from are satisfied. Then we will show that the conclusion of Theorem 1 in about \boldsymbol{\theta}_{N}^{*} leads to our statement about \boldsymbol{\beta}_{N}^{*}. We will use three lemmas:
###### .
The smallest and largest singular values of \boldsymbol{D}_{N} are bounded as follows: \sigma_{\text{max}}(\boldsymbol{D}_{N})\leq 3 and ≥\sigma_{\text{min}}\left(\boldsymbol{D}_{N}\right)\geq\frac{1}{T\sqrt{2T}}.
###### .
Under Assumption (R1) it holds that  
| 𝔼[4]≤∀i∈,j∈,t∈\mathbb{E}\left[(\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1})_{(it)j}^{4}\right]\leq\kappa_{4}\qquad\forall i\in[N],j\in[p_{N}],t\in[T]  |  
| --- |  
for some finite \kappa_{4}>0.
###### .
Under the assumptions of Theorem (b),  
| =​≥\displaystyle\rho_{1N}=\lambda_{\text{min}}\left(\boldsymbol{\hat{\Sigma}}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)\right)\geq\penalty\   | \displaystyle\frac{1}{9}e_{1N}\qquad\text{and}  |  
| --- | --- |  
| =​≤\displaystyle\rho_{2N}=\lambda_{\text{max}}\left(\boldsymbol{\hat{\Sigma}}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)\right)\leq\penalty\   | \displaystyle e_{2N}\cdot 2T^{3}.  |  
The proofs of all three of these lemmas are provided in Appendix . Now we will verify one assumption at a time.
     * •
In our notation, Assumption (FE1) from requires that (,,)N(\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1},c_{i},\boldsymbol{u}_{(i\cdot)})_{i=1}^{N} are iid. This is satisfied by our Assumption (F2) since \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot} is a deterministic, invertible function of (W_{i},\boldsymbol{X}_{i}), and \boldsymbol{D}_{N}^{-1} is also deterministic and invertible conditional on (W_{i},\boldsymbol{X}_{i}).
     * •
Assumption (FE2) requires that 𝔼⁡[4]\mathbb{E}[(\boldsymbol{\tilde{Z}}\boldsymbol{D}_{N}^{-1})_{(it)j}^{4}] and \mathbb{E}[u_{(it)}^{4}] are finite for all ,,i\in[N],j\in[d_{N}],t\in[T]. We assumed the latter in Assumption (R1). We proved that the former is satisfied in Lemma .
     * •
Next, Assumption (FE3) requires that  
| 𝔼[∣,]=\displaystyle\mathbb{E}\left[\boldsymbol{u}_{(i\cdot)}\mid\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1},c_{i}\right]=\penalty\   | \displaystyle\boldsymbol{0}\qquad\text{and}  |  
| --- | --- |  
| Var[∣,]=\displaystyle\mathrm{Var}\left[\boldsymbol{u}_{(i\cdot)}\mid\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1},c_{i}\right]=\penalty\   | \displaystyle\sigma^{2}\boldsymbol{I}_{T},\qquad i\in[N],  |  
and Assumption (RE4) requires  
| 𝔼[∣]=\displaystyle\mathbb{E}\left[c_{i}\mid\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1}\right]=\penalty\   | \displaystyle 0,  |  
| --- | --- |  
| Var[∣]=\displaystyle\mathrm{Var}\left[c_{i}\mid\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1}\right]=\penalty\   | \displaystyle\sigma_{c}^{2},  |  
and that and \sigma_{c}^{2} are known and finite. Our Assumption (F1) is sufficient for this because conditioning on W_{i},\boldsymbol{X}_{i} is the same as conditioning on \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1} by the following argument. Because \boldsymbol{D}_{N}^{-1} is invertible, the -algebra generated by \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1}, \sigma(\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1}), contains the same information as \sigma(\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}). Similarly, for each (W_{i},\boldsymbol{X}_{i}) there is only one valid \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}, and every valid \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot} corresponds to a unique (W_{i},\boldsymbol{X}_{i})—that is, the mapping from (W_{i},\boldsymbol{X}_{i}) to \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot} is also invertible, so σ=σ\sigma(W_{i},\boldsymbol{X}_{i})=\sigma(\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}).
     * •
Our Assumptions (R1), (R2), and (R3) exactly match assumptions (A1), (A3), and (A5), respectively, from (in our setting where we assume is fixed).
     * •
Finally, Assumption (A6) in requires that  
| 0,\frac{p_{N}+\lambda_{N}s_{N}}{N\rho_{1N}}\xrightarrow{a.s.}0,  |  
| --- |  
where is the smallest eigenvalue of the random matrix \boldsymbol{\hat{\Sigma}}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right). Under Lemma the smallest eigenvalue of \boldsymbol{\hat{\Sigma}}(\boldsymbol{Z}), , is within a constant factor of , so our Assumption (R2) suffices to satisfy this condition.
We have shown that the assumptions of Theorem 1 in are satisfied, so we have =​\lVert\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\theta}_{N}^{*}\rVert_{2}=\mathcal{O}_{\mathbb{P}}\left(\min\\{h_{N},h^{\prime}_{N}\\}\right). Finally, part (2) is proven because \left\lVert\boldsymbol{\hat{\beta}}^{(q)}-\boldsymbol{\beta}_{N}^{*}\right\rVert_{2} is within a constant factor of \lVert\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\theta}_{N}^{*}\rVert_{2}:  
| \displaystyle\lVert\boldsymbol{\hat{\beta}}^{(q)}-\boldsymbol{\beta}_{N}^{*}\rVert_{2}=\penalty\   | ‖​‖2≤​≤T​​,\displaystyle\left\lVert\boldsymbol{D}_{N}^{-1}\left(\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\theta}_{N}^{*}\right)\right\rVert_{2}\leq\left\lVert\boldsymbol{D}_{N}^{-1}\right\rVert_{\text{op}}\left\lVert\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\theta}_{N}^{*}\right\rVert_{2}\leq T\sqrt{2T}\left\lVert\boldsymbol{\hat{\theta}}^{(q)}-\boldsymbol{\theta}_{N}^{*}\right\rVert_{2},  |  
| --- | --- |  
where in the last step we applied Lemma .
  3. (c)
In light of the results from part (a), part (c) follows from Theorem below, which extends Lemma 3 from , under our assumptions. In particular, we will show that our Assumptions (R4) and (R5) suffice for ’s Assumptions (A4) and (A7), as well as the needed part of Assumption (A2) for Lemma 3, in .
###### (Extension of Lemma 3 from ).
Under the assumptions of Lemma 3 from ,  
| =1.\lim_{N\to\infty}\mathbb{P}\left(\hat{\mathcal{S}}=\mathcal{S}\right)=1.  |  
| --- |  
###### Proof.
Provided in Appendix . ∎
###### .
It appears that the strategy for the proof of of Theorem could be used to analogously extend Lemma 2, and Theorem 2(i), in .
Again we verify one assumption at a time.
     * •
Although Lemma 3 from requires their Assumption (A2) in the statement of the theorem, examining the proof we see that a lower bound on the minimum eigenvalue is not needed—the only thing that is needed is an almost sure upper bound on the maximum eigenvalue of \boldsymbol{\hat{\Sigma}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{S}}\right), the empirical Gram matrix of the columns of \boldsymbol{Z}\boldsymbol{D}_{N}^{-1} corresponding to the relevant features in \boldsymbol{\theta}_{N}^{*}. (In the proof, note the use of on p. 140, and note that is not needed.) Our Assumption (R4) is sufficient for this, since we assume the maximum eigenvalue of the empirical Gram matrix for \boldsymbol{\tilde{Z}}, is finite, which ensures that the maximum eigenvalue of the Gram matrix for the full \boldsymbol{Z}\boldsymbol{D}_{N}^{-1} is finite due to Lemma . This maximum eigenvalue upper-bounds the maximum eigenvalue of the submatrix \boldsymbol{\hat{\Sigma}}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{S}}\right).
     * •
Our Assumption (R5) matches Assumption (A4) in in our setting where is fixed.
     * •
Finally, we show that Assumption (A7) in is satisfied under our Assumption (R4). We need to show that  
| =.e_{1N}\sqrt{\frac{e_{2N}}{p_{N}}}=\mathcal{O}_{p}(1).  |  
| --- |  
But since e_{1N}\leq e_{2N}, almost surely  
| ≤≤=,e_{1N}\sqrt{\frac{e_{2N}}{p_{N}}}\leq\sqrt{\frac{e_{2N}^{3}}{p_{N}}}\leq\sqrt{\frac{e_{\text{max}}^{3}}{p_{N}}}=\mathcal{O}\left(\frac{1}{\sqrt{p_{N}}}\right),  |  
| --- |  
so =e_{1N}\sqrt{e_{2N}/p_{N}}=\mathcal{O}_{p}(1).
So the first part of the result is now immediate. It only remains to show that  
| ℙ⁡=1.\lim_{N\to\infty}\mathbb{P}\left(\sqrt{NT}\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=\boldsymbol{0}\right)=1.  |  
| --- |  
From part , we have that =1\lim_{n\to\infty}\mathbb{P}\left(\boldsymbol{\hat{\theta}}_{\mathcal{S}^{C}}^{(q)}=\boldsymbol{0}\right)=1, so using =\\{\sqrt{NT}\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=0\\}=\\{\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=0\\},  
| ℙ⁡==1.\displaystyle\lim_{N\to\infty}\mathbb{P}\left(\sqrt{NT}\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=0\right)=\lim_{N\to\infty}\mathbb{P}\left(\boldsymbol{\hat{\theta}}^{(q)}_{\mathcal{S}^{c}}=0\right)=1.  |  
| --- |  
  4. (d)
To prove parts (d) - (h), we will present an extension of Theorem 2(ii) from that yields our result when in applied in our setting. Then we will show that the needed assumptions are satisfied to conclude the proof. In particular, our extension (a) obtains the asymptotic convergence of linear combinations of the coefficients of both relevant and irrelevant features; (b) establishes the asymptotic convergence of a statistic formed using the estimated Gram matrix on the selected set of coefficients, not the population covariance matrix of the true active set; (c) is like (a) when \boldsymbol{\psi}_{N} is estimated rather than known and is independent of \hat{\beta}_{N}; (d) is like (b) when \boldsymbol{\psi}_{N} is estimated rather than known; and (e) is like (d) when \hat{\beta}_{N} and \boldsymbol{\hat{\psi}}_{N} are estimated on the same data set, so they are dependent.
###### (Extension of Theorem 2(ii) from ).
Suppose the assumptions of Theorem 2(ii) from are satisfied.
    1. (a)
(Asymptotic normality when coefficients equal to 0 are included.) Let \\{\boldsymbol{\psi}_{N}\\}_{N=1}^{\infty} be a sequence of real-valued vectors where \boldsymbol{\psi}_{N}\in\mathbb{R}^{p_{N}} has the structure (\boldsymbol{\alpha}^{\top},\boldsymbol{b}_{N}^{\top})^{\top} where \boldsymbol{\alpha}\in\mathbb{R}^{k} is a fixed, finite, nonzero vector and \\{\boldsymbol{b}_{N}\\}_{N=1}^{\infty} is any sequence of constants where each \boldsymbol{b}_{N}\in\mathbb{R}^{p_{N}-k} contains all finite entries. Then  
| 𝒩⁡(0,​​()​𝜶).\sqrt{NT}\boldsymbol{\psi}_{N}^{\top}(\hat{\beta}_{N}-\beta_{0})\xrightarrow{d}\mathcal{N}\left(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}\left[\Sigma_{1N}\right]\right)^{-1}\boldsymbol{\alpha}\right).  |  
| --- |  
    2. (b)
(Asymptotic normality when coefficients equal to 0 are included and the covariance matrix and selected set are estimated.) Define a function 𝜶\boldsymbol{\alpha} that maps any \mathcal{A}\in\mathcal{P}(\mathbb{N}) to a fixed vector \boldsymbol{\alpha}(\mathcal{A})\in(\mathbb{R}\setminus\\{0\\})^{|\mathcal{A}|}. Define the sequence of vectors \\{\boldsymbol{\psi}_{N}(\mathcal{A}_{N})\\}_{N=1}^{\infty} to have components in \mathcal{A}_{N} equal to \boldsymbol{\alpha}(\mathcal{A}_{N}) and components \\{\boldsymbol{b}_{N}(\mathcal{A}_{N})\\}_{N=1}^{\infty} in the components corresponding to [p_{N}]\setminus\mathcal{A}_{N} that are all finite for all  (but are otherwise arbitrary). Define the sequence of random variables U_{b}(\mathcal{A}) to equal  
| ​𝜶(𝒜)((𝑿))𝜶(𝒜)⋅(),\frac{1}{\sigma}\sqrt{\frac{NT}{\boldsymbol{\alpha}(\mathcal{A})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{A}})\right)^{-1}\boldsymbol{\alpha}(\mathcal{A})}}\cdot\left(\boldsymbol{\psi}_{N}(\mathcal{A})^{\top}\left(\hat{\beta}-\beta_{0}\right)\right),  |  
| --- |  
if \mathcal{A}\neq\emptyset and \boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{A}}) is invertible and 0 otherwise, where \hat{\beta} is the bridge estimator of . Let \hat{\mathcal{S}}_{N}:=\\{j:\hat{\beta}_{Nj}\neq 0\\} be the estimated selected set on a data set of size , and suppose \beta_{0}\neq\boldsymbol{0}. Then if \boldsymbol{\alpha}(\mathcal{S})\neq\boldsymbol{0}, the sequence \\{U_{b}(\hat{\mathcal{S}}_{N})\\} converges in distribution to a standard Gaussian random variable.
    3. (c)
(Asymptotic normality when coefficients equal to 0 are included and the weights are estimated.) For any \mathcal{A}\in\mathcal{P}(\mathbb{N}), suppose that \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) is a random vector with the property that for a given data set, \mathcal{A}=\mathcal{A}^{\prime} implies \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A})=\boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}^{\prime}). Define the sequence of random vectors \\{\boldsymbol{\hat{\psi}}_{N}(\mathcal{A}_{N})\\}_{N=1}^{\infty} to have components in \mathcal{A}_{N} equal to \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) and components \\{\boldsymbol{\hat{b}}_{N}(\mathcal{A}_{N})\\}_{N=1}^{\infty} in the components corresponding to [p_{N}]\setminus\mathcal{A}_{N} that are all almost surely finite for all  (but are otherwise arbitrary). Assume that for any fixed \beta_{0}\in\mathbb{R}^{p_{N}} with the  entries in the  positions finite and the remaining entries all equal to 0,  
| ​​.\sqrt{NT}\beta_{0}^{\top}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{S})-\boldsymbol{\psi}_{N}(\mathcal{S})\right)\xrightarrow{d}\mathcal{N}(0,v_{\psi}(\beta_{0},\mathcal{S})).  |  
| --- |  
Define the sequence of random variables U_{c}(\mathcal{A}) to equal  
| ​(−),\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})^{\top}\hat{\beta}_{N}-\boldsymbol{\psi}_{N}(\mathcal{A})\beta_{0}\right),  |  
| --- |  
where \hat{\beta}_{N} is the bridge estimator of . Assume \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) is independent of \hat{\beta}_{N}. Let \hat{\mathcal{S}}_{N}:=\\{j:\hat{\beta}_{Nj}\neq 0\\} be the estimated selected set on a data set of size , and suppose \beta_{0}\neq\boldsymbol{0}. Then  
| 𝒩⁡(0,​​()​𝜶+).U_{c}(\hat{\mathcal{S}}_{N})\xrightarrow{d}\mathcal{N}\left(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}\left[\Sigma_{1N}\right]\right)^{-1}\boldsymbol{\alpha}+v_{\psi}(\beta_{0},\mathcal{S})\right).  |  
| --- |  
    4. (d)
(Asymptotic normality when coefficients equal to 0 are included and the weights, covariance matrix, and selected set are estimated.) Define \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) and the sequence of random vectors \\{\boldsymbol{\hat{\psi}}_{N}(\mathcal{A}_{N})\\}_{N=1}^{\infty} as in part , and assume they have the same properties. Define the sequence of random variables U_{d}(\mathcal{A}) to equal  
| ​(−)\sqrt{\frac{NT}{\hat{v}_{N}(\mathcal{A})}}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})^{\top}\hat{\beta}_{N}-\boldsymbol{\psi}_{N}(\mathcal{A})^{\top}\beta_{0}\right)  |  
| --- |  
if 𝒜≠∅, is invertible, and ≠𝟎\mathcal{A}\neq\emptyset,\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{A}})\text{ is invertible, and }\boldsymbol{\hat{\alpha}}_{N}(\mathcal{A})\neq\boldsymbol{0}; and equals 0 otherwise; where \hat{\beta}_{N} is the bridge estimator of ,  
| :=+,\hat{v}_{N}(\mathcal{A}):=\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{A})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{A}})\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{A})+\hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{A}),  |  
| --- |  
and \hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{A}) is an estimator of v_{\psi}(\beta_{0},\mathcal{A}) that satisfies \hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{S})\xrightarrow{p}v_{\psi}(\beta_{0},\mathcal{S}). Assume \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) is independent of \hat{\beta}_{N}, and suppose \beta_{0}\neq\boldsymbol{0}. Then U_{d}(\hat{\mathcal{S}}_{N}) converges in distribution to a standard Gaussian random variable.
    5. (e)
(Asymptotic subgaussianity when coefficients equal to 0 are included and the weights, covariance matrix, and selected set are estimated.) Let the sequence of random variables U_{e}(\mathcal{A}) be defined the same was as U_{d}(\mathcal{A}), except that we assume that \boldsymbol{\hat{\alpha}}_{N}(\mathcal{A}_{N}) is estimated on the same data as \hat{\beta}_{N}, so they are not independent, and we use a different variance estimator:  
| :=​(−)U_{e}(\mathcal{A}):=\sqrt{\frac{NT}{\hat{v}_{N}^{(\text{cons})}(\mathcal{A})}}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})^{\top}\hat{\beta}_{N}-\boldsymbol{\psi}_{N}(\mathcal{A})^{\top}\beta_{0}\right)  |  
| --- |  
(unless \hat{v}_{N}^{(\text{cons})}(\mathcal{A}) is not well-defined or equals 0, in which case U_{e}(\mathcal{A})=0), where  
| \displaystyle\hat{v}_{N}^{(\text{cons})}(\mathcal{A}):=\penalty\   | +\displaystyle\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{A})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{A}})\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{A})+\hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{A})  |  
| --- | --- |  
| +2​σ(𝒜)((𝑿))(𝒜)⋅(,𝒜)\displaystyle+2\sqrt{\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{A})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{A}})\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{A})\cdot\hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{A})}  |  
is a conservative variance estimator. Assume again that \hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{S})\xrightarrow{p}v_{\psi}(\beta_{0},\mathcal{S}) and \beta_{0}\neq\boldsymbol{0}. Then U_{e}(\hat{\mathcal{S}}_{N}) converges in distribution to a mean-zero subgaussian random variable with variance at most 1.
###### Proof.
Provided in Appendix . ∎
###### .
It appears that the strategies used to prove Theorem could be used to analogously extend Theorem 2(ii) in .
It only remains to show that the assumptions of Theorem are satisfied in our setting. First, we note that states on p. 123 that a fixed is enough to satisfy the last two assumptions of Theorem 2(ii). We have already shown that Assumptions (FE1), (FE2), (FE3), (RE4), (A1), (A3), (A4), (A5), (A6), and (A7) in are satisfied. Due to Theorem 3(i) from , our fixed setting ensures that the uniform integrability assumption of Theorem 2(ii) from is satisfied. We have assumed fixed sparsity in Assumption S(), and on p. 122 explains why our fixed setting satisfies all of the remaining assumptions except for (A2).
We conclude by showing that the conditions implied by Assumption (A2) in are satisfied. makes an assumption that the minimum eigenvalue of the empirical Gram matrix of the columns of \boldsymbol{Z}\boldsymbol{D}_{N}^{-1} corresponding to the nonzero features in \boldsymbol{\theta}_{N}^{*},  
| ,\frac{1}{NT}\left((\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{S}}\right)^{\top}(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1})_{(\cdot\cdot)\mathcal{S}},  |  
| --- |  
is bounded away from 0 almost surely. Call this minimum eigenvalue e_{1\mathcal{S}N}. This assumption is used in two places, both in the proof of Theorem 2(ii).
    1. (a)
On p. 142, this assumption is used to establish that  
| ​​​​\displaystyle\frac{q}{2}\frac{1}{e_{1\mathcal{S}N}}\left(\frac{b_{0}}{2}\right)^{q-1}\lambda_{N}\sqrt{\frac{s_{N}}{NT}}\xrightarrow{p}\penalty\   | 0\displaystyle 0  |  
| --- | --- |  
| \displaystyle\iff\frac{1}{e_{1\mathcal{S}N}}\frac{\lambda_{N}}{\sqrt{N}}\xrightarrow{p}\penalty\   | 0\displaystyle 0  |  
| \displaystyle\iff\frac{1}{e_{1\mathcal{S}N}}\in\penalty\   | \displaystyle o_{\mathbb{P}}\left(\frac{\sqrt{N}}{\lambda_{N}}\right),  |  
where the first equivalence follows because the removed quantities are constant under our assumptions (specifically, that and are fixed). Note that our assumption () ensures that this condition holds for the full empirical Gram matrix due to Lemma . The minimum eigenvalue of the full empirical Gram matrix lower bounds e_{1\mathcal{S}N}, so () is sufficient for this condition to hold.
    2. (b)
The assumption is used again on p. 144 of , where it is sufficient for the minimum eigenvalue of the (population) matrix  
| 𝔼[]\mathbb{E}\left[\boldsymbol{\hat{\Sigma}}\left(\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)_{(\cdot\cdot)\mathcal{S}}\right)\right]  |  
| --- |  
to be bounded away from 0. This is a submatrix of (), so the minimum eigenvalue of () is a lower bound for the minimum eigenvalue of 𝔼[]\mathbb{E}\left[\boldsymbol{\hat{\Sigma}}\left(\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)_{(\cdot\cdot)\mathcal{S}}\right)\right]. Therefore our assumption in (R6) that the minimum eigenvalue of () is bounded away from 0 is enough to satisfy this assumption.
So our Assumption (R6), in combination with all of the previous assumptions, is enough to apply Theorem , and the proof is complete.


## Appendix G Proofs of Results Stated in the Appendix
In Appendix we prove results that were stated but not proven in Appendix . In Appendix we prove supporting results for our main theorems.
### G.1 Proofs of Results from Appendix 
###### Proof of Theorem .
First we provide a counterexample to show that (CCTS) does not imply (CTS). This counterexample is inspired by discussions in Sections 2 and 3 of that touch on closely related issues. For simplicity, let and \mathcal{R}=\\{2\\}. Assume that treatment status is not independent of the covariates, so ≠\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=2]\neq\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=0]. Consider a setting where () from Assumption (LINS) holds with \boldsymbol{\xi}_{2}^{*}\neq\boldsymbol{0}:  
| 𝔼[−∣=0,]=+a.s.,\mathbb{E}\left[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}\right]=\gamma_{2}^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\xi}_{2}^{*}\qquad a.s.,  |  
| --- |  
and the same condition holds for the untreated potential outcomes of the treated units:  
| 𝔼[−∣=2,]=+a.s.\mathbb{E}\left[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=2,\boldsymbol{X}_{i}\right]=\gamma_{2}^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\xi}_{2}^{*}\qquad a.s.  |  
| --- |  
(CCTS) holds since 𝔼[−∣=2,]=𝔼[−∣=0,]\mathbb{E}[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=2,\boldsymbol{X}_{i}]=\mathbb{E}[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}] almost surely. But for r\in\\{0,2\\},  
| 𝔼⁡[−∣]=\displaystyle\mathbb{E}[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r]=\penalty\   | 𝔼[𝔼[−∣=r,]∣=r]\displaystyle\mathbb{E}\left[\mathbb{E}\left[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}\right]\mid W_{i}=r\right]  |  
| --- | --- |  
| =\displaystyle=\penalty\   | +𝔼​​,\displaystyle\gamma_{2}^{*}+\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]^{\top}\boldsymbol{\xi}_{2}^{*},  |  
so 𝔼⁡[−∣]≠𝔼⁡[−∣]\mathbb{E}[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=2]\neq\mathbb{E}[\tilde{y}_{(i2)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0] since ≠\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=2]\neq\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=0].
Next we show an example where (CTS) holds but (CCTS) does not. Let \mathcal{R}=\\{2,\ldots,T\\}, and suppose \boldsymbol{X}_{i} has a discrete distribution, only taking on values \boldsymbol{x}_{1} and \boldsymbol{x}_{2}. To simplify notation, define :=−G_{it}:=\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0) for all t\in\\{2,\ldots,T\\}. Then (CTS) is equivalent to  
| \displaystyle\mathbb{E}[G_{it}\mid W_{i}]=\penalty\   | ,\displaystyle\mathbb{E}[G_{it}],\qquad i\in[N],t\in\\{2,\ldots,T\\}  |  
| --- | --- |  
| \displaystyle\iff\qquad\mathbb{E}[G_{it}\mid W_{i}=0]=\penalty\   | ,,,\displaystyle\mathbb{E}[G_{it}\mid W_{i}=r],\qquad r\in\mathcal{R},i\in[N],t\in\\{2,\ldots,T\\}  |  
and (CCTS) is equivalent to, for all and t\in\\{2,\ldots,T\\},  
| \displaystyle\mathbb{E}[G_{it}\mid W_{i},\boldsymbol{X}_{i}]=\penalty\   | \displaystyle\mathbb{E}[G_{it}\mid\boldsymbol{X}_{i}]  |  
| --- | --- |  
| ⇔𝔼[∣=0,=]=\displaystyle\iff\qquad\mathbb{E}[G_{it}\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}_{j}]=\penalty\   | 𝔼[∣=r,=],j∈,r∈ℛ.\displaystyle\mathbb{E}[G_{it}\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}_{j}],\qquad j\in\\{1,2\\},r\in\mathcal{R}.  |  
Our goal is to construct an example where () holds and () does not hold. Suppose that for some , for all , t\in\\{2,\ldots,T\\}  
| 𝔼[∣=0,=]=\displaystyle\mathbb{E}[G_{it}\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}_{1}]=\penalty\   | \displaystyle-a,  |  
| --- | --- |  
| 𝔼[∣=r,=]=\displaystyle\mathbb{E}[G_{it}\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}_{1}]=\penalty\   | \displaystyle-ra,\qquad r\in\mathcal{R},  |  
| 𝔼[∣=0,=]=\displaystyle\mathbb{E}[G_{it}\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}_{2}]=\penalty\   | \displaystyle a,\qquad\text{and}  |  
| 𝔼[∣=r,=]=\displaystyle\mathbb{E}[G_{it}\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}_{2}]=\penalty\   | \displaystyle ra,\qquad r\in\mathcal{R}.  |  
By inspection, one can see that () does not hold. We are done if we show that () does. Suppose and \boldsymbol{X}_{i} are independent: let ==\mathbb{P}(W_{i}=w,\boldsymbol{X}_{i}=\boldsymbol{x}_{j})=1/(2[R+1])=1/(2T) for w\in\\{0\\}\cup\mathcal{R} and j\in\\{1,2\\}. We have  
| \displaystyle\mathbb{E}[G_{it}\mid W_{i}=0]  |  
| --- |  
| =\displaystyle=\penalty\   | (𝔼[∣=0,=]ℙ\displaystyle\frac{1}{\mathbb{P}(W_{i}=0)}\big(\mathbb{E}[G_{it}\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}_{1}]\mathbb{P}(W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}_{1})  |  
| +𝔼[∣=0,=]ℙ)\displaystyle+\mathbb{E}[G_{it}\mid W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}_{2}]\mathbb{P}(W_{i}=0,\boldsymbol{X}_{i}=\boldsymbol{x}_{2})\big)  |  
| =\displaystyle=\penalty\   | T\displaystyle T\left(-a\cdot\frac{1}{2T}+a\cdot\frac{1}{2T}\right)  |  
| =\displaystyle=\penalty\   | 0\displaystyle 0  |  
and for any r\in\mathcal{R},  
| \displaystyle\mathbb{E}[G_{it}\mid W_{i}=r]  |  
| --- |  
| =\displaystyle=\penalty\   | (𝔼[∣=r,=]ℙ\displaystyle\frac{1}{\mathbb{P}(W_{i}=r)}\big(\mathbb{E}[G_{it}\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}_{1}]\mathbb{P}(W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}_{1})  |  
| +𝔼[∣=r,=]ℙ)\displaystyle+\mathbb{E}[G_{it}\mid W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}_{2}]\mathbb{P}(W_{i}=r,\boldsymbol{X}_{i}=\boldsymbol{x}_{2})\big)  |  
| =\displaystyle=\penalty\   | T(−ra⋅+ra⋅)\displaystyle T\left(-ra\cdot\frac{1}{2T}+ra\cdot\frac{1}{2T}\right)  |  
| =\displaystyle=\penalty\   | \displaystyle 0.  |  
###### Proof of Theorem (b) and (c).
For any r\in\mathcal{R} and t\in\\{r,\ldots,T\\},  
| \displaystyle\tau_{rt}^{*}  |  
| --- |  
| =\displaystyle=\penalty\   | 𝔼⁡[+()⊤​∣]\displaystyle\mathbb{E}[\tau_{rt}^{*}+\left(\boldsymbol{X}_{i}-\mathbb{E}\left[\boldsymbol{X}_{i}\mid W_{i}=r\right]\right)^{\top}\boldsymbol{\rho}_{rt}^{*}\mid W_{i}=r]  |  
| \displaystyle\stackrel{{\scriptstyle(a)}}{{=}}\penalty\   | 𝔼[𝔼[−∣=r,]∣=r]−𝔼[𝔼[−∣=0,]∣=r]\displaystyle\mathbb{E}\left[\mathbb{E}\left[\tilde{y}_{(it)}(r)-\tilde{y}_{(i1)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}\right]\mid W_{i}=r\right]-\mathbb{E}\left[\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}\right]\mid W_{i}=r\right]  |  
| \displaystyle\stackrel{{\scriptstyle(b)}}{{=}}\penalty\   | 𝔼[𝔼[∣=r,]∣=r]−𝔼[(𝔼[∣=r,]\displaystyle\mathbb{E}\left[\mathbb{E}[\tilde{y}_{(it)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}]\mid W_{i}=r\right]-\mathbb{E}\bigg[\bigg(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]  |  
| +𝔼[−∣=0,])∣=r]\displaystyle+\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}]\bigg)\mid W_{i}=r\bigg]  |  
| \displaystyle\stackrel{{\scriptstyle(c)}}{{=}}\penalty\   | 𝔼[𝔼[∣=r,]∣=r]−𝔼[(𝔼[∣=r,]\displaystyle\mathbb{E}\left[\mathbb{E}[\tilde{y}_{(it)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}]\mid W_{i}=r\right]-\mathbb{E}\bigg[\bigg(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]  |  
| +𝔼[−∣=0])∣=r]\displaystyle+\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0]\bigg)\mid W_{i}=r\bigg]  |  
| =\displaystyle=\penalty\   | 𝔼⁡[]−(𝔼⁡[]+𝔼⁡[−∣])\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(r)\mid W_{i}=r\right]-\Big(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r]+\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0\right]\Big)  |  
| \displaystyle\stackrel{{\scriptstyle(d)}}{{=}}\penalty\   | 𝔼⁡[]−𝔼⁡[]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(r)\mid W_{i}=r\right]-\mathbb{E}\left[\tilde{y}_{(it)}(0)\mid W_{i}=r\right]  |  
| =\displaystyle=\penalty\   | \displaystyle\tau_{\text{ATT}}(r,t),  |  
where in we used () from (LINS), in we used (CNAS), in we used (CIUN), and in we used (CTSA). This proves part (b).
Next we prove part (c). Observe that under () and () from (LINS), for any r\in\mathcal{R} and t\in\\{r,\ldots,T\\}, starting from () we have  
| \displaystyle\tau_{rt}^{*}  |  
| --- |  
| =\displaystyle=\penalty\   | 𝔼[𝔼[∣=r,]∣=r]−𝔼[(𝔼[∣=r,]\displaystyle\mathbb{E}\left[\mathbb{E}[\tilde{y}_{(it)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}]\mid W_{i}=r\right]-\mathbb{E}\bigg[\bigg(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]  |  
| +𝔼[−∣=0,])∣=r]\displaystyle+\mathbb{E}[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}]\bigg)\mid W_{i}=r\bigg]  |  
| \displaystyle\stackrel{{\scriptstyle(e)}}{{=}}\penalty\   | 𝔼[𝔼[∣=r,]∣=r]−𝔼[(𝔼[∣=r,]\displaystyle\mathbb{E}\left[\mathbb{E}[\tilde{y}_{(it)}(r)\mid W_{i}=r,\boldsymbol{X}_{i}]\mid W_{i}=r\right]-\mathbb{E}\bigg[\bigg(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r,\boldsymbol{X}_{i}]  |  
| ∣=r]\displaystyle+\gamma_{t}^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\xi}_{t}^{*}\bigg)\mid W_{i}=r\bigg]  |  
| =\displaystyle=\penalty\   | 𝔼⁡[]−(𝔼⁡[]++𝔼​​)\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(r)\mid W_{i}=r\right]-\left(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r]+\gamma_{t}^{*}+\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]^{\top}\boldsymbol{\xi}_{t}^{*}\right)  |  
| \displaystyle\stackrel{{\scriptstyle(f)}}{{=}}\penalty\   | 𝔼⁡[]−(𝔼⁡[]CLOSE\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(r)\mid W_{i}=r\right]-\Big(\mathbb{E}[\tilde{y}_{(i1)}(0)\mid W_{i}=r]  |  
| +𝔼[−∣=0,=𝔼])\displaystyle+\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right]\Big)  |  
| =\displaystyle=\penalty\   | 𝔼⁡[−∣]−𝔼⁡[−∣]\displaystyle\mathbb{E}\left[\tilde{y}_{(it)}(r)-\tilde{y}_{(i1)}(0)\mid W_{i}=r\right]-\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0\right]  |  
| +𝔼[−∣=0]−𝔼[−∣=0,=𝔼]\displaystyle+\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0\right]-\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}=\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right]  |  
| \displaystyle\stackrel{{\scriptstyle(g)}}{{=}}\penalty\   | +𝔼[𝔼[−∣=0,]∣=0]−(+𝔼)\displaystyle\tau_{\text{ATT}}(r,t)+\mathbb{E}\left[\mathbb{E}\left[\tilde{y}_{(it)}(0)-\tilde{y}_{(i1)}(0)\mid W_{i}=0,\boldsymbol{X}_{i}\right]\mid W_{i}=0\right]-\left(\gamma_{t}^{*}+\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]^{\top}\boldsymbol{\xi}_{t}^{*}\right)  |  
| \displaystyle\stackrel{{\scriptstyle(h)}}{{=}}\penalty\   | +𝔼⁡[∣]−(+𝔼​​)\displaystyle\tau_{\text{ATT}}(r,t)+\mathbb{E}\left[\gamma_{t}^{*}+\boldsymbol{X}_{i}^{\top}\boldsymbol{\xi}_{t}^{*}\mid W_{i}=0\right]-\left(\gamma_{t}^{*}+\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]^{\top}\boldsymbol{\xi}_{t}^{*}\right)  |  
| =\displaystyle=\penalty\   | ++𝔼​​−(+𝔼​​)\displaystyle\tau_{\text{ATT}}(r,t)+\gamma_{t}^{*}+\mathbb{E}\left[\boldsymbol{X}_{i}\mid W_{i}=0\right]^{\top}\boldsymbol{\xi}_{t}^{*}-\left(\gamma_{t}^{*}+\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]^{\top}\boldsymbol{\xi}_{t}^{*}\right)  |  
| =\displaystyle=\penalty\   | +(−)⊤​,\displaystyle\tau_{\text{ATT}}(r,t)+\left(\mathbb{E}\left[\boldsymbol{X}_{i}\mid W_{i}=0\right]-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)^{\top}\boldsymbol{\xi}_{t}^{*},  |  
where in (e), (f), and (h) we used () from (LINS) and (g) uses an equality we found under (CTSA) in the proof of part (b) above as well as () from (LINS).
### G.2 Proofs of Supporting Results for Main Theorems
###### Proof of Lemma .
First, \mathfrak{W}\leq(T-1)^{2} because treatment effects are interactions between the R\leq T-1 cohort indicators and (at most) treatment times.
Next, for any t\in\mathbb{N}, define  
| :=()∈.\boldsymbol{D}^{(1)}(t):=\begin{pmatrix}1&-1&0&0&\cdots&0&0\\\ 0&1&-1&0&\cdots&0&0\\\ 0&0&1&-1&\cdots&0&0\\\ \vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\\ 0&0&0&0&\cdots&1&-1\\\ 0&0&0&0&\cdots&0&1\end{pmatrix}\in\mathbb{R}^{t\times t}.  |  
| --- |  
Note that this is the linear transformation we apply to several groups of coefficients in 𝜷\boldsymbol{\beta}. For example, for the vector of cohort fixed effects \boldsymbol{\nu}\in\mathbb{R}^{R},  
| =,\boldsymbol{D}^{(1)}(R)\boldsymbol{\nu}=\begin{pmatrix}\nu_{2}-\nu_{1}\\\ \vdots\\\ \nu_{R}-\nu_{R-1}\\\ \nu_{R}\end{pmatrix},  |  
| --- |  
which are the transformed coefficients to which we apply an penalty. We similarly penalize 𝜸\boldsymbol{\gamma} as well as \boldsymbol{\zeta}_{j}, the coefficients for the interactions of the cohort fixed effects and covariate (separately for each of the covariates) and \boldsymbol{\xi}_{j}, the coefficients for the interactions of the time fixed effects and covariate . So if we rearrange the columns of \boldsymbol{\tilde{Z}} corresponding to the cohort-covariate interactions so that all of the cohort interactions with the base effect of feature come first, then the cohort interactions with feature 2, and so on, we can write this Rd_{N}\times Rd_{N} block diagonal part of \boldsymbol{D}_{N} as \boldsymbol{I}_{d_{N}}\otimes\boldsymbol{D}^{(1)}(R), where denotes the Kronecker product. We can likewise rearrange the columns of \boldsymbol{\tilde{Z}} and write the part of \boldsymbol{D}_{N} corresponding to the time-covariate interactions as \boldsymbol{I}_{d_{N}}\otimes\boldsymbol{D}^{(1)}(T-1).
We penalize the coefficients 𝜿\boldsymbol{\kappa} directly, so this part of 𝑫\boldsymbol{D} can be represented simply as \boldsymbol{I}_{d_{N}}.
Next we consider the matrix \boldsymbol{D}^{(2)}(\mathcal{R}) we use to transform the base treatment effects 𝝉\boldsymbol{\tau} for penalization. The matrix \boldsymbol{D}^{(2)}(\mathcal{R}) has a similar structure to \boldsymbol{D}^{(1)}(t), except that we directly penalize each cohort’s first treatment effect, not the last one, and we penalize each cohort’s first treatment effect towards the previous cohort’s first treatment effect. So denoting \mathcal{R}=(r_{1},r_{2},\ldots,r_{R}), \boldsymbol{D}^{(2)}(\mathcal{R}) has a nearly block diagonal structure  
| :=()∈,\boldsymbol{D}^{(2)}(\mathcal{R}):=\begin{pmatrix}\boldsymbol{\tilde{D}}^{(1)}(1)^{\top}&\boldsymbol{0}&\boldsymbol{0}&\cdots&\boldsymbol{0}&\boldsymbol{0}\\\ \boldsymbol{U}(r_{1})&\boldsymbol{\tilde{D}}^{(1)}(2)^{\top}&\boldsymbol{0}&\cdots&\boldsymbol{0}&\boldsymbol{0}\\\ \boldsymbol{0}&\boldsymbol{U}(r_{2})&\boldsymbol{\tilde{D}}^{(1)}(3)^{\top}&\cdots&\boldsymbol{0}&\boldsymbol{0}\\\ \vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\\ \boldsymbol{0}&\boldsymbol{0}&\boldsymbol{0}&\cdots&\boldsymbol{U}(r_{R})&\boldsymbol{\tilde{D}}^{(1)}(R)^{\top}\end{pmatrix}\in\mathbb{R}^{\mathfrak{W}\times\mathfrak{W}},  |  
| --- |  
where :=\boldsymbol{\tilde{D}}(x):=\boldsymbol{D}^{(1)}(T-r_{x}+1) and  
| :=()∈ℝ.\boldsymbol{U}(w):=\begin{pmatrix}-1&\boldsymbol{0}_{1\times(T-w)}\\\ \boldsymbol{0}_{(T-w)\times 1}&\boldsymbol{0}_{(T-w)\times(T-w)}\end{pmatrix}\in\mathbb{R}^{(T-w+1)\times(T-w+1)}.  |  
| --- |  
Then \boldsymbol{D}^{(2)}(\mathcal{R})\boldsymbol{\tau} yields the transformed treatment effects to which we apply an penalty, and similarly, for every j\in[d_{N}], \boldsymbol{D}^{(2)}(\mathcal{R})\boldsymbol{\rho}_{j} yields the transformed coefficients of the interactions between the j^{\text{th}} centered covariate and the treatment effects. Putting this all together, after re-ordering the columns appropriately we can write \boldsymbol{D}_{N} as ().
Next we show that these matrices are invertible by explicitly presenting their inverses. One can verify by matrix multiplication that  
| =()∈,\boldsymbol{D}^{(1)}(t)^{-1}=\begin{pmatrix}1&1&1&\cdots&1\\\ 0&1&1&\cdots&1\\\ 0&0&1&\cdots&1\\\ \vdots&\vdots&\vdots&\ddots&\vdots\\\ 0&0&0&\cdots&1\end{pmatrix}\in\mathbb{R}^{t\times t},  |  
| --- |  
Similarly,  
| \displaystyle(\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}=\penalty\   | (),\displaystyle\begin{pmatrix}\boldsymbol{\tilde{D}}^{-1}(1)^{\top}&\boldsymbol{0}&\cdots&\boldsymbol{0}&\boldsymbol{0}\\\ \boldsymbol{\tilde{U}}(r_{1},r_{2})&\boldsymbol{\tilde{D}}^{-1}(2)^{\top}&\cdots&\boldsymbol{0}&\boldsymbol{0}\\\ \vdots&\vdots&\ddots&\vdots&\vdots\\\ \boldsymbol{\tilde{U}}(r_{1},r_{R})&\boldsymbol{\tilde{U}}(r_{2},r_{R})&\cdots&\boldsymbol{\tilde{U}}(r_{R-1},r_{R})&\boldsymbol{\tilde{D}}^{-1}(R)^{\top}\end{pmatrix},  |  
| --- | --- |  
where  
| =()∈ℝ×.\boldsymbol{\tilde{U}}(w_{1},w_{2})=\begin{pmatrix}1&0&0&\cdots&0\\\ 1&0&0&\cdots&0\\\ \vdots&\vdots&\vdots&\ddots&\vdots\\\ 1&0&0&\cdots&0\end{pmatrix}\in\mathbb{R}^{(T-w_{2}+1)\times(T-w_{1}+1)}.  |  
| --- |  
Finally, using the properties of block diagonal matrices and the fact that \boldsymbol{D}^{(1)}(t) and \boldsymbol{D}^{(2)}(\mathcal{R}) are invertible, we have from () that \boldsymbol{D}_{N}^{-1} is  
| \displaystyle\operatorname{diag}\Big(  | ,(,,,⊗(CLOSECLOSE,\displaystyle\boldsymbol{D}^{(1)}(R)^{-1},((\boldsymbol{D}^{(1)}(T-1))^{-1},\boldsymbol{I}_{d_{N}},\boldsymbol{I}_{d_{N}}\otimes\boldsymbol{D}^{(1)}(R)^{-1},\boldsymbol{I}_{d_{N}}\otimes((\boldsymbol{D}^{(1)}(T-1))^{-1},  |  
| --- | --- |  
| OPEN,),\displaystyle(\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1},\boldsymbol{I}_{d_{N}}\otimes(\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}\Big),  |  
matching the expression from ().
Lastly, we show the claimed block diagonal structure of \boldsymbol{D}_{N} and \boldsymbol{D}_{N}^{-1}. Examining the expressions for \boldsymbol{D}_{N} and \boldsymbol{D}_{N}^{-1}, it is clear that they are both block diagonal with blocks of size R\leq T-1, , and \mathfrak{W}\leq(T-1)^{2}. We have shown in this proof that all of the blocks contain entries equal to 0, -1, or 1.
###### Proof of Lemma .
By Lemma , \boldsymbol{D}_{N}^{-1} has a block diagonal structure with blocks of size at most (T-1)^{2}\times(T-1)^{2} containing entries that equal either 0 or 1, so \lVert\boldsymbol{D}_{N}^{-1}\rVert_{\infty}\leq(T-1)^{2}, where for a matrix \lVert\cdot\rVert_{\infty} denotes the maximum absolute row sum. Therefore  
| \displaystyle\lVert\boldsymbol{\beta}_{N}^{*}\rVert_{\infty}=\penalty\   | ≤≤,\displaystyle\left\lVert\boldsymbol{D}_{N}^{-1}\boldsymbol{\theta}_{N}^{*}\right\rVert_{\infty}\leq\left\lVert\boldsymbol{D}_{N}^{-1}\right\rVert_{\infty}\left\lVert\boldsymbol{\theta}_{N}^{*}\right\rVert_{\infty}\leq(T-1)^{2}b_{1},  |  
| --- | --- |  
where in the last step we used Assumption (R3). ∎
###### Proof of Lemma .
To prove the result we will apply Theorem 5.1.7 in , which we reproduce here:
###### (Theorem 5.1.7 from ).
Let \boldsymbol{a}=(a_{1},\ldots,a_{k}) be a vector of constants, and let \\{\boldsymbol{U}^{(N)}\\} and \\{\boldsymbol{A}^{(N)}\\} be sequences of random variables, with =\boldsymbol{U}^{(N)}=(U_{1}^{(N)},\ldots,U_{k}^{(N)}) and =\boldsymbol{A}^{(N)}=(A_{1}^{(N)},\ldots,A_{k}^{(N)}). Suppose the sequence of random variables =T_{N}=\sum_{j=1}^{k}a_{j}U_{j}^{(N)} converges in distribution to a random variable , the random variables A_{j}^{(N)}\xrightarrow{p}a_{j} for each j\in\\{1,\ldots,k\\}, and the random variables U_{j}^{(N)}=\mathcal{O}_{\mathbb{P}}(1). Then the sequence of random variables \sum_{j=1}^{k}A_{j}^{(n)}U_{j}^{(N)} converges in distribution to .
Given the joint probability distribution \\{(W_{i},\boldsymbol{X}_{i})\\}_{i=1}^{N}, consider the iid random variables defined by  
| \displaystyle Z_{i}:=\penalty\   | ​​()\displaystyle\sum_{r\in\\{0\\}\cup\mathcal{R}}\frac{\mathbbm{1}\\{W_{i}=r\\}}{\mathbb{P}(W_{i}=r)}\boldsymbol{\psi}_{r}^{\top}\left(\boldsymbol{X}_{i}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r]\right)  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​​.\displaystyle\sum_{r\in\\{0\\}\cup\mathcal{R}}\frac{\mathbbm{1}\\{W_{i}=r\\}}{\mathbb{P}(W_{i}=r)}\boldsymbol{\psi}_{r}^{\top}\left(\boldsymbol{X}_{i}-\boldsymbol{\mu}_{r}\right).  |  
We have  
| \displaystyle\mathbb{E}[Z_{i}]=\penalty\   | =𝔼⁡[​​(−)]==0\displaystyle\mathbb{E}[\mathbb{E}[Z_{i}\mid W_{i}]]=\mathbb{E}\left[\frac{1}{\mathbb{P}(W_{i})}\boldsymbol{\psi}_{W_{i}}^{\top}\left(\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}]-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}]\right)\right]=\mathbb{E}\left[0\right]=0  |  
| --- | --- |  
| \displaystyle\mathrm{Var}(Z_{i})=\penalty\   | =\displaystyle\mathbb{E}[Z_{i}^{2}]=\mathbb{E}\left[\mathbb{E}\left[Z_{i}^{2}\mid W_{i}\right]\right]  |  
| --- | --- |  
| =\displaystyle=\penalty\   | 𝔼⁡[​​Cov​​]\displaystyle\mathbb{E}\left[\frac{1}{(\mathbb{P}(W_{i}))^{2}}\boldsymbol{\psi}_{W_{i}}^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i})\boldsymbol{\psi}_{W_{i}}\right]  |  
| =\displaystyle=\penalty\   | ​​Cov​​⋅\displaystyle\sum_{r\in\\{0\\}\cup\mathcal{R}}\frac{1}{\mathbb{P}(W_{i}=r)^{2}}\boldsymbol{\psi}_{r}^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\psi}_{r}\cdot\mathbb{P}(W_{i}=r)  |  
| =\displaystyle=\penalty\   | \displaystyle v_{R}.  |  
We can lower-bound uniformly across choices of \boldsymbol{\psi}_{r}: since \boldsymbol{\psi}_{r}\neq\boldsymbol{0} for at least one and by assumption the smallest eigenvalue of \mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r^{*}) is bounded below by some \lambda_{\text{min}}>0, we have  
| ≥​Cov​​≥0.v_{R}\geq\frac{\boldsymbol{\psi}_{r^{*}}^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r^{*})\boldsymbol{\psi}_{r^{*}}}{\mathbb{P}(W_{i}=r^{*})}\geq\frac{\lambda_{\text{min}}\lVert\boldsymbol{\psi}_{r^{*}}\rVert_{2}^{2}}{1}>0.  |  
| --- |  
Similarly, since the largest eigenvalue of ≤\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r)\leq\lambda_{\text{max}} for some finite \lambda_{\text{max}} for all r\in\\{0\\}\cup\mathcal{R} and ⁡0\min_{r\in\\{0\\}\cup\mathcal{R}}\\{\mathbb{P}(W_{i}=r)\\}>0 by Assumption (R1), we can upper-bound by a constant. So by the central limit theorem, for any choice of finite \\{\boldsymbol{\psi}_{r}\\} with at least one \boldsymbol{\psi}_{r^{*}}\neq\boldsymbol{0},  
| \displaystyle\frac{1}{\sqrt{N}}\sum_{i=1}^{N}Z_{i}\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,v_{R})  |  
| --- | --- |  
| ⇔\displaystyle\iff\qquad\frac{1}{\sqrt{N}}\sum_{r\in\\{0\\}\cup\mathcal{R}}\sum_{\\{i:W_{i}=r\\}}\frac{1}{\mathbb{P}(W_{i}=r)}\boldsymbol{\psi}_{r}^{\top}\left(\boldsymbol{X}_{i}-\boldsymbol{\mu}_{r}\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,v_{R})  |  
| ⇔(−)\displaystyle\iff\qquad\frac{1}{\sqrt{N}}\sum_{r\in\\{0\\}\cup\mathcal{R}}\frac{N_{r}}{\mathbb{P}(W_{i}=r)}\boldsymbol{\psi}_{r}^{\top}\left(\frac{1}{N_{r}}\sum_{\\{i:W_{i}=r\\}}\boldsymbol{X}_{i}-\boldsymbol{\mu}_{r}\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,v_{R})  |  
| ⇔\displaystyle\iff\qquad\sum_{r\in\\{0\\}\cup\mathcal{R}}\sqrt{N}\frac{N_{r}/N}{\mathbb{P}(W_{i}=r)}\boldsymbol{\psi}_{r}^{\top}\left(\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,v_{R}).  |  
Now consider the random variables  
| \displaystyle\boldsymbol{A}^{(N)}:=\penalty\   | (,⋅,…,⋅)\displaystyle\left(\pi_{0}\cdot\frac{N}{\sum_{i=1}^{N}\mathbbm{1}\\{W_{i}=0\\}},\pi_{r_{1}}\cdot\frac{N}{\sum_{i=1}^{N}\mathbbm{1}\\{W_{i}=r_{1}\\}},\ldots,\pi_{r_{R}}\cdot\frac{N}{\sum_{i=1}^{N}\mathbbm{1}\\{W_{i}=r_{R}\\}}\right)  |  
| --- | --- |  
| =\displaystyle=\penalty\   | (,,…,).\displaystyle\left(\pi_{0}\cdot\frac{N}{N_{0}},\pi_{r_{1}}\cdot\frac{N}{N_{r_{1}}},\ldots,\pi_{r_{R}}\cdot\frac{N}{N_{r_{R}}}\right).  |  
By the continuous mapping theorem \boldsymbol{A}^{(N)}\xrightarrow{p}\left(1,1,\ldots,1\right). Further, let  
| \displaystyle\boldsymbol{U}^{(N)}:=\penalty\   | (​​(),​​()CLOSE,\displaystyle\Bigg(\sum_{i=1}^{N}\frac{\mathbbm{1}\\{W_{i}=0\\}}{\pi_{0}}\boldsymbol{\psi}_{0}^{\top}\left(\boldsymbol{X}_{i}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=0]\right),\sum_{i=1}^{N}\frac{\mathbbm{1}\\{W_{i}=r_{1}\\}}{\pi_{r_{1}}}\boldsymbol{\psi}_{r_{1}}^{\top}\left(\boldsymbol{X}_{i}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r_{1}]\right),  |  
| --- | --- |  
| OPEN⋯,​​()).\displaystyle\cdots,\sum_{i=1}^{N}\frac{\mathbbm{1}\\{W_{i}=r_{R}\\}}{\pi_{r_{R}}}\boldsymbol{\psi}_{r_{R}}^{\top}\left(\boldsymbol{X}_{i}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r_{R}]\right)\Bigg).  |  
| ​,\sum_{r\in\\{0\\}\cup\mathcal{R}}1\cdot\frac{1}{\sqrt{N}}U_{r}^{(N)}\xrightarrow{d}\mathcal{N}(0,v_{R}),  |  
| --- |  
| ​\displaystyle\sum_{r\in\\{0\\}\cup\mathcal{R}}A_{r}^{(N)}\cdot\frac{1}{\sqrt{N}}U_{r}^{(N)}\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,v_{R})  |  
| --- | --- |  
| ⇔\displaystyle\iff\qquad\sum_{r\in\\{0\\}\cup\mathcal{R}}\sqrt{\frac{N}{v_{R}}}\boldsymbol{\psi}_{r}^{\top}\left(\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right)\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}(0,1).  |  
Finally we prove the last claim. It follows from the above that  
| N\lVert\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\rVert_{2}^{2}  |  
| --- |  
converges in distribution to a generalized chi-squared random variable, so by Theorem 2.4(i) in and Proposition 1.8(iii) in we have  
| \displaystyle N\lVert\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\rVert_{2}^{2}=\penalty\   | \displaystyle\mathcal{O}_{\mathbb{P}}(1)  |  
| --- | --- |  
| \displaystyle\implies\qquad\lVert\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\rVert_{2}^{2}=\penalty\   | \displaystyle\mathcal{O}_{\mathbb{P}}\left(\frac{1}{N}\right)  |  
| \displaystyle\implies\qquad\lVert\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\rVert_{2}=\penalty\   | \displaystyle\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right).  |  
###### Proof of Lemma .
For any r\in\mathcal{R},  
| |−|\displaystyle\left|\frac{\boldsymbol{a}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}-\frac{\boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}{\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right|  |  
| --- |  
| =\displaystyle=\penalty\   | |​𝝅​−​𝝅​​​​​​​​𝝅​|\displaystyle\left|\frac{\boldsymbol{a}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})\cdot\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})-\boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right|  |  
| =\displaystyle=\penalty\   | |​𝝅​−​𝝅​+​𝝅​−​𝝅​​​​​​​​𝝅​|\displaystyle\left|\frac{\boldsymbol{a}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})\cdot\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})-\boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})\cdot\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})+\boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})\cdot\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})-\boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right|  |  
| =\displaystyle=\penalty\   | |​+​​​​​𝝅​|\displaystyle\left|\frac{\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})\cdot\boldsymbol{a}^{\top}\left(\boldsymbol{\hat{\pi}}(\boldsymbol{x})-\boldsymbol{{\pi}}(\boldsymbol{x})\right)+\boldsymbol{a}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})\cdot\boldsymbol{b}^{\top}\left(\boldsymbol{{\pi}}(\boldsymbol{x})-\boldsymbol{\hat{\pi}}(\boldsymbol{x})\right)}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})\boldsymbol{b}^{\top}\boldsymbol{{\pi}}(\boldsymbol{x})}\right|  |  
| =\displaystyle=\penalty\   | ​​,\displaystyle\frac{1}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}\mathcal{O}_{\mathbb{P}}\left(a_{N}\right),  |  
where in the last step we used our consistency assumption and the finiteness of 𝒂\boldsymbol{a} and 𝒃\boldsymbol{b}. We are done if we can show that  
| =.\frac{1}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}=\mathcal{O}_{\mathbb{P}}(1).  |  
| --- |  
That is, we want to show that for any \epsilon>0 there exists a finite M_{\epsilon}>0 and N_{\epsilon}>0 such that for all N>N_{\epsilon},  
| ℙϵ⇔ℙϵ.\mathbb{P}\left(\frac{1}{\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})}>M_{\epsilon}\right)<\epsilon\qquad\iff\qquad\mathbb{P}\left(\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})<\frac{1}{M_{\epsilon}}\right)<\epsilon.  |  
| --- |  
Fix \epsilon>0. By assumption, for any 𝒙\boldsymbol{x} in the support of \boldsymbol{X}_{i} we have |​|=|\boldsymbol{b}^{\top}(\boldsymbol{\hat{\pi}}(\boldsymbol{x})-\boldsymbol{\pi}(\boldsymbol{x}))|=\mathcal{O}_{\mathbb{P}}(a_{N}), so there exists a finite M_{\epsilon}^{(1)}>0 and N_{\epsilon}^{(1)}>0 such that for all N>N_{\epsilon}^{(1)},  
| ℙ⁡(|​|)ϵ⟹\displaystyle\mathbb{P}\left(\frac{|\boldsymbol{b}^{\top}(\boldsymbol{\hat{\pi}}(\boldsymbol{x})-\boldsymbol{\pi}(\boldsymbol{x}))|}{a_{N}}>M_{\epsilon}^{(1)}\right)<\epsilon\qquad\implies\qquad  | ℙ⁡(−)ϵ.\displaystyle\mathbb{P}\left(\boldsymbol{b}^{\top}\boldsymbol{\hat{\pi}}(\boldsymbol{x})<\boldsymbol{b}^{\top}\boldsymbol{\pi}(\boldsymbol{x})-M_{\epsilon}^{(1)}a_{N}\right)<\epsilon.  |  
| --- | --- |  
Since by assumption \boldsymbol{b}^{\top}\boldsymbol{\pi}(\boldsymbol{x})>0, the result holds for N_{\epsilon}\geq N_{\epsilon}^{(1)} large enough so that for all N>N_{\epsilon} it holds that −0\boldsymbol{b}^{\top}\boldsymbol{\pi}(\boldsymbol{x})-M_{\epsilon}^{(1)}a_{N}>0 (which is assured since is decreasing), and :=1/(−)M_{\epsilon}:=1/(\boldsymbol{b}^{\top}\boldsymbol{\pi}(\boldsymbol{x})-M_{\epsilon}^{(1)}a_{N_{\epsilon}}). ∎
###### Proof of Lemma .
Observe that  
| −\displaystyle\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)}-\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)}  |  
| --- |  
| =\displaystyle=\penalty\   | ​(,,…,)​​−​(,,…,)​\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)\psi_{rt}\boldsymbol{\hat{\psi}}_{rt}^{*}-\sum_{r\in\mathcal{R}}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\sum_{t=r}^{T}\psi_{rt}\boldsymbol{\psi}_{rt}^{*}  |  
| =\displaystyle=\penalty\   | ​(,,…,)​​\displaystyle\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\psi_{rt}\left(\boldsymbol{\hat{\psi}}_{rt}^{*}-\boldsymbol{\psi}_{rt}^{*}\right)  |  
| +((,,…,)−(,,…,))\displaystyle+\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\left(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\right)\psi_{rt}\boldsymbol{\psi}_{rt}^{*}  |  
| +((,,…,)−(,,…,)),\displaystyle+\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\left(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\right)\psi_{rt}\left(\boldsymbol{\hat{\psi}}_{rt}^{*}-\boldsymbol{\psi}_{rt}^{*}\right),  |  
where \boldsymbol{\hat{\psi}}_{rt}^{*} was defined in () and \boldsymbol{\hat{\psi}}_{rt}^{*} was defined in (). We will show that after scaling by and multiplying by (\boldsymbol{\theta}_{N}^{*})^{\top}, terms of the first two forms converge in distribution to independent Gaussian random variables, and terms of the third form converge in probability to 0.
  * •
Applying Lemma , we have  
| ​​​(,,…,)​​\displaystyle\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\sum_{r\in\mathcal{R}}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\sum_{t=r}^{T}\psi_{rt}\left(\boldsymbol{\hat{\psi}}_{rt}^{*}-\boldsymbol{\psi}_{rt}^{*}\right)  |  
| --- |  
| =\displaystyle=\penalty\   | (,,…,)−(⊤)\displaystyle\sqrt{NT}\sum_{r\in\mathcal{R}}f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\sum_{t=r}^{T}-\psi_{rt}(\boldsymbol{\theta}_{N}^{*})^{\top}\left((\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}^{\top}\left(\boldsymbol{\overline{X}}_{r}-\boldsymbol{\mu}_{r}\right)\right)  |  
| \displaystyle\xrightarrow{d}\penalty\   | 𝒩⁡(0,vRCATT; prob),\displaystyle\mathcal{N}(0,v_{R}^{\text{CATT; prob}}),  |  
where  
| vRCATT; prob:=T⋅(𝝍rCATT; prob)⊤​Cov​​𝝍rCATT; probv_{R}^{\text{CATT; prob}}:=T\cdot\sum_{r\in\mathcal{R}}\frac{(\boldsymbol{\psi}_{r}^{\text{CATT; prob}})^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\psi}_{r}^{\text{CATT; prob}}}{\mathbb{P}(W_{i}=r)}  |  
| --- |  
for  
| 𝝍rCATT; prob:=\displaystyle\boldsymbol{\psi}_{r}^{\text{CATT; prob}}:=\penalty\   | ​(,,…,)​​​\displaystyle f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\sum_{t=r}^{T}\psi_{rt}(\boldsymbol{D}_{N}^{-1})_{i(r,t,\boldsymbol{x}),\cdot}\boldsymbol{\theta}_{N}^{*}  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​(,,…,)​,.\displaystyle f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\sum_{t=r}^{T}\psi_{rt}\boldsymbol{\rho}_{rt}^{*},\qquad r\in\mathcal{R}.  |  
  * •
Next, using the assumed joint asymptotic normality of (,,…,)(\hat{\pi}_{0}(\boldsymbol{x}),\hat{\pi}_{r_{1}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})) and the delta method (Theorem 3.1 in ), we have for each r\in\mathcal{R}  
| ​(​(,,…,)−​(,,…,),…CLOSE,\displaystyle\sqrt{NT}\big(f_{r_{1}}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)-f_{r_{1}}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right),\ldots,  |  
| --- |  
| OPEN​(,,…,)−​(,,…,))\displaystyle f_{r_{R}}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)-f_{r_{R}}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\big)  |  
| \displaystyle\xrightarrow{d}\penalty\   | \displaystyle\mathcal{N}\left(\boldsymbol{0},T\cdot\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{\pi}\nabla\boldsymbol{f}\right)  |  
Therefore  
| ​​(​(,,…,)−​(,,…,))​\displaystyle\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\sum_{r\in\mathcal{R}}\left(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\right)\sum_{t=r}^{T}\psi_{rt}\boldsymbol{\psi}_{rt}^{*}  |  
| --- |  
| =\displaystyle=\penalty\   | ​(​(,,…,)−​(,,…,))​​​\displaystyle\sqrt{NT}\sum_{r\in\mathcal{R}}\left(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\right)\sum_{t=r}^{T}\psi_{rt}(\boldsymbol{\theta}_{N}^{*})^{\top}\boldsymbol{\psi}_{rt}^{*}  |  
| \displaystyle\xrightarrow{d}\penalty\   | 𝒩(0,T⋅∇∇𝒇)\displaystyle\mathcal{N}\left(0,T\cdot\boldsymbol{\psi}_{\pi}^{\top}\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{\pi}\nabla\boldsymbol{f}\boldsymbol{\psi}_{\pi}\right)  |  
for  
| \displaystyle\boldsymbol{\psi}_{\pi}:=\penalty\   | (​​,​​,…,​​)\displaystyle\left(\sum_{t=r_{1}}^{T}\psi_{r_{1}t}(\boldsymbol{\theta}_{N}^{*})^{\top}\boldsymbol{\psi}_{r_{1}t}^{*},\sum_{t=r_{2}}^{T}\psi_{r_{2}t}(\boldsymbol{\theta}_{N}^{*})^{\top}\boldsymbol{\psi}_{r_{2}t}^{*},\ldots,\sum_{t=r_{R}}^{T}\psi_{r_{R}t}(\boldsymbol{\theta}_{N}^{*})^{\top}\boldsymbol{\psi}_{r_{R}t}^{*}\right)  |  
| --- | --- |  
| =\displaystyle=\penalty\   | (​(+()⊤​),…CLOSE,\displaystyle\bigg(\sum_{t=r_{1}}^{T}\psi_{r_{1}t}\left(\boldsymbol{\tau}_{r_{1}t}^{*}+\left(\boldsymbol{x}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r_{1}]\right)^{\top}\boldsymbol{\rho}_{r_{1}t}^{*}\right),\ldots,  |  
| OPEN​(+()⊤​)).\displaystyle\sum_{t=r_{R}}^{T}\psi_{r_{R}t}\left(\boldsymbol{\tau}_{r_{R}t}^{*}+\left(\boldsymbol{x}-\mathbb{E}[\boldsymbol{X}_{i}\mid W_{i}=r_{R}]\right)^{\top}\boldsymbol{\rho}_{r_{R}t}^{*}\right)\bigg).  |  
  * •
Finally, the above two results, along with Theorem 2.4(i) in and Proposition 1.8(iii) in , yield that for any r\in\mathcal{R}  
| (​(,,…,)−​(,,…,))​​\displaystyle\left(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\right)(\boldsymbol{\theta}_{N}^{*})^{\top}\sum_{t=r}^{T}\left(\boldsymbol{\hat{\psi}}_{rt}^{*}-\boldsymbol{\psi}_{rt}^{*}\right)  |  
| --- |  
| =\displaystyle=\penalty\   | =,\displaystyle\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right)\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right)=\mathcal{O}_{\mathbb{P}}\left(\frac{1}{N}\right),  |  
so  
| ​​(​(,,…,)CLOSE\displaystyle\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\sum_{r\in\mathcal{R}}\big(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)  |  
| --- |  
| OPEN−​(,,…,))​​\displaystyle-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\big)\sum_{t=r}^{T}\psi_{rt}\left(\boldsymbol{\hat{\psi}}_{rt}^{*}-\boldsymbol{\psi}_{rt}^{*}\right)  |  
| =\displaystyle=\penalty\   | ​​(​(,,…,)CLOSE\displaystyle\sqrt{NT}\sum_{r\in\mathcal{R}}\sum_{t=r}^{T}\psi_{rt}\big(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)  |  
| OPEN−​(,,…,))​​\displaystyle-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\big)(\boldsymbol{\theta}_{N}^{*})^{\top}\left(\boldsymbol{\hat{\psi}}_{rt}^{*}-\boldsymbol{\psi}_{rt}^{*}\right)  |  
| =\displaystyle=\penalty\   | \displaystyle\sqrt{N}\mathcal{O}_{\mathbb{P}}\left(\frac{1}{N}\right),  |  
and  
| ​​(​(,,…,)CLOSE\displaystyle\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\sum_{r\in\mathcal{R}}\big(f_{r}\left(\hat{\pi}_{r_{1}}(\boldsymbol{x}),\hat{\pi}_{r_{2}}(\boldsymbol{x}),\ldots,\hat{\pi}_{r_{R}}(\boldsymbol{x})\right)  |  
| --- |  
| OPEN−​(,,…,))​​0.\displaystyle-f_{r}\left(\pi_{r_{1}}(\boldsymbol{x}),\pi_{r_{2}}(\boldsymbol{x}),\ldots,\pi_{r_{R}}(\boldsymbol{x})\right)\big)\sum_{t=r}^{T}\psi_{rt}\left(\boldsymbol{\hat{\psi}}_{rt}^{*}-\boldsymbol{\psi}_{rt}^{*}\right)\xrightarrow{p}0.  |  


Putting this together and using the assumed independence, we have  
| ​​(−)𝒩⁡(0,)\sqrt{NT}(\boldsymbol{\theta}_{N}^{*})^{\top}\left(\boldsymbol{\hat{\psi}}_{N}^{(\text{C;CATT,}\pi)}-\boldsymbol{\psi}_{N}^{(\text{C;CATT,}\pi)}\right)\xrightarrow{d}\mathcal{N}\left(0,v_{R}^{(\text{C;CATT,}\pi)}\right)  |  
| --- |  
| v:=T⋅∑(𝝍rCATT; prob)⊤​Cov​​𝝍rCATT; prob+T⋅𝝍∇𝒇𝚺∇𝒇𝝍.v_{R}^{(\text{C;CATT,}\pi)}:=T\cdot\sum_{r\in\mathcal{R}}\frac{(\boldsymbol{\psi}_{r}^{\text{CATT; prob}})^{\top}\mathrm{Cov}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\psi}_{r}^{\text{CATT; prob}}}{\mathbb{P}(W_{i}=r)}+T\cdot\boldsymbol{\psi}_{\pi}^{\top}\nabla\boldsymbol{f}^{\top}\boldsymbol{\Sigma}_{\pi}\nabla\boldsymbol{f}\boldsymbol{\psi}_{\pi}.  |  
| --- |  
Finally, consider the plug-in estimator  
| :=T⋅∑(rCATT; prob)⊤​​​rCATT; prob+T⋅∇∇,\hat{v}_{R}^{(\text{C;CATT,}\pi)}:=T\cdot\sum_{r\in\mathcal{R}}\frac{(\boldsymbol{\hat{\psi}}_{r}^{\text{CATT; prob}})^{\top}\widehat{\mathrm{Cov}}(\boldsymbol{X}_{i}\mid W_{i}=r)\boldsymbol{\hat{\psi}}_{r}^{\text{CATT; prob}}}{N_{r}/N_{\tau}}+T\cdot\boldsymbol{\hat{\psi}}_{\pi}^{\top}\nabla\boldsymbol{\hat{f}}^{\top}\boldsymbol{\hat{\Sigma}}_{\pi}\nabla\boldsymbol{\hat{f}}\boldsymbol{\hat{\psi}}_{\pi},  |  
| --- |  
where  
| rCATT; prob:=\displaystyle\boldsymbol{\hat{\psi}}_{r}^{\text{CATT; prob}}:=\penalty\   | ​,,\displaystyle\frac{\hat{\pi}_{r}(\boldsymbol{x})}{\sum_{r^{\prime}\in\mathcal{R}}\hat{\pi}_{r^{\prime}}(\boldsymbol{x})}\sum_{t=r}^{T}\psi_{rt}\boldsymbol{\hat{\rho}}_{rt}^{(q)},\qquad r\in\mathcal{R},  |  
| --- | --- |  
| \displaystyle\boldsymbol{\hat{\psi}}_{\pi}:=\penalty\   | (​(+​),…CLOSE,\displaystyle\bigg(\sum_{t=r_{1}}^{T}\psi_{r_{1}t}\left(\boldsymbol{\hat{\tau}}_{r_{1}t}^{(q)}+\left(\boldsymbol{x}-\boldsymbol{\overline{X}}_{r_{1}}\right)^{\top}\boldsymbol{\hat{\rho}}_{r_{1}t}^{(q)}\right),\ldots,  |  
| OPEN​(+​)),\displaystyle\sum_{t=r_{R}}^{T}\psi_{r_{R}t}\left(\boldsymbol{\hat{\tau}}_{r_{R}t}^{(q)}+\left(\boldsymbol{x}-\boldsymbol{\overline{X}}_{r_{R}}\right)^{\top}\boldsymbol{\hat{\rho}}_{r_{R}t}^{(q)}\right)\bigg),  |  
and \nabla\boldsymbol{\hat{f}} is defined by replacing each instance of \pi_{r}(\boldsymbol{x}) in \nabla\boldsymbol{f} with its estimator \hat{\pi}_{r}(\boldsymbol{x}). Notice that \hat{v}_{R}^{(\text{C;CATT,}\pi)}\xrightarrow{p}v_{R}^{(\text{C;CATT,}\pi)} by the consistency of the individual terms and the continuous mapping theorem. ∎
## Appendix H Proofs of Lemmas From Proof of Theorem 
###### Proof of Lemma .
Notice from () and the expressions in the proof of Lemma that the norm of each row or column of \boldsymbol{D}_{N} is at most 3, so we have  
| ≤≤=3,\sigma_{\text{max}}(\boldsymbol{D}_{N})\leq\sqrt{\lVert\boldsymbol{D}_{N}\rVert_{1}\lVert\boldsymbol{D}_{N}\rVert_{\infty}}\leq\sqrt{3\cdot 3}=3,  |  
| --- |  
where for matrices \lVert\cdot\rVert_{1} denotes the maximum absolute column sum of a matrix and \lVert\cdot\rVert_{\infty} denotes the maximum absolute row sum. Now we lower bound \sigma_{\text{min}}\left(\boldsymbol{D}_{N}\right). We have that for any matrix 𝑨\boldsymbol{A}  
| ≥.\sigma_{\text{min}}\left(\boldsymbol{A}\right)\geq\frac{1}{\sqrt{\lVert\boldsymbol{A}^{-1}\rVert_{1}\lVert\boldsymbol{A}^{-1}\rVert_{\infty}}}.  |  
| --- |  
Recall our expression for \boldsymbol{D}_{N}^{-1} from (). The singular values of a block diagonal matrix are the union of the singular values of the individual matrices on the diagonal, so it is enough to lower bound the singular values of each of these individual matrices using () and then take the minimum of all of the lower bounds.
Using the expression for \boldsymbol{D}^{(1)}(t)^{-1} from () and (),  
| ​≤⟹​≥.\lVert\boldsymbol{D}^{(1)}(t)^{-1}\rVert_{1}\lVert\boldsymbol{D}^{(1)}(t)^{-1}\rVert_{\infty}\leq t^{2}\qquad\implies\qquad\sigma_{\text{min}}\left(\boldsymbol{D}^{(1)}(t)^{-1}\right)\geq 1/t.  |  
| --- |  
Next, using the expression for (\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1} from (), observe that the maximum column norm of (\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1} is 𝔚\mathfrak{W}, the norm of the first column. Similarly, we see that the maximum row norm, in the bottom block of (\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}, is at most R-1+T-r_{R}+1\leq R+T, because each of the blocks of \boldsymbol{\tilde{U}} matrices has a single 1 in each row and ()⊤\left(\boldsymbol{D}^{(1)}(T-r_{R}+1)^{-1}\right)^{\top} has a row of T-r_{R}+1 ones. So  
| ​≤\displaystyle\lVert(\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}\rVert_{1}\lVert(\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}\rVert_{\infty}\leq\penalty\   | \displaystyle\mathfrak{W}(T+R)  |  
| --- | --- |  
| ⟹​≥\displaystyle\implies\qquad\sigma_{\text{min}}\left((\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}\right)\geq\penalty\   | \displaystyle\frac{1}{\sqrt{\mathfrak{W}(T+R)}}.  |  
Using the fact that \mathfrak{W}\leq(T-1)^{2} from Lemma , we have  
| ​≥≥.\sigma_{\text{min}}\left((\boldsymbol{D}^{(2)}(\mathcal{R}))^{-1}\right)\geq\frac{1}{(T-1)\sqrt{T+R}}\geq\frac{1}{(T-1)\sqrt{2T-1}}.  |  
| --- |  
Finally, using () and (), , and the fact that the smallest singular value of the identity matrix is 1, we have  
| \displaystyle\sigma_{\text{min}}\left(\boldsymbol{D}_{N}\right)\geq\penalty\   | min⁡{,,1,}≥≥.\displaystyle\min\left\\{\frac{1}{R},\frac{1}{T-1},1,\frac{1}{(T-1)\sqrt{2T-1}}\right\\}\geq\frac{1}{(T-1)\sqrt{2T-1}}\geq\frac{1}{T\sqrt{2T}}.  |  
| --- | --- |  
###### Proof of Lemma .
We will first show that \boldsymbol{X}_{i} having finite fourth moments under Assumption (R1) is enough for \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot} to have finite fourth moments as well. We will show that each entry of \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1} is a bounded linear combination of a finite number of elements of \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}, and likewise for \boldsymbol{G}_{N}\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1}. Then the finite moments will follow using our Assumption (R1), Cauchy-Schwarz, and Hölder’s inequality.
Besides \boldsymbol{X}_{i}, the entries of \boldsymbol{\tilde{Z}}_{i} consist of indicator variables and interactions of the indicator variables with themselves and \boldsymbol{X}_{i}. All of these terms are upper-bounded by the greater of 1 or \boldsymbol{X}_{i}. Using this, our assumption that \mathbb{E}[X_{(i)j}^{4}] is finite for all j\in[p_{N}], and the fact that \boldsymbol{X}_{i} is time-invariant, we have that for any fixed , ⁡\max_{j\in[d_{N}],t\in[T]}\\{\mathbb{E}[Z_{(it)j}^{4}]\\} is finite.
Now we consider the entries of 𝑫∈ℝ\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1}\in\mathbb{R}^{NT\times p_{N}}. The columns of \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1} are linear combinations of the columns of \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}; in particular, the j^{\text{th}} column of \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1} is ∈\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}(\boldsymbol{D}_{N}^{-1})_{\cdot j}\in\mathbb{R}^{NT}, where (\boldsymbol{D}_{N}^{-1})_{\cdot j} is the j^{\text{th}} column of \boldsymbol{D}_{N}^{-1}. From () and the expressions in the proof of Lemma we see that \boldsymbol{D}_{N}^{-1} contains only entries that equal 1 or 0, so for any and j\in[p_{N}],  
| 4=[()]4=(∑{:})4.(\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1})_{(1t)j}^{4}=\left[\left(\boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}(\boldsymbol{D}_{N}^{-1})_{\cdot j}\right)_{(1t)}\right]^{4}=\left(\sum_{\\{j^{\prime}:(\boldsymbol{D}_{N}^{-1})_{j^{\prime}j}\neq 0\\}}Z_{(1t)j^{\prime}}\right)^{4}.  |  
| --- |  
From () and the expressions in the proof of Lemma we have that \boldsymbol{D}_{N}^{-1} has a block diagonal structure with blocks of size at most \mathfrak{W}\leq(T-1)^{2} (an upper bound we established in Lemma ), so this sum contains at most (T-1)^{2} terms before expansion. By the multinomial theorem, the number of terms after expansion is therefore at most  
| ().\binom{4+(T-1)^{2}-1}{(T-1)^{2}-1}.  |  
| --- |  
is fixed and finite, so the number of terms after expansion is finite as well. Therefore if we can show that each summand after expansion has finite 4^{\text{th}} moment, we have shown that each entry of \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}\boldsymbol{D}_{N}^{-1} has finite 4^{\text{th}} moment.
This sum when expanded contains terms of the form \tilde{Z}_{(1t)j^{\prime}}^{4}, ​\tilde{Z}_{(1t)j^{\prime}}^{2}\tilde{Z}_{(1t)j^{\prime\prime}}^{2} for j^{\prime}\neq j^{\prime\prime}, ​\tilde{Z}_{(1t)j^{\prime}}^{3}\tilde{Z}_{(1t)j^{\prime\prime}}, ​​\tilde{Z}_{(1t)j^{\prime}}^{2}\tilde{Z}_{(1t)j^{\prime\prime}}\tilde{Z}_{(1t)j^{\prime\prime\prime}} and ​​​\tilde{Z}_{(1t)j^{\prime}}\tilde{Z}_{(1t)j^{\prime\prime}}\tilde{Z}_{(1t)j^{\prime\prime\prime}}\tilde{Z}_{(1t)j^{\prime\prime\prime\prime}} (disregarding constant factors). We demonstrate that each of these terms has finite fourth moment below using Cauchy-Schwarz and Hölder’s inequality:
  * •
We have already shown that \mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}^{4}] is finite.
  * •
𝔼⁡[​]≤​𝔼​\mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}^{2}\tilde{Z}_{(1t)j^{\prime\prime}}^{2}]\leq\sqrt{\mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}^{4}]\mathbb{E}[\tilde{Z}_{(1t)j^{\prime\prime}}^{4}]}, and terms of this form are finite using the finite fourth moments of \boldsymbol{\tilde{Z}}_{(i\cdot)\cdot}.
  * •
Recall that by Hölder’s inequality, for any such that 1/p+1/q=1, if and are random variables with \left(\mathbb{E}|X|^{p}\right)^{1/p} and \left(\mathbb{E}|Y|^{q}\right)^{1/q} finite then  
| ≤​.\mathbb{E}|XY|\leq\left(\mathbb{E}|X|^{p}\right)^{1/p}\left(\mathbb{E}|Y|^{q}\right)^{1/q}.  |  
| --- |  
Choosing and , we have  
| 𝔼⁡[​]≤\displaystyle\mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}^{3}\tilde{Z}_{(1t)j^{\prime\prime}}]\leq\penalty\   | 𝔼⁡[​]≤(𝔼⁡[])​(𝔼⁡)\displaystyle\mathbb{E}[|\tilde{Z}_{(1t)j^{\prime}}|^{3}|\tilde{Z}_{(1t)j^{\prime\prime}}|]\leq\left(\mathbb{E}\left[(|\tilde{Z}_{(1t)j^{\prime}}|^{3})^{4/3}\right]\right)^{3/4}\left(\mathbb{E}\left[|\tilde{Z}_{(1t)j^{\prime\prime}}|^{4}\right]\right)^{1/4}  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ()​(),\displaystyle\left(\mathbb{E}|\tilde{Z}_{(1t)j^{\prime}}|^{4}\right)^{3/4}\left(\mathbb{E}|\tilde{Z}_{(1t)j^{\prime\prime}}|^{4}\right)^{1/4},  |  
which is finite.
  * •
𝔼[]≤𝔼[]\mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}^{2}\tilde{Z}_{(1t)j^{\prime\prime}}\tilde{Z}_{(1t)j^{\prime\prime\prime}}]\leq\sqrt{\mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}^{4}]}\sqrt{\mathbb{E}[\tilde{Z}_{(1t)j^{\prime\prime}}^{2}\tilde{Z}_{(1t)j^{\prime\prime\prime}}^{2}}], and we already showed that terms of this form are finite.
  * •
𝔼⁡[​​​]≤𝔼⁡[​]​𝔼​[​]\mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}\tilde{Z}_{(1t)j^{\prime\prime}}\tilde{Z}_{(1t)j^{\prime\prime\prime}}\tilde{Z}_{(1t)j^{\prime\prime\prime\prime}}]\leq\sqrt{\mathbb{E}[\tilde{Z}_{(1t)j^{\prime}}^{2}\tilde{Z}_{(1t)j^{\prime\prime}}^{2}]\mathbb{E}[\tilde{Z}_{(1t)j^{\prime\prime\prime}}^{2}\tilde{Z}_{(1t)j^{\prime\prime\prime\prime}}^{2}]}, and we already showed that terms of this form are finite.


###### Proof of Lemma .
Recall that for any matrix 𝑨\boldsymbol{A}, the eigenvalues of \boldsymbol{A}^{\top}\boldsymbol{A} are the squares of the singular values of 𝑨\boldsymbol{A}. So ​(​​𝒁​)=/\lambda_{\text{min}}\left(\frac{1}{NT}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)^{\top}\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)=\sigma_{\text{min}}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)^{2}/(NT) and  
| ​(​​𝒁​)=/,\lambda_{\text{max}}\left(\frac{1}{NT}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)^{\top}\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)=\sigma_{\text{max}}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)^{2}/(NT),  |  
| --- |  
and it is enough to bound the singular values of \boldsymbol{Z}\boldsymbol{D}_{N}^{-1}, which we can do with Lemma :
###### .
Let 𝐀\boldsymbol{A} and 𝐁\boldsymbol{B} be conformable real matrices.
  1. (a)
Suppose \boldsymbol{A}\neq\boldsymbol{0} and 𝑩\boldsymbol{B} has full row rank. Then  
| ≥​​​.\sigma_{\text{min}}\left(\boldsymbol{A}\boldsymbol{B}\right)\geq\sigma_{\text{min}}\left(\boldsymbol{A}\right)\sigma_{\text{min}}\left(\boldsymbol{B}\right).  |  
| --- |  
  2. (b)  
| ≤​​​.\sigma_{\text{max}}\left(\boldsymbol{A}\boldsymbol{B}\right)\leq\sigma_{\text{max}}\left(\boldsymbol{A}\right)\sigma_{\text{max}}\left(\boldsymbol{B}\right).  |  
| --- |  


We bounded the singular values of the deterministic \boldsymbol{D}_{N}^{-1} in Lemma . Using that and the fact that =​=\frac{1}{NT}\sigma_{\text{min}}\left(\boldsymbol{Z}\right)^{2}=\lambda_{\text{min}}\left(\frac{1}{NT}\boldsymbol{Z}^{\top}\boldsymbol{Z}\right)=e_{1N}, we have  
| ​(​​𝒁​)≥\displaystyle\lambda_{\text{min}}\left(\frac{1}{NT}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)^{\top}\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)\geq\penalty\   | ​​​​\displaystyle\frac{1}{NT}\sigma_{\text{min}}\left(\boldsymbol{Z}\right)^{2}\sigma_{\text{min}}\left(\boldsymbol{D}_{N}^{-1}\right)^{2}  |  
| --- | --- |  
| ≥\displaystyle\geq\penalty\   | ​​\displaystyle\lambda_{\text{min}}\left(\frac{1}{NT}\boldsymbol{\tilde{Z}}^{\top}\boldsymbol{\tilde{Z}}\right)\left(\frac{1}{3}\right)^{2}  |  
| ≥\displaystyle\geq\penalty\   | \displaystyle\frac{1}{9}e_{1N},  |  
where we used that from Lemma ,  
| ≤3⇔≥.\displaystyle\sigma_{\text{max}}(\boldsymbol{D}_{N})\leq 3\qquad\iff\qquad\sigma_{\text{min}}(\boldsymbol{D}_{N}^{-1})\geq\frac{1}{3}.  |  
| --- |  
Similarly,  
| ​(​​𝒁​)≤\displaystyle\lambda_{\text{max}}\left(\frac{1}{NT}\left(\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)^{\top}\boldsymbol{Z}\boldsymbol{D}_{N}^{-1}\right)\leq\penalty\   | ​​≤,\displaystyle\lambda_{\text{max}}\left(\frac{1}{NT}\boldsymbol{\tilde{Z}}^{\top}\boldsymbol{\tilde{Z}}\right)\left(T\sqrt{2T}\right)^{2}\leq e_{2N}\cdot 2T^{3},  |  
| --- | --- |  
where we used that from Lemma ,  
| ≥⇔≤T.\displaystyle\sigma_{\text{min}}(\boldsymbol{D}_{N})\geq\frac{1}{T\sqrt{2T}}\qquad\iff\qquad\sigma_{\text{max}}(\boldsymbol{D}_{N}^{-1})\leq T\sqrt{2T}.  |  
| --- |  
###### Proof of Theorem .
For clarity, we use the notation of . To extend Lemma 3 from , we use a similar strategy to the one used to prove Theorem 11.3(d) in . Since under Assumption (A5) from we have ≥\min_{1\leq j\leq k_{N}}\\{|\beta_{10j|}\\}\geq b_{0}, it is enough to show that with high probability \hat{\beta}_{2N}=\boldsymbol{0} and \lVert\hat{\beta}_{N}-\beta_{N}\rVert_{\infty}\leq b_{0}/2. That is,  
| ∩⊆\displaystyle\left\\{\hat{\beta}_{2N}=\boldsymbol{0}\right\\}\cap\left\\{\left\lVert\hat{\beta}_{N}-\beta_{N}\right\rVert_{\infty}\leq\frac{b_{0}}{2}\right\\}\subseteq\penalty\   | \displaystyle\left\\{\hat{\mathcal{S}}=\mathcal{S}\right\\}  |  
| --- | --- |  
| \displaystyle\implies\qquad\left\\{\hat{\mathcal{S}}\neq\mathcal{S}\right\\}\subseteq\penalty\   | ∪\displaystyle\left\\{\hat{\beta}_{2N}\neq\boldsymbol{0}\right\\}\cup\left\\{\left\lVert\hat{\beta}_{N}-\beta_{N}\right\rVert_{\infty}>\frac{b_{0}}{2}\right\\}  |  
| \displaystyle\stackrel{{\scriptstyle(a)}}{{\implies}}\qquad\mathbb{P}\left(\hat{\mathcal{S}}\neq\mathcal{S}\right)\leq\penalty\   | ℙ(∪)\displaystyle\mathbb{P}\left(\left\\{\hat{\beta}_{2N}\neq\boldsymbol{0}\right\\}\cup\left\\{\left\lVert\hat{\beta}_{N}-\beta_{N}\right\rVert_{2}>\frac{b_{0}}{2}\right\\}\right)  |  
| ≤\displaystyle\leq\penalty\   | +ℙ⁡,\displaystyle\mathbb{P}\left(\hat{\beta}_{2N}\neq\boldsymbol{0}\right)+\mathbb{P}\left(\left\lVert\hat{\beta}_{N}-\beta_{N}\right\rVert_{2}>\frac{b_{0}}{2}\right),  |  
where in we used \lVert\cdot\rVert_{\infty}\leq\lVert\cdot\rVert_{2}. Lemma 3 from establishes that  
| =0,\lim_{N\to\infty}\mathbb{P}\left(\hat{\beta}_{2N}\neq\boldsymbol{0}\right)=0,  |  
| --- |  
and Theorem 1 in (the assumptions of which are satisfied under the assumptions of Lemma 3 in ) establishes the consistency of \hat{\beta}_{N}, which is sufficient for  
| ℙ⁡=0.\lim_{N\to\infty}\mathbb{P}\left(\left\lVert\hat{\beta}_{N}-\beta_{N}\right\rVert_{2}>\frac{b_{0}}{2}\right)=0.  |  
| --- |  
| 0≤≤\displaystyle 0\leq\lim_{N\to\infty}\mathbb{P}\left(\hat{\mathcal{S}}\neq\mathcal{S}\right)\leq\penalty\   | +ℙ⁡=0,\displaystyle\lim_{N\to\infty}\mathbb{P}\left(\hat{\beta}_{2N}\neq\boldsymbol{0}\right)+\lim_{N\to\infty}\mathbb{P}\left(\left\lVert\hat{\beta}_{N}-\beta_{N}\right\rVert_{2}>\frac{b_{0}}{2}\right)=0,  |  
| --- | --- |  
proving the result.
###### Proof of Theorem .
  1. (a)
Again, for clarity we use the notation of . Since  
| ​​=\displaystyle\sqrt{NT_{N}}\boldsymbol{\psi}_{N}^{\top}(\hat{\beta}_{N}-\beta_{0})=\penalty\   | ​​+​​\displaystyle\sqrt{NT_{N}}\boldsymbol{\alpha}^{\top}(\hat{\beta}_{1N}-\beta_{10})+\sqrt{NT_{N}}\boldsymbol{b}_{N}^{\top}(\hat{\beta}_{2N}-\beta_{20})  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​​+,\displaystyle\sqrt{NT_{N}}\boldsymbol{\alpha}^{\top}(\hat{\beta}_{1N}-\beta_{10})+\sqrt{NT_{N}}\boldsymbol{b}_{N}^{\top}\hat{\beta}_{2N},  |  
if we can show that  
| 0\sqrt{NT_{N}}\boldsymbol{b}_{N}^{\top}\hat{\beta}_{2N}\xrightarrow{p}0  |  
| --- |  
then the result follows from Theorem 2(ii) from (in particular, Equation 7.5 in the proof) and Slutsky’s theorem. Theorem 2(i) from establishes that  
| =1.\displaystyle\lim_{N\to\infty}\mathbb{P}\left(\hat{\beta}_{2N}=\boldsymbol{0}\right)=1.  |  
| --- |  
Since for any   
| ⊆{||=0},\left\\{\hat{\beta}_{2N}=\boldsymbol{0}\right\\}\subseteq\left\\{\sqrt{NT_{N}}\sum_{j=1}^{p_{N}-k}\left|(\boldsymbol{b}_{N})_{j}(\hat{\beta}_{2N})_{j}\right|=0\right\\},  |  
| --- |  
we have  
| ≤ℙ⁡(​=0),\displaystyle\mathbb{P}\left(\hat{\beta}_{2N}=\boldsymbol{0}\right)\leq\mathbb{P}\left(\sqrt{NT_{N}}\sum_{j=1}^{p_{N}-k}\left|(\boldsymbol{b}_{N})_{j}(\hat{\beta}_{2N})_{j}\right|=0\right),  |  
| --- |  
so  
| 1≥ℙ⁡(​=0)≥=1,1\geq\lim_{N\to\infty}\mathbb{P}\left(\sqrt{NT_{N}}\sum_{j=1}^{p_{N}-k}\left|(\boldsymbol{b}_{N})_{j}\hat{\beta}_{2N}\right|=0\right)\geq\lim_{N\to\infty}\mathbb{P}\left(\hat{\beta}_{2N}=\boldsymbol{0}\right)=1,  |  
| --- |  
establishing  
| ℙ⁡(​=0)=1.\lim_{N\to\infty}\mathbb{P}\left(\sqrt{NT_{N}}\sum_{j=1}^{p_{N}-k}\left|(\boldsymbol{b}_{N})_{j}(\hat{\beta}_{2N})_{j}\right|=0\right)=1.  |  
| --- |  
Since this notion of convergence is stronger than convergence in probability, it follows that  
| ​\displaystyle\sqrt{NT_{N}}\sum_{j=1}^{p_{N}-k}(\boldsymbol{b}_{N})_{j}(\hat{\beta}_{2N})_{j}\xrightarrow{p}\penalty\   | \displaystyle 0,  |  
| --- | --- |  
proving the result.
  2. (b)
Since by the consistency of the sample covariance matrix and the continuous mapping theorem  
| ​T𝜶(𝒮)((𝑿))𝜶(𝒮)​T𝜶​​()​𝜶​,\frac{1}{\sigma}\sqrt{\frac{T}{\boldsymbol{\alpha}(\mathcal{S})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{S}})\right)^{-1}\boldsymbol{\alpha}(\mathcal{S})}}\xrightarrow{p}\frac{1}{\sigma}\sqrt{\frac{T}{\boldsymbol{\alpha}(\mathcal{S})^{\top}\left(\lim_{N\to\infty}\mathbb{E}\left[\Sigma_{1N}\right]\right)^{-1}\boldsymbol{\alpha}(\mathcal{S})}},  |  
| --- |  
by part (a) and Slutsky’s theorem we have that U_{b}(\mathcal{S}) converges in distribution to a standard Gaussian random variable. Further, ℙ⁡(=0)≥\mathbb{P}(U_{b}(\hat{\mathcal{S}}_{N})-U_{b}(\mathcal{S})=0)\geq\mathbb{P}(\hat{\mathcal{S}}_{N}=\mathcal{S}), so from Theorem we have  
| 1≥ℙ⁡(=0)≥=1,1\geq\lim_{N\to\infty}\mathbb{P}(U_{b}(\hat{\mathcal{S}}_{N})-U_{b}(\mathcal{S})=0)\geq\lim_{N\to\infty}\mathbb{P}(\hat{\mathcal{S}}_{N}=\mathcal{S})=1,  |  
| --- |  
which implies 0U_{b}(\hat{\mathcal{S}}_{N})-U_{b}(\mathcal{S})\xrightarrow{p}0. Therefore by Slutsky’s theorem,  
| =+().U_{b}(\hat{\mathcal{S}}_{N})=U_{b}(\mathcal{S})+\left(U_{b}(\hat{\mathcal{S}}_{N})-U_{b}(\mathcal{S})\right)\xrightarrow{d}\mathcal{N}(0,1).  |  
| --- |  
  3. (c)
We begin by decomposing our sequence of random variables:  
| \displaystyle U_{c}(\mathcal{A})=\penalty\   | ​(−)\displaystyle\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})^{\top}\hat{\beta}-\boldsymbol{\psi}_{N}(\mathcal{A})^{\top}\beta_{0}\right)  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​(−+⊤​)\displaystyle\sqrt{NT}\left(\boldsymbol{\psi}_{N}(\mathcal{A})^{\top}\hat{\beta}-\boldsymbol{\psi}_{N}(\mathcal{A})^{\top}\beta_{0}+\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})-\boldsymbol{\psi}_{N}(\mathcal{A})\right)^{\top}\hat{\beta}\right)  |  
| =\displaystyle=\penalty\   | ​(+⊤​CLOSE\displaystyle\sqrt{NT}\bigg(\boldsymbol{\psi}_{N}(\mathcal{A})^{\top}\left(\hat{\beta}-\beta_{0}\right)+\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})-\boldsymbol{\psi}_{N}(\mathcal{A})\right)^{\top}\beta_{0}  |  
| OPEN+⊤​).\displaystyle+\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})-\boldsymbol{\psi}_{N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}-\beta_{0}\right)\bigg).  |  
We examine these terms one at a time. By part (a),  
| ​​​𝒩⁡(0,​​()​𝜶).\sqrt{NT}\boldsymbol{\psi}_{N}(\mathcal{S})^{\top}(\hat{\beta}_{N}-\beta_{0})\xrightarrow{d}\mathcal{N}\left(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}\left[\Sigma_{1N}\right]\right)^{-1}\boldsymbol{\alpha}\right).  |  
| --- |  
By assumption, for any \beta_{0}\in\mathbb{R}^{p_{N}} with entries nonzero and finite and the remaining entries equal to 0,  
| ​⊤​.\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{S})-\boldsymbol{\psi}_{N}(\mathcal{S})\right)^{\top}\beta_{0}\xrightarrow{d}\mathcal{N}(0,v_{\psi}(\beta_{0},\mathcal{S})).  |  
| --- |  
Finally, similarly to we partition \boldsymbol{\hat{\psi}}_{N}(\mathcal{A}) into component \boldsymbol{\hat{\psi}}_{1N}(\mathcal{A}) that corresponds to the nonzero coefficients \beta_{10}\in\mathbb{R}^{k} and component \boldsymbol{\hat{\psi}}_{2N}(\mathcal{A}) corresponding to \beta_{20}=\boldsymbol{0}\in\mathbb{R}^{p_{N}-k}, and likewise for \boldsymbol{\psi}_{N}(\mathcal{A}). Then we have  
| ⊤​\displaystyle\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})-\boldsymbol{\psi}_{N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}-\beta_{0}\right)  |  
| --- |  
| =\displaystyle=\penalty\   | (−)⊤​+(−)⊤​\displaystyle\left(\boldsymbol{\hat{\psi}}_{1N}(\mathcal{A})-\boldsymbol{\psi}_{1N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}_{1N}-\beta_{10}\right)+\left(\boldsymbol{\hat{\psi}}_{2N}(\mathcal{A})-\boldsymbol{\psi}_{2N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}_{2N}-\beta_{20}\right)  |  
| =\displaystyle=\penalty\   | (−)⊤​+(−)⊤​.\displaystyle\left(\boldsymbol{\hat{\psi}}_{1N}(\mathcal{A})-\boldsymbol{\psi}_{1N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}_{1N}-\beta_{10}\right)+\left(\boldsymbol{\hat{\psi}}_{2N}(\mathcal{A})-\boldsymbol{\psi}_{2N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}_{2N}\right).  |  
For the first term, applying Theorem 2.4(i) in and Proposition 1.8(iii) in termwise we have that  
| (−)⊤​=\displaystyle\left(\boldsymbol{\hat{\psi}}_{1N}(\mathcal{A})-\boldsymbol{\psi}_{1N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}_{1N}-\beta_{10}\right)=\penalty\   | =\displaystyle\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right)\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right)=\mathcal{O}_{\mathbb{P}}\left(\frac{1}{N}\right)  |  
| --- | --- |  
| ⟹​(−)⊤​=\displaystyle\implies\qquad\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{1N}(\mathcal{A})-\boldsymbol{\psi}_{1N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}_{1N}-\beta_{10}\right)=\penalty\   | \displaystyle\mathcal{O}_{\mathbb{P}}\left(\frac{1}{\sqrt{N}}\right).  |  
For the second term, by a similar argument to the one we used in part (a), Theorem 2(i) from gives us that the ​(−)⊤​0\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{2N}(\mathcal{A})-\boldsymbol{\psi}_{2N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}_{2N}\right)\xrightarrow{p}0. Putting this together,  
| ​⊤​0.\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{A})-\boldsymbol{\psi}_{N}(\mathcal{A})\right)^{\top}\left(\hat{\beta}-\beta_{0}\right)\xrightarrow{p}0.  |  
| --- |  
Putting this all together, using Slutsky’s theorem and the independence of \hat{\beta} and \boldsymbol{\hat{\psi}}_{N}(\mathcal{S}) we have from () that  
| 𝒩⁡(0,​​()​𝜶+).U_{c}(\mathcal{S})\xrightarrow{d}\mathcal{N}\left(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}\left[\Sigma_{1N}\right]\right)^{-1}\boldsymbol{\alpha}+v_{\psi}(\beta_{0},\mathcal{S})\right).  |  
| --- |  
Finally, the same argument we used in part (b) yields 𝒩⁡(0,​​()​𝜶+)U_{c}(\hat{\mathcal{S}})\xrightarrow{d}\mathcal{N}(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}\left[\Sigma_{1N}\right]\right)^{-1}\boldsymbol{\alpha}+v_{\psi}(\beta_{0},\mathcal{S})).
  4. (d)
Using the consistency of the sample covariance matrix, the assumed consistency of \boldsymbol{\hat{\alpha}}(\mathcal{S}), the assumed consistency of \hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{S}), and the continuous mapping theorem,  
| \displaystyle\frac{1}{\sqrt{\hat{v}_{N}(\mathcal{S})}}  |  
| --- |  
| =\displaystyle=\penalty\   | 1(𝒮)((𝑿))(𝒮)+(,𝒮)\displaystyle\frac{1}{\sqrt{\boldsymbol{\hat{\alpha}}(\mathcal{S})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{S}})\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{S})+\hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{S})}}  |  
| \displaystyle\xrightarrow{p}\penalty\   | 1​()​𝜶+,\displaystyle\frac{1}{\sqrt{\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}[\Sigma_{1N}]\right)^{-1}\boldsymbol{\alpha}+v_{\psi}(\beta_{0},\mathcal{S})}},  |  
so part and Slutsky’s theorem yield that =U_{d}(\mathcal{S})=\frac{1}{\sqrt{\hat{v}_{N}(\mathcal{S})}}U_{c}(\mathcal{S})\xrightarrow{d}\mathcal{N}(0,1), where our finite-sample variance estimator \hat{v}_{N}(\mathcal{S}) was defined in (). Finally, the same argument we used in part (b) yields U_{d}(\hat{\mathcal{S}})\xrightarrow{d}\mathcal{N}(0,1).
  5. (e)
If \hat{\beta}_{N} and \boldsymbol{\hat{\psi}}_{N}(\mathcal{S}) are estimated on the same data set, they are not independent. So by the same argument we used in part we still have  
| ​(​​+⊤​)\displaystyle\sqrt{NT}\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{S})^{\top}(\hat{\beta}_{N}-\beta_{0})+\left(\boldsymbol{\hat{\psi}}_{N}(\mathcal{S})-\boldsymbol{\psi}_{N}(\mathcal{S})\right)^{\top}\beta_{0}\right)  |  
| --- |  
| \displaystyle\xrightarrow{d}\penalty\   | 𝒩⁡(0,​​​𝜶)+,\displaystyle\mathcal{N}\left(0,\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}[\Sigma_{1N}]\right)^{-1}\boldsymbol{\alpha}\right)+\mathcal{N}\left(0,v_{\psi}(\beta_{0},\mathcal{S})\right),  |  
but because the two Gaussian random variables are neither independent nor do we know that they are jointly Gaussian, we cannot say as much about the distribution of their sum. However, due to Lemma we do know that they are subgaussian.
###### .
If and are mean-zero subgaussian random variables, then is mean zero and subgaussian.
###### Proof.
Provided in Appendix . ∎
Further, since for any two random variables with finite variance by Cauchy-Schwarz  
| \displaystyle\mathrm{Var}(X+Y)=\penalty\   | ++\displaystyle\mathrm{Var}(X)+2\mathrm{Cov}(X,Y)+\mathrm{Var}(Y)  |  
| --- | --- |  
| ≤\displaystyle\leq\penalty\   | +2​+,\displaystyle\mathrm{Var}(X)+2\sqrt{\mathrm{Var}(X)\mathrm{Var}(Y)}+\mathrm{Var}(Y),  |  
we have that the above sequence of random variables converges in distribution to a subgaussian random variable with variance at most  
| ​​​𝜶+2​​​​𝜶⋅​+.\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}[\Sigma_{1N}]\right)^{-1}\boldsymbol{\alpha}+2\sqrt{\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}[\Sigma_{1N}]\right)^{-1}\boldsymbol{\alpha}\cdot v_{\psi}(\beta_{0},\mathcal{S})}+v_{\psi}(\beta_{0},\mathcal{S}).  |  
| --- |  
Using the assumed consistency of \hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{S}), we have by the continuous mapping theorem that for \hat{v}_{N}^{(\text{cons})}(\mathcal{A}) as defined in (),  
| \displaystyle\hat{v}_{N}^{(\text{cons})}(\mathcal{S})  |  
| --- |  
| =\displaystyle=\penalty\   | ++2σ(𝒮)((𝑿))(𝒮)⋅(,𝒮)\displaystyle\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{S})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{S}})\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{S})+\hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{S})+2\sqrt{\sigma^{2}\boldsymbol{\hat{\alpha}}(\mathcal{S})^{\top}\left(\boldsymbol{\hat{\Sigma}}(\boldsymbol{X}_{(\cdot\cdot)\mathcal{S}})\right)^{-1}\boldsymbol{\hat{\alpha}}(\mathcal{S})\cdot\hat{v}_{\psi}(\hat{\beta}_{N},\mathcal{S})}  |  
| \displaystyle\xrightarrow{p}\penalty\   | ​​​𝜶++2​​​​𝜶⋅​.\displaystyle\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}[\Sigma_{1N}]\right)^{-1}\boldsymbol{\alpha}+v_{\psi}(\beta_{0},\mathcal{S})+2\sqrt{\sigma^{2}\boldsymbol{\alpha}^{\top}\left(\lim_{N\to\infty}\mathbb{E}[\Sigma_{1N}]\right)^{-1}\boldsymbol{\alpha}\cdot v_{\psi}(\beta_{0},\mathcal{S})}.  |  
Therefore by Slutsky’s theorem, U_{e}(\mathcal{S}) converges in distribution to a mean-zero subgaussian random variable with variance at most 1. Finally, once again a similar argument to the one we used in part (b) yields that U_{e}(\hat{\mathcal{S}}) converges in distribution to the same random variable as U_{e}(\mathcal{S}).


## Appendix I Statements and Proofs of Other Results
###### Proof of Lemma .
The sum has mean zero due to linearity of expectation. To show subgaussianity, it is enough to show that 𝔼​≤\mathbb{E}\exp\left\\{t(X+Y)\right\\}\leq\exp\left\\{K^{2}t^{2}\right\\} for some finite for all t\in\mathbb{R} [[Vershynin, 2018](https://arxiv.org/html/2312.05985#bib.bib118), Proposition 2.5.2]. Because and are subgaussian, we have that ≤\mathbb{E}\left[\exp\left\\{tX\right\\}\right]\leq\exp\left\\{K_{X}^{2}t^{2}\right\\} and ≤\mathbb{E}\left[\exp\left\\{tY\right\\}\right]\leq\exp\left\\{K_{Y}^{2}t^{2}\right\\} for all t\in\mathbb{R}. So using these facts and Cauchy-Schwarz, we have for any t\in\mathcal{R}  
| 𝔼​=\displaystyle\mathbb{E}\exp\left\\{t(X+Y)\right\\}=\penalty\   | 𝔼⁡[​]\displaystyle\mathbb{E}\left[\exp\left\\{tX\right\\}\exp\left\\{tY\right\\}\right]  |  
| --- | --- |  
| ≤\displaystyle\leq\penalty\   | 𝔼⁡​𝔼​\displaystyle\sqrt{\mathbb{E}\left[\exp\left\\{2tX\right\\}\right]\mathbb{E}\left[\exp\left\\{2tY\right\\}\right]}  |  
| ≤\displaystyle\leq\penalty\   | exp⁡​\displaystyle\sqrt{\exp\left\\{K_{X}^{2}\cdot 4t^{2}\right\\}\exp\left\\{K_{Y}^{2}\cdot 4t^{2}\right\\}}  |  
| =\displaystyle=\penalty\   | exp⁡{},\displaystyle\exp\left\\{2t^{2}\left(K_{X}^{2}+K_{Y}^{2}\right)\right\\},  |  
and the result is proven with :=K^{2}:=2\left(K_{X}^{2}+K_{Y}^{2}\right).
###### .
The smallest and largest singular values of \boldsymbol{G}_{N} are bounded as follows: \sigma_{\text{max}}(\boldsymbol{G}_{N})\leq\sqrt{2} and ≥\sigma_{\text{min}}\left(\boldsymbol{G}_{N}\right)\geq\sigma/\sqrt{T\sigma_{c}^{2}+\sigma^{2}}.
###### Proof of Lemma .
We will prove the result in a similar way to the proof of Lemma . By properties of the Kronecker product, it is enough to bound the singular values of 𝛀\boldsymbol{\Omega}. We have  
| ≤≤=,\sigma_{\text{max}}(\boldsymbol{\Omega})\leq\sqrt{\lVert\boldsymbol{\Omega}\rVert_{1}\lVert\boldsymbol{\Omega}\rVert_{\infty}}\leq\sqrt{[T\cdot\sigma_{c}^{2}+\sigma^{2}]^{2}}=T\cdot\sigma_{c}^{2}+\sigma^{2},  |  
| --- |  
where for matrices \lVert\cdot\rVert_{1} denotes the maximum absolute column sum of a matrix and \lVert\cdot\rVert_{\infty} denotes the maximum absolute row sum. We will use () to lower-bound \sigma_{\text{min}}(\boldsymbol{\Omega}). First we will find \boldsymbol{\Omega}^{-1}. Observe that we can express 𝛀\boldsymbol{\Omega} as  
| \boldsymbol{\Omega}=\sigma^{2}\boldsymbol{I}_{T}+\boldsymbol{v}\boldsymbol{v}^{\top}  |  
| --- |  
where \boldsymbol{v}=(\sigma_{c},\ldots,\sigma_{c})\in\mathbb{R}^{T}. Therefore we can invert 𝛀\boldsymbol{\Omega} using the Sherman-Morrison formula:  
| \displaystyle\boldsymbol{\Omega}^{-1}=\penalty\   | −​​𝒗​​​=​(−)=​()\displaystyle\frac{1}{\sigma^{2}}\boldsymbol{I}_{T}-\frac{\frac{1}{\sigma^{2}}\boldsymbol{I}_{T}\boldsymbol{v}\boldsymbol{v}^{\top}\frac{1}{\sigma^{2}}\boldsymbol{I}_{T}}{1+\boldsymbol{v}^{\top}\frac{1}{\sigma^{2}}\boldsymbol{I}_{T}\boldsymbol{v}}=\frac{1}{\sigma^{2}}\left(\boldsymbol{I}_{T}-\frac{\frac{1}{\sigma^{2}}\boldsymbol{v}\boldsymbol{v}^{\top}}{1+\frac{1}{\sigma^{2}}\boldsymbol{v}^{\top}\boldsymbol{v}}\right)=\frac{1}{\sigma^{2}}\left(\boldsymbol{I}_{T}-\frac{\boldsymbol{v}\boldsymbol{v}^{\top}}{\sigma^{2}+T\sigma_{c}^{2}}\right)  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ​().\displaystyle\frac{1}{\sigma^{2}}\begin{pmatrix}1-\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}&-\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}&\cdots&-\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}\\\ -\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}&1-\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}&\cdots&-\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}\\\ \vdots&\vdots&\ddots&\vdots\\\ -\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}&-\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}&\cdots&1-\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}\end{pmatrix}.  |  
Therefore  
| \displaystyle\sigma_{\text{min}}\left(\boldsymbol{\Omega}\right)\geq\penalty\   | ≥​()=​()\displaystyle\frac{1}{\sqrt{\lVert\boldsymbol{\Omega}^{-1}\rVert_{1}\lVert\boldsymbol{\Omega}^{-1}\rVert_{\infty}}}\geq\sigma^{2}\left(1+T\frac{\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}\right)^{-1}=\sigma^{2}\left(\frac{\sigma^{2}+2T\sigma_{c}^{2}}{\sigma^{2}+T\sigma_{c}^{2}}\right)^{-1}  |  
| --- | --- |  
| =\displaystyle=\penalty\   | ≥=.\displaystyle\frac{\sigma^{2}\left(\sigma^{2}+T\sigma_{c}^{2}\right)}{\sigma^{2}+2T\sigma_{c}^{2}}\geq\frac{\sigma^{2}\left(\sigma^{2}+T\sigma_{c}^{2}\right)}{2\left(\sigma^{2}+T\sigma_{c}^{2}\right)}=\frac{\sigma^{2}}{2}.  |  
Finally, using these results we have ≥1/\sigma_{\text{min}}\left(\boldsymbol{\Omega}^{-1/2}\right)\geq 1/\sqrt{T\sigma_{c}^{2}+\sigma^{2}} and ≤/σ\sigma_{\text{max}}\left(\boldsymbol{\Omega}^{-1/2}\right)\leq\sqrt{2}/\sigma. By properties of the Kronecker product, these bounds hold for \boldsymbol{G}_{N}=\sigma\boldsymbol{I}_{T}\otimes\boldsymbol{\Omega}^{-1/2} as well after multiplying by .
###### .
Suppose one treated cohort has fewer than units. Then \boldsymbol{\tilde{Z}} does not have full column rank.
###### Proof of Lemma .
The treatment effect dummies can be seen as interaction effects between the cohort dummies and (a subset of) the time dummies. In particular, the treatment effect for cohort at time is equal to the column for the r^{\text{th}} cohort dummy, but with entries for all times except set equal to 0. The treatment dummies for cohort are mutually orthogonal and span a -dimensional subspace, and they occupy an N_{r}(T-r+1)-dimensional subspace, where is the number of units in cohort .
Because the covariates (like the cohort indicators) are time-invariant, the time-covariate interactions are closely related to the treatment effect dummies. The covariate-time interactions for cohort lie in the same N_{r}(T-r+1)-dimensional subspace as the treatment effects for cohort and are either linearly dependent or span a subspace of dimension d_{N}(T-r+1). \boldsymbol{\tilde{Z}} is rank deficient in the former setting, so we assume the latter. Then the treatment dummies and covariate-time interactions for cohort span a subspace of dimension (d_{N}+1)(T-r+1) within this N_{r}(T-r+1)-dimensional subspace. So if N_{r}<d_{N}+1, these columns must be linearly dependent, and \boldsymbol{\tilde{Z}} does not have full column rank.
###### .
Suppose is bounded away from 0 with high probability: for some finite N^{*},c>0, for all it holds that \mathbb{P}(e_{1N}<c)<a_{N} for some vanishing sequence . Let be a sequence of random variables, and suppose that =X_{N}=\mathcal{O}_{\mathbb{P}}(\frac{1}{e_{1N}}\sqrt{p_{N}/N}). Then X_{N}=\mathcal{O}_{\mathbb{P}}(\sqrt{p_{N}/N}).
###### Proof of Lemma .
By assumption, for every \epsilon>0 there exists a finite M^{(1)}_{\epsilon}>0 and N^{(1)}_{\epsilon}>0 such that for all N>N^{(1)}_{\epsilon},  
| ℙ⁡()ϵ.\mathbb{P}\left(\left|\frac{e_{1N}X_{N}}{\sqrt{p_{N}/N}}\right|>M^{(1)}_{\epsilon}\right)<\epsilon.  |  
| --- |  
Now for all Nmax⁡N>\max\\{N^{*},N^{(1)}_{\epsilon/2}\\},  
| ℙ⁡()\displaystyle\mathbb{P}\left(\left|\frac{X_{N}}{\sqrt{p_{N}/N}}\right|>\frac{M_{\epsilon/2}^{(1)}}{c}\right)  |  
| --- |  
| =\displaystyle=\penalty\   | ℙ⁡(∣)​ℙ​+ℙ⁡(∣)​ℙ​\displaystyle\mathbb{P}\left(\left|\frac{X_{N}}{\sqrt{p_{N}/N}}\right|>\frac{M_{\epsilon/2}^{(1)}}{c}\mid e_{1N}\geq c\right)\mathbb{P}(e_{1N}\geq c)+\mathbb{P}\left(\left|\frac{X_{N}}{\sqrt{p_{N}/N}}\right|>\frac{M_{\epsilon/2}^{(1)}}{c}\mid e_{1N}<c\right)\mathbb{P}(e_{1N}<c)  |  
| ≤\displaystyle\leq\penalty\   | ℙ⁡(∣)​ℙ​+\displaystyle\mathbb{P}\left(\left|\frac{X_{N}}{\sqrt{p_{N}/N}}\right|>\frac{M_{\epsilon/2}^{(1)}}{c}\mid e_{1N}\geq c\right)\mathbb{P}(e_{1N}\geq c)+\mathbb{P}(e_{1N}<c)  |  
| ≤\displaystyle\leq\penalty\   | ℙ⁡(∣)​ℙ​+\displaystyle\mathbb{P}\left(\left|\frac{X_{N}}{\sqrt{p_{N}/N}}\right|>\frac{M_{\epsilon/2}^{(1)}}{e_{1N}}\mid e_{1N}\geq c\right)\mathbb{P}(e_{1N}\geq c)+\mathbb{P}(e_{1N}<c)  |  
| =\displaystyle=\penalty\   | ℙ⁡(∣)​ℙ​+\displaystyle\mathbb{P}\left(\left|\frac{e_{1N}X_{N}}{\sqrt{p_{N}/N}}\right|>M_{\epsilon/2}^{(1)}\mid e_{1N}\geq c\right)\mathbb{P}(e_{1N}\geq c)+\mathbb{P}(e_{1N}<c)  |  
| ≤\displaystyle\leq\penalty\   | ℙ⁡(∣)​ℙ​+ℙ⁡(∣)​ℙ​\displaystyle\mathbb{P}\left(\left|\frac{e_{1N}X_{N}}{\sqrt{p_{N}/N}}\right|>M_{\epsilon/2}^{(1)}\mid e_{1N}\geq c\right)\mathbb{P}(e_{1N}\geq c)+\mathbb{P}\left(\left|\frac{e_{1N}X_{N}}{\sqrt{p_{N}/N}}\right|>M_{\epsilon/2}^{(1)}\mid e_{1N}<c\right)\mathbb{P}(e_{1N}<c)  |  
| \displaystyle+\mathbb{P}(e_{1N}<c)  |  
| =\displaystyle=\penalty\   | ℙ⁡()+\displaystyle\mathbb{P}\left(\left|\frac{e_{1N}X_{N}}{\sqrt{p_{N}/N}}\right|>M_{\epsilon/2}^{(1)}\right)+\mathbb{P}(e_{1N}<c)  |  
| \displaystyle<\penalty\   | \displaystyle\frac{\epsilon}{2}+a_{N}.  |  
Since is vanishing, there exists a finite N^{(2)}_{\epsilon} so that a_{N}\leq\epsilon/2 for all N>N^{(2)}_{\epsilon}. Then we have established that for any \epsilon>0, for M_{\epsilon}:=M_{\epsilon/2}^{(1)}/c and :=max⁡{,,}N_{\epsilon}:=\max\\{N^{*},N^{(1)}_{\epsilon/2},N^{(2)}_{\epsilon}\\}, for all N>N_{\epsilon} it holds that  
| \displaystyle\mathbb{P}\left(\left|\frac{X_{N}}{\sqrt{p_{N}/N}}\right|>M_{\epsilon}\right)<\penalty\   | \displaystyle\epsilon.  |  
| --- | --- |  
## Appendix J Other Extensions
### J.1 Incorporating Direct Ridge Regularization into FETWFE
Although FETWFE is a regularized estimation method, in settings where is comparable to it can be helpful in practice to add additional regularization to the estimated coefficients themselves. One practical way to do this is to add an additional (ridge) penalty directly to the estimated coefficients to enhance numerical stability and improve predictive performance. This idea is similar to the motivation behind the elastic net which combines the lasso and ridge penalties. In our context, ridge regularization applied to the untransformed coefficients 𝜷\boldsymbol{\beta} provides an optional way to stabilize fused extended two-way fixed effects (FETWFE) estimation in addition to the adaptive fusion penalties. Specifically, adding a ridge penalty to the optimization problem,  
| \lambda_{2N}\|\boldsymbol{\beta}\|_{2}^{2},  |  
| --- |  
serves this purpose by shrinking the coefficients toward zero. This additional regularization is conceptually similar to the elastic net approach and may reduce the variance of the estimator without sacrificing too much bias.
#### J.1.1 Estimation Approach
To incorporate ridge regularization, consider the following modified optimization problem:  
| ⁡{++}.\min_{\boldsymbol{\beta}\in\mathbb{R}^{p_{N}}}\left\\{\|\boldsymbol{y}-\boldsymbol{Z}\boldsymbol{\beta}\|_{2}^{2}+\lambda_{1N}\|\boldsymbol{D}_{N}\boldsymbol{\beta}\|_{q}^{q}+\lambda_{2N}\|\boldsymbol{\beta}\|_{2}^{2}\right\\}.  |  
| --- |  
Assuming that 𝒚\boldsymbol{y} is centered and the columns of 𝒁\boldsymbol{Z} are centered and scaled, his can be rewritten by augmenting the data, in an approach similar to the one mentioned by . Define the extended response and design matrix as  
| and.\tilde{\boldsymbol{y}}^{*}=\begin{pmatrix}\boldsymbol{y}\\\ \boldsymbol{0}\end{pmatrix}\quad\text{and}\quad\tilde{\boldsymbol{Z}}^{*}=\begin{pmatrix}\boldsymbol{Z}\\\ \sqrt{\lambda_{2N}}\,I_{p_{N}}\end{pmatrix}.  |  
| --- |  
Then () can be written as  
| ⁡{+}.\min_{\boldsymbol{\beta}\in\mathbb{R}^{p_{N}}}\left\\{\left\|\tilde{\boldsymbol{y}}^{*}-\tilde{\boldsymbol{Z}}^{*}\boldsymbol{\beta}\right\|_{2}^{2}+\lambda_{1N}\|D_{N}\boldsymbol{\beta}\|_{q}^{q}\right\\}.  |  
| --- |  
After again using the transformation from Appendix   
| \boldsymbol{\theta}=\boldsymbol{D}_{N}\boldsymbol{\beta}\quad\Longrightarrow\quad\boldsymbol{\beta}=\boldsymbol{D}_{N}^{-1}\boldsymbol{\theta},  |  
| --- |  
we can finally write () as  
| ⁡{‖−()​𝜽‖22+},\min_{\boldsymbol{\theta}\in\mathbb{R}^{p_{N}}}\left\\{\left\|\tilde{\boldsymbol{y}}^{*}-\begin{pmatrix}\boldsymbol{Z}D_{N}^{-1}\\\ \sqrt{\lambda_{2N}}\,D_{N}^{-1}\end{pmatrix}\boldsymbol{\theta}\right\|_{2}^{2}+\lambda_{1N}\|\boldsymbol{\theta}\|_{q}^{q}\right\\},  |  
| --- |  
which (given a fixed choice of \lambda_{2N}) can be solved with a standard bridge regression solver (for example, using packages such as grpreg in R) exactly as one would for the original FETWFE estimator.
Further, if \lambda_{2N} vanishes appropriately in then this ridge-penalized version of FETWFE is likely to retain its consistency and asymptotic normality properties, because the extra ridge rows will have a vanishing influence on the estimation as .
Such an estimator is analogous to the naive elastic net of ; it may be wise, by analogy, to scale this estimator by 1+\lambda_{2} in order to mitigate the “double" regularization.
### J.2 The Case with No Covariates
The fused extended two-way fixed effects (FETWFE) model simplifies when no covariates are included. In this setting, the model requires the estimation of only fixed effects and treatment parameters, and the identification assumptions can be different. In the standard FETWFE model with covariates, the full parameter vector comprises  
| p=R++d+++𝔚+p=R+(T-1)+d+dR+d(T-1)+\mathfrak{W}+\mathfrak{W}d  |  
| --- |  
parameters to estimate. When no covariates are present (i.e., ), the parameter vector reduces to a length of  
| p=R+(T-1)+\mathfrak{W}.  |  
| --- |  
Thus, only the fixed effects and the treatment effects must be estimated, reducing the dimensionality of the model and, in turn, the degrees of freedom required for estimation.
However, estimation without covariates also fundamentally changes the required causal inference assumptions. With covariates, identification of the treatment effects relies on the conditional no anticipation (CNAS) and conditional parallel trends (CCTS) assumptions. These assumptions require that, after conditioning on covariates, units do not anticipate treatment and that the untreated potential outcomes follow parallel trends.
In the absence of covariates, these conditions are replaced by their marginal (unconditional) counterparts, Assumption (CTS) along with marginal no-anticipation. The more key difference is marginal parallel trends. As we discussed in Appendix , there is some nuance, but overall Assumption (CTS) is generally more stringent than (CCTS).
Omitting covariates does allow us to omit Assumption (LINS). When covariates are included, we require Assumption (LINS) to impose a linear relationship between covariates and potential outcomes. Without covariates, there is no need for such a linearity assumption. Therefore, the identification of treatment effects does not depend on the correctness of a linear specification for covariate effects, which may enhance robustness against model misspecification.