### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
using BenchmarkTools, CSV, DataFrames, Distances, Distributions, HypothesisTests, HTTP, KernelDensity, LinearAlgebra, MCMCChains, MLDataUtils, Plots, PlutoUI, Random,  RDatasets, StatsBase, Statistics, StatsPlots, Turing, UrlDownload

# ╔═╡ 09a9d9f9-fa1a-4192-95cc-81314582488b
html"""
<div style="
position: absolute;
width: calc(100% - 30px);
border: 50vw solid #2e3440;
border-top: 200px solid #2e3440;
border-bottom: none;
box-sizing: content-box;
left: calc(-50vw + 15px);
top: -100px;
height: 100px;
pointer-events: none;
"></div>

<div style="
height: 200px;
width: 100%;
background: #2e3440;
color: #fff;
padding-top: 50px;
">
<span style="
font-family: Vollkorn, serif;
font-weight: 700;
font-feature-settings: 'lnum', 'pnum';
"> <p style="
font-size: 1.5rem;
opacity: 0.8;
">ATS 872: Lecture 5</p>
<p style="text-align: center; font-size: 1.8rem;">
 Bayesian basics and regression
</p>

<style>
body {
overflow-x: hidden;
}
</style>"""

# ╔═╡ 41eb90d1-9262-42b1-9eb2-d7aa6583da17
html"""
<style>
  main {
    max-width: 850px;
  }
</style>
"""

# ╔═╡ aa69729a-0b08-4299-a14c-c9eb2eb65d5c
md" # Introduction "

# ╔═╡ a681c4c5-1451-44d6-be27-f9003340883f
md"""

> These lecture notes draw heavily from the notes of [Joshua Chen](https://joshuachan.org/papers/BayesMacro.pdf), [Gary Koop](https://sites.google.com/site/garykoop/) and [Jamie Cross](https://github.com/Jamie-L-Cross/Bayes/blob/master/5_Linear_Regression_Normal.ipynb) 

"""

# ╔═╡ 000021af-87ce-4d6d-a315-153cecce5091
md"  
We will recap some of the basics on Bayesian econometrics. 
"

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
TableOfContents() # Uncomment to see TOC

# ╔═╡ bbcafd74-97f4-4b8f-bffa-937812d9a2eb
Random.seed!(0)

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐ if you want."

# ╔═╡ 38d2167d-2ca2-4398-9406-db418cc4bf17
md""" ## General overview """

# ╔═╡ 3f0c1adc-3efd-4ade-9a65-a208d8457b85
md" Today we will be looking at the following topics, 

- Recap of Bayesian basics
- Gibbs sampling routine for our Normal model with unknown $\sigma^{2}$
- The Bayesian approach to linear regression
- Gibbs sampling and regression analysis
- AR model example
- Regression with `Turing.jl` 

"

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Bayesian methods recap "

# ╔═╡ f3823457-8757-4665-86a8-bf536d80e24d
md"""

**Please read** the following section carefully. It is from the notes by Joshua Chen and it explains in clear detail what we have been trying to do up to this point. This first section is a quick recap of many of the concepts we have covered so far. You will notice that we adopt matrix notation from now on. This takes some getting used to, but in the end it is worth it. 

As we have established, the fundamental organizing principle in Bayesian econometrics is **Bayes' theorem**. It forms the unifying principle on how Bayesians estimate model parameters, conduct inference, compare models and so forth. 

Bayes' theorem states that for events $A$ and $B$, the conditional probability of $A$ given $B$ is:

$$\mathbb{P}(A \mid B)=\frac{\mathbb{P}(A) \mathbb{P}(B \mid A)}{\mathbb{P}(B)}$$

where $\mathbb{P}(A)$ and $\mathbb{P}(B)$ are the marginal probabilities for events $A$ and $B$, respectively. This expression tells us how our view about event $A$ should change in light of information in event $B$.

To apply Bayes' theorem to estimation and inference, we first introduce some notation. Notice that we are going to start talking about vectors and matrices. Which means that matrix algebra will feature heavily from now on. 

Suppose we have a model that is characterized by the likelihood function $p(\mathbf{y} \mid \boldsymbol{\theta})$, where $\boldsymbol{\theta}$ is a vector of model parameters. Intuitively, the likelihood function specifies how the observed data are generated from the model given a particular set of parameters. 

Now, suppose we have obtained an observed sample $\mathbf{y}=\left(y_{1}, \ldots, y_{T}\right)^{\prime}$, and we would like to learn about $\boldsymbol{\theta}$. How should we proceed? The goal of Bayesian methods is to obtain the posterior distribution $p(\boldsymbol{\theta} \mid \mathbf{y})$ that summaries all the information about the parameter vector $\boldsymbol{\theta}$ given the data.

Applying Bayes' theorem, the posterior distribution can be computed as

$$p(\boldsymbol{\theta} \mid \mathbf{y})=\frac{p(\boldsymbol{\theta}) p(\mathbf{y} \mid \boldsymbol{\theta})}{p(\mathbf{y})} \propto p(\boldsymbol{\theta}) p(\mathbf{y} \mid \boldsymbol{\theta})$$

where $p(\boldsymbol{\theta})$ is the prior distribution and

$$p(\mathbf{y})=\int p(\boldsymbol{\theta}) p(\mathbf{y} \mid \boldsymbol{\theta}) \mathrm{d} \boldsymbol{\theta}$$

is the marginal likelihood. Bayes' theorem says that knowledge of $\boldsymbol{\theta}$ comes from two sources: the prior distribution and an observed sample $y_{1}, \ldots, y_{T}$ summarized by the likelihood. The prior distribution $p(\boldsymbol{\theta})$ incorporates our subjective beliefs about $\boldsymbol{\theta}$ before we look at the data. 

The posterior distribution $p(\boldsymbol{\theta} \mid \mathbf{y})$ characterizes all relevant information about $\boldsymbol{\theta}$ given the data. For example, if we wish to obtain a point estimate of $\boldsymbol{\theta}$, we might compute the posterior mean $\mathbb{E}(\boldsymbol{\theta} \mid \mathbf{y})$. To characterize the uncertainty about $\boldsymbol{\theta}$, we might report the posterior standard deviations of the parameters. For instance, for the $i$ th element of $\boldsymbol{\theta}$, we can compute $\sqrt{\operatorname{Var}\left(\theta_{i} \mid \mathbf{y}\right)}$

In principle, these quantities can be computed given the posterior distribution $p(\boldsymbol{\theta} \mid \mathbf{y}) .$ In practice, however, they are often not available analytically. In those cases we would require simulation to approximate those quantities of interest.

To outline the main idea, suppose we obtain $R$ independent draws from $p(\boldsymbol{\theta} \mid \mathbf{y})$, say, $\boldsymbol{\theta}^{(1)}, \ldots, \boldsymbol{\theta}^{(R)}$. If we assume the first moment exists, i.e., $\mathbb{E}(\boldsymbol{\theta} \mid \mathbf{y})<\infty$, then by the weak law of large numbers the sample mean $\widehat{\boldsymbol{\theta}}=R^{-1} \sum_{r=1}^{R} \boldsymbol{\theta}^{(r)}$ converges in probability to $\mathbb{E}(\boldsymbol{\theta} \mid \mathbf{y})$ as $R$ tends to infinity. Since we can control the simulation size, we could approximate the posterior mean arbitrarily well - if we are patient enough. Similarly, other moments or quantiles can be estimated using the sample analogs.

Hence, estimation and inference become essentially a computational problem of obtaining draws from the posterior distribution. In general, sampling from arbitrary distribution is a difficult problem. Fortunately, there is now a large family of algorithms generally called Markov chain Monte Carlo (MCMC) methods to sample from complex distributions.

The basic idea behind these algorithms is to construct a Markov chain so that its limiting distribution is the target distribution - in our case the target is the posterior distribution. By construction, samples from the MCMC algorithms are autocorrelated. Fortunately, similar convergence theorems - called ergodic theorems-hold for these correlated samples. Under some weak regularity conditions, we can use draws from these MCMC algorithms to estimate any functions of the parameters arbitrary well, provided that the population analogs exist.

"""

# ╔═╡ f95ccee4-a2d3-4492-b869-551e61acf995
md"""

## Normal model with unknown $\sigma^{2}$
"""

# ╔═╡ 6c027ac7-ef68-4582-ba17-8f34a879a21d
md"""

We have already covered the normal model with known variance in great detail in a previous lecture. We also covered the case of a normal model with unknown variance in theory. However, now we will introduce our knowledge of Markov chain Monte Carlo methods to estimate the posterior distribution of a normal model with unknown variance. We will use Gibbs sampling to achieve this. You might notice that the notation is slightly different here, but the ideas are the same. 

"""

# ╔═╡ 66c936a3-055d-42a0-bcad-5176d10e5994
md"""

### Prior and likelihood selection 

"""

# ╔═╡ 3aad749e-1256-4f93-b119-4717d2b95607
md"""

The model in question is

$$\left(y_{n} \mid \mu, \sigma^{2}\right) \sim \mathcal{N}\left(\mu, \sigma^{2}\right), \quad n=1, \ldots, N$$

where both $\mu$ and $\sigma^{2}$ are now unknown. We assume the following prior for $\mu: \mathcal{N}\left(\mu_{0}, \sigma_{0}^{2}\right)$.

As for $\sigma^{2}$, which takes only positive values, a convenient prior is the **inverse-gamma prior**. It turns out that this prior is conjugate for the likelihood of the model.

A random variable $X$ is said to have an inverse-gamma distribution with shape parameter $\alpha>0$ and scale parameter $\beta>0$ if its density is given by

$$f(x ; \alpha, \beta)=\frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{-(\alpha+1)} \mathrm{e}^{-\beta / x}$$

We write $X \sim \mathcal{I} G(\alpha, \beta)$. This is the parameterization we use throughout these notes. There are other common parameterizations for the inverse-gamma distribution when comparing derivations and results across books and papers, it is important to first determine the parameterization used.

For $X \sim \mathcal{I} G(\alpha, \beta)$, its mean and variance are given by

$$\mathbb{E} X=\frac{\beta}{\alpha-1}$$

for $\alpha>1$, and

$$\operatorname{Var}(X)=\frac{\beta^{2}}{(\alpha-1)^{2}(\alpha-2)}$$

for $\alpha>2$.

"""

# ╔═╡ 83ae194c-e04f-4615-9b71-7c389513898c
md"""

### Posterior derivation

"""

# ╔═╡ 727a703f-1233-4098-86f3-3192e4de08d4
md"""

Given the likelihood and priors, we can now derive the joint posterior distribution of $\left(\mu, \sigma^{2}\right) .$ Again, by Bayes' theorem, the joint posterior density is given by

$$\begin{aligned}
p\left(\mu, \sigma^{2} \mid \mathbf{y}\right) & \propto p\left(\mu, \sigma^{2}, \mathbf{y}\right) \\
& \propto p(\mu) p\left(\sigma^{2}\right) p\left(\mathbf{y} \mid \mu, \sigma^{2}\right) \\
& \propto \mathrm{e}^{-\frac{1}{2 \sigma_{0}^{2}}\left(\mu-\mu_{0}\right)^{2}}\left(\sigma^{2}\right)^{-\left(\nu_{0}+1\right)} \mathrm{e}^{-\frac{S_{0}}{\sigma^{2}}} \prod_{n=1}^{N}\left(\sigma^{2}\right)^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2 \sigma^{2}}\left(y_{n}-\mu\right)^{2}}
\end{aligned}$$

"""

# ╔═╡ 14f11fc0-dc9b-4b77-b2da-ae534b911cd6
md"""
Even though we have an explicit expression for the joint posterior density, it is not obvious how we can compute analytically various quantities of interest, such as, $\mathbb{E}(\mu \mid \mathbf{y})$, the posterior mean of $\mu$ or $\operatorname{Var}\left(\sigma^{2} \mid \mathbf{y}\right)$, the posterior variance of $\sigma^{2}$. One way forward is to use Monte Carlo simulation to approximate those quantities. 
"""

# ╔═╡ d92d80ef-256c-443a-a81c-8d5f02e01e66
md"""

### Quick detour on Gibbs sampling
"""

# ╔═╡ 48c98f0d-d880-4abc-91d3-6c79be5fcf8a
md"""
Generally, if we wanted to approximate for example $\operatorname{Var}\left(\sigma^{2} \mid \mathbf{y}\right)$, we first obtain draws from the posterior distribution $\left(\mu, \sigma^{2} \mid \mathbf{y}\right), \operatorname{say},\left(\mu^{(1)}, \sigma^{2(1)}\right), \ldots,\left(\mu^{(R)}, \sigma^{2(R)}\right) .$ Then, we compute

$$\frac{1}{R} \sum_{r=1}^{R}\left(\sigma^{2(r)}-\bar{\sigma}^{2}\right)^{2}$$

where $\bar{\sigma}^{2}$ is the mean of $\sigma^{2(1)}, \ldots, \sigma^{2(R)}$

Now the problem becomes: How do we sample from the posterior distribution? This brings us to Markov chain Monte Carlo (MCMC) methods, which are a broad class of algorithms for sampling from arbitrary probability distributions. This is achieved by constructing a Markov chain such that its limiting distribution is the desired distribution. Below we discuss one such method, called Gibbs sampling.

Specifically, suppose we wish to sample from the target distribution $p(\boldsymbol{\Theta})=p\left(\boldsymbol{\theta}_{1}, \ldots, \boldsymbol{\theta}_{n}\right)$. A Gibbs sampler constructs a Markov chain $\boldsymbol{\Theta}^{(1)}, \boldsymbol{\Theta}^{(2)}, \ldots$ using the full conditional distributions $p\left(\boldsymbol{\theta}_{i} \mid \boldsymbol{\theta}_{1}, \ldots, \boldsymbol{\theta}_{i-1}, \boldsymbol{\theta}_{i+1}, \ldots, \boldsymbol{\theta}_{n}\right)$ as the transition kernels. Under certain regularity conditions, the limiting distribution of the Markov chain thus constructed is the target distribution.

Operationally, we start from an initial state $\boldsymbol{\Theta}^{(0)}$. Then, we repeat the following steps from $r=1$ to $R$ :

Given the current state $\boldsymbol{\Theta}=\boldsymbol{\Theta}^{(r)}$, generate $\mathbf{Y}=\left(\mathbf{Y}_{1}, \ldots, \mathbf{Y}_{n}\right)$ as follows:

1. Draw $\mathbf{Y}_{1} \sim p\left(\mathbf{y}_{1} \mid \boldsymbol{\theta}_{2}, \ldots, \boldsymbol{\theta}_{n}\right)$.

2. Draw $\mathbf{Y}_{i} \sim p\left(\mathbf{y}_{i} \mid \mathbf{Y}_{1}, \ldots, \mathbf{Y}_{i-1}, \boldsymbol{\theta}_{i+1}, \ldots, \boldsymbol{\theta}_{n}\right), i=2, \ldots, n-1$

3. Draw $\mathbf{Y}_{n} \sim p\left(\mathbf{y}_{n} \mid \mathbf{Y}_{1}, \ldots, \mathbf{Y}_{n-1}\right)$ 

It is important to note that the Markov chain $\boldsymbol{\Theta}^{(1)}, \boldsymbol{\Theta}^{(2)}, \ldots$ does not converge to a fixed vector of constants. Rather, it is the distribution of $\boldsymbol{\Theta}^{(r)}$ that converges to the target distribution.

In practice, one typically discards the first $R_{0}$ draws to eliminate the effect of the initial state $\boldsymbol{\Theta}^{(0)}$. The discarded draws are often refereed to as the 'burn-in'. There are a number of convergence diagnostics to test if the Markov chain has converged to the limiting distribution. One popular test is the Geweke's convergence diagnostics.

"""

# ╔═╡ d811cfd9-3bdd-4830-8e9d-ecd4d7d2c890
md""" 

### Posterior and Gibbs sampling
"""

# ╔═╡ d3ceb4ea-6d45-4545-be09-8446f103c2e5
md"  

Now, after this discussion of Gibbs sampling, we return to the estimation of the two parameter normal model. To construct a Gibbs sampler to draw from the posterior distribution $p\left(\mu, \sigma^{2} \mid \mathbf{y}\right)$, we need to derive two conditional distributions: $p\left(\mu \mid \mathbf{y}, \sigma^{2}\right)$ and $p\left(\sigma^{2} \mid \mathbf{y}, \mu\right)$

To derive the first conditional distribution, note that given $\sigma^{2}$, this is the same normal model with known variance discussed in last section. Thus, using the same derivation before, we have

$$\begin{aligned}
p\left(\mu \mid \mathbf{y}, \sigma^{2}\right) & \propto p(\mu) p\left(\mathbf{y} \mid \mu, \sigma^{2}\right) \\
& \propto \mathrm{exp}{\left[-\frac{1}{2}\left(\left(\frac{1}{\sigma_{0}^{2}}+\frac{N}{\sigma^{2}}\right) \mu^{2}-2 \mu\left(\frac{\mu_{0}}{\sigma_{0}^{2}}+\frac{N \bar{y}}{\sigma^{2}}\right)\right)\right]}
\end{aligned}$$

Hence, $\left(\mu \mid \mathbf{y}, \sigma^{2}\right) \sim \mathcal{N}\left(\widehat{\mu}, D_{\mu}\right)$, where

$$D_{\mu}=\left(\frac{1}{\sigma_{0}^{2}}+\frac{N}{\sigma^{2}}\right)^{-1}, \quad \widehat{\mu}=D_{\mu}\left(\frac{\mu_{0}}{\sigma_{0}^{2}}+\frac{N \bar{y}}{\sigma^{2}}\right)$$

Next, we derive the conditional distribution of $\sigma^{2}$ :

$$\begin{aligned}
p\left(\sigma^{2} \mid \mathbf{y}, \mu\right) & \propto p\left(\sigma^{2}\right) p\left(\mathbf{y} \mid \mu, \sigma^{2}\right) \\
& \propto\left(\sigma^{2}\right)^{-\left(\nu_{0}+1\right)} \mathrm{e}^{-\frac{S_{0}}{\sigma^{2}}}\left(\sigma^{2}\right)^{-N / 2} \mathrm{e}^{-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(y_{n}-\mu\right)^{2}} \\
& \propto\left(\sigma^{2}\right)^{-\left(\nu_{0}+N / 2+1\right)} \mathrm{e}^{-\frac{S_{0}+\sum_{n=1}^{N}\left(y_{n}-\mu\right)^{2} / 2}{\sigma^{2}}}
\end{aligned}$$

Is this a known distribution? It turns out that this is an inverse-gamma distribution. Comparing this expression with the generic inverse-gamma distribution, we have

$$\left(\sigma^{2} \mid \mathbf{y}, \mu\right) \sim \mathcal{I} G\left(\nu_{0}+\frac{N}{2}, S_{0}+\frac{1}{2} \sum_{n=1}^{N}\left(y_{n}-\mu\right)^{2}\right)$$


To summarize, the Gibbs sampler for the two-parameter model is given as below. Pick some initial values $\mu^{(0)}=a_{0}$ and $\sigma^{2(0)}=b_{0}>0 .$ Then, repeat the following steps from $r=1$ to $R$ :

1. Draw $\mu^{(r)} \sim p\left(\mu \mid \mathbf{y}, \sigma^{2(r-1)}\right)$.

2. Draw $\sigma^{2(r)} \sim p\left(\sigma^{2} \mid \mathbf{y}, \mu^{(r)}\right)$

After discarding the first $R_{0}$ draws as burn-in, we can use the draws $\mu^{\left(R_{0}+1\right)}, \ldots, \mu^{(R)}$ and $\sigma^{2\left(R_{0}+1\right)}, \ldots, \sigma^{2(R)}$ to compute various quantities of interest. For example, we can use

$$\frac{1}{R-R_{0}} \sum_{r=R_{0}+1}^{R} \mu^{(r)}$$

as an estimate of $\mathbb{E}(\mu \mid \mathbf{y})$.

"

# ╔═╡ 5cb5ab74-da6b-439a-bea4-75a3d0e43c63
md" ### Practical implementation "

# ╔═╡ bade1741-5e84-413a-9c67-932bd6748c49
md"""

As an illustration, the following code first generates a dataset of $N=100$ observations from the two-parameter normal model with $\mu=3$ and $\sigma^{2}=0.1$. Then, it implements a Gibbs sampler to sequentially to sample from the two conditional distributions: $p\left(\mu \mid \mathbf{y}, \sigma^{2}\right)$ and $p\left(\sigma^{2} \mid \mathbf{y}, \mu\right)$.

"""

# ╔═╡ b94db7f0-9f38-4761-a3c3-4d6fc4729ae9
begin
	# Parameters for first example
	nsim 	= 10_000
	burnin 	= 1000
	μ 		= 3.0
	σ2 		= 0.1
	N 		= 100
	μ_0 	= 0.0
	σ2_0 	= 100
	ν_0 	= 3.0
	Σ_0 	= 0.5
	μ_1 	= 0.0
	σ2_1 	= 1.0
	
	# Additional parameters for second example
	T 		= 500
	β 		= [1.0 5.0]
	β_0 	= [0.0 0.0]
end;

# ╔═╡ 1a6c859c-e3e7-4ad9-9299-091b6b1d2bbf
function data_gen(μ, σ2, N)
    μ .+ sqrt(σ2) .* randn(N, 1)
end;

# ╔═╡ 0980d7a1-129b-4724-90fb-b46e3088d2d6
data_gen(μ, σ2, N);

# ╔═╡ 0919cb0d-ba03-49c8-b2b9-53a467c39f87
function gibbs(nsim, burnin, μ, σ2, N, μ_0, σ2_0, ν_0, Σ_0, μ_1, σ2_1)
    y = data_gen(μ, σ2, N) # Generated data
    store_θ = zeros(nsim, 2) # Initialise the store_θ array

    # Start the Gibbs sampling procedure
    for i in 1:nsim + burnin
        # Sample from μ 
        D_μ   = 1 / (1 / σ2_0 .+ N / σ2_1)
        μ_hat = D_μ .* (μ_0 / σ2_0 .+ sum(y) / σ2_1)
        μ_1   = μ_hat .+ sqrt(D_μ) .* randn() # Affine transformation is also normal

        # Sample from σ2
        σ2_1  = 1/rand(Gamma(ν_0 .+ N/2, 1/(Σ_0 .+ sum((y .- μ_1).^2) / 2)))

        if i > burnin
            isave = i .- burnin
            store_θ[isave, :] = [μ_1, σ2_1]
        end
    end
    mean(store_θ[:, 1]), mean(store_θ[:, 2]), store_θ
end;

# ╔═╡ 343202b3-23b5-4600-b912-7db4ab58deaf
post_gibbs = gibbs(nsim, burnin, μ, σ2, N, μ_0, σ2_0, ν_0, Σ_0, μ_1, σ2_1); # posterior mean of μ and σ^2

# ╔═╡ 3aeab073-c98a-4213-a835-3098233ba90c
md" Let us see what this looks like when we plot the posterior. "

# ╔═╡ afc56ef2-d5d2-4624-9ad9-11f15429607f
begin
	x_l = collect(1:(nsim-burnin));
	histogram(x_l, post_gibbs[3][:,1], normalize=:pdf, legend = false)
end

# ╔═╡ 41365136-5d0e-4a4e-808f-d0e90a14c5dd
md""" We can also draw nice contour plots... """

# ╔═╡ 3335094d-a67b-471c-834d-e22089933104
begin
	gr()
	const N₁ = 100_000
	const μ₁ = [post_gibbs[1], post_gibbs[1]]
	const Σ = [1 post_gibbs[2]; post_gibbs[2] 1]

	const mvnormal = MvNormal(μ₁, Σ)

	data = rand(mvnormal, N₁)'
	x₁ = 0:0.01:6
	y₁ = 0:0.01:6
	dens_mvnormal = [pdf(mvnormal, [i, j]) for i in x₁, j in y₁]
	contour(x₁, y₁, dens_mvnormal, xlabel="X", ylabel="Y", fill=true, fillcolour = :ice)
end

# ╔═╡ 330bd7cb-7102-41c2-b7a4-f053669960c3
md""" Or even fancy surface plots """

# ╔═╡ 80e6619b-ac42-453b-8f38-850b2b99d000
surface(x₁, y₁, dens_mvnormal, fillcolour = :ice, backgroundinside = :ghostwhite)

# ╔═╡ 82b96729-33c2-49b0-b908-562faf903a1e
md"""

## Bayesian linear regression

"""

# ╔═╡ 1f2c9795-0b2c-4a14-9f28-1fef68f6b467
md"""

The workhorse model in econometrics is the normal linear regression model. Virtually all other more flexible models are built upon this foundation. It is therefore vital to fully understand how one estimates this model. In this section we will provide the details in deriving the likelihood and the posterior sampler. 

"""

# ╔═╡ 70193cca-ce19-49ee-aa0c-06997affe2a6
md""" ### Linear regression in matrix notation """


# ╔═╡ 1703eb19-aeca-4ebe-a9b3-18dfbf4efdfe
md"""

To start, suppose we have data on a dependent variable $y_{t}$ for $t=1, \ldots, T$. Then, consider the following linear regression model:

$$y_{t}=\beta_{1}+x_{2, t} \beta_{2}+\cdots+x_{k, t} \beta_{k}+\varepsilon_{t}$$

where $\varepsilon_{1}, \ldots, \varepsilon_{T}$ are assumed to be iid $\mathcal{N}\left(0, \sigma^{2}\right)$, and $x_{1, t}, x_{2, t}, \ldots, x_{k, t}$ are the $k$ regressors (explanatory variables) and $\beta_{1}, \ldots, \beta_{k}$ are the associated regression coefficients.

Note that in this construction we set $x_{1, t} = 1$ for all $t = 1, \ldots, T$ to allow for an intercept. In addition, if we replace the $x_{i, t}$ values to be lags of the dependent variable then this would be an autoregressive model. 

To derive the likelihood, it is more convenient to write this in matrix notation. In particular, we stack the observations over $t=1, \ldots, T$ so that each row represents the observation at time $t$. Let $\mathbf{y}=\left(y_{1}, \ldots, y_{T}\right)^{\prime}$ and $\boldsymbol{\beta}=\left(\beta_{1}, \ldots, \beta_{k}\right)^{\prime} .$ Finally, we have that $\varepsilon$ is a $T \times 1$ vector that is stacked in the same way as $\mathbf{Y}$. Then, rewrite the whole system of $T$ equations as:

$$\left(\begin{array}{c}
y_{1} \\
y_{2} \\
\vdots \\
y_{T}
\end{array}\right)=\left(\begin{array}{cccc}
1 & x_{2,1} & \cdots & x_{k, 1} \\
1 & x_{2,2} & \cdots & x_{k, 2} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{2, T} & \cdots & x_{k, T}
\end{array}\right)\left(\begin{array}{c}
\beta_{1} \\
\vdots \\
\beta_{k}
\end{array}\right)+\left(\begin{array}{c}
\varepsilon_{1} \\
\varepsilon_{2} \\
\vdots \\
\varepsilon_{T}
\end{array}\right)$$

Make sure the dimensions above make sense to you. For our example, $\mathbf{y}$ is a $T \times 1$ vector, $\mathbf{X}$ is a $T \times k$ matrix, $\boldsymbol{\beta}$ is a $k \times 1$ vector and $\varepsilon$ is a $T \times 1$ vector.

Our system can be written more succinctly as,

$$\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\varepsilon$$

Since we assume that $\varepsilon_{1}, \ldots, \varepsilon_{T}$ are iid $\mathcal{N}\left(0, \sigma^{2}\right), \varepsilon$ has a multivariate normal distribution with mean vector $\mathbf{0}_{T}$ and covariance matrix $\mathbf{\Sigma} = \sigma^{2} \mathbf{I}_{T}$, where $\mathbf{0}_{T}$ is a $T \times 1$ vector of zeros and $\mathbf{I}_{T}$ is the $T \times T$ identity matrix. That is,

$$\varepsilon \sim \mathcal{N}\left(0, \sigma^{2} \mathbf{I}_{T}\right)$$

Now we move on to the derivation of the likelihood function. 

"""

# ╔═╡ 9ffed7a8-f8d3-4c1a-affa-5d6646683829
md"""

### Estimation

"""

# ╔═╡ 993c9137-d708-4feb-8a85-51d85c38fc8d
md"""

The first thing that we do in the estimation procedure is derive the likelihood function. This function is defined as the joint density of the data given the parameters. In our case, the likelihood function is the joint density of $\mathbf{y}$ given $\boldsymbol{\beta}$ and $\sigma^{2}$. To derive the likelihood of the normal linear regression model, we use two useful results. 

First, an affine transformation, i.e. a linear transformation followed by a translation of a normal random vector, is also a normal random vector. You can find a discussion on this results [here](https://math.stackexchange.com/questions/332441/affine-transformation-applied-to-a-multivariate-gaussian-random-variable-what). Now, $\mathbf{y}$ is an affine transformation of $\varepsilon$, which is assumed to have a multivariate normal distribution. Thus, $\mathbf{y}$ also has a normal distribution. Since a normal distribution is uniquely determined by its mean vector and covariance matrix, it suffices to compute the mean and covariance matrix of $\mathbf{y}$.

This brings us to the next useful result: suppose $\mathbf{u}$ has a mean vector $\boldsymbol{\mu}_{\mathrm{u}}$ and covariance matrix $\boldsymbol{\Sigma}_{\mathbf{u}}$. Let $\mathbf{v}=\mathbf{A} \mathbf{u}+\mathbf{c}$ for constant matrices $\mathbf{A}$ and $\mathbf{c} .$ Then the mean vector and covariance matrix of $\mathbf{v}$ are given by

$$\mathbb{E} \mathbf{v}=\mathbf{A} \boldsymbol{\mu}_{\mathbf{u}}+\mathbf{c}, \quad \operatorname{Cov}(\mathbf{u})=\mathbf{A} \boldsymbol{\Sigma}_{\mathbf{u}} \mathbf{A}^{\prime}$$

Using this result, it is easy to see that given $\boldsymbol{\beta}$ and $\sigma^{2}$,

$$\mathbb{E} \mathbf{y}=\mathbf{X} \boldsymbol{\beta}, \quad \operatorname{Cov}(\mathbf{y})=\sigma^{2} \mathbf{I}_{T}$$

Putting it all together, we have

$$\left(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^{2}\right) \sim \mathcal{N}\left(\mathbf{X} \boldsymbol{\beta}, \sigma^{2} \mathbf{I}_{T}\right)$$

This means that estimating the linear regression model is the same as estimating the parameters of a multivariate Nomral distribution with unknown mean and covariance. In order to estimate the model parameters with Bayesian methods we need to apply Bayes' rule. 

$$\begin{align}
p(\boldsymbol{\beta},\sigma^2|\mathbf{y}) \propto p(\mathbf{y}|\boldsymbol{\beta},\sigma^2)p(\boldsymbol{\beta},\sigma^2)
\end{align}$$

The posterior is comprised of the likelihood function $p(\mathbf{y}|\boldsymbol{\beta},\sigma^2)$ and the joint prior distribution $p(\boldsymbol{\beta},\sigma^2)$. 

"""

# ╔═╡ dcf7adfb-0d6e-4459-a3cc-6a20cc632c72
md"""

#### Likelihood function 

"""

# ╔═╡ a541e83e-22c5-48c7-9beb-705d26f602ee
md""" 


For this model the likelihood function is then given by:

$$\begin{aligned}
p\left(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^{2}\right) &=\left|2 \pi \sigma^{2} \mathbf{I}_{T}\right|^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}\left(\sigma^{2} \mathbf{I}_{T}\right)^{-1}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})} \\
&=\left(2 \pi \sigma^{2}\right)^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2 \sigma^{2}}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})}
\end{aligned}$$

where $|\cdot|$ denotes the determinant. Also note that the second equality follows from the result that for an $n \times n$ matrix $\mathbf{A}$ and scalar $c,|c \mathbf{A}|=c^{n}|\mathbf{A}|$. In our example this means that $|\sigma^{2} \mathbf{I}_{t}| = (\sigma^{2})^{T}$. For this equation we also have that $\sigma^{2} \mathbf{I}_{t} = \frac{1}{\sigma^{2}}$.

"""

# ╔═╡ 5b442d42-b6ff-4677-afff-a00cfdb00fbc
md""" 

#### Priors
"""

# ╔═╡ 8a8dea4a-3f49-4095-a1a0-d4d3c803c171
md"""

In specifying priors we need to take a stance on the dependence between the model parameters $\boldsymbol{\beta}$ and $\sigma^{2}$. 

If we believe that the regression coefficients might depend on the error variance, i.e. $p(\boldsymbol{\beta},\sigma^2)=p(\boldsymbol{\beta}|\sigma^2)p(\sigma^2)$, then it turns out (see [Chapter 2 of Koop's Bayesian Econometrics](https://www.amazon.com/Bayesian-Econometrics-Gary-Koop/dp/0470845678)) that the natural conjugate prior for $\boldsymbol{\beta}$ and $\sigma^2$ is the [Normal-Gamma distribution](https://en.wikipedia.org/wiki/Normal-gamma_distribution) denoted by

$$\boldsymbol{\beta},\sigma^2 \sim NG(m_0,v_0,\nu_0,S_0)$$

in which $m_0,v_0,\nu_0$ and $S_0$ are hyperparameters. 

If we believe that the regression coefficients and the error variance are independent, i.e. $p(\boldsymbol{\beta},\sigma^2)=p(\boldsymbol{\beta})p(\sigma^2)$, then it's convenient to specify the following conjugate priors
1. Multivariate Normal prior for $\boldsymbol{\beta}$, i.e. $\boldsymbol{\beta}\sim N(\mathbf{m}_0,\mathbf{V}_0)$
2. Inverse-Gamma prior for $\sigma^2$, i.e. $\sigma^2\sim IG(\nu_0,S_0)$

The explicit prior densities are given by, 

$$\begin{aligned}
p(\boldsymbol{\beta}) &=(2 \pi)^{-\frac{k}{2}}\left|\mathbf{V}_{0}\right|^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2}\left(\boldsymbol{\beta}-\boldsymbol{m}_{0}\right)^{\prime} \mathbf{V}_{\boldsymbol{0}}^{-1}\left(\boldsymbol{\beta}-\boldsymbol{m}_{0}\right)} \\
p\left(\sigma^{2}\right) &=\frac{S_{0}^{\nu_{0}}}{\Gamma\left(\nu_{0}\right)}\left(\sigma^{2}\right)^{-\left(\nu_{0}+1\right)} \mathrm{e}^{-\frac{S_{0}}{\sigma^{2}}}
\end{aligned}$$

We will assume the latter is true.  

"""

# ╔═╡ be364d9e-c04d-47fd-ac2f-2be93bdd3f87
md"""

#### Posterior

"""

# ╔═╡ 4a985421-7513-407a-8edf-01df7b405c55
md"""

To get the posterior, we combine the prior and likelihood

$$\begin{align}
p(\boldsymbol{\beta},\sigma^2|\mathbf{y}) &\propto p(\mathbf{y}|\boldsymbol{\beta},\sigma^2)p(\boldsymbol{\beta})p(\sigma^2)\\
                     &\propto \underset{p(\mathbf{y}|\boldsymbol{\beta},\sigma^2)}{\underbrace{(\sigma^2)^{-\frac{T}{2}}\exp(-\frac{1}{2\sigma^2}(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})'(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}))}}\underset{p(\boldsymbol{\beta})}{\underbrace{\exp(-\frac{1}{2}(\boldsymbol{\beta}-\mathbf{m}_0)'\mathbf{V}_0^{-1}(\boldsymbol{\beta}-\mathbf{m}_0))}}\underset{p(\sigma^2)}{\underbrace{(\sigma^2)^{-(\nu_0+1)}\exp(-\frac{S_0}{\sigma^2})}}\\
                     &\propto (\sigma^2)^{-(\frac{T}{2}\nu_0+1)}\exp(-\frac{1}{2\sigma^2}(S_0 + (\mathbf{y}-\mathbf{X}\boldsymbol{\beta})'(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})) -\frac{1}{2}(\boldsymbol{\beta}-\mathbf{m}_0)'\mathbf{V}_0^{-1}(\boldsymbol{\beta}-\mathbf{m}_0)))\\
\end{align}$$

This final distribution is an unknown distribution. However, we know that we can use a two-block Gibbs sampler to estimate the parameters of a univariate Normal distribution (from above). Now we need to see whether we can extend this to parameters of a multivariate Normal distribution. 

"""

# ╔═╡ 5b3bd2d1-2697-4516-aa58-99b3d229e5f3
md"""

#### Gibbs sampling (full derivation)

"""

# ╔═╡ 8cf7fa3c-aaf2-4684-869e-d099218fe6e9
md"""

Now, we derive a Gibbs sampler for the normal linear regression. Specifically, we need to derive the two conditional densities $p\left(\sigma^{2} \mid \mathbf{y}, \boldsymbol{\beta}\right)$ and $p\left(\boldsymbol{\beta} \mid \mathbf{y}, \sigma^{2}\right)$. The full conditional distributions are

$p(\boldsymbol{\beta} \mid \mathbf{y},\sigma^2)\propto p(\mathbf{y} \mid \boldsymbol{\beta},\sigma^2)p(\boldsymbol{\beta})$
$p(\sigma^2 \mid \mathbf{y},\boldsymbol{\beta})\propto p(\mathbf{y} \mid \boldsymbol{\beta},\sigma^2) p(\sigma^2)$


**First block:**

First, we can show that the conditional density $p\left(\sigma^{2} \mid \mathbf{y}, \boldsymbol{\beta}\right)$ is inverse-gamma:

$$\begin{aligned}
p\left(\sigma^{2} \mid \mathbf{y}, \boldsymbol{\beta}\right) & \propto p\left(\mathbf{y} \mid \boldsymbol{\beta}, \sigma^{2}\right) p\left(\sigma^{2}\right) \\
& \propto\left(\sigma^{2}\right)^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2 \sigma^{2}}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})} \times\left(\sigma^{2}\right)^{-\left(\nu_{0}+1\right)} \mathrm{e}^{-\frac{S_{0}}{\sigma^{2}}} \\
&=\left(\sigma^{2}\right)^{-\left(\frac{T}{2}+\nu_{0}+1\right)} \mathrm{e}^{-\frac{1}{\sigma^{2}}\left(S_{0}+\frac{1}{2}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})\right)}
\end{aligned}$$ 

We recognize this as the kernel of an inverse-gamma density. In fact, we have

$$\left(\sigma^{2} \mid \mathbf{y}, \boldsymbol{\beta}\right) \sim \mathcal{I} G\left(\nu_{0}+\frac{T}{2}, S_{0}+\frac{1}{2}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})\right)$$

In other words it is the the kernel of an inverse-gamma density with scale parameter $\nu = \frac{T}{2}+\nu_0$ and shape parameter $S = S_0 + (\mathbf{y}-\mathbf{X}\boldsymbol{\beta})'(\mathbf{y}-\mathbf{X}\boldsymbol{\beta})$. Thus

$$\sigma^2 \mid \mathbf{y},\boldsymbol{\beta} \sim \mathcal{I} G(\nu,S)$$

**Second block:** 

Next, we derive the conditional density $p\left(\boldsymbol{\beta} \mid \mathbf{y}, \sigma^{2}\right)$. To that end, first note that the likelihood involves the quadratic term $(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})$, which can be expanded as

$$\begin{aligned}
(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta}) &=\mathbf{y}^{\prime} \mathbf{y}-\mathbf{y}^{\prime} \mathbf{X} \boldsymbol{\beta}-\boldsymbol{\beta}^{\prime} \mathbf{X}^{\prime} \mathbf{y}+\boldsymbol{\beta}^{\prime} \mathbf{X}^{\prime} \mathbf{X} \boldsymbol{\beta} \\
&=\boldsymbol{\beta}^{\prime} \mathbf{X}^{\prime} \mathbf{X} \boldsymbol{\beta}-2 \boldsymbol{\beta}^{\prime} \mathbf{X}^{\prime} \mathbf{y}+\mathbf{y}^{\prime} \mathbf{y}
\end{aligned}$$

where we have used the fact that $\mathbf{y}^{\prime} \mathbf{X} \boldsymbol{\beta}$ is a scalar, and therefore it is equal to its transpose:

$$\mathbf{y}^{\prime} \mathbf{X} \boldsymbol{\beta}=\left(\boldsymbol{\beta}^{\prime} \mathbf{X}^{\prime} \mathbf{y}\right)^{\prime}=\boldsymbol{\beta}^{\prime} \mathbf{X}^{\prime} \mathbf{y}$$

Similarly, the quadratic term in the prior can be expanded as

$$\left(\boldsymbol{\beta}-\boldsymbol{m}_{0}\right)^{\prime} \mathbf{V}_{\boldsymbol{\beta}}^{-1}\left(\boldsymbol{\beta}-\boldsymbol{m}_{0}\right)=\boldsymbol{\beta}^{\prime} \mathbf{V}_{\boldsymbol{0}}^{-1} \boldsymbol{\beta}-2 \boldsymbol{\beta}^{\prime} \mathbf{V}_{\boldsymbol{0}}^{-1} \boldsymbol{m}_{0}+\boldsymbol{m}_{0}^{\prime} \mathbf{V}_{\boldsymbol{0}}^{-1} \boldsymbol{m}_{0}$$

For notational convenience, let $\frac{1}{\sigma^{2}} = \Sigma^{-1}$ where $\Sigma = \sigma^{2}\mathbf{I}_{T}$. The conditional density $p\left(\boldsymbol{\beta} \mid \mathbf{y}, \sigma^{2}\right)$ is then given by

$$\begin{align}
p(\boldsymbol{\beta} \mid \mathbf{y},\sigma^2)&\propto p(\mathbf{y} \mid \boldsymbol{\beta},\sigma^2)p(\boldsymbol{\beta})\\
                          &\propto\exp(-\frac{1}{2}((\mathbf{y}-\mathbf{X}\boldsymbol{\beta})'\boldsymbol{\Sigma}^{-1}(\mathbf{y}-\mathbf{X}\boldsymbol{\beta}))\exp(-\frac{1}{2}(\boldsymbol{\beta}-\mathbf{m}_0)'\mathbf{V}_0^{-1}(\boldsymbol{\beta}-\mathbf{m}_0))\\
                          &\propto\exp(-\frac{1}{2}((\boldsymbol{\beta}'\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}\boldsymbol{\beta}-2\boldsymbol{\beta}'\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{y}))\exp(-\frac{1}{2}(\boldsymbol{\beta}'\mathbf{V}_0^{-1}\boldsymbol{\beta}-2\boldsymbol{\beta}'\mathbf{V}_0^{-1}\mathbf{m}_0))\\
                          &=\exp(-\frac{1}{2}(\boldsymbol{\beta}'(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{X}+\mathbf{V}_0^{-1})\boldsymbol{\beta}-2\boldsymbol{\beta}'(\mathbf{X}'\boldsymbol{\Sigma}^{-1}\mathbf{y} + \mathbf{V}_0^{-1}\mathbf{m}_0))\\
\end{align}$$

Since the exponent is quadratic in $\boldsymbol{\beta}, p\left(\boldsymbol{\beta} \mid \mathbf{y}, \sigma^{2}\right)$ is a multivariate normal density, 

$$\boldsymbol{\beta} \mid \mathbf{y}, \sigma^{2} \sim \mathcal{N}\left(\widehat{\boldsymbol{\beta}}, \mathbf{D}_{\boldsymbol{\beta}}\right)$$

for some mean vector $\widehat{\boldsymbol{\beta}}$ and covariance matrix $\mathbf{D}_{\beta}$. Next, we derive explicit expressions for $\widehat{\boldsymbol{\beta}}$ and $\mathbf{D}_{\boldsymbol{\beta}}$

The kernel of $\mathcal{N}\left(\widehat{\boldsymbol{\beta}}, \mathbf{D}_{\boldsymbol{\beta}}\right)$ is simply

$$p(\boldsymbol{\beta} \mid \mathbf{y}, \sigma^{2}) \propto \mathrm{exp}\left({-\frac{1}{2}\left(\boldsymbol{\beta}^{\prime} \mathbf{D}_{\boldsymbol{\beta}}^{-1} \boldsymbol{\beta}-2 \boldsymbol{\beta}^{\prime} \mathbf{D}_{\boldsymbol{\beta}}^{-1} \widehat{\boldsymbol{\beta}}\right)}\right)$$

Comparing this with the kernel in a previous expression, we have that 

$$\mathbf{D}_{\boldsymbol{\beta}}=\left(\mathbf{V}_{0}^{-1}+\frac{1}{\sigma^{2}} \mathbf{X}^{\prime} \mathbf{X}\right)^{-1}, \quad \widehat{\boldsymbol{\beta}}=\mathbf{D}_{\beta}\left(\mathbf{V}_{0}^{-1} \boldsymbol{m}_{0}+\frac{1}{\sigma^{2}} \mathbf{X}^{\prime} \mathbf{y}\right)$$

Even though one can use the built-in functionality in _Julia_ to sample from $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$, it is instructive to see how it can be done by simply transforming independent standard normal random variables.






"""

# ╔═╡ 0999d280-3d93-4a0d-a487-215a3f068fee
md""" 

### Sampling from a Multivariate Normal distribution

"""

# ╔═╡ f43f8154-5bd3-4a33-8316-991417549d32
md"""

To generate $R$ independent draws from $\mathcal{N}(\boldsymbol{\mu}, \mathbf{\Sigma})$ of dimension $n$, carry out the following steps:

1. Compute the lower Cholesky factorization $\mathbf{\Sigma}=\mathbf{B B}^{\prime}$.

2. Generate $\mathbf{Z}=\left(Z_{1}, \ldots, Z_{n}\right)^{\prime}$ by drawing $Z_{1}, \ldots, Z_{n} \sim \mathcal{N}(0,1)$.

3. Return $\mathbf{U}=\boldsymbol{\mu}+\mathbf{B Z}$.

4. Repeat Steps 2 and 3 independently $R$ times.

We can show what this process looks like below. 

"""

# ╔═╡ 6ec571a0-6407-439a-b7f3-fb15085d73bf
begin
	## Sample from a Multivariate Normal Distribution
	# Set up Normal random vector
	n = 2 # Dimension of Normal random vector 
	mu = ones(n);
	A = randn(n, n); 
	Sig = A' * A; # positive definite matrix
	
	# Simulate
	R = 10000; # Simulation size
	U = zeros(R, 2); # storage matrix
	for i in 1:R
	    U[i,:] = transpose(mu + (cholesky(Sig).L) * rand(Normal(0, 1), n));
	end
	
	# Summary statistics
	muhat = mean(U, dims=1);
	Sighat = cov(U);
end;

# ╔═╡ 3752db0a-3c4f-48e9-89ec-c14927584a1d
md""" In _Julia_ we could have simply run the following command to sample from this distribution """

# ╔═╡ 6e288af8-8f7d-4d4a-9fc3-4bb7df212e8e
rand(MvNormal(mu, Sig));

# ╔═╡ fd643db4-ad7f-4203-b7ed-96c730343013
md""" However, it is computationially more efficient to use the first method.  """

# ╔═╡ c5c4da02-8b6b-4318-b625-ce4f31703c79
md"""

### Gibbs Sampler for Regression

"""

# ╔═╡ 428eb291-d516-4b56-91e7-df3a92cd3a4f
md"""

Pick some initial values $\boldsymbol{\beta}^{(0)}=\mathbf{a}_{0}$ and $\sigma^{2(0)}=b_{0}>0 .$ Then, repeat the following steps from $r=1$ to $R:$

1. Draw $\sigma^{2(r)} \sim p\left(\sigma^{2} \mid \mathbf{y}, \boldsymbol{\beta}^{(r-1)}\right)$ (inverse-gamma).

2. Draw $\boldsymbol{\beta}^{(r)} \sim p\left(\boldsymbol{\beta} \mid \mathbf{y}, \sigma^{2(r)}\right)$ (multivariate normal).

Next we show how to implement this process in Julia. 

"""

# ╔═╡ 223f4b6f-8321-4142-9dfa-1afe371d40ac
md"""

### Practical implementation

"""

# ╔═╡ edaf930a-4933-4047-8854-bbb02ea9c39c
md"""

As an example, the following code first generates a sample of $T=500$ observations from a normal linear regression model. It then implements the Gibbs sampler in our second algorithm, where the sampler is initialized using the least squares estimate. The posterior means of the model parameters are stored in the variable `theta_hat` and the corresponding $95$ percent credible intervals are stored in `theta_CI`.

"""

# ╔═╡ c9c8d57b-3010-4056-aaa3-a0354cf456fd
function data_gen2(T, β, σ2)
	
    X = [ones(T, 1) 1 .+ randn(T, 1)]
    return X * β' .+ sqrt(σ2) .* randn(T, 1)
end;

# ╔═╡ bb10296b-1b93-46d9-bd5f-c11a4ed8bcae
y_dg = data_gen2(T, β, σ2)

# ╔═╡ bfd8fab5-64fb-4bee-aee6-e9691cc887f8
mean(y_dg)

# ╔═╡ 1c38646c-b750-4402-bc70-571670cd7acd
var(y_dg)

# ╔═╡ 000cf50f-d3f8-452b-b661-11545c2ec0c4
function get_prior(ν_0)

    Vβ_0 = I(2) ./ 100 # Prior for beta
    Σ_0  = 1 * (ν_0 - 1) # Prior for Sig2

    return Vβ_0, Σ_0
end;

# ╔═╡ 7b70270e-2991-40ee-97a9-082691e68701
function ols_est(T, β, σ2)

    y = data_gen2(T, β, σ2) # Generated data
    X = [ones(T, 1) 1 .+ randn(T, 1)]

    β_1 = (X' * X) \ (X' * y)
    σ2_1 = sum((y .- X * β_1) .^ 2) / T

    return β_1, σ2_1
end;

# ╔═╡ 791939ae-3e00-4d6b-968c-5a82a78f55f8
function gibbs_linear(nsim, burnin, T, β, σ2, β_0, ν_0)

    y = data_gen2(T, β, σ2) # Generated data
    X = [ones(T, 1) 1 .+ randn(T, 1)]

    Vβ_0, Σ_0 = get_prior(ν_0)
    β_1, σ2_1 = ols_est(T, β, σ2)

    # Initialise Markov Chain
    store_θ = zeros(nsim, 3)

    # Start the Gibbs sampling procedure
    for i in 1:nsim + burnin

        # Sample from beta
        D_β   = (Vβ_0 .+ X' * X / σ2_1) \ I(2)
        β_hat = D_β * (Vβ_0 * β_0' .+ X' * y ./ σ2_1)

        # Refer to section on sampling from multivariate normal
        C = cholesky(Hermitian(D_β)).L
        β_1 = β_hat .+ C * randn(2,1)

        # Sample from σ2
        e = y - X * β_1
        sig2 = rand(InverseGamma(ν_0 .+ T/2, Σ_0 + (e' * e / 2)[1]))

        if i > burnin
            isave = i .- burnin
            store_θ[isave, :] = [β_1' σ2_1]
        end
    end
    mean(store_θ[:, 1]), mean(store_θ[:, 2]), store_θ
end;

# ╔═╡ 9d1361c3-6de1-4326-85f8-4a272856d16b
posterior_beta, posterior_sig2, store_θ  = gibbs_linear(nsim, burnin, T, β, σ2, β_0, ν_0);

# ╔═╡ 01f5e6cd-1efa-49a2-8d0d-b5536bdd7388
begin
	x_n = collect(1:(nsim-burnin));
	pg1 = histogram(x_n, store_θ[:,1], normalize=:pdf, title = "Posterior: Beta", legend = false)
	
	pg2 = histogram(x_n, store_θ[:,3], normalize=:pdf, title = "Posterior: Sigma2", legend = false)

	plot(pg1,pg2, layout = (1,2), legend = false)
end

# ╔═╡ 3006d4f8-1cca-4947-8dc6-3b77b278fbb8
md"""

### Example: Autoregressive model 

"""

# ╔═╡ 22bdce87-a1f2-440b-965a-391148b010ac
md""" As you know from the first semester, the autoregressive model is a workhorse model in economics and finance. This model is represented by 

$$y_t = \rho_0 + \rho_1 y_{t-1} + \dots + \rho_p y_{t-p} + e_t, \quad e_t\sim N(0,\sigma^2)$$

where $\rho_0$ is an unknown real-valued constant and $\rho_i$ for $i = 1, \ldots, p$ are the **unknown** coefficients and $\sigma^2 \geq 0$ is the **unknown** error variance. The number of lagged terms determines the order of the AR process. We estimate an AR(1) model on simulated data as follows. 

"""

# ╔═╡ 54d73a3a-5804-44cf-b771-7313790129e4
begin
	## Simulate Data for AR(1) process
	true_beta0 = 0;   # true intercept
	true_beta1 = 0.6; # true AR(1) coefficient
	true_sig2 = 1;    # true variance

	T_1 = 1000; # no. of dates
	y0 = true_beta0/(1 - true_beta1); # initial condition
	y = zeros(T_1); # storage vector
	
	y[1] = y0;
	for t = 2:T_1
	    y[t] = true_beta0 + true_beta1*y[t-1] + rand(Normal(0,sqrt(true_sig2)));
	end
	
	data_new = y;
	x_1 = collect(1:1:T_1);
	plot(x_1, data_new, label="Simulated Data")
end

# ╔═╡ f539d376-0f23-4667-9b26-25848b5c635a
begin
	# Define key variables in linear regression model
	p = 1;              # AR order
	y0_new = data_new[1:p];     # Initial conditions
	y_new = data_new[p+1:end];  # Observations
	T_new = size(y_new, 1);      # Dates after removing initial conditions
	X = [ones(T_new) data_new[1:end-1]]; # intercept and first lag
	
	x_new = collect(1:1:T_new);
	plot(x_new, y_new, label="Data")
	plot!(x_new, X[:,2], label="First lag")
end

# ╔═╡ 78f29328-9670-4f77-a0d0-4386c6aa71ae
function gibbs_AR(X, T, y)
	# Controls
	nburn = 1000;
	ndraws = nburn + 10000;

	# Prior for beta
	k = size(X,2);                    # number of regressors
	pri_m = zeros(k);                 # prior mean
	pri_v = Diagonal(10*ones(k));     # prior covariance
	pri_beta = MvNormal(pri_m,pri_v); # prior distribution

	# Prior for sig2
	pri_nu = 3;                           # prior shape parameter  
	pri_S = 1*(pri_nu-1);                 # prior scale parameter - sets E(pri_sig2) = 1
	pri_sig2 = InverseGamma(pri_S,pri_S); # prior distribution

	# Storage
	s_beta = zeros(ndraws-nburn,k);
	s_sig2 = zeros(ndraws-nburn,1);

	# Deterministic terms in posterior
	post_nu = pri_nu + T/2;   # posterior shape parameter for sig2 
	inv_priV = (pri_v)\I(k);  # inverse prior covariance for beta
	XpX = X'*X; 
	Xpy = X'*y; 

	# Initial conditions
	MC_beta = XpX\Xpy;
	MC_sig2 = (y-X*MC_beta)'*(y-X*MC_beta)/T;

	# Markov chain
	for loop in 1:ndraws
	# Draw beta
    	post_v = (XpX/MC_sig2 + inv_priV)\I(k);
    	post_m = post_v*(Xpy/MC_sig2 + inv_priV*pri_m);
    	#MC_beta = post_m + (cholesky(post_v).L)*rand(Normal(0,1),k);
    	MC_beta = post_m + (cholesky(Hermitian(post_v)).L)*rand(Normal(0,1),k);

		# Draw sig2
    	post_S = pri_S +0.5*(y-X*MC_beta)'*(y-X*MC_beta);
    	MC_sig2 = rand(InverseGamma(post_nu,post_S));

		# Store
    	if loop > nburn
        	count_loop = loop - nburn;
        	s_beta[count_loop,:] = transpose(MC_beta);
        	s_sig2[count_loop] = MC_sig2;
    	end
	end
	
	## Summarize results
	# Compute posterior mean using Monte Carlo Integration
	post_beta = mean(s_beta,dims=1);
    post_sig2 = mean(s_sig2);

	post_beta, post_sig2, s_beta, s_sig2
end;

# ╔═╡ 2502b96c-7bdc-4749-b92e-f171f60a4508
begin
	nburn = 1000;
	ndraws = nburn + 10000;
	
	post_beta, post_sig2, s_beta, s_sig2 = gibbs_AR(X, T_new, y_new)
end;

# ╔═╡ 01b84089-7a99-4bc4-93f4-5259fabde2cb
begin
	# Trace plots
	x_t = collect(1:(ndraws-nburn))
	p1a = plot(x_t,s_beta[:,1], title = "Markov chain: beta0", legend = false);
	p1b = plot(x_t,s_beta[:,2], title = "Markov chain: beta1", legend = false);
	p1c = plot(x_t,s_sig2, title = "Markov chain: sig2", label="posterior draws");
end;

# ╔═╡ 82fb8ff1-8017-4554-ba79-98045b1fccf3
begin
	# Plot marginal posterior distributions
	x_h = collect(1:(ndraws-nburn));
	histogram(x_h, s_beta[:,1], normalize=:pdf, title = "Posterior: beta0", legend = false)
	plot!([true_beta0], seriestype="vline", legend = false)
	p2a = plot!([post_beta[1]], seriestype="vline", legend = false)
	
	histogram(x_h, s_beta[:,2], normalize=:pdf, title = "Posterior: beta1", legend = false)
	plot!([true_beta1], seriestype="vline", legend = false)
	p2b = plot!([post_beta[2]], seriestype="vline", legend = false)
	
	histogram(x_h, s_sig2, normalize=:pdf, title = "Posterior: sig2", label="Empirical pdf")
	plot!([true_sig2], seriestype="vline", label="True value")
	p2c = plot!([post_sig2], seriestype="vline", label="MC mean")

	plot(p2a,p2b,p2c,layout = (1,3), legend = false)
end

# ╔═╡ 4b4dde8d-e34d-44ac-9010-8ca22feae0e2
md""" We can extend this example to a moving average model and thereby also the more general class of ARMA models. In the next section we will be looking at how to implement linear regression and ARMA models in `Turing.jl`.
"""

# ╔═╡ 442f79e7-425b-43fd-a760-099f5005d4b1
md"""

## Regression with `Turing.jl`

"""

# ╔═╡ d5e73cc5-0a28-4a84-ba7e-7d493c53114b
md"""

We will be doing a quick regression using a dataset from the `RDatasets` package. You can pick whatever you find interesting. I wanted it to be related to economics in some way, so I chose the `Ecdat` dataset, which is supposed to be relevant for econometrics  

"""

# ╔═╡ f3109dad-1b97-4f68-babd-65751142486a
ic_data = RDatasets.dataset("Ecdat", "Icecream")

# ╔═╡ 692704dd-4be7-4c51-a421-2273f2b0711b
md" First we split the dataset into two subsets. One for the training model and the other for evaluation. "

# ╔═╡ 8d35cc9d-e37a-4a42-83af-8f48ffa67252
# Split our dataset 70%/30% into training/test sets.
trainset, testset = splitobs(shuffleobs(ic_data), 0.7)

# ╔═╡ 71172cf3-be3e-416b-a35f-9ba724c528f0
md" Convert everything to matrix form, not DataFrames. "

# ╔═╡ bd6bcc2d-dd6d-4140-a62f-61a0c27bcda1
begin
	# Turing requires data in matrix form.
	target = :Cons
	train = Matrix(select(trainset, Not(target)))
	test = Matrix(select(testset, Not(target)))
	train_target = trainset[:, target]
	test_target = testset[:, target]
	
	# Standardize the features.
	μ₂, σ₂ = rescale!(train; obsdim = 1)
	rescale!(test, μ₂, σ₂; obsdim = 1)
	
	# Standardize the targets.
	μtarget, σtarget = rescale!(train_target; obsdim = 1)
	rescale!(test_target, μtarget, σtarget; obsdim = 1);
end

# ╔═╡ 0b23e994-d1c7-466e-bb40-e8b1db183885
md""" #### Model specification """

# ╔═╡ 8c803ed9-051b-45a0-8e8a-c3754b0a1f10
# Bayesian linear regression.
@model function linear_regression(x, y)
    # Set variance prior.
    σ₂ ~ truncated(Normal(0, 100), 0, Inf)
    
    # Set intercept prior.
    intercept ~ Normal(0, sqrt(3))
    
    # Set the priors on our coefficients.
    nfeatures = size(x, 2)
    coefficients ~ MvNormal(nfeatures, sqrt(10))
    
    # Calculate all the mu terms.
    mu = intercept .+ x * coefficients
    y ~ MvNormal(mu, sqrt(σ₂))
end;

# ╔═╡ 7cf8df8f-aa2d-41f9-95b7-b50fc191cb9d
model = linear_regression(train, train_target);

# ╔═╡ 7967b6e7-8391-4489-9e36-fe281dfe3fcc
chain = sample(model, NUTS(0.65), 2_000);

# ╔═╡ beaa1dfc-e2a7-4125-a45e-bfba93f8f02a
plot(chain)

# ╔═╡ 89353f76-27d3-4675-8867-d4edf0dab9a5
md"""

### ARIMA model in `Turing.jl`

"""

# ╔═╡ 34307b89-0a10-4339-9356-fd21fd877e95
md" So we have looked at a basic regression model. What if we want to do an ARIMA type model, like modelling an AR(1) process? How would we go about doing this? Let us look at some financial data in this example. Ice cream data might not have been the best bet for real econometric analysis. "

# ╔═╡ d81f725d-a250-44e8-98ba-5c0efecdb19c
df = urldownload("https://raw.githubusercontent.com/inertia7/timeSeries_sp500_R/master/data/data_master_1.csv") |> DataFrame

# ╔═╡ dfbe27b2-bfdd-40e9-a63c-3115c308de75
s = df.sp_500;

# ╔═╡ 455a97c8-db6e-42d8-82b1-9c21f2794ab7
plot(s, legend = false, lw = 1.5, alpha = 0.8)

# ╔═╡ 3f656347-0b61-4b7e-b52f-c4f34d724f5c
begin
	# Split into training and test sets. 
	train_percentage = 0.95
	s_train = s[1:floor(Int, train_percentage*length(s))]
	N₂ = length(s_train)
end

# ╔═╡ 536ecb71-f33f-402d-8d6f-16291aa9530f
md" We can test for stationarity with a Dickey Fuller test. "

# ╔═╡ ccea5b69-dfb1-47ee-bd54-7b58d87f09f5
ADFTest(s_train, Symbol("constant"), 5)

# ╔═╡ d90046ec-2bb6-4224-a26d-65550d86dc96
md" We observe stationarity and the easiest way to resolve this is by taking a first difference. "

# ╔═╡ 6c7f89c4-cbeb-42c6-aba4-cff79a68ab8f
begin
	s_diff = diff(s_train)
	plot(s_diff, legend = false, lw = 1.5, alpha = 0.8)
end

# ╔═╡ 1c1d4271-6cec-4d86-be3b-1e89417154ee
ADFTest(s_diff, Symbol("constant"), 5)

# ╔═╡ 9b5622fa-9caf-4f54-a15d-f8b731dc842b
md" Seems stationary, so let's move to our next step. We want to figure out the number of MA and AR terms by using ACF and PACF plots. You should have this type of process in the first semester.   "

# ╔═╡ 596fdda0-0d0e-4e45-8fa2-5618122a08b6
begin
	total_lags = 20
	s1 = plot(line = :stem, collect(1:total_lags), autocor(s_diff, collect(1:total_lags)), title = "ACF", ylim = [-0.3,0.5], lw = 10)
	s2 = plot(line = :stem, collect(1:total_lags), pacf(s_diff, collect(1:total_lags)), title = "PACF", ylim = [-0.3,0.5], lw = 10)
	plot(s1, s2, layout = (2, 1), legend = false)
end

# ╔═╡ e680e488-f1e2-4da8-8ef4-612f60cd030b
md" For the sake of argument, let us say that we can then model with with an AR(0, 1, 1) model."

# ╔═╡ 7eecc27c-5af0-42d4-96f6-2a689e945e7f
@model ARIMA011(x) = begin
    T = length(x)

    # Set up error vector.
    ϵ = Vector(undef, T)
    x_hat = Vector(undef, T)

    θ ~ Uniform(-5, 5)

    # Treat the first x_hat as a parameter to estimate.
    x_hat[1] ~ Normal(550, 220)
    ϵ[1] = x[1] - x_hat[1]

    for t in 2:T
        # Predicted value for x.
        x_hat[t] = x[t-1] - θ * ϵ[t-1]
        # Calculate observed error.
        ϵ[t] = x[t] - x_hat[t]
        # Observe likelihood.
        x[t] ~ Normal(x_hat[t], 1)
    end
end;

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
HypothesisTests = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
MLDataUtils = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RDatasets = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
UrlDownload = "856ac37a-3032-4c1c-9122-f86d88358c8b"

[compat]
BenchmarkTools = "~0.7.0"
CSV = "~0.8.5"
DataFrames = "~1.2.2"
Distances = "~0.10.4"
Distributions = "~0.24.18"
HTTP = "~0.9.14"
HypothesisTests = "~0.10.4"
KernelDensity = "~0.6.3"
MCMCChains = "~4.14.1"
MLDataUtils = "~0.5.4"
Plots = "~1.22.0"
PlutoUI = "~0.7.1"
RDatasets = "~0.7.5"
StatsBase = "~0.33.10"
StatsPlots = "~0.14.27"
Turing = "~0.15.1"
UrlDownload = "~1.0.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.0-rc1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "7fcd8ce8931c56ba62827c87a291ea72ee07ce31"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "2.5.0"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[deps.AdvancedHMC]]
deps = ["ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "Parameters", "ProgressMeter", "Random", "Requires", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7e85ed4917716873423f8d47da8c1275f739e0b7"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.2.27"

[[deps.AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "57bda8215ba78990ce600972b533e2f6516287e8"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.5.9"

[[deps.AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "130d6b17a3a9d420d9a6b37412cae03ffd6a64ff"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[deps.Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[deps.ArrayInterface]]
deps = ["LinearAlgebra", "Requires", "SparseArrays"]
git-tree-sha1 = "a2a1884863704e0414f6f164a1f6f4a2a62faf4e"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "2.14.17"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "e239020994123f08905052b9603b4ca14f8c5807"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.31"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "068fda9b756e41e6c75da7b771e6f89fa8a43d15"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "0.7.0"

[[deps.Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "Compat", "Distributions", "LinearAlgebra", "MappedArrays", "NNlib", "NonlinearSolve", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "e85c654943615af69d504b05fc7d5408dc40a53f"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.8.13"

[[deps.BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "JSON", "Missings", "Printf", "RecipesBase", "Statistics", "StructTypes", "Unicode"]
git-tree-sha1 = "1562002780515d2573a4fb0c3715e4e57481075e"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.0"

[[deps.ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Reexport", "Requires", "Statistics"]
git-tree-sha1 = "422db294d817de46668a3bf119175080ab093b23"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.7.70"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "4b28f88cecf5d9a07c85b9ce5209a361ecaff34a"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.9.45"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "4866e381721b30fac8dda4c8cb1d9db45c8d2994"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.37.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[deps.DataAPI]]
git-tree-sha1 = "bec2532f8adb82005476c141ec23e921fc20971b"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.8.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "3ed8fa7178a10d1cd0f1ca524f249ba6937490c0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "9f46deb4d4ee4494ffb5a40a27a2aced67bdd838"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.4"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a837fdf80f333415b69684ba8e8ae6ba76de6aaa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.24.18"

[[deps.DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "f773f784beca655b28ec1b235dbb9f5a6e5e151f"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.29"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.DynamicPPL]]
deps = ["AbstractMCMC", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "NaturalSort", "Random"]
git-tree-sha1 = "1f4b0135233d616c65ef210a63384c5500119bf1"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.10.7"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.EllipsisNotation]]
git-tree-sha1 = "18ee049accec8763be17a933737c1dd0fdf8673a"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.0.0"

[[deps.EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "4471d36a75e9168f80708155df33e1601b11c13c"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.3.1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[deps.ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "3c041d2ac0a52a12a27af2782b34900d9c3ee68c"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.1"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "ff537e5a3cba92fb48f30fec46723510450f2c0e"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.10.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "c2178cfbc0a5a552e16d097fae508f2024de61a3"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.59.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ef49a187604f865f4708c90e3f431890724e9012"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.59.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "60ed5f1643927479f845b0135bb369b031b541fa"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.14"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[deps.HypothesisTests]]
deps = ["Combinatorics", "Distributions", "LinearAlgebra", "Random", "Rmath", "Roots", "Statistics", "StatsBase"]
git-tree-sha1 = "a82a0c7e790fc16be185ce8d6d9edc7e62d5685a"
uuid = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
version = "0.10.4"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InitialValues]]
git-tree-sha1 = "26c8832afd63ac558b98a823265856670d898b6c"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.10"

[[deps.InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["AxisAlgorithms", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "1e0e51692a3a77f1eeb51bf741bdd0439ed210e7"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.2"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1a8c6237e78b714e901e406c096fc8a65528af7d"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LearnBase]]
deps = ["LinearAlgebra", "StatsBase"]
git-tree-sha1 = "47e6f4623c1db88570c7a7fa66c6528b92ba4725"
uuid = "7f8f8fb0-2700-5f03-b4bd-41f8cfc144b6"
version = "0.3.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "71be1eb5ad19cb4f61fa8c73395c0338fd092ae0"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.2"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtask]]
deps = ["BinaryProvider", "Libdl", "Pkg"]
git-tree-sha1 = "83e082fccb4e37d93df6440cdbd41dcbe5e46cb6"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.4.2"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "BenchmarkTools", "DiffResults", "DocStringExtensions", "Random", "Requires", "TransformVariables", "UnPack"]
git-tree-sha1 = "b8a3c29fdd8c512a7e80c4ec27d609e594a89860"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "0.10.6"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[deps.MCMCChains]]
deps = ["AbstractFFTs", "AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MLJModelInterface", "NaturalSort", "PrettyTables", "Random", "RecipesBase", "Serialization", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "2cc8ea543a3dbb951e8b3310a4ab7605790aa9f1"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "4.14.1"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[deps.MLDataPattern]]
deps = ["LearnBase", "MLLabelUtils", "Random", "SparseArrays", "StatsBase"]
git-tree-sha1 = "e99514e96e8b8129bb333c69e063a56ab6402b5b"
uuid = "9920b226-0b2a-5f5f-9153-9aa70a013f8b"
version = "0.5.4"

[[deps.MLDataUtils]]
deps = ["DataFrames", "DelimitedFiles", "LearnBase", "MLDataPattern", "MLLabelUtils", "Statistics", "StatsBase"]
git-tree-sha1 = "ee54803aea12b9c8ee972e78ece11ac6023715e6"
uuid = "cc2ba9b6-d476-5e6d-8eaf-a92d5412d41d"
version = "0.5.4"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "1b780b191a65dbefc42d2a225850d20b243dde88"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.3.0"

[[deps.MLLabelUtils]]
deps = ["LearnBase", "MappedArrays", "StatsBase"]
git-tree-sha1 = "3211c1fdd1efaefa692c8cf60e021fb007b76a08"
uuid = "66a33bbf-0c2b-5fc8-a008-9da813334f0a"
version = "0.5.6"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[deps.MappedArrays]]
deps = ["FixedPointNumbers"]
git-tree-sha1 = "b92bd220c95a8bbe89af28f11201fd080e0e3fe7"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.3.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "e991b6a9d38091c4a0d7cd051fcb57c05f98ac03"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "748f6e1e4de814b101911e64cc12d83a6af66782"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.2"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "5203a4532ad28c44f82c76634ad621d7c90abcbd"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.29"

[[deps.NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[deps.NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "35585534c0c79c161241f2e65e759a11a79d25d0"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.10"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c870a0d713b51e4b49be6432eff0e26a4325afee"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.6"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "2537ed3c0ed5e03896927187f5f2ee6a4ab342db"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.14"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "b1a708d607125196ea1acf7264ee1118ce66931b"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.0"

[[deps.PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "Logging", "Markdown", "Random", "Suppressor"]
git-tree-sha1 = "45ce174d36d3931cd4e37a47f93e07d1455f038d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.1"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[deps.RData]]
deps = ["CategoricalArrays", "CodecZlib", "DataFrames", "Dates", "FileIO", "Requires", "TimeZones", "Unicode"]
git-tree-sha1 = "19e47a495dfb7240eb44dc6971d660f7e4244a72"
uuid = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
version = "0.8.3"

[[deps.RDatasets]]
deps = ["CSV", "CodecZlib", "DataFrames", "FileIO", "Printf", "RData", "Reexport"]
git-tree-sha1 = "06d4da8e540edb0314e88235b2e8f0429404fdb7"
uuid = "ce6b1742-4840-55fa-b093-852dadbb1d8b"
version = "0.7.5"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "7dff99fbc740e2f8228c6878e2aad6d7c2678098"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.1"

[[deps.RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "7ad0dfa8d03b7bcf8c597f59f5292801730c55b8"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.4.1"

[[deps.RecursiveArrayTools]]
deps = ["ArrayInterface", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "b3f4e34548b3d3d00e5571fd7bc0a33980f01571"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.11.4"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6761a5d1f9646affb2a369ff932841fff77934a3"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.1.0"

[[deps.Reexport]]
deps = ["Pkg"]
git-tree-sha1 = "7b1d07f411bc8ddb7977ec7f377b97b158514fe0"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "0.2.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.Roots]]
deps = ["CommonSolve", "Printf", "Setfield"]
git-tree-sha1 = "e8102d671f2343d2e9f12af0ecbd85fd5f86c4fa"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "1.3.4"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "0b5b04995a3208f82ae69a17389c60b762000793"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.13.6"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "9c1a0dea3b442024c54ca6a318e8acf842eab06f"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.2.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["OpenSpecFun_jll"]
git-tree-sha1 = "d8d8b8a9f4119829410ecd706da4cc8594a1e020"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "0.10.3"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "edef25a158db82f4940720ebada14a60ef6c4232"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.13"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "730732cae4d3135e2f2182bd47f8d8b795ea4439"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.1.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8cbbc098554648c84f79a463c9ff0fd277144b6c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.10"

[[deps.StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[deps.StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "233bc83194181b07b052b4ee24515564b893faf6"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.27"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "8445bf99a36d703a09c601f9a57e2f83000ef2ae"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.3"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "019acfd5a4a6c5f0f38de69f2ff7ed527f1881da"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.1.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "1162ce4a6c4b7e31e0e6b14486a6986951c73be9"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.2"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "d620a061cb2a56930b52bdf5cf908a5c4fa8e76a"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.4"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimeZones]]
deps = ["Dates", "Future", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "6c9040665b2da00d30143261aea22c7427aada1c"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.7"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "34f27ac221cb53317ab6df196f9ed145077231ff"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.65"

[[deps.TransformVariables]]
deps = ["ArgCheck", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "Pkg", "Random", "UnPack"]
git-tree-sha1 = "9433efc8545a53a9a34de0cdb9316f9982a9f290"
uuid = "84d833dd-6860-57f9-a1a7-6da5db126cff"
version = "0.4.1"

[[deps.TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[deps.Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedVI", "BangBang", "Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "LogDensityProblems", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "2b57320e0f4c59e264dd34d5f0fb3bcc67ac65ae"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.15.1"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UrlDownload]]
deps = ["HTTP", "ProgressMeter"]
git-tree-sha1 = "05f86730c7a53c9da603bd506a4fc9ad0851171c"
uuid = "856ac37a-3032-4c1c-9122-f86d88358c8b"
version = "1.0.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "eae2fbbc34a79ffd57fb4c972b08ce50b8f6a00d"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─09a9d9f9-fa1a-4192-95cc-81314582488b
# ╟─41eb90d1-9262-42b1-9eb2-d7aa6583da17
# ╟─aa69729a-0b08-4299-a14c-c9eb2eb65d5c
# ╟─a681c4c5-1451-44d6-be27-f9003340883f
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# ╠═bbcafd74-97f4-4b8f-bffa-937812d9a2eb
# ╟─d65de56f-a210-4428-9fac-20a7888d3627
# ╟─38d2167d-2ca2-4398-9406-db418cc4bf17
# ╟─3f0c1adc-3efd-4ade-9a65-a208d8457b85
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─f3823457-8757-4665-86a8-bf536d80e24d
# ╟─f95ccee4-a2d3-4492-b869-551e61acf995
# ╟─6c027ac7-ef68-4582-ba17-8f34a879a21d
# ╟─66c936a3-055d-42a0-bcad-5176d10e5994
# ╟─3aad749e-1256-4f93-b119-4717d2b95607
# ╟─83ae194c-e04f-4615-9b71-7c389513898c
# ╟─727a703f-1233-4098-86f3-3192e4de08d4
# ╟─14f11fc0-dc9b-4b77-b2da-ae534b911cd6
# ╟─d92d80ef-256c-443a-a81c-8d5f02e01e66
# ╟─48c98f0d-d880-4abc-91d3-6c79be5fcf8a
# ╟─d811cfd9-3bdd-4830-8e9d-ecd4d7d2c890
# ╟─d3ceb4ea-6d45-4545-be09-8446f103c2e5
# ╟─5cb5ab74-da6b-439a-bea4-75a3d0e43c63
# ╟─bade1741-5e84-413a-9c67-932bd6748c49
# ╠═b94db7f0-9f38-4761-a3c3-4d6fc4729ae9
# ╠═1a6c859c-e3e7-4ad9-9299-091b6b1d2bbf
# ╠═0980d7a1-129b-4724-90fb-b46e3088d2d6
# ╠═0919cb0d-ba03-49c8-b2b9-53a467c39f87
# ╠═343202b3-23b5-4600-b912-7db4ab58deaf
# ╠═3aeab073-c98a-4213-a835-3098233ba90c
# ╟─afc56ef2-d5d2-4624-9ad9-11f15429607f
# ╟─41365136-5d0e-4a4e-808f-d0e90a14c5dd
# ╟─3335094d-a67b-471c-834d-e22089933104
# ╟─330bd7cb-7102-41c2-b7a4-f053669960c3
# ╟─80e6619b-ac42-453b-8f38-850b2b99d000
# ╟─82b96729-33c2-49b0-b908-562faf903a1e
# ╟─1f2c9795-0b2c-4a14-9f28-1fef68f6b467
# ╟─70193cca-ce19-49ee-aa0c-06997affe2a6
# ╟─1703eb19-aeca-4ebe-a9b3-18dfbf4efdfe
# ╟─9ffed7a8-f8d3-4c1a-affa-5d6646683829
# ╟─993c9137-d708-4feb-8a85-51d85c38fc8d
# ╟─dcf7adfb-0d6e-4459-a3cc-6a20cc632c72
# ╟─a541e83e-22c5-48c7-9beb-705d26f602ee
# ╟─5b442d42-b6ff-4677-afff-a00cfdb00fbc
# ╟─8a8dea4a-3f49-4095-a1a0-d4d3c803c171
# ╟─be364d9e-c04d-47fd-ac2f-2be93bdd3f87
# ╟─4a985421-7513-407a-8edf-01df7b405c55
# ╟─5b3bd2d1-2697-4516-aa58-99b3d229e5f3
# ╟─8cf7fa3c-aaf2-4684-869e-d099218fe6e9
# ╟─0999d280-3d93-4a0d-a487-215a3f068fee
# ╟─f43f8154-5bd3-4a33-8316-991417549d32
# ╠═6ec571a0-6407-439a-b7f3-fb15085d73bf
# ╟─3752db0a-3c4f-48e9-89ec-c14927584a1d
# ╠═6e288af8-8f7d-4d4a-9fc3-4bb7df212e8e
# ╟─fd643db4-ad7f-4203-b7ed-96c730343013
# ╟─c5c4da02-8b6b-4318-b625-ce4f31703c79
# ╟─428eb291-d516-4b56-91e7-df3a92cd3a4f
# ╟─223f4b6f-8321-4142-9dfa-1afe371d40ac
# ╟─edaf930a-4933-4047-8854-bbb02ea9c39c
# ╠═c9c8d57b-3010-4056-aaa3-a0354cf456fd
# ╠═bb10296b-1b93-46d9-bd5f-c11a4ed8bcae
# ╠═bfd8fab5-64fb-4bee-aee6-e9691cc887f8
# ╠═1c38646c-b750-4402-bc70-571670cd7acd
# ╠═000cf50f-d3f8-452b-b661-11545c2ec0c4
# ╠═7b70270e-2991-40ee-97a9-082691e68701
# ╠═791939ae-3e00-4d6b-968c-5a82a78f55f8
# ╠═9d1361c3-6de1-4326-85f8-4a272856d16b
# ╠═01f5e6cd-1efa-49a2-8d0d-b5536bdd7388
# ╟─3006d4f8-1cca-4947-8dc6-3b77b278fbb8
# ╟─22bdce87-a1f2-440b-965a-391148b010ac
# ╟─54d73a3a-5804-44cf-b771-7313790129e4
# ╠═f539d376-0f23-4667-9b26-25848b5c635a
# ╠═78f29328-9670-4f77-a0d0-4386c6aa71ae
# ╠═2502b96c-7bdc-4749-b92e-f171f60a4508
# ╟─01b84089-7a99-4bc4-93f4-5259fabde2cb
# ╟─82fb8ff1-8017-4554-ba79-98045b1fccf3
# ╟─4b4dde8d-e34d-44ac-9010-8ca22feae0e2
# ╟─442f79e7-425b-43fd-a760-099f5005d4b1
# ╟─d5e73cc5-0a28-4a84-ba7e-7d493c53114b
# ╠═f3109dad-1b97-4f68-babd-65751142486a
# ╟─692704dd-4be7-4c51-a421-2273f2b0711b
# ╠═8d35cc9d-e37a-4a42-83af-8f48ffa67252
# ╟─71172cf3-be3e-416b-a35f-9ba724c528f0
# ╠═bd6bcc2d-dd6d-4140-a62f-61a0c27bcda1
# ╟─0b23e994-d1c7-466e-bb40-e8b1db183885
# ╠═8c803ed9-051b-45a0-8e8a-c3754b0a1f10
# ╠═7cf8df8f-aa2d-41f9-95b7-b50fc191cb9d
# ╠═7967b6e7-8391-4489-9e36-fe281dfe3fcc
# ╠═beaa1dfc-e2a7-4125-a45e-bfba93f8f02a
# ╟─89353f76-27d3-4675-8867-d4edf0dab9a5
# ╟─34307b89-0a10-4339-9356-fd21fd877e95
# ╠═d81f725d-a250-44e8-98ba-5c0efecdb19c
# ╠═dfbe27b2-bfdd-40e9-a63c-3115c308de75
# ╠═455a97c8-db6e-42d8-82b1-9c21f2794ab7
# ╠═3f656347-0b61-4b7e-b52f-c4f34d724f5c
# ╟─536ecb71-f33f-402d-8d6f-16291aa9530f
# ╠═ccea5b69-dfb1-47ee-bd54-7b58d87f09f5
# ╟─d90046ec-2bb6-4224-a26d-65550d86dc96
# ╠═6c7f89c4-cbeb-42c6-aba4-cff79a68ab8f
# ╠═1c1d4271-6cec-4d86-be3b-1e89417154ee
# ╟─9b5622fa-9caf-4f54-a15d-f8b731dc842b
# ╟─596fdda0-0d0e-4e45-8fa2-5618122a08b6
# ╟─e680e488-f1e2-4da8-8ef4-612f60cd030b
# ╠═7eecc27c-5af0-42d4-96f6-2a689e945e7f
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
