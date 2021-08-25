### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
using AbstractGPsMakie, AlgebraOfGraphics, BenchmarkTools, CairoMakie, CSV, DataFrames, Distributions, KernelDensity, LinearAlgebra, Plots, PlutoUI, Random, SparseArrays, StatsBase, Statistics, UrlDownload

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
">ATS 872: Lecture 6</p>
<p style="text-align: center; font-size: 1.8rem;">
 Bayesian vector autoregression (BVAR)
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
    max-width: 800px;
  }
</style>
"""

# ╔═╡ aa69729a-0b08-4299-a14c-c9eb2eb65d5c
md" # Introduction "

# ╔═╡ 000021af-87ce-4d6d-a315-153cecce5091
md" In this session we will move toward understanding Bayesian vector autoregression models."

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
TableOfContents() # Uncomment to see TOC

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐ if you want."

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Vector Autoregressions "

# ╔═╡ 41221332-0a22-43c7-b1b2-8a8d84e61dd4
md"""

Vector autoregressions (VARs) have been widely used for macroeconomic forecasting and structural analysis since the seminal work of Sims $(1980)$. In particular, VARs are often served as the benchmark for comparing forecast performance of new models and methods. VARs are also used to better understand the interactions between macroeconomic variables, often through the estimation of impulse response functions that characterize the effects of a variety of structural shocks on key economic variables.

Despite the empirical success of the standard constant-coefficient and homoscedastic VAR, there is a lot of recent work in extending these conventional VARs to models with time-varying regression coefficients and stochastic volatility. These extensions are motivated by the widely observed structural instabilities and time-varying volatility in a variety of macroeconomic time series.

In this chapter we will study a few of these more flexible VARs, including the timevarying parameter (TVP) VAR and VARs with stochastic volatility. An excellent review paper that covers many of the same topics is Koop and Korobilis (2010). We will begin with a basic VAR.

"""

# ╔═╡ 77fde8ac-f019-4b4c-9d03-9a9c0d7ca2a0
md"""

### Basic VAR
"""

# ╔═╡ 7e0462d4-a307-465b-ae27-737431ecb565
md"""

Suppose $\mathbf{y}_{t}=\left(y_{1 t}, \ldots, y_{n t}\right)^{\prime}$ is a vector of dependent variables at time $t$. Consider the following $\operatorname{VAR}(p)$ :

$$\mathbf{y}_{t}=\mathbf{b}+\mathbf{A}_{1} \mathbf{y}_{t-1}+\cdots+\mathbf{A}_{p} \mathbf{y}_{t-p}+\varepsilon_{t}$$

where $\mathbf{b}$ is an $n \times 1$ vector of intercepts, $\mathbf{A}_{1}, \ldots, \mathbf{A}_{p}$ are $n \times n$ coefficient matrices and $\varepsilon_{t} \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$. In other words, $\operatorname{VAR}(p)$ is simply a multiple-equation regression where the regressors are the lagged dependent variables.

To fix ideas, consider a simple example with $n=2$ variables and $p=1$ lag. Then the equation above can be written explicitly as:

$$\left(\begin{array}{l}
y_{1 t} \\
y_{2 t}
\end{array}\right)=\left(\begin{array}{l}
b_{1} \\
b_{2}
\end{array}\right)+\left(\begin{array}{ll}
A_{1,11} & A_{1,12} \\
A_{1,21} & A_{1,22}
\end{array}\right)\left(\begin{array}{l}
y_{1(t-1)} \\
y_{2(t-1)}
\end{array}\right)+\left(\begin{array}{c}
\varepsilon_{1 t} \\
\varepsilon_{2 t}
\end{array}\right)$$

where

$$\left(\begin{array}{l}
\varepsilon_{1 t} \\
\varepsilon_{2 t}
\end{array}\right) \sim \mathcal{N}\left(\left(\begin{array}{l}
0 \\
0
\end{array}\right),\left(\begin{array}{ll}
\sigma_{11} & \sigma_{12} \\
\sigma_{21} & \sigma_{22}
\end{array}\right)\right)$$

The model runs from $t=1, \ldots, T$, and it depends on the $p$ initial conditions $\mathbf{y}_{-p+1}, \ldots, \mathbf{y}_{0}$. In principle these initial conditions can be modeled explicitly. Here all the analysis is done conditioned on these initial conditions. If the series is sufficiently long (e.g., $T>50)$, both approaches typically give essentially the same results.

"""


# ╔═╡ a34d8b43-152c-42ff-ae2c-1d439c538c8a
md""" ### Likelihood """

# ╔═╡ 01213f94-8dee-4475-b307-e8b18806d453
md"""

To derive the likelihood for the $\operatorname{VAR}(p)$, we aim to write the system as the linear regression model

$$\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\varepsilon$$

Then, we can simply apply the linear regression results to derive the likelihood.

Let's first work out our example with $n=2$ variables and $p=1$ lag. To that end, we stack the coefficients equation by equation, i.e., $\boldsymbol{\beta}=\left(b_{1}, A_{1,11}, A_{1,12}, b_{2}, A_{1,21}, A_{1,22}\right)^{\prime}$. Equivalent, we can write it using the vec operator that vectorizes a matrix by its columns: $\boldsymbol{\beta}=\operatorname{vec}\left(\left[\mathbf{b}, \mathbf{A}_{1}\right]^{\prime}\right)$. Given our definition of $\boldsymbol{\beta}$, we can easily work out the corresponding regression matrix $\mathbf{X}_{t}$ :

$$\left(\begin{array}{l}
y_{1 t} \\
y_{2 t}
\end{array}\right)=\left(\begin{array}{cccccc}
1 & y_{1(t-1)} & y_{2(t-1)} & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & y_{1(t-1)} & y_{2(t-1)}
\end{array}\right)\left(\begin{array}{c}
b_{1} \\
A_{1,11} \\
A_{1,12} \\
b_{2} \\
A_{1,21} \\
A_{1,22}
\end{array}\right)+\left(\begin{array}{c}
\varepsilon_{1 t} \\
\varepsilon_{2 t}
\end{array}\right)$$

Or

$$\mathbf{y}_{t}=\left(\mathbf{I}_{2} \otimes\left[1, \mathbf{y}_{t-1}^{\prime}\right]\right) \boldsymbol{\beta}+\varepsilon_{t}$$

where $\otimes$ is the Kronecker product. More generally, we can write the $\operatorname{VAR}(p)$ as:

$$\mathbf{y}_{t}=\mathbf{X}_{t} \boldsymbol{\beta}+\varepsilon_{t}$$

where $\mathbf{X}_{t}=\mathbf{I}_{n} \otimes\left[1, \mathbf{y}_{t-1}^{\prime}, \ldots, \mathbf{y}_{t-p}^{\prime}\right]$ and $\boldsymbol{\beta}=\operatorname{vec}\left(\left[\mathbf{b}, \mathbf{A}_{1}, \cdots \mathbf{A}_{p}\right]^{\prime}\right) .$ Then, stack the observations over $t=1, \ldots, T$ to get

$$\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\varepsilon$$

where $\varepsilon \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)$.



"""

# ╔═╡ 868019c7-eeed-449f-9343-a541f6daefe5
md"""

Before we continue, let us show what Kronecker products and `vec` operators actually do in relation to predefined matrices. This might help make the mathematical operations easier to internalise. 

"""

# ╔═╡ 17c1a647-7ebc-4703-bed8-148d3a35ac1d
md"""

### Kronecker product and `vec` operator 

"""

# ╔═╡ 538be57f-54e9-45dd-95f5-12e7a3df51a7
md""" ###  Likelihood contd. """

# ╔═╡ 057a49fa-a68a-4bf0-8b70-7ccd5b8d7931
md"""

Let us now continue with our calculation. Since, 

$$(\mathbf{y} \mid \boldsymbol{\beta}, \mathbf{\Sigma}) \sim \mathcal{N}\left(\mathbf{X} \boldsymbol{\beta}, \mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)$$

the likelihood function is given by:

$$\begin{aligned}
p(\mathbf{y} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma}) &=\left|2 \pi\left(\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)\right|^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}\right)^{-1}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})} \\
&=(2 \pi)^{-\frac{T n}{2}}|\mathbf{\Sigma}|^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2}(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})^{\prime}\left(\mathbf{I}_{T} \otimes \Sigma^{-1}\right)(\mathbf{y}-\mathbf{X} \boldsymbol{\beta})}
\end{aligned}$$

where the second equality holds because $\left|\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right|=|\boldsymbol{\Sigma}|^{T}$ and $\left(\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}\right)^{-1}=\mathbf{I}_{T} \otimes \boldsymbol{\Sigma}^{-1}$. Note that the likelihood can also be written as

$$p(\mathbf{y} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma})=(2 \pi)^{-\frac{T n}{2}}|\mathbf{\Sigma}|^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2} \sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)}$$

"""

# ╔═╡ 3efef390-b791-40dd-a950-26dcc84c3485
md""" ### Independent priors """

# ╔═╡ 349c3025-3b88-4e7e-9e31-574943f31dc6
md"""

Recall that in the normal linear regression model, we assume independent normal and inverse-gamma priors for the coefficients $\boldsymbol{\beta}$ and the variance $\sigma^{2}$, respectively. Both are conjugate priors and the model can be easily estimated using the Gibbs sampler.

Here a similar result applies. But instead of an inverse-gamma prior, we need a multivariate generalization for the covariance matrix $\boldsymbol{\Sigma}$.

An $m \times m$ random matrix $\mathbf{Z}$ is said to have an inverse-Wishart distribution with shape parameter $\alpha>0$ and scale matrix $\mathbf{W}$ if its density function is given by

$$f(\mathbf{Z} ; \alpha, \mathbf{W})=\frac{|\mathbf{W}|^{\alpha / 2}}{2^{m \alpha / 2} \Gamma_{m}(\alpha / 2)}|\mathbf{Z}|^{-\frac{\alpha+m+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left(\mathbf{W} \mathbf{Z}^{-1}\right)}$$

where $\Gamma_{m}$ is the multivariate gamma function and $\operatorname{tr}(\cdot)$ is the trace function. We write $\mathbf{Z} \sim \mathcal{I} W(\alpha, \mathbf{W})$. For $\alpha>m+1, \mathbb{E} \mathbf{Z}=\mathbf{W} /(\alpha-m-1)$.

For the $\operatorname{VAR}(p)$ with parameters $\boldsymbol{\beta}$ and $\boldsymbol{\Sigma}$, we consider the independent priors:

$$\boldsymbol{\beta} \sim \mathcal{N}\left(\boldsymbol{\beta}_{0}, \mathbf{V}_{\boldsymbol{\beta}}\right), \quad \boldsymbol{\Sigma} \sim \mathcal{I} W\left(\nu_{0}, \mathbf{S}_{0}\right)$$

"""

# ╔═╡ 5e825c74-431e-4055-a864-d2b366e8ae11
md""" ### Gibbs sampler """

# ╔═╡ 6d451af1-2288-4ee1-a37c-432c14888e16
md"""

Now, we derive a Gibbs sampler for the $\operatorname{VAR}(p)$ with likelihood and priors given in the previous section. Specifically, we derive the two conditional densities $p(\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{\Sigma})$ and $p(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta})$.

The first step is easy, as standard linear regression results would apply. In fact, we have

$$(\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{\Sigma}) \sim \mathcal{N}\left(\widehat{\boldsymbol{\beta}}, \mathbf{K}_{\boldsymbol{\beta}}^{-1}\right)$$

where

$$\mathbf{K}_{\boldsymbol{\beta}}=\mathbf{V}_{\boldsymbol{\beta}}^{-1}+\mathbf{X}^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}\right) \mathbf{X}, \quad \widehat{\boldsymbol{\beta}}=\mathbf{K}_{\boldsymbol{\beta}}^{-1}\left(\mathbf{V}_{\boldsymbol{\beta}}^{-1} \boldsymbol{\beta}_{0}+\mathbf{X}^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}\right) \mathbf{y}\right)$$

and we have used the result $\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}\right)^{-1}=\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}$.

Next, we derive the conditional density $p(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta})$. Recall that for conformable matrices $\mathbf{A}, \mathbf{B}, \mathbf{C}$, we have

$$\operatorname{tr}(\mathbf{A B C})=\operatorname{tr}(\mathbf{B C A})=\operatorname{tr}(\mathbf{C A B})$$

Now, combining the likelihood and the prior, we obtain

$$\begin{aligned}
p(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta}) & \propto p(\mathbf{y} \mid \boldsymbol{\beta}, \boldsymbol{\Sigma}) p(\boldsymbol{\Sigma}) \\
& \propto|\boldsymbol{\Sigma}|^{-\frac{T}{2}} \mathrm{e}^{-\frac{1}{2} \sum_{t=1}^{T}\left(\mathbf{y} t-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime} \boldsymbol{\Sigma}^{-1}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)} \times|\boldsymbol{\Sigma}|^{-\frac{\nu_{0}+n+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left(\mathbf{S}_{0} \Sigma^{-1}\right)} \\
& \propto|\boldsymbol{\Sigma}|^{-\frac{\nu_{0}+n+T+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left(\mathbf{S}_{0} \Sigma^{-1}\right)} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left[\sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime} \mathbf{\Sigma}^{-1}\right]} \\
&\left.\propto|\boldsymbol{\Sigma}|^{-\frac{\nu_{0}+n+T+1}{2}} \mathrm{e}^{-\frac{1}{2} \operatorname{tr}\left[\left(\mathbf{S}_{0}+\sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime}\right) \Sigma^{-1}\right.}\right]
\end{aligned}$$

which is the kernel of an inverse-Wishart density. In fact, we have

$$(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta}) \sim \mathcal{I} W\left(\nu_{0}+T, \mathbf{S}_{0}+\sum_{t=1}^{T}\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)\left(\mathbf{y}_{t}-\mathbf{X}_{t} \boldsymbol{\beta}\right)^{\prime}\right)$$

"""

# ╔═╡ 2b30e958-6134-4c8c-8dbf-54cf5babef88
md""" ### Gibbs Sampler for the $\operatorname{VAR}(p)$ """

# ╔═╡ 1e22e482-c467-4a57-bf19-96361e2896e6
md"""

We summarize the Gibbs sampler as follows

Pick some initial values $\boldsymbol{\beta}^{(0)}=\mathbf{c}_{0}$ and $\boldsymbol{\Sigma}^{(0)}=\mathbf{C}_{0}>0 .$ Then, repeat the following steps from $r=1$ to $R$

1. Draw $\boldsymbol{\beta}^{(r)} \sim p\left(\boldsymbol{\beta} \mid \mathbf{y}, \boldsymbol{\Sigma}^{(r-1)}\right)$ (multivariate normal).

2. Draw $\boldsymbol{\Sigma}^{(r)} \sim p\left(\boldsymbol{\Sigma} \mid \mathbf{y}, \boldsymbol{\beta}^{(r)}\right)$ (inverse-Wishart).

"""

# ╔═╡ 193f5509-0733-4409-9b88-1d2bc68e3aee
md""" ### Empirical example: Small model of SA economy """

# ╔═╡ c8e7e82f-415e-4b9d-bd62-47c1c60d0741
md"""

In this empirical example we estimate a 3-variable VAR(2) using SA quarterly data on CPI inflation rate, unemployment rate and repo rate from $1994 \mathrm{Q} 1$ to $2020 \mathrm{Q} 4$. These three variables are commonly used in forecasting (e.g., Banbura, Giannone and Reichlin, 2010; Koop and Korobilis, 2010; Koop, 2013) and small DSGE models (e.g., An and Schorfheide, 2007 ).

Following Primiceri (2005), we order the interest rate last and treat it as the monetary policy instrument. The identified monetary policy shocks are interpreted as "non-systematic policy actions" that capture both policy mistakes and interest rate movements that are responses to variables other than inflation and unemployment.

We first implement the Gibbs sampler. Then, given the posterior draws of $\boldsymbol{\beta}$ and $\boldsymbol{\Sigma}$, we compute the impulse-response functions of the three variables to a 100-basis-point monetary policy shock.

"""

# ╔═╡ 7557b795-839c-41bf-9586-7b2b3972b28d
begin
	# Parameters
	p = 2            # Number of lags
	nsim = 2000      # Number of simulation in Gibbs sampler
	burnin = 10      # Burnin for Gibbs sampler
	n_hz = 40        # Horizon for the IRF
end;

# ╔═╡ 8b97f4dd-31cf-4a55-a72b-66775d79445c
md""" After specifying the parameters for the model, we load the data that we are going to be working with."""

# ╔═╡ 5592c6a8-defc-4b6e-a303-813bdbacaffe
begin
	# Load the dataset and transform to array / matrix
	df = urldownload("https://raw.githubusercontent.com/DawievLill/ATS-872/main/data/sa-data.csv") |> DataFrame
end

# ╔═╡ 6c697f4b-c928-43c2-9490-27b0f195857c
data = Matrix(df[:, 1:end]);

# ╔═╡ 8d0a4b08-5cbf-43a8-91f3-5f448893a4c6
md""" In order to estimate the model we use the following function to construct to regression matrix $\boldsymbol{X}$. Recall that $\mathbf{X}_{t}=\mathbf{I}_{n} \otimes\left[1, \mathbf{y}_{t-1}^{\prime}, \ldots, \mathbf{y}_{t-p}^{\prime}\right]$ and we stack $\mathbf{X}_{t}$ over $t=1, \ldots, T$ to obtain $\mathbf{X}$."""

# ╔═╡ 9e3795cf-8695-46c1-9857-728d765caa02
# SUR representation of the VAR(p)
function SUR_form(X, n)

    repX = kron(X, ones(n, 1))
    r, c = size(X)
    idi  = kron((1:r * n), ones(c, 1))
    idj  = repeat((1:n * c), r, 1)

    # Some prep for the out return value.
    d    = reshape(repX', n * r * c, 1)
    out  = sparse(idi[:, 1], idj[:, 1], d[:, 1])
end;

# ╔═╡ 171c9c8f-a4a9-4aad-8bab-f8f38d36a94b
md""" Given the posterior draws of $\boldsymbol{\beta}$ and $\boldsymbol{\Sigma}$, we then use the function `construct_IR` to compute the impulse-response functions of the three variables to a 100 -basis-point monetary policy shock. More specifically, we consider two alternative paths: in one a 100 -basis-point monetary policy shock hits the system, and in the other this shock is absent. We then let the two systems evolve according to the $\operatorname{VAR}(p)$ written as the regression

$$\mathbf{y}_{t}=\mathbf{X}_{t} \boldsymbol{\beta}_{t}+\mathbf{C} \widetilde{\varepsilon}_{t}, \quad \widetilde{\varepsilon}_{t} \sim \mathcal{N}\left(\mathbf{0}, \mathbf{I}_{3}\right)$$

for $t=1, \ldots, n_{\mathrm{hz}}$, where $n_{\mathrm{hz}}$ denotes the number of horizons and $\mathbf{C}$ is the Cholesky factor of $\boldsymbol{\Sigma}$. Each impulse-response function is then the difference between these two paths. """

# ╔═╡ 741e914f-4d6d-4249-a324-4dd54fd0f277
function construct_IR(β, Σ, shock)

    n      = size(Σ, 1)
    CΣ     = cholesky(Σ).L
    tmpZ1  = zeros(p, n)
    tmpZ   = zeros(p, n)
    Yt1    = CΣ * shock
    Yt     = zeros(n, 1)
    yIR    = zeros(n_hz,n)
    yIR[1,:] = Yt1'

    for t = 2:n_hz
        # update the regressors
        tmpZ = [Yt'; tmpZ[1:end-1,:]]
        tmpZ1 = [Yt1'; tmpZ1[1:end-1,:]]

        # evolution of variables if a shock hits
        Random.seed!(12)
        e = CΣ*randn(n,1)
        Z1 = reshape(tmpZ1',1,n*p)
        Xt1 = kron(I(n), [1 Z1])
        Yt1 = Xt1 * β + e

        # evolution of variables if no shocks hit
        Z = reshape(tmpZ',1,n*p)
        Xt = kron(I(n), [1 Z])
        Yt = Xt * β + e

        # the IR is the difference of the two scenarios
        yIR[t,:] = (Yt1 - Yt)'
    end
    return yIR
end;

# ╔═╡ 3bb39875-2f99-4a1f-a7ab-a107b4b95716
md"""

The main function `bvar` is given next. It first loads the dataset, constructs the regression matrix $\mathbf{X}$ using the above function, and then implements the 2 -block Gibbs sampler. Note that within the for-loop we compute the impulse-response functions for each set of posterior draws $(\boldsymbol{\beta}, \boldsymbol{\Sigma})$. Also notice that the shock variable shock is normalized so that the impulse responses are to 100 basis points rather than one standard deviation of the monetary policy shock.

"""

# ╔═╡ 9e949115-a728-48ed-8c06-5cc26b6733bf
function bvar(data)

	data   = data[1:end, :]
    Y0     = data[1:4, :]
    Y      = data[5:end, :]
    T      = size(Y, 1)
    n      = size(Y, 2)
    y      = reshape(Y', T * n, 1)
    k      = n * p + 1 # number of coefficients in each equation (21 in total, 3 equations)

    # Specification of the prior (diffuse prior in this case)
    ν_0    = n + 3
    Σ_0    = I(n)
    β_0    = zeros(n * k, 1) # Best way to initialise with zeros?
    #β_0    = fill( NaN, n * k, 1) # Alternative?k

    # Precision for coefficients and intercepts
    tmp    = ones(k * n, 1)
    tmp[1: p * n + 1 : k * n] .= 1/10
    A      = collect(1:k*n)
    Vβ     = sparse(A, A, tmp[:, 1]) # This works! I seem to have figured out the sparse array struct. However, is this what we want to use? What is the benefit here?
    Vβ_d     = Matrix(Vβ) # Dense version

    # Working on the lagged matrix (preparation of the data) / method is similar to Korobilis
    tmpY   = [Y0[(end-p+1): end,:]; Y]
    X_til  = zeros(T, n * p)

    for i=1:p
        X_til[:, ((i - 1) * n + 1):i * n] = tmpY[(p - i + 1):end - i,:]
    end
    X_til  = [ones(T, 1) X_til] # This is the dense X matrix
    X      = SUR_form(X_til, n) # Creates a sparse regression array...

    # Initialise these arrays (used for storage)
    store_Sig  = zeros(nsim, n, n) # For the sigma values
    store_beta = zeros(nsim, k*n) # For the beta values
    store_yIR  = zeros(n_hz, n) # For the impulse responses

    X_d   = Matrix(X)
    # Initialise chain
    β     = (X_d'*X_d) \ Matrix{Float64}(X_d'*y)
    e     = reshape(y - X_d*β, n, T)
    Σ     = e*e'/T

    iΣ    = Σ\I(n)
    iΣ    = Symmetric(iΣ)

    for isim = 1:nsim + burnin

        # sample beta
        XiΣ = X_d'*kron(I(T), iΣ)
        XiΣX = XiΣ*X_d
        Kβ = Vβ_d + XiΣX
        β_hat = Kβ\(Vβ_d*β_0 + XiΣ*y)

        Kβ = Symmetric(Kβ)
        Random.seed!(12)
        β = β_hat + Kβ'\randn(n*k, 1)
        # β = β_hat + CKβL'\randn(n*k, 1) # sparse version

        # sample Sig
        e = reshape(y - X_d*β, n, T)
        Σ = rand(InverseWishart(ν_0 + T, Σ_0 + e*e'))


        if isim > burnin
            # store the parameters
            isave = isim - burnin;
            store_beta[isave,:] = β';
            store_Sig[isave,:,:] = Σ;

            # compute impulse-responses
            CΣ = cholesky(Σ).L

            # 100 basis pts rather than 1 std. dev.
            shock_d = [0; 0; 1]/Σ[n,n]
            shock = [0; 0; 1]/CΣ[n,n]
            yIR = construct_IR(β, Σ, shock)
            store_yIR = store_yIR + yIR
        end
    end
    yIR_hat = store_yIR/nsim
    #return store_beta, store_Sig
end;

# ╔═╡ c21aa46d-08ac-4256-a021-83194bad3a5e
md""" Code seems to be running, good sign. Now I just need to fix the data and plot the IRFs for this session. """

# ╔═╡ abe69a74-74ff-4cc5-9a93-90bd36c48e8a
begin
	x = 0:39
	plot(x, bvar(data)[:, 2]) # Figure out how to insert a line here
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractGPsMakie = "7834405d-1089-4985-bd30-732a30b92057"
AlgebraOfGraphics = "cbdf2221-f076-402e-a563-3d30da359d67"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
UrlDownload = "856ac37a-3032-4c1c-9122-f86d88358c8b"

[compat]
AbstractGPsMakie = "~0.2.1"
AlgebraOfGraphics = "~0.5.2"
BenchmarkTools = "~1.1.4"
CSV = "~0.8.5"
CairoMakie = "~0.6.2"
DataFrames = "~1.2.2"
Distributions = "~0.25.11"
KernelDensity = "~0.6.3"
Plots = "~1.15.2"
PlutoUI = "~0.7.9"
StatsBase = "~0.33.9"
UrlDownload = "~1.0.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractGPs]]
deps = ["ChainRulesCore", "Distributions", "FillArrays", "KernelFunctions", "LinearAlgebra", "Random", "RecipesBase", "Reexport", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "d1b311e6f0358fa73c82473daf23de6ba1148900"
uuid = "99985d1d-32ba-4be9-9821-2ec096f28918"
version = "0.3.10"

[[AbstractGPsMakie]]
deps = ["AbstractGPs", "LinearAlgebra", "Makie"]
git-tree-sha1 = "bd44f422fcbdeada1528711d56135fa0652ae8ea"
uuid = "7834405d-1089-4985-bd30-732a30b92057"
version = "0.2.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[AlgebraOfGraphics]]
deps = ["Colors", "Dates", "FileIO", "GLM", "GeoInterface", "GeometryBasics", "GridLayoutBase", "KernelDensity", "Loess", "Makie", "PlotUtils", "PooledArrays", "RelocatableFolders", "StatsBase", "StructArrays", "Tables"]
git-tree-sha1 = "8e4d8d012a5fb4f12cf60ae8658afa3aeffe5873"
uuid = "cbdf2221-f076-402e-a563-3d30da359d67"
version = "0.5.2"

[[Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "baf4ef9082070477046bd98306952292bfcb0af9"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.25"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "42ac5e523869a84eac9669eaceed9e4aa0e1587b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.4"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "d0b3f8b4ad16cb0a2988c6788646a5e6a17b6b1b"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.0.5"

[[CairoMakie]]
deps = ["Base64", "Cairo", "Colors", "FFTW", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "SHA", "StaticArrays"]
git-tree-sha1 = "68628add03c2c7c2235834902aad96876f07756a"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.6.2"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "9995eb3977fbf67b86d0a0a0508e83017ded03f2"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.14.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "32a2b8af383f11cbb65803883837a149d10dfe8a"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.10.12"

[[ColorVectorSpace]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "StatsBase"]
git-tree-sha1 = "4d17724e99f357bfd32afa0a9e2dda2af31a9aea"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.8.7"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "727e463cfebd0c7b999bbf3e9e7e16f254b94193"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.34.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "abe4ad222b26af3337262b8afb28fab8d215e9f8"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3889f646423ce91dd1055a76317e9a1d3a23fff1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.11"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "92d8f9f208637e8d2d28c664051a00569c01493d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.1.5+1"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b3bfd02e98aedfa5cf885665493c5598c350cd2f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.2.10+0"

[[FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f985af3b9f4e278b1d24434cbb546d6092fca661"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.3"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "937c29268e405b6808d958a9ac41bfe1a31b08e7"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "7c365bdef6380b29cfc5caaf99688cd7489f9b87"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.2"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "cabd77ab6a6fdff49bfd24af2ebe76e6e018a2b4"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.0.0"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "StaticArrays"]
git-tree-sha1 = "d51e69f0a2f8a3842bca4183b700cf3d9acce626"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.9.1"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
git-tree-sha1 = "39007773fd6097164ab537f78d3ac78ad2b8b695"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.4"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[GR]]
deps = ["Base64", "DelimitedFiles", "LinearAlgebra", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "1185d50c5c90ec7c0784af7f8d0d1a600750dc4d"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.49.1"

[[GeoInterface]]
deps = ["RecipesBase"]
git-tree-sha1 = "38a649e6a52d1bea9844b382343630ac754c931c"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "0.5.5"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "4136b8a5668341e58398bb472754bff4ba0456ff"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.12"

[[Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "7bf67e9a481712b3dbe9cb3dac852dc4b1162e02"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+0"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Match", "Observables"]
git-tree-sha1 = "d44945bdc7a462fa68bb847759294669352bd0a4"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.5.7"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "13c826abd23931d909e4c5538643d9691f62a617"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.8"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[KernelFunctions]]
deps = ["ChainRulesCore", "Compat", "CompositionsBase", "Distances", "FillArrays", "Functors", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Random", "Requires", "SpecialFunctions", "StatsBase", "TensorCore", "Test", "ZygoteRules"]
git-tree-sha1 = "6f46fb7fa868699dfbb6ae7973ba2825d3558ade"
uuid = "ec8451be-7e33-11e9-00cf-bbf324bd1392"
version = "0.10.16"

[[LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "761a393aeccd6aa92ec3515e428c26bf99575b3b"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+0"

[[Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b5254a86cf65944c68ed938e575f5c81d5dfe4cb"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.3"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[Makie]]
deps = ["Animations", "Artifacts", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "Distributions", "DocStringExtensions", "FFMPEG", "FileIO", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "IntervalSets", "Isoband", "KernelDensity", "LinearAlgebra", "MakieCore", "Markdown", "Match", "Observables", "Packing", "PlotUtils", "PolygonOps", "Printf", "Random", "Serialization", "Showoff", "SignedDistanceFields", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "UnicodeFun"]
git-tree-sha1 = "82f9d9de6892e5470372720d23913d14a08991fd"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.14.2"

[[MakieCore]]
deps = ["Observables"]
git-tree-sha1 = "7bcc8323fb37523a6a51ade2234eee27a11114c8"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.1.3"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[Match]]
git-tree-sha1 = "5cf525d97caf86d29307150fcba763a64eaa9cbe"
uuid = "7eb4fadd-790c-5f42-8a69-bfa0b872bfbf"
version = "1.1.0"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "2ca267b08821e86c5ef4376cffed98a46c2cb205"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0f4a4836e5f3e0763243b8324200af6d0e0f90c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.5"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "15003dcb7d8db3c6c857fda14891a539a8f2705a"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.10+0"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "520e28d4026d16dcf7b8c8140a3041f0e20a9ca8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.7"

[[Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "f4049d379326c2c7aa875c702ad19346ecb2b004"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.4.1"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9bc1871464b12ed19297fbc56c4fb4ba84988b0d"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.47.0+0"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "bfd7d8c7fd87f04543810d9cbd3995972236ba1b"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.2"

[[Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "f3a57a5acc16a69c03539b3684354cbbbb72c9ad"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.15.2"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PolygonOps]]
git-tree-sha1 = "c031d2332c9a8e1c90eca239385815dc271abb22"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.1"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "7dff99fbc740e2f8228c6878e2aad6d7c2678098"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.1"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "0529f4188bc8efee85a7e580aca1c7dff6b103f8"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "54f37736d8934a12a200edea2f9206b03bdf3159"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.7"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a322a9493e49c5f3a10b50df3aedaf1cdb3244b7"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.1"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "62701892d172a2fa41a1f829f66d2b0db94a9a63"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3240808c6d463ac46f1c1cd7638375cd22abbccb"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.12"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "20d1bb720b9b27636280f751746ba4abb465f19d"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.9"

[[StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "a209a68f72601f8aa0d3a7c4e50ba3f67e32e6f8"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.24"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "Tables"]
git-tree-sha1 = "44b3afd37b17422a62aea25f04c1f7e09ce6b07f"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.5.1"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "03fb246ac6e6b7cb7abac3b3302447d55b43270e"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.1"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[UrlDownload]]
deps = ["HTTP", "ProgressMeter"]
git-tree-sha1 = "05f86730c7a53c9da603bd506a4fc9ad0851171c"
uuid = "856ac37a-3032-4c1c-9122-f86d88358c8b"
version = "1.0.0"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[isoband_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "a1ac99674715995a536bbce674b068ec1b7d893d"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.2+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "c45f4e40e7aafe9d086379e5578947ec8b95a8fb"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"
"""

# ╔═╡ Cell order:
# ╟─09a9d9f9-fa1a-4192-95cc-81314582488b
# ╟─41eb90d1-9262-42b1-9eb2-d7aa6583da17
# ╟─aa69729a-0b08-4299-a14c-c9eb2eb65d5c
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# ╟─d65de56f-a210-4428-9fac-20a7888d3627
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─41221332-0a22-43c7-b1b2-8a8d84e61dd4
# ╟─77fde8ac-f019-4b4c-9d03-9a9c0d7ca2a0
# ╟─7e0462d4-a307-465b-ae27-737431ecb565
# ╟─a34d8b43-152c-42ff-ae2c-1d439c538c8a
# ╟─01213f94-8dee-4475-b307-e8b18806d453
# ╟─868019c7-eeed-449f-9343-a541f6daefe5
# ╟─17c1a647-7ebc-4703-bed8-148d3a35ac1d
# ╟─538be57f-54e9-45dd-95f5-12e7a3df51a7
# ╟─057a49fa-a68a-4bf0-8b70-7ccd5b8d7931
# ╟─3efef390-b791-40dd-a950-26dcc84c3485
# ╟─349c3025-3b88-4e7e-9e31-574943f31dc6
# ╟─5e825c74-431e-4055-a864-d2b366e8ae11
# ╟─6d451af1-2288-4ee1-a37c-432c14888e16
# ╟─2b30e958-6134-4c8c-8dbf-54cf5babef88
# ╟─1e22e482-c467-4a57-bf19-96361e2896e6
# ╟─193f5509-0733-4409-9b88-1d2bc68e3aee
# ╟─c8e7e82f-415e-4b9d-bd62-47c1c60d0741
# ╠═7557b795-839c-41bf-9586-7b2b3972b28d
# ╟─8b97f4dd-31cf-4a55-a72b-66775d79445c
# ╠═5592c6a8-defc-4b6e-a303-813bdbacaffe
# ╠═6c697f4b-c928-43c2-9490-27b0f195857c
# ╟─8d0a4b08-5cbf-43a8-91f3-5f448893a4c6
# ╠═9e3795cf-8695-46c1-9857-728d765caa02
# ╟─171c9c8f-a4a9-4aad-8bab-f8f38d36a94b
# ╠═741e914f-4d6d-4249-a324-4dd54fd0f277
# ╟─3bb39875-2f99-4a1f-a7ab-a107b4b95716
# ╠═9e949115-a728-48ed-8c06-5cc26b6733bf
# ╟─c21aa46d-08ac-4256-a021-83194bad3a5e
# ╠═abe69a74-74ff-4cc5-9a93-90bd36c48e8a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
