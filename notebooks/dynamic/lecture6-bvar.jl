### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
using BenchmarkTools, CSV, DataFrames, Distributions, KernelDensity, LinearAlgebra, Plots, PlutoUI, Random, SparseArrays, StatsBase, Statistics, UrlDownload

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

In some of the future lectures we will study a few of these more flexible VARs, including the timevarying parameter (TVP) VAR and VARs with stochastic volatility. An excellent review paper that covers many of the same topics is Koop and Korobilis (2010). We will, however, begin with a basic VAR.

"""

# ╔═╡ 77fde8ac-f019-4b4c-9d03-9a9c0d7ca2a0
md"""

### Basic (reduced form) VAR
"""

# ╔═╡ 7e0462d4-a307-465b-ae27-737431ecb565
md"""

Suppose $\mathbf{y}_{t}=\left(y_{1 t}, \ldots, y_{n t}\right)^{\prime}$ is a vector of dependent variables at time $t$. Consider the following $\operatorname{VAR}(p)$ :

$$\mathbf{y}_{t}=\mathbf{b}+\mathbf{A}_{1} \mathbf{y}_{t-1}+\cdots+\mathbf{A}_{p} \mathbf{y}_{t-p}+\boldsymbol{\varepsilon}_{t}$$

where $\mathbf{b} = \left(b_{1}, \ldots, b_{n}\right)^{\prime}$ is an $n \times 1$ vector of intercepts , $\mathbf{A}_{1}, \ldots, \mathbf{A}_{p}$ are $n \times n$ matrices of autoregressive coefficients and $\boldsymbol{\varepsilon}_{t} = \left(\varepsilon_{1 t}, \ldots, \varepsilon_{n t}\right)^{\prime} \sim \mathcal{N}(\mathbf{0}, \mathbf{\Sigma})$. 

In other words, $\operatorname{VAR}(p)$ is simply a multiple-equation regression where the regressors are the lagged dependent variables.

Each equation in this system has $k = np + 1$ regressors, and the total system has $nk = n(np + 1)$ coefficients.

To fix ideas, consider a simple example with $n=2$ variables and $p=1$ lag. Then the equation above can be written explicitly as:

$$\underset{\mathbf{y}_t}{\underbrace{\left(\begin{array}{l}
y_{1 t} \\
y_{2 t}
\end{array}\right)}}=\underset{\mathbf{b}}{\underbrace{\left(\begin{array}{l}
b_{1} \\
b_{2}
\end{array}\right)}}
+
\underset{\mathbf{A}}{\underbrace{\left(\begin{array}{ll}
A_{1,11} & A_{1,12} \\
A_{1,21} & A_{1,22}
\end{array}\right)}}
\underset{\mathbf{y}_{t-1}}{\underbrace{\left(\begin{array}{l}
y_{1,t-1} \\
y_{2,t-1}
\end{array}\right)}}
+
\underset{\boldsymbol{\varepsilon}_t}{\underbrace{\left(\begin{array}{c}
\varepsilon_{1 t} \\
\varepsilon_{2 t}
\end{array}\right)}}$$

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
md""" ### Link to linear regression """

# ╔═╡ 01213f94-8dee-4475-b307-e8b18806d453
md"""

To derive the likelihood for the $\operatorname{VAR}(p)$, we aim to write the system as the linear regression model

$$\mathbf{y}=\mathbf{X} \boldsymbol{\beta}+\varepsilon$$

Then, we can simply apply the linear regression results to derive the likelihood.

Let's first work out our example with $n=2$ variables and $p=1$ lag. To that end, we stack the coefficients equation by equation, i.e., $\boldsymbol{\beta}=\left(b_{1}, A_{1,11}, A_{1,12}, b_{2}, A_{1,21}, A_{1,22}\right)^{\prime}$. 

Equivalently, we can write it using the `vec` operator that vectorizes a matrix by its columns: $\boldsymbol{\beta}=\operatorname{vec}\left(\left[\mathbf{b}, \mathbf{A}_{1}\right]^{\prime}\right)$. Given our definition of $\boldsymbol{\beta}$, we can easily work out the corresponding regression matrix $\mathbf{X}_{t}$ :

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

# ╔═╡ 138d3f7c-4575-4232-97e5-5d78ee64ffb1
md""" To create our matrix $\mathbf{X_t}$ we need to define block diagonal matrices with the same matrix repeated a number of times along the diagonal. One of the easiest ways to do this is via the Kronecker product of the two matrices. Let us consider an example for the Kronecker product. We start with a 2x2 identity matrix. """

# ╔═╡ ec2cb83a-39b9-4e4a-a939-85ff25f0c4ac
Id2 = I(2)

# ╔═╡ 13fa6dcc-26c7-4dcb-ba91-532549063191
md""" Now we need to generate a 2x2 matrix to match our example from above. """

# ╔═╡ e6248785-88ae-43ad-967e-924b0024bfb6
z = randn(1, 3)

# ╔═╡ 20c1c904-1a06-48da-b77e-b6a97047163b
md""" Now we create a block diagonal matrix with $\mathbf{z} = \left[1, \mathbf{y}_{t-1}^{\prime}\right]$ repeated two times on the diagonal. """

# ╔═╡ 56ff8382-f0c1-472c-bf85-b34360823a04
kron(Id2, z)

# ╔═╡ 67f2cc01-074c-410b-859c-b79f670cc74d
md""" Now that we have some idea what the Kronecker product does, let us move to the `vec` operator. In this case we are just stacking coefficients. Consider an example with a $\mathbf{b}$ vector and $\mathbf{A_1}$ matrix (as above). """

# ╔═╡ 0244ce0e-42e7-4013-bedd-9e618565d43e
b = randn(1, 2)

# ╔═╡ fa2e2cc2-d103-48d2-88b1-4f62df4f0f3f
A1 = randn(2, 2)

# ╔═╡ 2e388e43-9fc7-4dd1-8856-0c734ce22cf3
vec([b; A1])

# ╔═╡ 07eeab9e-dd7b-4ff2-9b0f-6f0d5d9a60ec
md""" Hopefully these examples provide some intuition. """

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

**Step 1: Sample $\boldsymbol{\beta}$**

The first step is easy, as standard linear regression results would apply. In fact, we have

$$(\boldsymbol{\beta} \mid \mathbf{y}, \mathbf{\Sigma}) \sim \mathcal{N}\left(\widehat{\boldsymbol{\beta}}, \mathbf{K}_{\boldsymbol{\beta}}^{-1}\right)$$

where

$$\mathbf{K}_{\boldsymbol{\beta}}=\mathbf{V}_{\boldsymbol{\beta}}^{-1}+\mathbf{X}^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}\right) \mathbf{X}, \quad \widehat{\boldsymbol{\beta}}=\mathbf{K}_{\boldsymbol{\beta}}^{-1}\left(\mathbf{V}_{\boldsymbol{\beta}}^{-1} \boldsymbol{\beta}_{0}+\mathbf{X}^{\prime}\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}\right) \mathbf{y}\right)$$

and we have used the result $\left(\mathbf{I}_{T} \otimes \mathbf{\Sigma}\right)^{-1}=\mathbf{I}_{T} \otimes \mathbf{\Sigma}^{-1}$.

**Step 2: Sample $\boldsymbol{\Sigma}$**

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
    Vβ     = sparse(A, A, tmp[:, 1]) 
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
md""" IRFs look strange here, but this could simply be reflecting the prize puzzle? Someone have a good explanation? """

# ╔═╡ abe69a74-74ff-4cc5-9a93-90bd36c48e8a
begin
	plot(bvar(data)[:, 1], lw = 1.5, label = "Inflation") 
	plot!(bvar(data)[:, 2], lw = 1.5, label = "GDP") 
	plot!(bvar(data)[:, 3], lw = 1.5, label = "Repo") 
end

# ╔═╡ 878760b6-0a98-4955-b061-6e56ca83dfbf
md""" Now let us move to an example of a BVAR in R. There are many packages that one could use here. However, I am only going to illustrate one. For your project you could use any package you want. """

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
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
BenchmarkTools = "~1.1.4"
CSV = "~0.8.5"
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

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

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

[[CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a325370b9dd0e6bf5656a6f1a7ae80755f8ccc46"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.7.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

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
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

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

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["ChainRulesCore", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "e13d3977b559f013b3729a029119162f84e93f5b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.19"

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
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

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
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8756f9935b7ccc9064c6eef0bff0ad643df733a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.7"

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

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "dba1e8614e98949abfa60480b13653813d8f0157"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+0"

[[GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "b83e3125048a9c3158cbb7ca423790c7b1b57bea"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.57.5"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cafe0823979a5c9bff86224b3b8de29ea5a44b2e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.61.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "15ff9a14b9e1218958d3530cc288cf31465d9ae2"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.3.13"

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

[[Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "f6532909bf3d40b308a0f360b6a0e626c0e263a8"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.1"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

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

[[InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
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

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

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

[[Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

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

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "34dc30f868e368f8a17b728a1238f3fcda43931a"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.3"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

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
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

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

[[PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "8c3a370130925787b1efed55716c7026f5b3a9fa"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.15.3"

[[PlutoUI]]
deps = ["Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "4c8a7d080daca18545c56f1cac28710c362478f3"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.16"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a193d6ad9c45ada72c14b731a318bedd3c2f00cf"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.3.0"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "69fd065725ee69950f3f58eceb6d144ce32d627d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "ad368663a5e20dbb8d6dc2fddeefe4dae0781ae8"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[RecipesBase]]
git-tree-sha1 = "44a75aa7a527910ee3d1751d1f0e4148698add9e"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.2"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "1f27772b89958deed68d2709e5f08a5e5f59a5af"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.7"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

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

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

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
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "793793f1df98e3d7d554b65a107e9c9a6399a6ed"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.7.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "65fb73045d0e9aaa39ea9a29a5e7506d9ef6511f"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.11"

[[StatsFuns]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "95072ef1a22b057b1e80f73c2a89ad238ae4cfff"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.12"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

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
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[UrlDownload]]
deps = ["HTTP", "ProgressMeter"]
git-tree-sha1 = "05f86730c7a53c9da603bd506a4fc9ad0851171c"
uuid = "856ac37a-3032-4c1c-9122-f86d88358c8b"
version = "1.0.0"

[[Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll"]
git-tree-sha1 = "2839f1c1296940218e35df0bbb220f2a79686670"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.18.0+4"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9398e8fefd83bde121d5127114bd3b6762c764a6"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.4"

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

[[Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

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

[[Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

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

[[Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

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

[[xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
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
# ╟─138d3f7c-4575-4232-97e5-5d78ee64ffb1
# ╠═ec2cb83a-39b9-4e4a-a939-85ff25f0c4ac
# ╟─13fa6dcc-26c7-4dcb-ba91-532549063191
# ╠═e6248785-88ae-43ad-967e-924b0024bfb6
# ╟─20c1c904-1a06-48da-b77e-b6a97047163b
# ╠═56ff8382-f0c1-472c-bf85-b34360823a04
# ╟─67f2cc01-074c-410b-859c-b79f670cc74d
# ╠═0244ce0e-42e7-4013-bedd-9e618565d43e
# ╠═fa2e2cc2-d103-48d2-88b1-4f62df4f0f3f
# ╠═2e388e43-9fc7-4dd1-8856-0c734ce22cf3
# ╟─07eeab9e-dd7b-4ff2-9b0f-6f0d5d9a60ec
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
# ╟─abe69a74-74ff-4cc5-9a93-90bd36c48e8a
# ╟─878760b6-0a98-4955-b061-6e56ca83dfbf
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
