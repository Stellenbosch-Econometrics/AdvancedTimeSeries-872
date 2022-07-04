### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
using BenchmarkTools, Compat, DataFrames, Distributed, Distributions, KernelDensity, LinearAlgebra, Plots, PlutoUI, Random, RCall,  SpecialFunctions, StatsBase, Statistics, StatsPlots, Turing

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
">ATS 872: Lecture 3</p>
<p style="text-align: center; font-size: 1.8rem;">
 Simulation with the Normal distribution
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
md" In this session we will be looking at the normal distribution in detail. This distribution features in many places in Bayesian econometrics and it is quite important to understand some of its properties.  We will then move on to a discussion of marginalisation and look at the normal distribution with different types of priors. This lecture almost serves as a reminder of why we don't want to work with analytical distributions, we would much rather perform computational exercises. 

In this session we use some mathematics and theorethical constructs. Make sure you follow the logic. The math is not particularly difficult, you should have all the necesarry tools. If you are struggling with any part, please let me know.  "

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
TableOfContents() # Uncomment to see TOC

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐."

# ╔═╡ 9480a241-6bd6-41a7-bda3-406e1fc8d94c
md""" ## General overview """

# ╔═╡ 82f8bccb-49f4-4c74-8e49-d0e30250b787
md""" 

In this session we plan to cover the following topics. 

- Simulation of $\pi$ with Monte Carlo methods
- Introduction to parallel programming in _Julia_
- Sampling from a posterior Normal distribution
- Posterior simulation
- Models with more than one unknown parameter
- Marginalisation
- More examples with `Turing.jl`

"""

# ╔═╡ db39d95e-81c8-4a40-942c-6507d2f08274
md" ## Basic Monte Carlo methods "

# ╔═╡ 675dfafa-46cb-44b8-bd7b-55395100e1ca
md" Before we start with the Normal distribution and marginalisation, let us quickly discuss Monte Carlo methods, since we will be using them throughout the rest of the course. According to Wikipedia, Monte Carlo methods, or Monte Carlo experiments, are a broad class of computational algorithms that rely on repeated random sampling to obtain numerical results. The underlying concept is to use randomness to solve problems that might be deterministic in principle. They are often used in physical and mathematical problems and are most useful when it is difficult or impossible to use other approaches. Monte Carlo methods are mainly used in three problem classes: optimization, numerical integration, and generating draws from a probability distribution.  Our first Monte Carlo example will be to try and estimate the value for $\pi$. 

"

# ╔═╡ 2e83e207-2405-44f4-a5d9-13dd69b741a9
md" ### Estimating π "

# ╔═╡ 961747f8-c884-43bf-941d-4545fb4510e6
md"""
In this section we will have some fun trying to approximate $\pi$ using Monte Carlo methods. We will compare the speed of computation between Julia and R for this example. The purpose of this section is twofold. First, to establish a motivation for the use of simulation techniques. Second, to encourage some general problem solving through coding. Think about what this program is doing and see whether it makes sense to you. The code is easy enough to understand, it is the underlying logic that might be tricky.    
 
"""

# ╔═╡ e3f722c1-739f-45b4-8fbf-db07e9483d92
function compute_pi_naive(n::Int)
	
	n_landed_in_circle = 0
	
	for i in 1:n
		
		u = rand(2)
		
		if u[1]^2 + u[2]^2 < 1.0
			n_landed_in_circle += 1
		end
	end
	
	return n_landed_in_circle / n * 4.0
end;		

# ╔═╡ 59df263a-a284-40b2-8d9d-bb34fbf13b3a
# Similar code to first instance above 
R"mc_pi_loop <- function(n){
  n_landed_in_circle <- 0 
  for (i in 1:n) {
    u <- runif(2)
    if (u[1]^2 + u[2]^2 < 1) {
      n_landed_in_circle <- n_landed_in_circle + 1
    } 
  }
  area_est <- n_landed_in_circle / n
  return(area_est * 4)
}";

# ╔═╡ 16b216ff-81ed-481a-896b-4ddf2cd139f6
function compute_pi(n::Int)
  
    n_landed_in_circle = 0  
  
    for i = 1:n
        x = rand()   
        y = rand()   
  
        r2 = x*x + y*y  
        if r2 < 1.0
            n_landed_in_circle += 1
        end
    end
  
    return n_landed_in_circle / n * 4.0    
end;

# ╔═╡ 6b5886d8-d783-4cbc-9a8c-286b741cb16f
# Vectorised version of the code (should be much faster)
R"mc_piv <- function(n) {
  x <- runif(n, 0, 1)
  y <- runif(n, 0, 1)
  is_inside <- (x^2 + y^2) < 1
  pi_estimate <- 4 * sum(is_inside) / n
  return(pi_estimate)
}";

# ╔═╡ 09b953aa-7329-48df-b152-a532d997ceea
PlutoUI.with_terminal() do
	@time compute_pi_naive(1000000)
end

# ╔═╡ 9e6a45a9-07da-4859-861c-db0c6ca30ec1
#@benchmark compute_pi_naive(10000000); 

# ╔═╡ 0d10bec1-e358-4b9d-a452-751df45b863c
R"system.time(a <- mc_pi_loop(1000000))"

# ╔═╡ e57f06bd-941b-422a-91e6-004f92ef4b4f
PlutoUI.with_terminal() do
	@time compute_pi(1000000)
end

# ╔═╡ 4dfbbd9f-0f34-4b7c-a018-b89df5a81dfe
#@benchmark compute_pi(10000000);

# ╔═╡ aac33160-18d2-4d0e-87c9-1eb9dc363c88
R"system.time(a <- mc_piv(1000000))" # This is why we are advised against loops in R...

# ╔═╡ 9d1a94ac-7ca2-4c8c-a801-021fd2cfc90e
md" So we have managed to find an approximation for $\pi$. Let us look at some other methods and also draw some nice graphs in the process to better explain what is going on. Our code above is optimised to a certain extent. However, when writing code, do not worry too much about optimisation. You can always go back to code to optimise. Try and write code that makes sense to you before you expirment with optimisation.  "

# ╔═╡ 880792b2-02b7-444e-a787-d02ee685ee72
md"""
 > Programmers waste enormous amounts of time thinking about, or worrying about, the speed of noncritical parts of their programs, and these attempts at efficiency actually have a strong negative impact when debugging and maintenance are considered. We should forget about small efficiencies, say about 97% of the time: **premature optimization is the root of all evil**. Yet we should not pass up our opportunities in that critical 3% -- **Donald Knuth**
"""

# ╔═╡ 309c4a9b-cd46-43ed-8a59-001057549fc4
md" ### Parallel programming "

# ╔═╡ 963e30cf-16af-44bf-b28e-f7ea4fe96dcc
md" Suppose that we want this code to run even faster without changing the underlying code muc. In the data science course for the first semester we talked about parallel programming. This type of problem is embarassingly parallel and can be easily extended to multiple cores. You can read more on this in the following blog post by [Cory Simon](https://corysimon.github.io/articles/parallel-monte-carlo-in-julia/). "

# ╔═╡ 94dc15cd-31ea-46a0-8401-aff7f2a74e5e
md" This does not work so well in Pluto, so we will illustrate this in `VSCode`. "

# ╔═╡ 11a1cec2-2e95-4ebd-a7ce-bbe77f3c9a1d
md" ### Plotting π "

# ╔═╡ a0981303-008c-4e4a-b96e-581b52ab15f2
md" Let us try to plot the estimate for $\pi$ "

# ╔═╡ 924ea0e4-0133-4711-9e19-662b4d753e37
N = @bind N₁ Slider(100:100:5000, default=100)

# ╔═╡ 3ef808e6-c49a-4c25-befe-d7bfd502ff64
begin
	#N₁ = 10^5
	data     = [[rand(),rand()] for _ in 1:N₁]
	indata   = filter((x)-> (norm(x) <= 1), data)
	outdata  = filter((x)-> (norm(x) > 1), data)
	piApprox = 4*length(indata)/N₁
	
	scatter(first.(indata),last.(indata), c=:blue, ms=2.9, msw=0, alpha = 0.7)
	scatter!(first.(outdata),last.(outdata), c=:red, ms=2.9, msw=0,
		xlims=(0,1), ylims=(0,1), legend=:none, ratio=:equal, alpha = 0.8)
end

# ╔═╡ ceaaccf2-ae65-4731-afc9-f6763bc2b3a1
piApprox

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Gaussian (Normal) "

# ╔═╡ e4730930-c3cd-4a01-a4d9-420bd15004ad
md"""
This is a distribution that we will be dealing with a lot in the course, so it is worthwhile getting used to its functional form and different properties. It appears almost everywhere! 

The central limit theorem helps justify our usage of the normal distribution. Important authors responsible for development of this theorem include De Moivre, Laplace, Gauss, Chebysev, Liapounov and Markov. Given certain conditions the sum (and mean) of independent random variables approach a Gaussian distribution as $n \rightarrow \infty$ even if original variables are not normally distributed.

There are some problems

1. It does not hold for all distributions, e.g., Cauchy
2. This may require very large values of $n$. See the case of Binomial, when $\theta$ close to $0$ or $1$

The first thing that we will do is derive the posterior distribution for the mean in the Gaussian model. We will start with the single parameter case, so we will assume that the variance $\sigma^{2}$ is known. This is rarely true in practice, so we will then move on to a discussion where both parameters are unknown. 
"""

# ╔═╡ f82ce58d-f292-4ecd-86ae-06d4fa79bcd4
md" ### Gaussian model with known $\sigma^2$ "

# ╔═╡ 8ffaa0dc-36f8-49da-9742-74db90c6d5a8
md""" 

**Note**: As we have mentioned, we start with a single parameter model. The parameter of interest for this model is the mean, which we will refer to as $\theta$. We could have named it $\mu$, but that can create confusion as to whether the quantity is known or not. We know $\sigma^{2}$ but are looking for information on $\theta$ so that we can build our posterior, $p(\theta \mid y)$. In other books you might see the same calculations, but with $\mu$ instead of $\theta$. We will only do this for the first example, so that you can get comfortable with the math. 

To illustrate the basic mechanics of Bayesian analysis, we start with a toy example. Suppose we take $N$ independent measurements $y_{1}, \ldots, y_{N}$ of an unknown quantity $\theta$, where the magnitude of measurement error is known. In addition, from a small pilot study $\theta$ is estimated to be about $\mu_{0}$. In other words, we have some information to help us decide on a prior for $\theta$. 

Our goal is to obtain the posterior distribution $p(\theta \mid {y})$ given the sample ${y}=$ $\left(y_{1}, \ldots, y_{N}\right)^{\prime} .$ To that end, we need two ingredients: a likelihood function and a prior for the parameter $\theta$.

One simple model for this measurement problem is the normal model:

$$\left(y_{n} \mid \theta \right) \sim \mathcal{N}\left(\theta, \sigma^{2}\right), \quad n=1, \ldots, N$$

where the variance $\sigma^{2}$ is **assumed to be known**. Then, the model defines the likelihood function $p({y} \mid \theta)$. 

Since the scale of the study is small, there is uncertainty around the estimate. A reasonable prior would be

$$\theta \sim \mathcal{N}\left(\mu_{0}, \tau_{0}^{2}\right)$$

where both $\mu_{0}$ and $\tau_{0}^{2}$ are known. **Note**: this is a distribution around our unknown parameter $\theta$. Our prior distribution is Normal with these specific parameters. 

Relevant information about $\theta$ is summarized by posterior distribution, which can be obtained by Bayes' theorem:

$$p(\theta \mid {y})=\frac{p(\theta) p({y} \mid \theta)}{p({y})}$$

It turns out that $p(\theta \mid y)$ is a Gaussian distribution. We will now try and show this. The derivation can get a bit messy, but the logic is important. If you have done mathematical statistics at any stage you will feel right at home! 😉

"""

# ╔═╡ 5d97bab1-346d-4edd-bc5e-bc6b1a510912
md"""

#### Derivation (NB for Exam)

"""

# ╔═╡ 39d705ff-4540-4103-ae10-694d5b64e82b
md"""

We need a prior and likelihood function to produce the posterior distribution. 

We start with the form of the likelihood function $p(y \mid {\theta})$. 

Recall from our first lecture that we say a random variable $X$ follows a normal or Gaussian distribution, and we write $X \sim \mathcal{N}\left(a, b^{2}\right)$, its density is given by

$$f_{X}\left(X = x ; a, b^{2}\right)=\left(2 \pi b^{2}\right)^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2 b^{2}}(x-a)^{2}}$$

The likelihood function is a product of $N$ normal densities, so we can write

$$\begin{aligned}
p(y \mid {\theta}) &=\prod_{n=1}^{N}\left(2 \pi \sigma^{2}\right)^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2 \sigma^{2}}\left(y_{n}-\theta\right)^{2}} \\
&=\left(2 \pi \sigma^{2}\right)^{-\frac{N}{2}} \mathrm{e}^{-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(y_{n}-\theta\right)^{2}}
\end{aligned}$$

Similarly, the prior density $p(\theta)$ is given by

$$p(\theta)=\left(2 \pi \tau_{0}^{2}\right)^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2 \tau_{0}^{2}}\left(\theta-\mu_{0}\right)^{2}}$$

Remember the assumption that we made about the distribution for the prior density in the previous section. 

Now, as you would expect from Bayes' theorem, we will combine the prior and likelihood to obtain the posterior distribution. Note that the variable in $p(\theta \mid {y})$ is $\theta$, and we can ignore any constants that do not involve $\theta$.

$$\begin{aligned}
p(\theta \mid {y}) & \propto p(\theta) p({y} \mid \theta) \\
& \propto \mathrm{e}^{-\frac{1}{2 \tau_{0}^{2}}\left(\theta-\mu_{0}\right)^{2}} \mathrm{e}^{-\frac{1}{2 \sigma^{2}} \sum_{n=1}^{N}\left(y_{n}-\theta\right)^{2}} \\
& \propto \exp \left[-\frac{1}{2}\left(\frac{\theta^{2}-2 \theta \mu_{0}}{\tau_{0}^{2}}+\frac{N \theta^{2}-2 \theta \sum_{n=1}^{N} y_{n}}{\sigma^{2}}\right)\right] \\
& \propto \exp \left[-\frac{1}{2}\left(\left(\frac{1}{\tau_{0}^{2}}+\frac{N}{\sigma^{2}}\right) \theta^{2}-2 \theta\left(\frac{\mu_{0}}{\tau_{0}^{2}}+\frac{N \bar{y}}{\sigma^{2}}\right)\right)\right]
\end{aligned}$$

where $\bar{y}=N^{-1} \sum_{n=1}^{N} y_{n}$ is the sample mean. Since the exponent is quadratic in $\theta$, $p(\theta \mid {y})$ is Gaussian, and we next determine the mean and variance of the distribution.

In this example, the posterior distribution is in the same family as the prior - both are Gaussian. In this case, the prior is called a conjugate prior for the likelihood function.

Now, suppose $(\theta \mid {y}) \sim \mathcal{N}\left(\mu_1, \tau_{1}^{2}\right)$ for some mean $\mu_{1}$ and variance $\tau_{1}^{2}$. Using the definition of the Gaussian density, we can rewrite the posterior distribution as

$$\begin{aligned}
p(\theta \mid {y}) &=\left(2 \pi \tau_{1}^{2}\right)^{-\frac{1}{2}} \mathrm{e}^{-\frac{1}{2 \tau_{1}^{2}}(\theta-\mu_1)^{2}} \\
& \propto \exp {-\frac{1}{2}\left(\frac{1}{\tau^{2}_1} \theta^{2}-2 \theta \frac{\mu_1}{\tau_{1}^{2}}\right)}
\end{aligned}$$

We now have two expressions for the posterior distribution, which are identical for any $\theta \in \mathbb{R}$. The coefficients on $\theta^{2}$ must be the same, in other words we have that 

$$\tau_{1}^{2}=\left(\frac{1}{\tau_{0}^{2}}+\frac{N}{\sigma^{2}}\right)^{-1}$$

Similarily the coefficients on $\theta$ must be the same as well, 

$$\begin{aligned}
\frac{\mu_1}{\tau_{1}^{2}} &=\frac{\mu_{0}}{\tau_{0}^{2}}+\frac{N \bar{y}}{\sigma^{2}} \\
\mu_{1} &=\tau_{1}^{2}\left(\frac{\mu_{0}}{\tau_{0}^{2}}+\frac{N \bar{y}}{\sigma^{2}}\right)
\end{aligned}$$

Subtituting the definition of $\tau_{1}^{2}$ into this equation we can write the posterior mean as, 

$$\begin{aligned}
\mu_{1} &=\left(\frac{1}{\tau_{0}^{2}}+\frac{N}{\sigma^{2}}\right)^{-1}\left(\frac{\mu_{0}}{\tau_{0}^{2}}+\frac{N \bar{y}}{\sigma^{2}}\right) \\
&=\frac{\frac{1}{\tau_{0}^{2}}}{\frac{1}{\tau_{0}^{2}}+\frac{N}{\sigma^{2}}} \mu_{0}+\frac{\frac{N}{\sigma^{2}}}{\frac{1}{\tau_{0}^{2}}+\frac{N}{\sigma^{2}}} \bar{y}
\end{aligned}$$

If we write the posterior mean as $\omega \bar{y} + (1-\omega) \mu_{0}$ where 

$\omega = \frac{N/\sigma^{2}}{N/\sigma^{2} + 1/\tau_{0}^{2}}$

then we can say that the posterior mean is a weighted average of the data mean and the prior mean. Posterior therefore puts more emphasis on the data when the value of $N$ is large. This is something that we also found in our previous lecture with the Bernoulli distribution. 
"""

# ╔═╡ f5184e69-55c6-4688-8b0e-d543a07be97b
md"""

#### Summary

"""

# ╔═╡ 3b2e280f-d83a-4f1c-86bb-1397b95210cb
md"""
To make things a bit easier in terms of notation, set $N = 1$. For unknown mean $\theta$ and known variance $\sigma^2$ we have the folowing summary. Some simplifications have been made to make things more readable,   

$$\begin{align*}
    p(y|\theta)&=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{1}{2\sigma^2}(y-\theta)^2\right), \quad y \sim \mathcal{N}(\theta,\sigma^2)
\end{align*}$$

Assume $\sigma^2$ known. Remember that model defines likelihood function, 

$$\begin{align*}
      &\text{Likelihood} & p(y|\theta) & \propto
\exp\left(-\frac{1}{2\sigma^2}(y-\theta)^2\right) \\ 
&\text{Prior} & p(\theta) & \propto
      \exp\left(-\frac{1}{2\tau_0^2}(\theta-\mu_0)^2\right) \\ 
& {\exp(a)\exp(b)}  {= \exp(a+b)} \\ \\

&\text{Posterior} & p(\theta|y)& \propto \exp\left(-\frac{1}{2}\left[\frac{(y-\theta)^2}{\sigma^2}+\frac{(\theta-\mu_0)^2}{\tau_0^2} \right]\right)

\end{align*}$$

Furthermore, we have that, 

$$\begin{align*}
      & &
      p(\theta|y)&\propto \exp\left(-\frac{1}{2}\left[
          \frac{(y-\theta)^2}{\sigma^2}+\frac{(\theta-\mu_0)^2}{\tau_0^2} \right]\right) \\
      & & & \propto \exp \left(-\frac{1}{2\tau_1^2}(\theta-\mu_1)^2
      \right)
\end{align*}$$

$\begin{equation*}
      \theta|y \sim \mathcal{N}(\mu_1,\tau_1^2), \quad
      \text{where} \quad
      \mu_1=\frac{\frac{1}{\tau_0^2}\mu_0+\frac{1}{\sigma^2}y}{\frac{1}{\tau_0^2}+\frac{1}{\sigma^2}} \quad  \text{and}  \quad \frac{1}{\tau_1^2} = \frac{1}{\tau_0^2}+\frac{1}{\sigma^2}
\end{equation*}$

"""

# ╔═╡ 9741e08b-3f54-49d7-8db0-3125f4f90d3c
md" ### Practical implementation "

# ╔═╡ e99e1925-6219-4bf2-b743-bb2ea725dfcd
md" We know from our previous discussion that a Gaussian model with known variance delivers Gaussian posterior. We generate some fake data in the height of males in South Africa, with different variances as well. These can be thought of as two samples that we drew from the population. We will construct a DataFrame, which is similar to the tibble / dataframe in R.  "

# ╔═╡ c2897578-b910-41db-886f-a9b8688bff48
md"""
h₁ = $(@bind h₁ PlutoUI.Slider(150:190, show_value=true, default=165));
h₂ = $(@bind h₂ PlutoUI.Slider(150:190, show_value=true, default=175))
"""

# ╔═╡ 0fc998e7-d824-412b-a138-b626ba118904
df = DataFrame(id = [1, 2], height_μ = [h₁, h₂], height_σ = [4, 2]);

# ╔═╡ 01607a33-ad7e-4fab-b5cf-8ddf20a69a52
md" Our prior belief on the mean and standard deviation in the population is given as follows, "

# ╔═╡ aac22072-c3f2-4b66-bf3f-3dbf967fe6f9
begin
	pop_μ = 171 # Prior for population mean
	pop_σ = 6 # Prior for population standard deviation
end;

# ╔═╡ 2af6f4a8-7b64-405f-ab2e-e77a2699ceb2
grid = range(150, 210, length = 601) |> collect;

# ╔═╡ 1316fbc2-700d-4106-99f0-0b9243a99aba
begin
	
	plot(grid, Normal(df.height_μ[1], df.height_σ[1]), lw = 0, fill = (0, 0.3, :steelblue), labels = "Group 1")
	plot!([df.height_μ[1]], seriestype = :vline, lw = 2, color = :steelblue, ls = :dash, alpha =0.7, labels = "Group 1 Mean")
	plot!(grid, Normal(df.height_μ[2], df.height_σ[2]), lw = 0, color = :blue, fill = (0, 0.3, :black), labels = "Group 2")
	plot!([df.height_μ[2]], seriestype = :vline, lw = 2, color = :black, ls = :dash, alpha =0.4, labels = "Group 2 Mean")
	plot!(grid, Normal(pop_μ, pop_σ), xlims = (150, 190), lw = 2.5, color = :green4, fill = (0, 0.6, :green), size = (600,400), labels = "Prior")
end

# ╔═╡ 3c409c93-545a-4d10-a972-509dff7c0120
md" The above graph shows the data and prior information from the population. One can think of the fake guesses as likelihood functions. Next we provide the posterior function as analytically calculated above." 

# ╔═╡ cef4da14-ee03-44b8-bd86-c11c263b26b3
post_σ(prior_σ, obs_σ) = sqrt.(1 ./ ((1 ./ prior_σ .^ 2) .+ (1 ./ obs_σ .^ 2)));

# ╔═╡ d4f3331d-3dbd-4bad-b043-dff9409005c3
post_μ(prior_μ, prior_σ, obs_μ, obs_σ) = ((prior_μ ./ (prior_σ .^ 2)) .+ (obs_μ ./ (obs_σ .^ 2))) ./ ((1 ./ (prior_σ .^ 2)) .+ (1 ./ (obs_σ .^ 2)));

# ╔═╡ ec24134e-a9c0-491e-9a68-c95a618f0b1d
md" The posterior combines information from the prior and the likelihood. We will show in each case how the information retrieved from the data alters the posterior. " 

# ╔═╡ 572653c4-0e3d-4619-bec6-64a01f4b2e06
posterior_σ = post_σ([df.height_σ[1], df.height_σ[2]], pop_σ)

# ╔═╡ 168df9b2-340c-4e06-8985-d2a1fad60bea
post_μ₁ = post_μ(df.height_μ[1], df.height_σ[1], pop_μ, pop_σ)

# ╔═╡ 74e1be5b-503d-48cb-97fd-c22b634cc361
post_μ₂ = post_μ(df.height_μ[2], df.height_σ[2], pop_μ, pop_σ)

# ╔═╡ fd9fe53b-3c5c-4e09-837e-bef6f00e485f
posterior_μ = [post_μ₁, post_μ₂];

# ╔═╡ 39b604b9-a4c8-4f54-afaa-faa84d99ad73
begin
	plot(grid, Normal(df.height_μ[1], df.height_σ[1]), lw = 0, fill = (0, 0.3, :steelblue), labels = "Group 1")
	plot!(grid, Normal(df.height_μ[2], df.height_σ[2]), lw = 0, color = :blue, fill = (0, 0.3, :black), labels = "Group 2")
	plot!(grid, Normal(pop_μ, pop_σ), lw = 0, color = :black, fill = (0, 0.5, :green), size = (600,400), labels = "Prior")
	plot!(grid, Normal(posterior_μ[1], posterior_σ[1]), lw = 2, color = :steelblue4, size = (600,400), labels = "Posterior 1")
	plot!(grid, Normal(posterior_μ[2], posterior_σ[2]), xlims = (150, 190), lw = 2, color = :grey20, size = (600,400), labels = "Posterior 2") # Something went wrong with this calculation. 
end

# ╔═╡ a0670eb1-2091-47d0-9d9f-bba76834fbed
md" One can see that doing this analytically is quite cumbersome. That is why we will often rely on numerical methods to draw from the posterior. Our next lecture will focus exclusively on such methods. In particular, we will consider different types of **Markov chain Monte Carlo** methods to draw from the posterior distribution of interest. We can use a basic Monte Carlo simulation to get an answer, before we move to another analytical example.  "

# ╔═╡ 8af7a479-7ac5-4a21-a354-fcaee618367b
md""" ## Monte Carlo simulation """

# ╔═╡ 8010f3a3-bc04-488f-b4de-d613be8f6f84
md""" 
> These notes on simulation are based of the lecture notes by [Jamie Cross](https://github.com/Jamie-L-Cross/Bayes/blob/master/3_Monte_Carlo.ipynb). Go check out his notes for more detail. He has some good discussions on Bayesian econometrics.  
"""

# ╔═╡ 0a4bb68a-f82a-426b-842f-23c405af2e64
md""" Monte Carlo methods can be used to approximate our parameter of interest by random sampling from a probability distribution. We can construct posterior distributions with these techniques and then find different moments or probabilities. We will discuss two methods Monte Carlo methods that are commonly employed in Bayesian statistics. First, we look at posterior simulation and then posterio integration.  """

# ╔═╡ f691ceb8-31c8-47dc-8ba0-8a4a6bbe811f
md""" ### Posterior simulation """

# ╔═╡ 0d42559a-721e-4084-bbec-911c5c9bc722
md""" With posterior simulation we are using Monte Carlo methods to draw from the posterior distribution. We will employ direct sampling here, which means that we draw from the posterior distribution directly and then from these draws we approximate the posterior distribution. In other words we take 

$\theta^{(s)} \sim p(\theta \mid y)$ 

where $s = 1, \ldots, S$. The distribution of the simulations, $q(\theta^{(1)}, \ldots, \theta^{(S)} \mid y)$ will converge to the true posterior as $S \rightarrow \infty$. Increasing the number of draws will get us a better approximation. Approximation of true posterior via simulations is then known as posterior simulation.  """


# ╔═╡ 3160f1bc-85e9-408d-8562-6306b86eace0
md"""
Draws = $(@bind S1 PlutoUI.Slider(10:1000, show_value=true, default=10));
"""

# ╔═╡ 381154f0-e0bb-4bcf-aba4-81ad30477496
begin 
	dist = Normal(0,1); # Distribution from which we will sample
	y = rand(dist, S1);
	x_axis = collect(-5:0.1:5);
	histogram(x_axis, y, normalize = :pdf, labels = "MC draws", alpha = 0.3)
    p1 = plot!(x_axis -> pdf(dist, x_axis), labels = "True PDF", color = :black, lw = 2)
end

# ╔═╡ 116982c6-80c9-4cc8-8140-804a631a77a8
md" ### Posterior integration "

# ╔═╡ 9e8aa12a-540b-4153-9dbf-8b503c16b091
md" After we have derived (or approximated) a posterior we often want to calculate certain values, such as the mean or variance. Calculating these values normally requires that we compute integrals, like the one below,

$$\mathbb{E}(g(\theta) \mid {y})=\int_{\Theta} g(\theta) p(\theta \mid {y}) \mathrm{d} \theta$$

As an example, if we wanted to calculate the posterior mean we would set $g(\theta) = \theta$ and calculate, 

$$\mathbb{E}(\theta \mid {y})=\int_{\Theta} \theta p(\theta \mid {y}) \mathrm{d} \theta$$

In the case of the posterior variance one would need to have the integral for $g(\theta) = \theta^{2}$. There are cases, such as the coin flipping model, in which these values can be calculated by hand. In general, this integration cannot be solved analytically. However, we can estimate this quantity using Monte Carlo integration. The one prerequisite is that we know how to obtain samples from the posterior. If we are able to sample from the posterior then we generate $S$ draws $\theta^{(1)}, \ldots, \theta^{(S)}$ from $p(\theta \mid {y})$, and compute

$$\widehat{g}=\frac{1}{S} \sum_{s=1}^{S} g\left(\theta^{(s)}\right)$$

By the weak law of large numbers, $\widehat{g}$ converges weakly in probability to $\mathbb{E}(g(\theta) \mid {y})$ as $S$ tends to infinity. Since we control the simulation size $S$, we can in principle estimate $\mathbb{E}(g(\theta) \mid {y})$ arbitrarily well.

If we wanted to calculate the posterior mean, this would mean that the Monte Carlo approximation is, 

$\hat \theta =\frac{1}{S}\sum_{s=1}^{S} \theta^{(s)}$ 

We could do a similar approximation for the variance using the following formula,

$\hat \sigma^{2} =\frac{1}{S} \sum_{s=1}^{S} (\theta^{(s)} - \hat \theta)^{2}$ 

Let us show how to calculate these quantities with some code. 

"

# ╔═╡ 41915620-c22b-49f3-bb13-299f853dfb9f
md"""
Draws = $(@bind S₁ PlutoUI.Slider(10:1000, show_value=true, default=10));
"""

# ╔═╡ 8f5de37a-6c2c-400d-97c2-7f04e0fa4857
begin
	Random.seed!(1234)
	θ_s = rand(dist, S₁)
	θ_hat = mean(θ_s) # alternative: sum(θ_s)/S₁
	σ2 = var(θ_s, corrected = false) # alternative: sum((θ_s .- mean(θ_s).^2)/S₁)
end;

# ╔═╡ e5d0ad86-5bba-4f1c-a2da-b232bff9b4f2
θ_hat # estimate for the posterior mean

# ╔═╡ 0375ebdc-1221-4dc1-89bb-b73a0fd6cd20
σ2 # estimate for posterior variance

# ╔═╡ 135aa2d9-05e0-4ab0-8838-5c77326d43b2
md""" ### Random walk model """

# ╔═╡ c79eaac5-f0db-40cf-b32f-3e1be6504549
md""" Let us work through a time series example where the mean is known, but the variance is unknown. This is a single parameter model, but slightly different from the derivation we explored above.  """

# ╔═╡ 3bbb7b4e-b8cd-4af8-b1c5-dec8008dc448
md""" We let $\mathbf{Y}=(Y_1,\dots,Y_T)'$ be the time series. The random walk model is then given by, 

$$Y_t = Y_{t-1} + e_t$$ 

where $Y_0 = 0$ and $e_t$ is a random vairable with distribution $\mathcal{N}(0, \sigma^{2})$. In this case **$\sigma^{2}$ is unknown**. You will have encountered this type of model in previous econometrics courses, specifically the financial econometrics course as this underlies the random walk hypothesis of stock prices. 

Next we simulate the data for such a random walk model. In another lecture, we will take data from the S&P 500 or even JSE. However, for today let us work with simulated data, as we are on the topic of simulation. """

# ╔═╡ 8301f15f-a1e8-4910-abd6-ccd758719381
begin
	
	Random.seed!(1232)
	
	## Simulate Data from random walk model
	σ2_true = 1; # true σ2
	T = 1000; # number of dates
	Y0 = 0;   # initial condition
	
	Y = zeros(T); # storage vector
	
	Y[1] = Y0;
	for t = 2:T
	    Y[t] = Y[t-1] + rand(Normal(0,sqrt(σ2_true)));
	end
	
	X = collect(1:T);
	plot(X, Y, title = "Simulated data", legend = :false, lw = 1)
end

# ╔═╡ ccf3fce7-436c-46e3-ad62-935240891259
md""" We can estimate the model using Bayesian methods. If $e_t \sim \mathcal{N}(0, \sigma^2)$ then our probability model representation is given by, 

$$Y_t\sim \mathcal{N}(Y_{t-1},\sigma^2)$$

Estimating the random walk model in this case is then the same as estimating parameters of the normal distribution with known mean and unknown variance. Since the mean of the distribution is known we can rewrite things in terms of first differences (subtract $Y_{t-1}$) to get, 

$$\Delta Y_t\sim N(0,\sigma^2)$$

We show the first difference representation below. 
"""

# ╔═╡ ad298e39-3abb-45b8-bc5a-a7e1c8e7024f
begin
	DY = Y[2:end] - Y[1:end-1];
	T1 = length(DY);
	X1 = collect(1:T1);
	plot(X1, DY, title = "First-difference of simulated data", legend = false)
end

# ╔═╡ 286e4c9a-3546-41c0-9e71-7a1d89c705f4
md""" Now that we have our probability model, we need to consider our likelihood function and prior to gain access to our posterior. The likelihood of this model is given by, 

$$\begin{align}
p(\mathbf{Y}|\sigma^2) &= \prod_{t=1}^{T}p(\Delta Y_t|\sigma^2)\\
&= \prod_{t=1}^{T}(2\pi\sigma^2)^{-\frac{1}{2}}\exp(-\frac{1}{2\sigma^2}(\Delta Y_t)^2)\\
&= (2\pi\sigma^2)^{-\frac{T}{2}}\exp(-\frac{1}{2\sigma^2}\sum_{t=1}^{T}(\Delta Y_t)^2)
\end{align}$$ 

In the case of the prior we have that $\sigma^{2}$ is positive, so it makes sense to choose a prior with support over $(0, \infty)$. One such an example is the Gamma distribution or the invers-Gamma distribution will also work. You might not have heard about the Gamma distribution, but the Chi-sqaured distribution is a special case of the Gamma. Once again, Wikipedia really is your friend here. As a Bayesian econometrician it is worthwhile to think about different distributions and their properties.  """

# ╔═╡ 962b5c8a-dcc0-464c-adad-04e3109dfe4a
md""" For the prior we will use inverse-Gamma with the shape parameter $\nu$ and scale parameter $S$. We denote this distribution $IG(\nu,S)$. The probability desnity function of this distribution is, 

$$p(\sigma^2) = \frac{S^\nu}{\Gamma(\nu)}(\sigma^2)^{-(\nu+1)}\exp(-\frac{\nu}{\sigma^2})$$

"""

# ╔═╡ 92c6c3a1-f046-44f5-ae3c-f17d342a9399
md""" Changing the hyperparameters will change the shape of the probability density function, as shown below. """

# ╔═╡ 872dc6b1-0b13-418b-a614-732d2a4d8963
md"""
ν = $(@bind ν PlutoUI.Slider(1:0.1:20, show_value=true, default=4));
S = $(@bind S PlutoUI.Slider(0.1:0.1:5, show_value=true, default=0.5))
"""

# ╔═╡ d02bd87a-50ef-4e37-b11a-cfc0b023608e
Plots.plot(
    InverseGamma(ν, S);
    title="\$\\mathrm{InvGamma}\\,($ν, $S)\$",
    label=false,
    linewidth=2,
    fill=true,
    fillalpha=0.3,
	color = :steelblue
)

# ╔═╡ d172fe2c-d842-4c6a-9097-3a0e51058bb0
md""" Now to get the posterior we will combine the likelihood function and posterior, 

$$\begin{align}
p(\sigma^2|\mathbf{Y}) &\propto p(\mathbf{Y}|\sigma^2) p(\sigma^2)\\
                       &\propto (\sigma^2)^{-\frac{T}{2}}\exp(-\frac{1}{2\sigma^2}\sum_{t=1}^{T}(\Delta Y_t)^2)(\sigma^2)^{-(\nu_0+1)}\exp(-\frac{S_0}{\sigma^2})\\
                       &= (\sigma^2)^{-(\frac{T}{2}+\nu_0+1)}\exp(-\frac{1}{\sigma^2}(S_0+\frac{1}{2}\sum_{t=1}^{T}(\Delta Y_t)^2)
\end{align}$$

While it might we tricky to see, the kernel of the posterior is inverse-Gamma with scale parameter $\nu = \frac{T}{2}+\nu_0+1$ and shape parameter $S=S_0+\frac{1}{2}\sum_{t=1}^{T}(\Delta Y_t)^2$. This means we have a nice analytical form to compute different moments and also to compare with our Monte Carlo approximation. 

"""

# ╔═╡ 6cf5de34-b256-471f-a8c7-ecc14f469967
begin
	# Code is directly from Jamie Cross

	
	# Prior
	ν_pri = 3;
	S_pri = 1 * (ν_pri-1); # sets E(sig2) = 1
	pri = InverseGamma(ν_pri, S_pri);
	
	# Posterior
	ν_post = ν_pri + T/2;
	S_post = S_pri + 0.5 * sum(DY.^2);
	post = InverseGamma(ν_post, S_post);
	
	## Approximate posterior using direct MC
	ndraws = 1000;
	σ2_MC = rand(post, ndraws);
	
	# Compute posterior mean using Monte Carlo Integration
	σ2hat_MC = mean(σ2_MC);
	
	# Compute posterior mean using theoretical formula
	σ2hat_theory = S_post/(ν_post-1);
	
	# Plot empirical and theoretical posterior distributions
	X2 = collect(0.6:0.01:1.5);
	histogram(X2, σ2_MC, normalize = :pdf, label="Empirical pdf", alpha = 0.3)
	plot!(X2, pdf(post, X2), label = "Theoretical pdf", lc = :black, lw = 2)
	plot!([σ2_true], seriestype="vline", label="True value", lw = 2,  ls = :dash, lc = :green)
	plot!([σ2hat_MC], seriestype="vline", label="MC mean", lw = 2, lc = :black, ls = :dashdot)
	plot!([σ2hat_theory], seriestype="vline", label="theoretical mean", lc = :green4, lw = 3, alpha = 0.6)
end

# ╔═╡ f0aee9ca-9b53-40b8-8f9a-0b955362118d
md""" In this model we can still sample from the posterior using Monte Carlo methods. In the next lecture we will look toward approximating unknown distributions when we can't sample directly from them using Markov chain Monte Carlo methods. """

# ╔═╡ 5bf3c91c-cac2-4259-85eb-d798b296355e
md" ## Gaussian with unknown $\mu$ and $\sigma^{2}$ "

# ╔═╡ 9781d63d-23ed-4d43-8446-9a495c31e85d
md"""

In our models thus far we have mostly dealt with one unknown parameter. However, most models deal with more than one parameter. Many models have a rich selection of parameters. With the increasing size of datasets, we are estimating increasingly more complex models and this will increase the parameter space. 

In order to look at a simple multiparameter model we look at the multivariate normal distribution. Below is a three dimensional representation of the mulitivariate normal. 

"""

# ╔═╡ 0c5f78a2-7fbd-4455-a7dd-24766bf78d90
begin
	
	# Uncomment the last two lines for the plot. 
	gr()
	Random.seed!(123)
	
	mvnorm = fit(MvNormal, [rand(0.0:100.0, 100) rand(0.0:100.0, 100)]')
	x₁ = 0:100
	y₁ = 0:(1/length(x₁))+1:101 # because x and y are of different lengths
	z = [pdf(mvnorm, [i, j]) for i in x₁, j in y₁]
	
	plot(x₁, y₁, z, linetype=:wireframe, legend=false, color=:blues, width=0.4)
	plot!(x₁, y₁, z, linetype=:surface, legend=false, color=:ice, alpha = 0.8)
	
end

# ╔═╡ a793de54-cb93-4127-944b-30d23dbd8ff5
md""" #### Marginalisation """

# ╔═╡ b31d550f-3cdf-44ba-b1e6-116cfe84c1c4
md""" 

Please go through this section slowly and make sure you understand everything. I have tried to type out as many of the steps as I could, but perhaps I have skipped some without knowing it.  

Seeing as we now have multiple parameters of interest, we are going to interested in a joint distribution. The joint posterior distribution is given by the following proportional relationship

$$\begin{align*}
      p(\theta_1,\theta_2 \mid y) \propto p(y \mid \theta_1,\theta_2)p(\theta_1,\theta_2)
\end{align*}$$

This joint posterior distribution contains the posterior information about $\mathbf{\theta} = \{\theta_1, \theta_2\}$. Obviously it is going to be difficult to visualise this is we did for the single parameter model. In many instances we are interested in a subset of the parameters, and the other parameters of no real interest to our analysis. These parameters are often referred to as nuisance parameters. The Bayesian solution to this problem is quite natural. We have already established that the posterior is a joint probability distribution, therefore we can simply marginalise out the nuisance parameter through integration. 

Marginalisation entails averaging over the parameter $\theta_2$ to gain access to $\theta_1$, 

$$\begin{align*}
        p(\theta_1 \mid y) = \int p(\theta_1,\theta_2 \mid y) d\theta_2
\end{align*}$$

Using the decomposition $p\left(\theta_{1}, \theta_{2} \mid y\right)=p\left(\theta_{1} \mid \theta_{2}, y\right) p\left(\theta_{2} \mid y\right)$ we can also express this as 

$p\left(\theta_{1} \mid y\right)=\int p\left(\theta_{1} \mid \theta_{2}, y\right) p\left(\theta_{2} \mid y\right) d \theta_{2}$



where $p(\theta_1 \mid y)$ is a marginal distribution. It is a mixture of conditional posterior distributions given the nuisance parameter $\theta_2$. The goal is to find marginal posterior of parameter of interest. We can do this using Monte Carlo approximation.

$$\begin{align*}
      p(\theta_1 \mid y) \approx  \frac{1}{S}\sum_{s=1}^{S} p(\theta_1,\theta_2^{(s)}\mid y)
\end{align*}$$

where the $\theta_2^{(s)}$ values are draws from $p(\theta_2 \mid y)$. We can illustrate this with an example using the Gaussian. In the case of the Gaussian we have the following likelihood model

$$\begin{align*}
    p({y} \mid  \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{1}{2\sigma^2}({ y}-\mu)^2 \right)
\end{align*}$$

We would like to draw from the joint posterior, in other words, we are going to use Monte Carlo methods to draw $\mu^{(s)}, \sigma^{(s)} \sim p(\mu, \sigma  \mid  y)$, where the last part is the joint posterior. We assume that we have a noninformative prior so that $p(\mu,\sigma^2)  \propto \sigma^{-2}$. Note that $p(\mu,\sigma^2)$ is the joint prior here. 

The idea behind the marginalisation concept is depicted in the graph below. Marginal distributions gained from the multivariate normal joint density are depicted below. 

"""

# ╔═╡ 679c11cf-97fd-4cab-b12d-52a2b1166402
begin
	gr()
	Random.seed!(123)
	x₂ = randn(2000)
	marginalkde(x₂, x₂, levels = 15, color = :ice, fill = (0, 0.8, :steelblue3),  lw = 1.1, clip=((-2.5, 2.5), (-2.5, 2.5)))
end

# ╔═╡ 4074ea94-617c-44de-9f88-0f62826acca4
md"""

#### Derivation continued (section optional)

"""

# ╔═╡ 9ca2715b-c6a3-48e5-80b5-131f2eeb0840
md"""

Now let us get ready for some math to show what the posterior will look like given our prior choice and the likelihood function provided. This section still needs some work, the notation is not nearly clear enough. However, we will run through the basic logic in class. 

$$\begin{align*}
    & p(\mu,\sigma^2 \mid y) \propto  \sigma^{-2}\prod_{i=1}^n\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{1}{2\sigma^2}(y_i-\mu)^2\right) \\
    &{=  \sigma^{-n-2}\exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n(y_i-\mu)^2\right)}\\
    &{  = \sigma^{-n-2}\exp\left(-\frac{1}{2\sigma^2}\left[\sum_{i=1}^n(y_i-\bar{y})^2+n(\bar{y}-\mu)^2\right]\right)}\\
    &{\color{gray} \text{where } \bar{y}  \color{gray} = \frac{1}{n}\sum_{i=1}^n y_i }\\
    &{  = \sigma^{-n-2}\exp\left(-\frac{1}{2\sigma^2}\left[(n-1)s^2+n(\bar{y}-\mu)^2\right]\right)}\\
    &{\color{gray} \text{where }  s^2  \color{gray} =\frac{1}{n-1}\sum_{i=1}^n(y_i-\bar{y})^2}
\end{align*}$$

It is useful to know the remember the following trick to complete the derivation. If you have done mathematical statistics you will have seen this before,  


$$\begin{align*}
   &{\sum_{i=1}^n(y_i-\mu)^2}\\
   &{\sum_{i=1}^n(y_i^2-2 y_i \mu + \mu^2)}\\
   &{\sum_{i=1}^n(y_i^2-2 y_i \mu + \mu^2 -\bar{y}^2 + \bar{y}^2 - 2 y_i \bar{y} + 2 y_i \bar{y})}\\
   &{\sum_{i=1}^n(y_i^2-2 y_i \bar{y} + \bar{y}^2) + \sum_{i=1}^n(\mu^2 - 2 y_i \mu -\bar{y}^2  + 2 y_i \bar{y})}\\
   &{\sum_{i=1}^n(y_i-\bar{y})^2 + n(\mu^2 -  2\bar{y}\mu -\bar{y}^2  + 2 \bar{y}\bar{y})}\\
   &{\sum_{i=1}^n(y_i-\bar{y})^2 + n(\bar{y}-\mu)^2}
\end{align*}$$

We take a look at the marginals since we have a funtional form for the joint posterior (as derived above),

$$\begin{align*}
p(\mu \mid y)  = \int p(\mu,\sigma \mid y) d \sigma \\
       {p(\sigma \mid y)  = \int p(\mu,\sigma \mid y) d \mu }\end{align*}$$

The marginal posterior for $p(\sigma^{2} | y)$ (easier for $\sigma^{2}$ than $\sigma$) is the following, 

$$\begin{align*}
    & {p(\sigma^2 \mid y)} \quad {\propto  \int  p(\mu,\sigma^2 \mid y) d\mu} \\
    &{=\int \sigma^{-n-2}\exp\left(-\frac{1}{2\sigma^2}\left[(n-1)s^2+n(\bar{y}-\mu)^2\right]\right) d\mu} \\ 
    &{= \sigma^{-n-2}\exp\left(-\frac{1}{2\sigma^2}(n-1)s^2\right)} \\
    &{\int \exp\left(-\frac{n}{2\sigma^2}(\bar{y}-\mu)^2\right) d\mu}\\
    &{\color{gray} \int \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{1}{2\sigma^2}(y-\theta)^2\right) d\theta = 1} \\
    &{= \sigma^{-n-2}\exp\left(-\frac{1}{2\sigma^2}(n-1)s^2\right)\sqrt{2\pi\sigma^2/n}} \\
    &{= (\sigma^2)^{-(n+1)/2}\exp\left(-\frac{(n-1)s^2}{2\sigma^2}\right)}
\end{align*}$$

From this we can see that $${p(\sigma^2 \mid y) = \text{Inv}\chi^{2}(\sigma^2 \mid n-1,s^2)}$$. With uninformative prior and unknown mean, 

$$\begin{align*}
      \sigma^2 \mid y  \sim \text{Inv}\chi^{2}(n-1,s^2)\\
      \text{where} \quad s^2 =\frac{1}{n-1}\sum_{i=1}^{n}(y_i-\bar{y})^2
  \end{align*}$$

Now we know the marginal distribution for $\sigma$, which means that we can use factorisation of the joint distribution to sample from the joint posterior. The join posterior can be factorised as follows 

$p(\mu,\sigma^2 \mid y) = {\color{darkgreen} p(\mu \mid \sigma^2,y)}{\color{blue} p(\sigma^2 \mid y)}$ 

We have just found the marginal for $\sigma$ as the following, 

${\color{blue} p(\sigma^2 \mid y)}  = \text{Inv}\chi^{2}(\sigma^2 \mid  n-1,s^2)$

This means we can sample from this distribution, 

$(\sigma^2)^{(s)} \sim {\color{blue} p(\sigma^2 \mid y)}$ 

With the variance now known, we know from our previous discussion (Gaussin model with known $\sigma^{2}$ how to sample from the marginal distribution of $\mu$, 

${\color{darkgreen} p(\mu \mid \sigma^2,y)} = \mathcal{N}(\mu \mid \bar{y},\sigma^2/n)\, 
{ \color{gray} {\textstyle \propto \exp\left(-\frac{n}{2\sigma^2}(\bar{y}-\mu)^2\right)}}$

We will then get a sample from this distribution as follows, 

${\mu^{(s)} \sim {\color{darkgreen} p(\mu \mid \sigma^2,y)} }$

In essence we are now done, we can sample from the joint posterior, 

${{\color{red} \mu^{(s)}, \sigma^{(s)}} \sim p(\mu, \sigma  \mid  y)}$

We can write the analytic form of the marginal posterior distribution of $\mu$ here. It is one of the few multiparameter models that is simple enough to solve in closed form.

$$\begin{align*}
     & p(\mu \mid y) =\int_0^\infty p(\mu,\sigma^2 \mid y)d\sigma^2\\
     & { \propto \int_0^\infty \sigma^{-n-2}\exp\left(-{{\color{blue}}\frac{1}{2\sigma^2}\left[{{\color{blue}}(n-1)s^2+n(\bar{y}-\mu)^2}\right]}\right) d\sigma^2}
\end{align*} $$

Transformation

$$\begin{align*}
     & {A={\color{blue}}(n-1)s^2+n(\mu-\bar{y})^2}{\quad \text{and} \quad {{\color{blue}}z=\frac{A}{2\sigma^2}}} \\
     & {p(\mu \mid y) \propto {{\color{blue}}A^{-n/2}}\int_0^\infty {{\color{blue}}z}^{(n-2)/2}\exp(-{{\color{blue}}z})d{{\color{blue}}z}} \\
& \color{gray} \Gamma(u) = \int_0^\infty x^{u-1}\exp(-x)dx \\
    &{\propto {{\color{blue}}[(n-1)s^2+n(\mu-\bar{y})^2]^{-n/2}}}\\
    &{\propto \left[1+\frac{n(\mu-\bar{y})^2}{(n-1)s^2}\right]^{-n/2}} \\
    &{p(\mu \mid y) = t_{n-1}(\mu \mid \bar{y},s^2/n) \color{gray} \quad \text{Student's $t$}}
\end{align*}$$


We will encounter this idea of marginalisation in many of our Markov chain Monte Carlo algorithms, so make sure that you understand the basic logic. This will be something that features in many of the future lectures. 

"""

# ╔═╡ 0e18fe2a-0965-4ef3-b3a8-c0f880e7a719
md"""

## Practical implementation with `Turing.jl`

"""

# ╔═╡ 41d3b953-cdaa-468e-b3b7-bce844726ddb
md""" Before we get into the practical implementation, let us quickly discuss what `Turing.jl` is and what it is doing. For those interested in this package I recommend going to their [main webpage](https://turing.ml/stable/) and doing some of the worked examples there. The alternative to `Turing.jl` in Python is [PyMC3](https://docs.pymc.io/). The best probabilistic programming language for R is [Stan](https://mc-stan.org/users/interfaces/rstan). There are many more packages for the different programming languages. If you want to know more please let me know. 

Probabilistic programming languages like `Turing.jl` allow you to specify variables as random variables. In our case we have worked with Bernoulli, Binomial and Normal random variables. One can include known and unknown parameters in these models. In constructing the model you have specify how the variables relate to each other and then inference of the variables' unknown parameters is performed. 

This means that we will specify priors and likelihoods and let the PPL worry about computing the posterior. """

# ╔═╡ 2e125ab3-136f-44ef-a51b-475aacda96b5
md"""

Inputting the direct functional forms is labour intensive and it is much easier to simply run our model using a PPL. Below is the code to replicate our model with two unknown parameters. We see that the results coincide with out expectation of the mean and variance. 

"""

# ╔═╡ eb0634ba-9bca-4973-ae62-42ef9cd5b6cd
md""" #### Putting together a model """

# ╔═╡ 43f3b350-b61e-4577-b96a-13aea4188551
md" We begin with specfiying a model by using the `@model` macro. Within this model block we can assign variables, either with `~` or `=`. In the first case the variable follows some probability distribution, where in the second case the value is deterministic. 

**Note**: At this stage the notation for the model below does not match up to our derivation, but the idea is right. I just need to relabel things. "

# ╔═╡ 75215065-16a5-4c54-8966-f4bdc8b15054
begin
	ScaledInverseChiSq(ν,τ²) = InverseGamma(ν/2,ν*τ²/2) # Inv-χ² distribution
	
	# Setting up the Turing model:
	@model function iidnormal(x, μ₀, κ₀, ν₀, σ²₀)
	    σ² ~ ScaledInverseChiSq(ν₀, σ²₀)
	    θ ~ Normal(μ₀,σ²/κ₀)  # prior
	    n = length(x)  # number of observations
	    for i in 1:n
	        x[i] ~ Normal(θ, √σ²) # model
	    end
	end
	
	# Set up the observed data
	x_data = [15.77,20.5,8.26,14.37,21.09]
	
	# Set up the prior
	μ₀ = 20; κ₀ = 1; ν₀ = 5; σ²₀ = 5^2
	
	# Settings of the Hamiltonian Monte Carlo (HMC) sampler.
	niter = 2500
	nburn = 1000
	α = 0.65 # target acceptance probability in No U-Turn sampler
	       
	# Sample the posterior using HMC
	postdraws = sample(iidnormal(x_data, μ₀, κ₀, ν₀, σ²₀), NUTS(α), niter, discard_initial = nburn)

end

# ╔═╡ a9052ded-2181-47b0-b9d7-1faa985e7b3a
plot(postdraws, line = 1.7, color = :steelblue, alpha = 0.8)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Compat = "34da2185-b29b-5c13-b0c7-acf172513d20"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
BenchmarkTools = "~0.7.0"
Compat = "~3.39.0"
DataFrames = "~1.2.2"
Distributions = "~0.24.18"
KernelDensity = "~0.6.3"
Plots = "~1.22.4"
PlutoUI = "~0.7.1"
RCall = "~0.13.12"
SpecialFunctions = "~0.10.3"
StatsBase = "~0.33.10"
StatsPlots = "~0.14.28"
Turing = "~0.15.1"
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

[[deps.AbstractPPL]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "ba9984ea1829e16b3a02ee49497c84c9795efa25"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.1.4"

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
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "0ad226aa72d8671f20d0316e03028f0ba1624307"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.32"

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
deps = ["ArgCheck", "ChainRulesCore", "Compat", "Distributions", "Functors", "LinearAlgebra", "MappedArrays", "NNlib", "NonlinearSolve", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "88a1303ee10c24b6df86eeafb98e502115c7be58"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.8.16"

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

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f2202b55d816427cd385a9a4f3ffb226bee80f99"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "fbc5c413a005abdeeb50ad0e54d85d000a1ca667"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.1"

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

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "a851fec56cb73cfdf43762999ec72eff5b86882a"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.15.0"

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
git-tree-sha1 = "31d0151f5716b655421d9d75b7fa74cc4e744df2"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.39.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

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
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

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
git-tree-sha1 = "7220bc21c33e990c14f4a9a319b1d242ebc5b269"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.3.1"

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
deps = ["AbstractMCMC", "AbstractPPL", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "NaturalSort", "Random", "ZygoteRules"]
git-tree-sha1 = "c32726683fc17742ece85ac63e8368b033cffa44"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.10.20"

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

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

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
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "63777916efbcb0ab6173d09a658fb7f2783de485"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.21"

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

[[deps.Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "f40adc6422f548176bb4351ebd29e4abf773040a"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.1.0"

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
git-tree-sha1 = "d189c6d2004f63fd3c91748c458b09f26de0efaa"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.61.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "cafe0823979a5c9bff86224b3b8de29ea5a44b2e"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.61.0+0"

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
git-tree-sha1 = "14eece7a3308b4d8be910e265c724a6ba51a9798"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.16"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "8a954fed8ac097d5be04921d595f741115c1b2ad"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+0"

[[deps.Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "ffdcd4272a7cc36442007bca41aa07ca3cc5fda4"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "1.3.0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3395d4d4aeb3c9d31f5929d32760d8baeee88aaf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.5.0+0"

[[deps.IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InitialValues]]
git-tree-sha1 = "7f6a4508b4a6f46db5ccd9799a3fc71ef5cad6e6"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.11"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

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
git-tree-sha1 = "95d36f32dde312e694c1de5714821efc4b010815"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.7"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "DocStringExtensions", "IfElse", "LinearAlgebra", "OffsetArrays", "Requires", "SLEEFPirates", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "5f275de503982d59bd82eb1e4fbc273f55a72dee"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.10.0"

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

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "0174e9d180b0cae1f8fe7976350ad52f0e70e0d8"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.3.3"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "5a5bc6bf062f0f95e62d0fe0a2d99699fed82dd9"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.8"

[[deps.MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

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
git-tree-sha1 = "4f65bdbbe93475f6ff9ea6969b21532f88d359be"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

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
git-tree-sha1 = "e9ffc92217b8709e0cf7b8808f6223a4a0936c95"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.11"

[[deps.Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "c0e9e582987d36d5a61e650e6e543b9e44d9914b"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.7"

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
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "98f59ff3639b3d9485a03a72f3ab35bab9465720"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.6"

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
git-tree-sha1 = "b084324b4af5a438cd63619fd006614b3b20b87b"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.15"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "ba43b248a1f04a9667ca4a9f782321d9211aa68e"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.22.6"

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
git-tree-sha1 = "d940010be611ee9d67064fe559edbb305f8cc0eb"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.3"

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
git-tree-sha1 = "78aadffb3efd2155af139781b8a8df1ef279ea39"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.2"

[[deps.RCall]]
deps = ["CategoricalArrays", "Conda", "DataFrames", "DataStructures", "Dates", "Libdl", "Missings", "REPL", "Random", "Requires", "StatsModels", "WinReg"]
git-tree-sha1 = "80a056277142a340e646beea0e213f9aecb99caa"
uuid = "6f49c342-dc21-5d91-9882-a32aef131414"
version = "0.13.12"

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
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

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
deps = ["LinearAlgebra", "LoopVectorization"]
git-tree-sha1 = "2e1a88c083ebe8ba69bc0b0084d4b4ba4aa35ae0"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.1.13"

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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Libdl", "VectorizationBase"]
git-tree-sha1 = "3d44bb7517298fd262915924fdc1645c61a6ef17"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.8"

[[deps.SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "0b5b04995a3208f82ae69a17389c60b762000793"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.13.6"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "185e373beaf6b381c1e7151ce2c2a722351d6637"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "2.3.0"

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
git-tree-sha1 = "def0718ddbabeb5476e51e5a43609bee889f285d"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

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
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

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
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "eb35dcc66558b2dda84079b9a1be17557d32091a"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.12"

[[deps.StatsFuns]]
deps = ["Rmath", "SpecialFunctions"]
git-tree-sha1 = "ced55fd4bae008a8ea12508314e725df61f0ba45"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.7"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "5cfe2d754634d9f11ae19e7b45dad3f8e4883f54"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.27"

[[deps.StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "eb007bb78d8a46ab98cd14188e3cec139a4476cf"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.28"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "2ce41e0d042c60ecd131e9fb7154a3bfadbf50d3"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.3"

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
git-tree-sha1 = "fed34d0e71b91734bf0a7e10eb1bb05296ddbcd0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.0"

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

[[deps.ThreadingUtilities]]
deps = ["VectorizationBase"]
git-tree-sha1 = "64a2be7c73951d7c402eb40a16055e5e6fdda468"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.2.3"

[[deps.Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "dec7b7839f23efe21770b3b1307ca77c13ed631d"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.66"

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

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "Hwloc", "IfElse", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "9f27ddc74f04319574749bb0a5b3f231a5e7db16"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.16.2"

[[deps.VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

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
git-tree-sha1 = "80661f59d28714632132c73779f8becc19a113f2"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.4"

[[deps.WinReg]]
deps = ["Test"]
git-tree-sha1 = "808380e0a0483e134081cc54150be4177959b5f4"
uuid = "1b915085-20d7-51cf-bf83-8f477d6f5128"
version = "0.3.1"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9398e8fefd83bde121d5127114bd3b6762c764a6"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.4"

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
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

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
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# ╟─d65de56f-a210-4428-9fac-20a7888d3627
# ╟─9480a241-6bd6-41a7-bda3-406e1fc8d94c
# ╟─82f8bccb-49f4-4c74-8e49-d0e30250b787
# ╟─db39d95e-81c8-4a40-942c-6507d2f08274
# ╟─675dfafa-46cb-44b8-bd7b-55395100e1ca
# ╟─2e83e207-2405-44f4-a5d9-13dd69b741a9
# ╟─961747f8-c884-43bf-941d-4545fb4510e6
# ╠═e3f722c1-739f-45b4-8fbf-db07e9483d92
# ╠═59df263a-a284-40b2-8d9d-bb34fbf13b3a
# ╠═16b216ff-81ed-481a-896b-4ddf2cd139f6
# ╠═6b5886d8-d783-4cbc-9a8c-286b741cb16f
# ╠═09b953aa-7329-48df-b152-a532d997ceea
# ╠═9e6a45a9-07da-4859-861c-db0c6ca30ec1
# ╠═0d10bec1-e358-4b9d-a452-751df45b863c
# ╠═e57f06bd-941b-422a-91e6-004f92ef4b4f
# ╠═4dfbbd9f-0f34-4b7c-a018-b89df5a81dfe
# ╠═aac33160-18d2-4d0e-87c9-1eb9dc363c88
# ╟─9d1a94ac-7ca2-4c8c-a801-021fd2cfc90e
# ╟─880792b2-02b7-444e-a787-d02ee685ee72
# ╟─309c4a9b-cd46-43ed-8a59-001057549fc4
# ╟─963e30cf-16af-44bf-b28e-f7ea4fe96dcc
# ╟─94dc15cd-31ea-46a0-8401-aff7f2a74e5e
# ╟─11a1cec2-2e95-4ebd-a7ce-bbe77f3c9a1d
# ╟─a0981303-008c-4e4a-b96e-581b52ab15f2
# ╟─924ea0e4-0133-4711-9e19-662b4d753e37
# ╟─3ef808e6-c49a-4c25-befe-d7bfd502ff64
# ╠═ceaaccf2-ae65-4731-afc9-f6763bc2b3a1
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─e4730930-c3cd-4a01-a4d9-420bd15004ad
# ╟─f82ce58d-f292-4ecd-86ae-06d4fa79bcd4
# ╟─8ffaa0dc-36f8-49da-9742-74db90c6d5a8
# ╟─5d97bab1-346d-4edd-bc5e-bc6b1a510912
# ╟─39d705ff-4540-4103-ae10-694d5b64e82b
# ╟─f5184e69-55c6-4688-8b0e-d543a07be97b
# ╟─3b2e280f-d83a-4f1c-86bb-1397b95210cb
# ╟─9741e08b-3f54-49d7-8db0-3125f4f90d3c
# ╟─e99e1925-6219-4bf2-b743-bb2ea725dfcd
# ╟─c2897578-b910-41db-886f-a9b8688bff48
# ╠═0fc998e7-d824-412b-a138-b626ba118904
# ╟─01607a33-ad7e-4fab-b5cf-8ddf20a69a52
# ╠═aac22072-c3f2-4b66-bf3f-3dbf967fe6f9
# ╟─2af6f4a8-7b64-405f-ab2e-e77a2699ceb2
# ╟─1316fbc2-700d-4106-99f0-0b9243a99aba
# ╟─3c409c93-545a-4d10-a972-509dff7c0120
# ╠═cef4da14-ee03-44b8-bd86-c11c263b26b3
# ╠═d4f3331d-3dbd-4bad-b043-dff9409005c3
# ╟─ec24134e-a9c0-491e-9a68-c95a618f0b1d
# ╠═572653c4-0e3d-4619-bec6-64a01f4b2e06
# ╠═168df9b2-340c-4e06-8985-d2a1fad60bea
# ╠═74e1be5b-503d-48cb-97fd-c22b634cc361
# ╠═fd9fe53b-3c5c-4e09-837e-bef6f00e485f
# ╟─39b604b9-a4c8-4f54-afaa-faa84d99ad73
# ╟─a0670eb1-2091-47d0-9d9f-bba76834fbed
# ╟─8af7a479-7ac5-4a21-a354-fcaee618367b
# ╟─8010f3a3-bc04-488f-b4de-d613be8f6f84
# ╟─0a4bb68a-f82a-426b-842f-23c405af2e64
# ╟─f691ceb8-31c8-47dc-8ba0-8a4a6bbe811f
# ╟─0d42559a-721e-4084-bbec-911c5c9bc722
# ╟─3160f1bc-85e9-408d-8562-6306b86eace0
# ╟─381154f0-e0bb-4bcf-aba4-81ad30477496
# ╟─116982c6-80c9-4cc8-8140-804a631a77a8
# ╟─9e8aa12a-540b-4153-9dbf-8b503c16b091
# ╟─41915620-c22b-49f3-bb13-299f853dfb9f
# ╠═8f5de37a-6c2c-400d-97c2-7f04e0fa4857
# ╠═e5d0ad86-5bba-4f1c-a2da-b232bff9b4f2
# ╠═0375ebdc-1221-4dc1-89bb-b73a0fd6cd20
# ╟─135aa2d9-05e0-4ab0-8838-5c77326d43b2
# ╟─c79eaac5-f0db-40cf-b32f-3e1be6504549
# ╟─3bbb7b4e-b8cd-4af8-b1c5-dec8008dc448
# ╠═8301f15f-a1e8-4910-abd6-ccd758719381
# ╟─ccf3fce7-436c-46e3-ad62-935240891259
# ╠═ad298e39-3abb-45b8-bc5a-a7e1c8e7024f
# ╟─286e4c9a-3546-41c0-9e71-7a1d89c705f4
# ╟─962b5c8a-dcc0-464c-adad-04e3109dfe4a
# ╟─92c6c3a1-f046-44f5-ae3c-f17d342a9399
# ╟─872dc6b1-0b13-418b-a614-732d2a4d8963
# ╟─d02bd87a-50ef-4e37-b11a-cfc0b023608e
# ╟─d172fe2c-d842-4c6a-9097-3a0e51058bb0
# ╠═6cf5de34-b256-471f-a8c7-ecc14f469967
# ╟─f0aee9ca-9b53-40b8-8f9a-0b955362118d
# ╟─5bf3c91c-cac2-4259-85eb-d798b296355e
# ╟─9781d63d-23ed-4d43-8446-9a495c31e85d
# ╟─0c5f78a2-7fbd-4455-a7dd-24766bf78d90
# ╟─a793de54-cb93-4127-944b-30d23dbd8ff5
# ╟─b31d550f-3cdf-44ba-b1e6-116cfe84c1c4
# ╟─679c11cf-97fd-4cab-b12d-52a2b1166402
# ╟─4074ea94-617c-44de-9f88-0f62826acca4
# ╟─9ca2715b-c6a3-48e5-80b5-131f2eeb0840
# ╟─0e18fe2a-0965-4ef3-b3a8-c0f880e7a719
# ╟─41d3b953-cdaa-468e-b3b7-bce844726ddb
# ╟─2e125ab3-136f-44ef-a51b-475aacda96b5
# ╟─eb0634ba-9bca-4973-ae62-42ef9cd5b6cd
# ╟─43f3b350-b61e-4577-b96a-13aea4188551
# ╠═75215065-16a5-4c54-8966-f4bdc8b15054
# ╠═a9052ded-2181-47b0-b9d7-1faa985e7b3a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
