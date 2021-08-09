### A Pluto.jl notebook ###
# v0.15.1

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
using BenchmarkTools, Distributions, KernelDensity, LinearAlgebra, Plots, PlutoUI, QuadGK, Random, StatsBase, Statistics, StatsPlots, Turing

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
">ATS 872: Lecture 2</p>
<p style="text-align: center; font-size: 1.8rem;">
 Introduction to Bayesian econometrics
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
md" In this session we will be looking at the basics of Bayesian econometrics / statistics. We will start with a discussion on probability and Bayes' rule and then we will move on to discuss single parameter models. Some math will be interlaced with the code. I assume some familiarity with linear algebra, probability and calculus for this module. The section is on probability is simply a high level overview that leads us to our derivation of Bayes' theorem / rule. "

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
TableOfContents() # Uncomment to see TOC

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐ to show support."

# ╔═╡ 4666fae5-b072-485f-bb3a-d28d54fb5274
md" ## Basic probability (preliminaries) "

# ╔═╡ ec9e8512-85fb-4d23-a4ad-7c758cd87b62
md"
The probability of an **event** is a real number between $0$ and $1$. Here $0$ indicates the 'impossible event' and $1$ the certain event. 

> **Note:** Probability $0$ does not mean impossible -- see this [cool video](https://www.youtube.com/watch?v=ZA4JkHKZM50) on this topic. Possibility is tied to the idea of probability density, but we won't go into detail on this idea here. If you are interested in this topic you will have to learn a little [measure theory](https://en.wikipedia.org/wiki/Measure_(mathematics)) to fully understand. If you want to learn more about probability theory in a semi-rigorous set of notes, I recommend the following by [Michael Betancourt](https://betanalpha.github.io/writing/). 

We define $A$ as an event and $P(A)$ as the probability of the event $A$. Therefore, 

$\{P(A) \in \mathbb{R}: 0 \leq P(A) \leq 1\}$

This means that the probability of the event to occur is the set of all real numbers between $0$ and $1$ including $0$ and $1$. We have three axioms of probability, namely, 

1. **Non-negativity**: For all $A$, $P(A) \geq 0$
2. **Additivity**: For two mutually exclusive $A$ and $B$, $P(A) = 1 - P(B)$ and $P(B) = 1 - P(A)$
3. **Normalisation**: Probability of all possible events $A_1, A_2, \ldots$ must add up to one, i.e. $\sum_{n \in \mathbb{N}} A_n = 1$

"

# ╔═╡ 14339a8b-4486-4869-9cf2-6a5f1e4363ac
md" With the axioms established, we are able construct all mathematics pertaining to probability. The first topic of interest to us in the Bayesian school of thought is conditional probability. "

# ╔═╡ 2acbce7f-f528-4047-a2e0-0d7c710e37a1
md" ### Conditional probability "

# ╔═╡ f7592ef8-e966-4113-b60e-559902911e65
md" This is the probability that one event will occur if another has occurred or not. We use the notation $P(A | B)$, which can be read as, the probability that we have observed $A$ given that we have already observed $B$."

# ╔═╡ 6d0bb8df-7211-4fca-b021-b1c35da7f7db
md" We can illustrate with an example. Think about a deck of cards, with 52 cards in a deck. The probability that you are dealt a 🂮 is $4/52$. In other words, $P(🂮 )=\left(\frac{1}{52}\right)$, while the probability of begin a dealt a 🂱 is $P(🂱)=\left(\frac{1}{52}\right)$. However, the probability that you will be dealt 🂱 given that you have been dealt 🂮 is

$P(🂱 | 🂮)=\left(\frac{1}{51}\right)$

This is because we have one less card, since you have already been dealt 🂮. Next we consider joint probability. "

# ╔═╡ 4dfcf3a2-873b-42be-9f5e-0c93cf7220fc
md" ### Joint probability "

# ╔═╡ b8a11019-e87e-4ccb-b9dd-f369ba51a182
md" Join probability is the probability that two events will both occur. Let us extend our card problem to all the kings and aces in the deck. Probability that you will receive an Ace ($A$) and King ($K$) as the two starting cards:

$P(A, K) = P(A) \cdot P(K \mid A)$ 

We can obviously calculate this by using Julia as our calculator... "


# ╔═╡ d576514c-88a3-4292-93a8-8d23edefb2e1
begin
	p_a = 1/ 13 # Probabilty of A
	p_kga = 4 / 51 # Probability of K given A
	p_ak = p_a * p_kga # Joint probability
end

# ╔═╡ 3abcd374-cb1b-4aba-bb3d-09e2819bc842
md" One should note that $P(A, K) = P(K, A)$ and that from that we have 

$P(A) \cdot P(K \mid A) =  P(K) \cdot P(A \mid K)$

One **NB note** here: Conditional probability is not commutative, which means that $P(A \mid B) \neq P(B \mid A)$. In our example above we have some nice symmetry, but this very rarely occurs.  "

# ╔═╡ 411c06a3-c8f8-4d1d-a247-1f0054701021
md" ### Bayes' Theorem "

# ╔═╡ 46780616-1282-4e6c-92ec-5a965f1fc701
md" From the previous example we now know that for two events $A$ and $B$ the following probability identities hold, 

$\begin{aligned} P(A, B) &=P(B, A) \\ P(A) \cdot P(B \mid A) &=P(B) \cdot P(A \mid B) \end{aligned}$

From this we are ready to derive Bayes' rule. 

$\begin{aligned} P(A) \cdot P(B \mid A) &=\overbrace{P(B)}^{\text {this goes to the left}} P(A \mid B) \\ \frac{P(A) \cdot P(B \mid A)}{P(B)} &=P(A \mid B) \\ P(A \mid B) &=\frac{P(A) \cdot P(B \mid A)}{P(B)}  \end{aligned}$

Bayesian statistics uses this theorem as method to conduct inference on parameters of the model given the observed data. "

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Bayesian thinking "

# ╔═╡ 03288125-5ebd-47d3-9e1a-32e2308dbd51
md" Consider an economic model that describes an AR($1$) process

$\begin{equation*} y_{t}=\mu+\alpha y_{t-1}+\varepsilon_{t}, \quad \varepsilon_{t} \sim \mathcal{N}\left[0, \sigma^{2}\right] \end{equation*}$ 

where $\mu$, $\alpha$ and $\sigma^{2}$ are parameters in a vector $\theta$. In the usual time series econometrics course one would try and estimte these unkown parameters with methods such as maximum likelihood estimation, as you did in the first part of the course. 

Unobserved variables are usually called **parameters** and can be inferred from other variables. $\theta$ represents the unobservable parameter of interest, where $y$ is the observed data. Bayesian conclusions about the parameter $\theta$ is made in terms of **probability statements**. Statements are conditional on the observed values of $y$ and can be written $p(\theta \mid y)$: given the data, what do we know about $\theta$? 

**Notation remark**: You will see that we have switched to a small letter $p$ for probability distribution of a random variable. Previously we have used a capital $P$ to relate probability of events. You will often see probability of events written as $\mathbb{P}$ in textbooks as well. 

The Bayesian view is that may be many possible values for $\theta$ from the population of parameter values $\Theta$, while the frequentist view is that only one such a $\theta$ exists. In other words, $\theta$ is regarded as a random variable in the Bayesian setting.

Bayesians will test initial assertions regarding $\theta$ using data on $y$ to investigate probability of assertions. This provides probability distribution over possible values for $\theta \in \Theta$. 

For our model we start with a numerical formulation of joint beliefs about $y$ and $\theta$ expressed in terms of probability distributions.

1. For each $\theta \in \Theta$ the prior distribution $p(\theta)$ describes belief about true population characteristics
2. For each $\theta \in \Theta$ and $y \in \mathcal{Y}$, our sampling model $p(y \mid \theta)$ describes belief that $y$ would be the outcome of the study if we knew $\theta$ to be true.

Once data is obtained, the last step is to update beliefs about $\theta$. For each $\theta \in \Theta$ our posterior distribution $p(\theta \mid y)$ describes our belief that $\theta$ is the true value having observed the dataset.

To make probability statements about $\theta$ given $y$, we begin with a model providing a **joint probability distribution** for $\theta$ and $y$. Joint probability density can be written as product of two densities: the prior $p(\theta)$ and sampling distribution $p(y \mid \theta)$

$p(\theta, y) = p(\theta)p(y \mid \theta)$

However, using the properties of conditional probability we can also write the joint probability density as

$p(\theta, y) = p(y)p(\theta \mid y)$

Setting these equations equal and rearranging provides us with Bayes' theorem / rule, as discussed before. 

$p(y)p(\theta \mid y) = p(\theta)p(y \mid \theta) \rightarrow p(\theta \mid y) = \frac{p(y \mid \theta)p(\theta)}{p(y)}$
"

# ╔═╡ eed2582d-1ba9-48a1-90bc-a6a3bca139ba
md" We are then left with the following formulation:

$$\underbrace{p(\theta \mid y)}_{\text {Posterior }}=\frac{\overbrace{p(y \mid \theta)}^{\text {Likelihood }} \cdot \overbrace{p(\theta)}^{\text {Prior }}}{\underbrace{p(y)}_{\text {Normalizing Constant }}}$$ 

We can safely ignore $p(y)$ in Bayes' rule since it does not involve the parameter of interest $(\theta)$, which means we can write 

$p(\theta|y)\propto p(\theta)p(y|\theta)$

The **posterior density** $p(\theta \mid y)$ summarises all we know about $\theta$ after seeing the data, while the **prior density** $p(\theta)$ does not depend on the data (what you know about $\theta$ prior to seeing data). The **likelihood function** $p(y \mid \theta)$ is the data generating process (density of the data conditional on the parameters in the model). "

# ╔═╡ 0c0d89c0-d70d-42ac-a55c-cd1afbc051ed
md" ### Model vs. likelihood (notational sloppiness) "

# ╔═╡ c6d22671-26e8-4ba3-865f-5cd449a6c9be
md" The following is important to point out, since it can create some confusion (at least I found it confusing at first). The sampling model is shown as $p_{Y}(Y \mid \Theta = \theta) = p(y \mid \theta)$ as a function of $y$ given **fixed** $\theta$ and describes the aleatoric (unknowable) uncertainty.  

On the other hand, likelihood is given as $p_{\Theta}(Y=y \mid \Theta) = p(y \mid \theta) =L(\theta \mid y)$ which is a function of $\theta$ given **fixed** $y$ and provides information about epistemic (knowable) uncertainty, but is **not a probability distribution** 

Bayes' rule combines the **likelihood** with **prior** uncertainty $p(\theta)$ and transforms them to updated **posterior** uncertainty."

# ╔═╡ 34b792cf-65cf-447e-a30e-40ebca992427
md"""

#### What is a likelihood? 

"""

# ╔═╡ b7c5cbe9-a27c-4311-9642-8e0ed80b3d51
md"""

This is a question that has bother me a fair bit since I got started with econometrics. So I will try and give an explanation with some code integrated to give a better idea. This has been mostly inspired by the blog post of [Jim Savage](https://khakieconomics.github.io/2018/07/14/What-is-a-likelihood-anyway.html).

In order to properly fit our Bayesian models we need to construction some type of function that lets us know whether the values of the unknown components of the model are good or not. Imagine that we have some realised values and we can form a histogram of those values. Let us quickly construct a dataset. 

"""

# ╔═╡ f364cb91-99c6-4b64-87d0-2cd6e8043142
data_realised = 3.5 .+ 3.5 .* randn(100) # This is a normally distributed dataset that represents our true data. 

# ╔═╡ 6087f997-bcb7-4482-a343-4c7830967e49
histogram(data_realised, color = :black, legend = false, lw = 1.5,  fill = (0, 0.3, :black), bins  = 20)

# ╔═╡ bf5240e8-59fc-4b1f-8b0d-c65c196ab402
md"""

We would like to fit a density function on this histogram so that we can make probabilistic statements about observations that we have not yet observed. Our proposed density should try and match model unknowns (like to location and scale). Let us observe to potential proposed densities. One is bad guess and the other quite good. 

"""

# ╔═╡ 8165a49f-bd0c-4ad6-8631-cae7425ca4a6
begin
	Random.seed!(1234) 
	bad_guess = -1 .+ 1 .* randn(100);
	good_guess = 3 .+ 3 .* randn(100);
end

# ╔═╡ 5439c858-6ff3-4faa-9145-895390240d76
begin
	density(bad_guess, color = :red,lw = 1.5,  fill = (0, 0.3, :red), title = "Bad Guess")
	histogram!(data_realised, color = :black, legend = false, lw = 1.5,  fill = (0, 0.3, :black), norm = true, bins = 20)
end

# ╔═╡ 0504929d-207c-4fb7-a8b9-14e21aa0f74b
begin
	density(good_guess, color = :steelblue,lw = 1.5,  fill = (0, 0.3, :steelblue), title = "Good Guess")
	histogram!(data_realised, color = :black, legend = false, lw = 1.5,  fill = (0, 0.3, :black), norm = true, bins = 20)
end

# ╔═╡ 169fbcea-4e82-4756-9b4f-870bcb92cb93
md"""

A likelihood function would return a higher value for the proposed density in the case of the good guess. There are many functions that we could use to determine wheter proposed model unknowns result in a better fit. Likelihood functions are one particular approach. Like we mentioned before, these likelihood functions represent a data generating process. The assumption is that the proposed generative distribution gave rise to the data in question. We will continue this discussion on the likelihood function in our next section.  

"""

# ╔═╡ 699dd91c-1141-4fb6-88fc-f7f05906d923
md" ## Bernoulli and Binomial "

# ╔═╡ 6e1de0ff-0fef-48b4-ac5b-0279ea8f2d4d
md" In this section we will be looking at some single parameter models. In other words, models where we only have a single parameter of interest. This will draw on our knowledge from random variables and their distributions in the last lecture.  "

# ╔═╡ 284d0a23-a329-4ea7-a069-f387e21ba797
md"""

### Bernoulli random variable 

"""

# ╔═╡ bb535c41-48cb-44fd-989b-a6d3e310406f
md"""

Let us give a general description of the Bernoulli and Binomial random variables and their relation to each other. We will start with data that has a Bernoulli distribution. In the section on estimating bias in a coin we will work through a specific example. Here we provide the framework.

Consider an experiment (such as tossing a coin) that is repeated $N$ times. Each time we conduct this experiment / trial we can evaluate the outcome as being a success or failure.

In this case the $y_{i}$'s, for $i = 1, \ldots, N$, are random variables for each repetition of the experiment. A random variable is a function that maps the outcome to a value on the real line. In our example, the realisation of $y_{i}$ can be $0$ or $1$ depending on whether the experiment was a success or failure.  

The probability of success is represented by $\theta$, while the probability of failure is given by $1- \theta$. This is considered an Bernoulli event. Our goal is to gain an estimate for $\theta$.

A binary random variable  $y_{i} \in \{0, 1\}$, $0 \leq \theta \leq 1$ follows a Bernoulli distribution if

$p\left(y_{i} \mid \theta\right)=\left\{\begin{array}{cl}\theta & \text { if } y_{i}=1 \\ 1-\theta & \text { if } y_{i}=0\end{array}\right.$

Let $y$ be the number of successes in $N$ repetitions of the experiment then our likelihood function is


$\begin{aligned} p(y \mid \theta) &=\prod_{i=1}^{N} p\left(y_{i} \mid \theta\right) \\ &=\theta^{y_{i}}(1-\theta)^{N-y_{i}} \end{aligned}$


"""

# ╔═╡ 9e7a673e-7cb6-454d-9c57-b6b4f9181b06
md" We can write some code for the likelihood for the binomial random variable. We should have some sort of mental model of what this function looks like when thining about statistical modelling. Consider coin tossing for this Bernoulli distribution. The likelihood function tells us what the probability is of observing a particular sequence of heads and tails if the probability of heads is $\theta$."

# ╔═╡ 5046166d-b6d8-4473-8823-5209aac59c84
begin
	Random.seed!(1244)
	coin_seq = Int.(rand(Bernoulli(0.4), 5))
end

# ╔═╡ 82d0539f-575c-4b98-8679-aefbd11f268e
md"""

Let us say that we think the probability of heads is $0.3$, then our likelihood will be 

$p(y = (1, 0, 0, 1, 1) \mid \theta) = \prod_{i=1}^{N} \theta^{y_{i}} \times (1 - \theta)^{1 - y_{i}} = \theta ^ m (1- \theta) ^{N - m}$

Do we think that the proposed probability of heads is a good one? We can use the likelihood function to perhaps determine this. We plot the values of the likelihood function for this data evaluated over the possible values that $\theta$ can take. 
"""

# ╔═╡ 00cb5973-3047-4c57-9213-beae8f116113
begin
	grid_θ = range(0, 1, length = 1001) |> collect;
	bernoulli(grid_θ, m, N) = (grid_θ .^ m) .* ((1 .- grid_θ) .^ (N - m))
end

# ╔═╡ c0bba3aa-d52c-4192-8eda-32d5d9f49a28
md"""

!!! note "Coin flippling with Bernoulli likelihood"

heads = $(@bind m Slider(1:50, show_value = true, default=3));
flips = $(@bind N Slider(1:50, show_value = true, default=5)); 

"""

# ╔═╡ 9e3c0e01-8eb6-4078-bc0f-019466afba5e
bern  = bernoulli(grid_θ, m, N);

# ╔═╡ ab9195d6-792d-4603-8605-228d479226c6
max_index = argmax(bernoulli(grid_θ, m, N)); # Get argument that maximises this function 

# ╔═╡ e42c5e18-a647-4281-8a87-1b3c6c2abd33
likelihood_max = grid_θ[max_index]; # Value at which the likelihood function is maximised. Makes sense, since we have 3 successes in 5 repetitions. 

# ╔═╡ 5d6a485d-90c4-4f76-a27e-497e8e12afd8
begin
	plot(grid_θ, bern, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Unnormalised likelihood", legend = false)
	plot!([likelihood_max], seriestype = :vline, lw = 2, color = :black, ls = :dash, alpha =0.5, xticks = [likelihood_max])
end

# ╔═╡ 9eaf73e9-0f68-4e8b-9ae1-a42f050f695a
md"""

What do we notice about the likelihood function? 

1. It is **NOT** a probability distribution, since it doesn't integrate to one (we check this with some code above)
2. We notice that in our particular example the function is maximised at $likelihood_max. This maximum point of the likelihood function is known as the maximum likelihood estimate of our parameter given our data. You have dealt with this quantity before. Formally it is, 

$\hat{\theta}_{M L E}=\operatorname{argmax}_{\theta}(p(y \mid \theta))$

This example shows that it can be dangerous to use maximum likelihood with small samples. The true success rate is $0.4$ but our estimate provided a value of $0.6$.

In this case, our prior information (subjective belief) was that the probability of heads should be $0.3$. This could have helped is in this case get to a better estimate, but unfortunately maximimum likelihood does not reflect this prior belief. This means that we are left with a success rate equal to the frequency of occurence. 

"""

# ╔═╡ 828166f7-1a69-4952-9e3b-50a99a99789f
md" #### Estimating bias in a coin (Bernoulli)  "

# ╔═╡ 24c4d724-5911-4534-a5c6-3ab86999df43
md"""
Now we move on to a specific example that uses the knowledge we gained in the previous sections. 

We look at estimating bias in a coin. We observe the number of heads that result from flipping a coin and we estimate its underlying probability of coming up heads. Want to create a descriptive model with meaningful parameters. The outcome of a flip will be given by $y$, with $y=1$ indicating heads and $y = 0$ tails. 

We need underlying probability of heads as value of parameter $\theta$. This can be written as $p(y = 1 \mid \theta) = \theta$. The probability that the outcome is heads, given a parameter value of $\theta$, is the value $\theta$. 

We also need the probability of tails, which is the complement of probability of heads $p(y = 0 \mid \theta) = 1 - \theta$. 

Combine the equations for the probability of heads and tails 

$$\begin{align*}
  p(y \mid \theta)  = \theta^{y}(1-\theta)^{1-y}
\end{align*}$$

We have established this probability distribution is called the Bernoulli distribution. This is a distribution over two discrete values of $y$ for a fixed value of $\theta$. The sum of the probabilities is $1$ (which must be the case for a probability distribution).

$$\begin{align*}
  \sum_{y} p(y \mid \theta) = p(y = 1 \mid \theta) + p(y = 0 \mid \theta) = \theta + (1-\theta) = 1
\end{align*}$$

If we consider $y$ fixed and the value of $\theta$ as variable, then our equation is a **likelihood function** of $\theta$.

This likelihood function is not a probability distribution, suppose that $y = 1$ then $\int_{0}^{1}\theta^{y}(1-\theta)^{1-y}\text{d}\theta = \int_{0}^{1}\theta^{y}\text{d}\theta = 1/2$

Let us take a look at what happens for multiple flips. Outcome of $i$th flip is given by $y_i$ and set of outcomes is $\{y_i\}$. Formula for the probability of the set of outcomes is given by

$$\begin{align*}
  p(y \mid \theta)  =& \prod_{i} p(y_i \mid \theta)  \\
  =& \prod_{i} \theta^{y_i}(1-\theta)^{(1-y_i)} \\
  =& \theta^{\sum_{i} {y_i}}(1-\theta)^{\sum_{i}(1-y_i)} \\
  =& \theta^{\#\text{heads}}(1-\theta)^{\#\text{tails}} \\
	=& \theta^m(1-\theta)^{N - m}
\end{align*}$$

Next we establish the prior, which will be an arbitrary choice here. One assumption could be that the factory producing the coins tends to produce mostly fair coins. Indicate number of heads by $m$ and number of flips by $N$. We need to specify some prior, and we will use the Triangular distribution for our prior in the next section.

Let us code up the likelihood, prior and updated posterior for this example. In order to do this let us implement the grid method. There are many other ways to do this. However, this method is easy to do and gives us some good coding practice.  

"""

# ╔═╡ 11b8b262-32d2-4620-bc6b-afca4a5ce977
md" #### The grid method "

# ╔═╡ 89f7f633-4f75-4ef5-aa5b-80e318d14ee5
md" There are four basic steps behind the grid method.

1. Discretize the parameter space if it is not already discrete.
2. Compute prior and likelihood at each “grid point” in the (discretized) parameter space.
3. Compute (kernel of) posterior as 'prior $\times$ likelihood' at each “grid point”.
4. Normalize to get posterior, if desired."

# ╔═╡ 11552b20-3407-4d0b-b07d-1488c8e8a759
md" The first step then in the grid method is to create a grid. The parameter is $\theta$ and we will discretise by selecting grid points between $0$ and $1$. For our first example let us choose $1001$ grid points. "

# ╔═╡ 599c2f09-ad5e-4f39-aa7d-c1ba155725d6
coins_grid = range(0, 1, length = 1001) |> collect;

# ╔═╡ 09ec10d9-a604-480d-8e82-59e84a843749
md" Now we will add a triangular prior across the grid points with a peak at $\theta = 0.5$ and plot the resulting graph."

# ╔═╡ 071761f8-a187-47a6-8fee-5fc91e65d04c
m₁ = 1 # Number of heads

# ╔═╡ f001b040-2ae7-4e97-b229-eebaabb537b0
N₁ = 4 # Number of flips

# ╔═╡ 9a2d5bdf-9597-40c7-ac18-bb27f187912d
triangle_prior = TriangularDist(0, 1); # From the Distributions.jl package

# ╔═╡ f6e6c4bf-9b2f-4047-a6cc-4ab9c3ae1420
plot(triangle_prior, coins_grid, xlab = "theta", ylab = "prior", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))

# ╔═╡ 8382e073-433b-4e42-a6fa-d5a051586457
md" In this small dataset we have $1$ success out of $4$ attempts. Our distribution function will calculate the probability that we want for a given value of $\theta$. We want to do this for each value of $\theta$, but using same values for $y$ each time. This is the hardest part of our computation (for this example, not in general). "

# ╔═╡ 9678396b-d42c-4c7c-821c-08126895efd3
bern₁ = bernoulli(grid_θ, m₁, N₁);

# ╔═╡ 0a1d46ed-0295-4000-9e30-3ad838552a7e
begin
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))
	plot!(coins_grid, bern₁, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Unnormalised likelihood")
end

# ╔═╡ e5aade9a-4593-4903-bc3a-3a37f9f71c98
md"""
We can normalise the likelihood for the purpose of plotting. We can do this by dividng by the sum of the likelihoods and by the width of the spaces betwen the grid points.
"""

# ╔═╡ 87db6122-4d28-45bf-b5b0-41189792199d
likelihood_norm = bern₁ / sum(bern₁) / 0.001; # Normalised

# ╔═╡ c6e9bb86-dc67-4f42-89da-98581a0c3c98
md" Likelihoods are **NOT** probability mass functions or probability density functions so the total area under the likelihood function is not generally going to be $1$.  "

# ╔═╡ b81924b8-73f6-4b28-899c-ec417d538dd4
begin
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Normalised likelihood")
end

# ╔═╡ c3d2ba03-e676-4f0f-bafd-feecd0e4e414
md" The hardest part is done, now we only need to multiply by the prior and likelihood to get the posterior. "

# ╔═╡ 97c0c719-fb73-4571-9a6c-629a98cc544d
prior = pdf(triangle_prior, coins_grid); # Extract the values for the prior at the grid points

# ╔═╡ 0b3945a8-0ae3-4c18-a9b7-a249eb530bcb
posterior = prior .* likelihood_norm; # Calculate the posterior

# ╔═╡ 4e790ffa-554d-4c46-af68-22ecb461fb7b
begin
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior, color = :black,lw = 2,  fill = (0, 0.4, :green))
end

# ╔═╡ 4b141ffc-4100-47e3-941a-4e72c784ccf0
md" Play around with the sliders here so that you can see what happens to the posterior once it gets updated with new information from the data. "

# ╔═╡ 219aafcb-17b1-4f5f-9c2b-9b713ba78b18
md"""
heads = $(@bind y₂ Slider(1:10, show_value = true, default=1));
flips = $(@bind n₂ Slider(1:10, show_value = true, default=4))
"""

# ╔═╡ 2833e081-45d6-4f64-8d1e-b3a5895b7952
begin
	b₂ = Binomial.(n₂, coins_grid)
	likelihood_2 = pdf.(b₂, y₂)
	likelihood_norm_2 = likelihood_2 / sum(likelihood_2) / 0.001
	posterior_2 = prior .* likelihood_norm;
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm_2, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior_2, color = :black,lw = 2,  fill = (0, 0.4, :green))
end

# ╔═╡ e6a32850-6656-4482-b57b-61f774896514
md"""

We will think about some other potential priors and how they affect the posterior soon. However, we will first look at the Binomial random variable and its relation to the Bernoulli random variable. """

# ╔═╡ fe8f71b2-4198-4a12-a996-da254d2cc656
md" ### Binomial random variable "

# ╔═╡ 7e89eee0-dd19-4fec-b7c0-7783a9ffb83c
md"""

It is worthwhile mentioning the Binomial distribution at this stage. The Bernoulli distribution represents the success or failure of a **single Bernoulli trial**. The Binomial distribution represents the number of successes and failues in $n$ independent Bernoulli trials for some given value of $n$. 

The probability of several events in independent trials is $\theta \cdot \theta(1-\theta)\cdot\theta(1-\theta)(1-\theta)\ldots$ 

If there are $n$ trials then the probability that a success occurs $y$ times is

$$\begin{align*}
      p(y \mid \theta, n) & = \frac{n!}{y!(n-y)!} \theta^y(1-\theta)^{n-y} \\
      &= \binom{n}{y} \theta^y(1-\theta)^{n-y}
    \end{align*}$$

We can easily use the `Distributions.jl` package discussed in the previous lecture to draw from the distribution, but let us try and see what happens if we try and code this up ourselves. 
"""

# ╔═╡ f45eb380-7b43-4fd0-af34-89ffd126a63f
md" Working with distributions is a big part of Bayesian statistics, so for our first example today we will write a function that generates draws from a **binomial distribution**."

# ╔═╡ 4084f646-bce6-4a21-a529-49c7774f5ad1
md" The binomial random variable $y \sim \text{Bin}(n, p)$ represents the number of successes in $n$ binary trials where each trial succeeds with probability $p$. We are going to use the `rand()` command to write a function called `binomial_rv` that generates one draw of $y$. "

# ╔═╡ 3c49d3e4-3555-4d18-9445-5347247cf639
function binomial_rv(n, p)
    count = 0
    for i in 1:n
        if rand(n)[i] < p
            count += 1 # or count = count + 1
        end
    end
    return count
end

# ╔═╡ 98db344c-2ada-4781-bb4a-f3ec2ea7ccfd
md" Given a value of $n = 10000$ indepedent trials, how many times will we observe success? "

# ╔═╡ f7b158af-537e-4d9f-9c4c-318281097dce
binomial_rv(10000, 0.5) # Compare this with the time it takes to run in R. 

# ╔═╡ 69a1f4bb-35f6-42bf-9a2a-e3631bf4e43e
md" Now let us conduct some experiments with our new binomial random variable. "

# ╔═╡ b6da2479-1545-4b1d-8d7f-07d6d1f67635
md"""

!!! note "Interactive sliders for Binomial random variable"

n = $(@bind n Slider(1:1:100, show_value = true, default=1));
p = $(@bind p Slider(0:0.01:1, show_value = true, default=0.5)); 

> Shift these sliders around to see what happens to the graph below. Try fixing values for $p$ and increase the number of $n$, what happens to the distribution? What theorem is at play here?

"""

# ╔═╡ d6316b4f-9882-4d25-87d0-31fa3c1f3935
b = [binomial_rv(n, p) for _ in 1:1000]; # Using an array comprehension

# ╔═╡ c4cc482b-815b-4747-9f5a-5779d69086f7
begin
	#dens_b = kde(b)
	#plot(dens_b, line = 2, color = :black, legend = false, norm = true)
	histogram(b, alpha = 0.5, c = :steelblue, legend = false, size = (700, 500), norm = true)
end

# ╔═╡ 9016cba4-58f0-4b7f-91af-66faaf3fe99c
md" Naturally, one would want to use a pre-packaged solution to sampling with a binomial random variable. The `Distributions.jl` package contains optimised routines that work faster than our code, but is still a good idea to code some things yourself to fully understand the mechanisms. " 

# ╔═╡ 6b1e8fc3-48ee-471b-9c04-7c75cfef156c
# @benchmark (rand(Binomial(n, p), 1000)  # Will generally give the same result as above, but likely much faster. 

# ╔═╡ 2eb59993-4ace-4acb-9810-ba064ea1eb3e
# @benchmark [(rand(Binomial(n, p))) for _ in 1:1000] # What is different with this?

# ╔═╡ 7c04e47c-eeed-47ec-9c6f-e2b710d0b746
# @benchmark [binomial_rv(n, p) for _ in 1:1000] # We can see from our benchmarking that this is much slower. 

# ╔═╡ 79b45389-fa2a-46df-9869-1992c8afb397
md"""

Now that we know more about the Binomial random variable, let us get back to our discussion on priors. 

"""

# ╔═╡ cb53475c-cc56-46b3-94b0-3ded33eb18d4
md"""
### Eliciting a prior 



"""

# ╔═╡ 75ef279d-2c7b-4776-b93f-5b28cbc67f63
md""" We start our discussion with the idea of prior elicitation. This is basically extracting a prior distribution from a person, normally an expert. Common practice is to settle on a distributional family and then elicit **hyperparameters** within the family.

The Beta distribution, for example, is a two parameter family with great flexibility. For different values of the parameters $a$ and $b$ we get different functional forms for the distribution. We refer to the parameters as hyperparemeters in Bayesian econometrics. One of the things that the researcher might have some information is the expected value of $\theta$.

"""

# ╔═╡ 2844b7a6-002e-4459-9e37-30e3a16c88f0
md" Here are some nice facts about the Beta distribution, you don't need to memorise these. 

$$\begin{equation}
\begin{array}{ll}
\text { Notation } & \operatorname{Beta}(a, b) \\
\hline \text { Parameters } & \begin{array}{l}
a>0 \text { shape (real) } \\
b>0 \text { shape (real) }
\end{array} \\
\hline \text { Support } & x \in[0,1] \text { or } x \in(0,1) \\
\text { PDF } & \frac{x^{a-1}(1-x)^{b-1}}{\mathrm{~B}(a, b)} \\
\hline \text { Mean } & \frac{a}{a+b} \\
\hline \text { Mode } & \frac{a-1}{a+b-2} \text { for } a, b>1 \\
& 0 \text { for } a=1, b>1 \\
& 1 \text { for } a>1, b=1 \\
\hline \text { Variance } & \frac{a b}{(a+b)^{2}(a+b+1)} \\
\text { Concentration } & \kappa=a+b
\end{array}
\end{equation}$$
"

# ╔═╡ 68bb2bfb-6643-4f59-9d2b-c59fd1dc3273
md"""
In the case of the Beta distribution the prior mean is given above as 

$\frac{a}{a + b}$

The prior mean will, by the fact that the prior is conjugate, also translate to a posterior distribution that has a Beta functional form. Therefore, if you choose the values for $a$ and $b$ properly you are in fact stating something about $\mathbb{E}(\theta)$.

Suppose you believe that $\mathbb{E}(\theta) = 1/2$. This can be obtained by setting $a = b$. 

As an example, set $a = b = 2$, then we have 

$\mathbb{E}(\theta) = \frac{2}{2+2} = 1/2$

We could also choose a completely noninformative prior with $a = b = 1$, which implies that $p(\theta) \propto 1$. This is simply a uniform distribution over the interval $[0, 1]$. Every value for $\theta$ receives the same probability. 

Obviously there are multiple values of $a$ and $b$ will work, play around with the sliders above to see what happens for a choice of different $a$ and $b$ under this restriction for the expected value of $\theta$.
In the case of the Beta distribution the prior mean is given above as 

$\frac{a}{a + b}$

The prior mean will, by the fact that the prior is conjugate, also translate to a posterior distribution that has a Beta functional form. Therefore, if you choose the values for $a$ and $b$ properly you are in fact stating something about $\mathbb{E}(\theta)$.

Suppose you believe that $\mathbb{E}(\theta) = 1/2$. This can be obtained by setting $a = b$. 

As an example, set $a = b = 2$, then we have 

$\mathbb{E}(\theta) = \frac{2}{2+2} = 1/2$

We could also choose a completely noninformative prior with $a = b = 1$, which implies that $p(\theta) \propto 1$. This is simply a uniform distribution over the interval $[0, 1]$. Every value for $\theta$ receives the same probability. 

Obviously there are multiple values of $a$ and $b$ will work, play around with the sliders above to see what happens for a choice of different $a$ and $b$ under this restriction for the expected value of $\theta$.

"""

# ╔═╡ 5c714d3b-ac72-40dc-ba98-bb2b24435d4c
md"""

#### Coin flipping contd. 

"""

# ╔═╡ 573b8a38-5a9b-4d5f-a9f6-00a5255914f0
md"""
In our coin flipping model we have derived the posterior credibilities of parameter values given certain priors. Generally, we need a mathematical description of the **prior probability** for each value of the parameter $\theta$ on interval $[0, 1]$. Any relevant probability density function would work, but there are two desiderata for mathematical tractability.

1. Product of $p(y \mid \theta)$ and $p(\theta)$ results in same form as $p(\theta)$.
2. Necesarry for $\int p(y \mid \theta)p(\theta) \text{d} \theta$ to be solvable analytically

When the forms of $p(y \mid \theta)$ and $p(\theta)$ combine so that the posterior has the same form as the prior distribution then $p(\theta)$ is called **conjugate prior** for $p(y \mid \theta)$. 

Prior is conjugate with respect to particular likelihood function. We are looking for a functional form for a prior density over $\theta$ that is conjugate to the **Bernoulli / Binomial likelihood function**.

If the prior is of the form, $\theta^{a}(1-\theta)^b$ then when you multiply with Bernoulli likelihood you will get

$$\begin{align*}
  \theta^{y + a}(1-\theta)^{(1-y+b)}
\end{align*}$$

A probability density of this form is called the Beta distribution. Beta distribution has two parameters, called $a$ and $b$.

$$\begin{align*}
  p(\theta \mid a, b) =& \text{Beta}(\theta \mid a, b) \\
  =& \frac{\theta^{a-1}(1-\theta)^{(b-1)}}{B(a,b)}
\end{align*}$$

In this case $B(a,b)$ is a normalising constant, to make sure area under Beta density integrates to $1$. 

Beta function is given by $\int_{0}^{1} \theta^{a-1}(1-\theta)^{(b-1)}\text{d}\theta$.

Another way in which we can express the Beta function,

$$\begin{align*}
  B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)  \quad \text{where} \quad \Gamma(a) = \int_{0}^{\infty} t^{(a-1)}\text{exp}(-t)
\end{align*}$$

The variables $a$ and $b$ are called the shape parameters of the Beta distribution (they determine the shape). We can use the `Distributions.jl` package to see what Beta looks like for different values of $a$ and $b$ over $\theta$.

"""

# ╔═╡ 1ca20976-757f-4e30-94d4-ee1276a614fb
md"""
a = $(@bind α Slider(1:0.1:4, show_value = true, default=1)); 
b = $(@bind β Slider(1:1:4, show_value = true, default=1))
"""

# ╔═╡ aa69d0e8-cbbb-436c-b488-5bb113cdf97f
prior_beta = Beta(α, β);

# ╔═╡ 84d7e4dd-23a9-4412-a8de-ab8ee8351770
plot(prior_beta, coins_grid, xlab = "theta", ylab = "prior", color = :black, legend = false, lw = 1.5,  fill = (0, 0.4, :black))

# ╔═╡ 43d563ae-a435-417f-83c6-19b3b7d6e6ee
md"""
a1 = $(@bind α₁ Slider(1:0.1:4, show_value = true, default=1));
b1 = $(@bind β₁ Slider(1:1:4, show_value = true, default=1))
"""

# ╔═╡ 448b4ddf-5be2-4843-8e8e-4afb60aa8843
md" Using this Beta distribution then as prior will give us the following posterior " 

# ╔═╡ 11a5614b-c195-45a8-8be0-b99fda6c60fd
begin
	prior_beta₁ = Beta(α₁, β₁)
	prior_beta_pdf = pdf(prior_beta₁, coins_grid); # Beta distribution
	posterior_beta = prior_beta_pdf .* likelihood_norm;
	plot(prior_beta₁, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm_2, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior_beta, color = :black,lw = 2,  fill = (0, 0.4, :green))
end

# ╔═╡ f004ec01-1e27-4e30-9a53-23a299208846
md" Initially, with $a = 1$ and $b = 1$ this will be the same as the uniform prior. However, play around with the values on the slider to see how it changes for a different parameterisation of the Beta distribution. "

# ╔═╡ e7669fea-5aff-4522-81bf-3356ce126b1f
md"""

## Analytical derivation (Bernoulli)

"""

# ╔═╡ a32faf6c-a5bb-4005-ad42-188af732fba5
md"""
We have established the Beta distribution as a convenient prior for the Bernoulli likelikood function. Now we can figure out, mathematically, what the posterior would look like when we apply Bayes' rule. Suppose we have set of data with $N$ flips and $m$ heads, then we can calculate the posterior as,

$$\begin{align*}
  p(\theta \mid m, N) =& p(m, N \mid \theta)p(\theta)/p(m, N) \\
  =& \theta^{m}(1-\theta)^{(N-m)}\frac{\theta^{a-1}(1-\theta)^{(b-1)}}{B(a,b)} /p(m, N) \\
  =& \theta^{m}(1-\theta)^{(N-m)}{\theta^{a-1}(1-\theta)^{(b-1)}} / [B(a,b)p(m, N)] \\
  =& \theta^{((N + a) -1)}(1-\theta)^{((N-m + b)-1)}/ [B(a,b)p(m, N)] \\
  =& \theta^{((m + a) -1)}(1-\theta)^{((N-m + b)-1)}/ B(m + a, N-m+b)
\end{align*}$$

Last step was made by considering what the normalising factor should be for the numerator of the Beta distribution. From this we see that if prior is $\text{Beta}(\theta \mid a,b)$ then the posterior will be $\text{Beta}(\theta \mid m + a, N - m + b)$. Multiplying the likelihood and prior leads to a posterior with the same form as the prior. We refer to this as a **conjugate prior** (for a particular likelihood function). Beta priors are conjugate priors for the Bernoulli likelihood. If we use the Beta prior, we will in turn receive a Beta posterior. 

From this we can see that posterior is a compromise between the prior and likelihood. We illustrated this with graphs in the previous sections, but now we can see it analytically. For a $\text{Beta}(\theta \mid a, b)$ prior distribution, the prior mean of $\theta$ is $\frac{a}{a+b}$. If we observe $m$ heads in $N$ flips, this results in a proportion of $m/N$ heads in the data.  The posterior mean is 

$\frac{(m + a)}{m + a + N - m + b} = \frac{m+a}{N+a+b}$

This can then be rearranged algebraically into a weighted average of the prior mean and data proportion, 

$\underbrace{\frac{m+a}{N+a+b}}_{\text {posterior }}=\underbrace{\frac{m}{N}}_{\text {data }} \underbrace{\frac{N}{N+a+b}}_{\text {weight }}+\underbrace{\frac{a}{a+b}}_{\text {prior }} \underbrace{\frac{a+b}{N+a+b}}_{\text {weight }}$

This indicates that the posterior mean is somewhere between the prior mean and the proportion in the data. The more data we have, the less influence of the prior. 


"""

# ╔═╡ 92a4aa17-2e2d-45c2-a9a2-803d389077d5
md" ## Coin toss with `Turing.jl` 🤓"

# ╔═╡ 0a3ed179-b60b-4740-ab73-b176bba08d84
md" In this section I will construct a coin tossing model in `Turing.jl`. You don't need to worry for now what this package is, we will devote an entire session to it later in the course. However, let us just run the example and I will explain what is going on in class. 

Important to note that in this example we are utilising the Binomial distribution, since we are working with multiple independent instances of the coin tossing experiment. "

# ╔═╡ 0259f04b-6739-4803-a3eb-4641a6af8361
md" First, we need to construct our model. We specify the prior and then the likelihood function. " 

# ╔═╡ eeb9d3b0-ab6b-49fd-9d3c-87489ccd7c26
begin
	Random.seed!(1)
	
	y₃ = 1
	n₃ = 4
	
	@model function coin_toss(n, y)
	    θ ~ Beta(1, 1) # Prior 
	    y ~ Binomial(n, θ) # Likelihood (model)
	    return y, θ
	end
	
	chns = sample(coin_toss(n₃, y₃), NUTS(), 1000) # Using the No U Turn Sampler
end

# ╔═╡ 54f47150-282e-434c-a588-c7c530c438b9
StatsPlots.plot(chns)

# ╔═╡ bf1a74e4-cf55-470e-843d-a8e6b90517e9
md" We will take a look at a more involved coin toss example later in the course, but this should be easy enough to understand. "

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuadGK = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
BenchmarkTools = "~1.1.1"
Distributions = "~0.25.11"
KernelDensity = "~0.6.3"
Plots = "~1.19.4"
PlutoUI = "~0.7.9"
QuadGK = "~2.4.1"
StatsBase = "~0.33.9"
StatsPlots = "~0.14.26"
Turing = "~0.16.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "db0a7ff3fbd987055c43b4e12d2fa30aaae8749c"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "3.2.1"

[[AbstractPPL]]
deps = ["AbstractMCMC"]
git-tree-sha1 = "ba9984ea1829e16b3a02ee49497c84c9795efa25"
uuid = "7a57a42e-76ec-4ea3-a279-07e840d6d9cf"
version = "0.1.4"

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[AdvancedHMC]]
deps = ["AbstractMCMC", "ArgCheck", "DocStringExtensions", "InplaceOps", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "Setfield", "Statistics", "StatsBase", "StatsFuns", "UnPack"]
git-tree-sha1 = "38dc9bd338445735b7c11b07ddcfe5a117012e5e"
uuid = "0bf59076-c3b1-5ca4-86bd-e02cd72cde3d"
version = "0.3.0"

[[AdvancedMH]]
deps = ["AbstractMCMC", "Distributions", "Random", "Requires"]
git-tree-sha1 = "6fcaabc5def4dcb20218a12c73a261090182b0c1"
uuid = "5b7e9947-ddc0-4b3f-9b55-0d8042f74170"
version = "0.6.3"

[[AdvancedPS]]
deps = ["AbstractMCMC", "Distributions", "Libtask", "Random", "StatsFuns"]
git-tree-sha1 = "06da6c283cf17cf0f97ed2c07c29b6333ee83dc9"
uuid = "576499cb-2369-40b2-a588-c64705576edc"
version = "0.2.4"

[[AdvancedVI]]
deps = ["Bijectors", "Distributions", "DistributionsAD", "DocStringExtensions", "ForwardDiff", "LinearAlgebra", "ProgressMeter", "Random", "Requires", "StatsBase", "StatsFuns", "Tracker"]
git-tree-sha1 = "130d6b17a3a9d420d9a6b37412cae03ffd6a64ff"
uuid = "b5ca4192-6429-45e5-a2d9-87aec30a685c"
version = "0.1.3"

[[ArgCheck]]
git-tree-sha1 = "dedbbb2ddb876f899585c4ec4433265e3017215a"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.1.0"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra"]
git-tree-sha1 = "2ff92b71ba1747c5fdd541f8fc87736d82f40ec9"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.4.0"

[[Arpack_jll]]
deps = ["Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "e214a9b9bd1b4e1b4f15b22c0994862b66af7ff7"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.0+3"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "be3671b34caec1d28a7915ca59cf8ba5a89a34fb"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.20"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "e239020994123f08905052b9603b4ca14f8c5807"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.31"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "c31ebabde28d102b602bada60ce8922c266d205b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.1"

[[Bijectors]]
deps = ["ArgCheck", "ChainRulesCore", "Compat", "Distributions", "Functors", "LinearAlgebra", "MappedArrays", "NNlib", "NonlinearSolve", "Random", "Reexport", "Requires", "SparseArrays", "Statistics", "StatsFuns"]
git-tree-sha1 = "f032f0b27318b0ea5e35fc510759971fbba65179"
uuid = "76274a88-744f-5084-9051-94815aaf08c4"
version = "0.9.7"

[[Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c3598e525718abcc440f69cc6d5f60dda0a1b61e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.6+5"

[[Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "e2f47f6d8337369411569fd45ae5753ca10394c6"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.0+6"

[[ChainRules]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "0902fc7f416c8f1e3b1e014786bb65d0c2241a9b"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "0.8.24"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f53ca8d41e4753c41cdafa6ec5f7ce914b34be54"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.13"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random", "StaticArrays"]
git-tree-sha1 = "ed268efe58512df8c7e224d2e170afd76dd6a417"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.13.0"

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

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

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

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DefineSingletons]]
git-tree-sha1 = "77b4ca280084423b728662fe040e5ff8819347c5"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.1"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "85d2d9e2524da988bffaf2a381864e20d2dae08d"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.2.1"

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

[[DistributionsAD]]
deps = ["Adapt", "ChainRules", "ChainRulesCore", "Compat", "DiffRules", "Distributions", "FillArrays", "LinearAlgebra", "NaNMath", "PDMats", "Random", "Requires", "SpecialFunctions", "StaticArrays", "StatsBase", "StatsFuns", "ZygoteRules"]
git-tree-sha1 = "1c0ef4fe9eaa9596aca50b15a420e987b8447e56"
uuid = "ced4e74d-a319-5a8a-b0ac-84af2272839c"
version = "0.6.28"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DynamicPPL]]
deps = ["AbstractMCMC", "AbstractPPL", "Bijectors", "ChainRulesCore", "Distributions", "MacroTools", "Random", "ZygoteRules"]
git-tree-sha1 = "94c766fb4432d359a6968094ffce36660cbaa05a"
uuid = "366bfd00-2699-11ea-058f-f148b4cae6d8"
version = "0.12.4"

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

[[EllipticalSliceSampling]]
deps = ["AbstractMCMC", "ArrayInterface", "Distributions", "Random", "Statistics"]
git-tree-sha1 = "254182080498cce7ae4bc863d23bf27c632688f7"
uuid = "cad2338a-1db2-11e9-3401-43bc07c9ede2"
version = "0.4.4"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "LibVPX_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "3cc57ad0a213808473eafef4845a74766242e05f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.3.1+4"

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

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "8b3c09b56acaf3c0e581c66638b85c8650ee9dca"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.8.1"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "35895cf184ceaab11fd778b4590144034a167a2f"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.1+14"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "NaNMath", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "b5e930ac60b613ef3406da6d4f42c35d8dc51419"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.19"

[[FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "cbd58c9deb1d304f5a245a0b7eb841a2560cfec6"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.1+5"

[[FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[Functors]]
deps = ["MacroTools"]
git-tree-sha1 = "4cd9e70bf8fce05114598b663ad79dfe9ae432b3"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.2.3"

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
git-tree-sha1 = "9f473cdf6e2eb360c576f9822e7c765dd9d26dbc"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.58.0"

[[GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "eaf96e05a880f3db5ded5a5a8a7817ecba3c7392"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.58.0+0"

[[GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "58bcdf5ebc057b085e58d95c138725628dd7453c"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.1"

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

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "c6a1fff2fd4b1da29d3dccaffb1e1001244d844e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.12"

[[Hwloc]]
deps = ["Hwloc_jll"]
git-tree-sha1 = "92d99146066c5c6888d5a3abc871e6a214388b91"
uuid = "0e44f5e4-bd66-52a0-8798-143a42290a1d"
version = "2.0.0"

[[Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3395d4d4aeb3c9d31f5929d32760d8baeee88aaf"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.5.0+0"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[InitialValues]]
git-tree-sha1 = "26c8832afd63ac558b98a823265856670d898b6c"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.2.10"

[[InplaceOps]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "50b41d59e7164ab6fda65e71049fee9d890731ff"
uuid = "505f98c9-085e-5b2c-8e89-488be7bf1f34"
version = "0.3.0"

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
git-tree-sha1 = "1470c80592cf1f0a35566ee5e93c5f8221ebc33a"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.3"

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

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "1a8c6237e78b714e901e406c096fc8a65528af7d"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.1"

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
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

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

[[LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "71be1eb5ad19cb4f61fa8c73395c0338fd092ae0"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.2"

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

[[LibVPX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "12ee7e23fa4d18361e7c2cde8f8337d4c3101bc7"
uuid = "dd192d2f-8180-539f-9fb4-cc70b1dcf69a"
version = "1.10.0+0"

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

[[Libtask]]
deps = ["Libtask_jll", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "90c6ed7f9ac449cddacd80d5c1fca59c97d203e7"
uuid = "6f1fad26-d15e-5dc8-ae53-837a1d7b8c9f"
version = "0.5.3"

[[Libtask_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "901fc8752bbc527a6006a951716d661baa9d54e9"
uuid = "3ae2931a-708c-5973-9c38-ccf7496fb450"
version = "0.4.3+0"

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
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "dfeda1c1130990428720de0024d4516b1902ce98"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.7"

[[LoopVectorization]]
deps = ["ArrayInterface", "DocStringExtensions", "IfElse", "LinearAlgebra", "OffsetArrays", "Polyester", "Requires", "SLEEFPirates", "Static", "StrideArraysCore", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "2daac7e480432fd48fb05805772ba018053b935e"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.59"

[[MCMCChains]]
deps = ["AbstractFFTs", "AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "LinearAlgebra", "MLJModelInterface", "NaturalSort", "PrettyTables", "Random", "RecipesBase", "Serialization", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "09e3390e2c9825ec1cdcacaa470f738b7ed61ae0"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "4.13.1"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "55c785a68d71c5fd7b64b490e0d9ab18cf13a04c"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.1.1"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[ManualMemory]]
git-tree-sha1 = "71c64ebe61a12bad0911f8fc4f91df8a448c604c"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.4"

[[MappedArrays]]
git-tree-sha1 = "18d3584eebc861e311a552cbb67723af8edff5de"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.0"

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

[[MicroCollections]]
deps = ["BangBang", "Setfield"]
git-tree-sha1 = "e991b6a9d38091c4a0d7cd051fcb57c05f98ac03"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "8d958ff1854b166003238fe191ec34b9d592860a"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.8.0"

[[NNlib]]
deps = ["Adapt", "ChainRulesCore", "Compat", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "d27c8947dab6e3a315f6dcd4d2493ed3ba541791"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.7.26"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NamedArrays]]
deps = ["Combinatorics", "DataStructures", "DelimitedFiles", "InvertedIndices", "LinearAlgebra", "Random", "Requires", "SparseArrays", "Statistics"]
git-tree-sha1 = "2fd5787125d1a93fbe30961bd841707b8a80d75b"
uuid = "86f7a689-2022-50b4-a561-43c23ac3c673"
version = "0.9.6"

[[NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[NonlinearSolve]]
deps = ["ArrayInterface", "FiniteDiff", "ForwardDiff", "IterativeSolvers", "LinearAlgebra", "RecursiveArrayTools", "RecursiveFactorization", "Reexport", "SciMLBase", "Setfield", "StaticArrays", "UnPack"]
git-tree-sha1 = "ef18e47df4f3917af35be5e5d7f5d97e8a83b0ec"
uuid = "8913a72c-1f9b-4ce2-8d82-65094dcecaec"
version = "0.3.8"

[[Observables]]
git-tree-sha1 = "fe29afdef3d0c4a8286128d4e45cc50621b1e43d"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.4.0"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "4f825c6da64aebaa22cc058ecfceed1ab9af1c7e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.3"

[[Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7937eda4681660b4d6aeeecc2f7e1c81c8ee4e2f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+0"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

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
git-tree-sha1 = "94bf17e83a0e4b20c8d77f6af8ffe8cc3b386c0a"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.1"

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
git-tree-sha1 = "501c20a63a34ac1d015d5304da0e645f42d91c9f"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.0.11"

[[Plots]]
deps = ["Base64", "Contour", "Dates", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs"]
git-tree-sha1 = "1e72752052a3893d0f7103fbac728b60b934f5a5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.19.4"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[Polyester]]
deps = ["ArrayInterface", "IfElse", "ManualMemory", "Requires", "Static", "StrideArraysCore", "ThreadingUtilities", "VectorizationBase"]
git-tree-sha1 = "4b692c8ce1912bae5cd3b90ba22d1b54eb581195"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.3.7"

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

[[ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

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
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
git-tree-sha1 = "37d210f612d70f3f7d57d488cb3b6eff56ad4e41"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.0"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "2a7a2469ed5d94a98dea0e85c46fa653d76be0cd"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.3.4"

[[RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "0426474f50756b3b47b08075604a41b460c45d17"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.16.1"

[[RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization"]
git-tree-sha1 = "2e1a88c083ebe8ba69bc0b0084d4b4ba4aa35ae0"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.1.13"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

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

[[SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "bfdf9532c33db35d2ce9df4828330f0e92344a52"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.25"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "f0bf114650476709dd04e690ab2e36d88368955e"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.18.2"

[[ScientificTypesBase]]
git-tree-sha1 = "3f7ddb0cf0c3a4cff06d9df6f01135fa5442c99b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "1.0.0"

[[Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "35927c2c11da0a86bcd482464b93dadd09ce420f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.5"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

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
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "508822dca004bf62e210609148511ad03ce8f1d8"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.0"

[[SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "edef25a158db82f4940720ebada14a60ef6c4232"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.13"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "62701892d172a2fa41a1f829f66d2b0db94a9a63"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "885838778bb6f0136f8317757d7803e0d81201e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.9"

[[StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "93f7326079b73910e5a81f8848e7a633f99a2946"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "2.0.1"

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
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[StatsPlots]]
deps = ["Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "e7d1e79232310bd654c7cef46465c537562af4fe"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.14.26"

[[StrideArraysCore]]
deps = ["ArrayInterface", "ManualMemory", "Requires", "ThreadingUtilities", "VectorizationBase"]
git-tree-sha1 = "e1c37dd3022ba6aaf536541dd607e8d5fb534377"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.1.17"

[[StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "000e168f5cc9aded17b6999a560b7c11dda69095"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.0"

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

[[TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "a7cf690d0ac3f5b53dd09b5d613540b230233647"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.0.0"

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

[[TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "d620a061cb2a56930b52bdf5cf908a5c4fa8e76a"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.4"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "03013c6ae7f1824131b2ae2fc1d49793b51e8394"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.4.6"

[[Tracker]]
deps = ["Adapt", "DiffRules", "ForwardDiff", "LinearAlgebra", "MacroTools", "NNlib", "NaNMath", "Printf", "Random", "Requires", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "bf4adf36062afc921f251af4db58f06235504eff"
uuid = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
version = "0.2.16"

[[Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "34f27ac221cb53317ab6df196f9ed145077231ff"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.65"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[Turing]]
deps = ["AbstractMCMC", "AdvancedHMC", "AdvancedMH", "AdvancedPS", "AdvancedVI", "BangBang", "Bijectors", "DataStructures", "Distributions", "DistributionsAD", "DocStringExtensions", "DynamicPPL", "EllipticalSliceSampling", "ForwardDiff", "Libtask", "LinearAlgebra", "MCMCChains", "NamedArrays", "Printf", "Random", "Reexport", "Requires", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tracker", "ZygoteRules"]
git-tree-sha1 = "a330a52cbbc2b926b4e5b4296105fe1fc7d656b9"
uuid = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"
version = "0.16.6"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VectorizationBase]]
deps = ["ArrayInterface", "Hwloc", "IfElse", "Libdl", "LinearAlgebra", "Static"]
git-tree-sha1 = "a4bc1b406dcab1bc482ce647e6d3d53640defee3"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.20.25"

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

[[Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "eae2fbbc34a79ffd57fb4c972b08ce50b8f6a00d"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.3"

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

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "acc685bcf777b2202a904cdcb49ad34c2fa1880c"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.14.0+4"

[[libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7a5780a0d9c6864184b3a2eeeb833a0c871f00ab"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "0.1.6+4"

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
git-tree-sha1 = "d713c1ce4deac133e3334ee12f4adff07f81778f"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2020.7.14+2"

[[x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "487da2f8f2f0c8ee0e83f39d13037d6bbf0a45ab"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.0.0+3"

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
# ╟─4666fae5-b072-485f-bb3a-d28d54fb5274
# ╟─ec9e8512-85fb-4d23-a4ad-7c758cd87b62
# ╟─14339a8b-4486-4869-9cf2-6a5f1e4363ac
# ╟─2acbce7f-f528-4047-a2e0-0d7c710e37a1
# ╟─f7592ef8-e966-4113-b60e-559902911e65
# ╟─6d0bb8df-7211-4fca-b021-b1c35da7f7db
# ╟─4dfcf3a2-873b-42be-9f5e-0c93cf7220fc
# ╟─b8a11019-e87e-4ccb-b9dd-f369ba51a182
# ╠═d576514c-88a3-4292-93a8-8d23edefb2e1
# ╟─3abcd374-cb1b-4aba-bb3d-09e2819bc842
# ╟─411c06a3-c8f8-4d1d-a247-1f0054701021
# ╟─46780616-1282-4e6c-92ec-5a965f1fc701
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─03288125-5ebd-47d3-9e1a-32e2308dbd51
# ╟─eed2582d-1ba9-48a1-90bc-a6a3bca139ba
# ╟─0c0d89c0-d70d-42ac-a55c-cd1afbc051ed
# ╟─c6d22671-26e8-4ba3-865f-5cd449a6c9be
# ╟─34b792cf-65cf-447e-a30e-40ebca992427
# ╟─b7c5cbe9-a27c-4311-9642-8e0ed80b3d51
# ╠═f364cb91-99c6-4b64-87d0-2cd6e8043142
# ╟─6087f997-bcb7-4482-a343-4c7830967e49
# ╟─bf5240e8-59fc-4b1f-8b0d-c65c196ab402
# ╟─8165a49f-bd0c-4ad6-8631-cae7425ca4a6
# ╟─5439c858-6ff3-4faa-9145-895390240d76
# ╟─0504929d-207c-4fb7-a8b9-14e21aa0f74b
# ╟─169fbcea-4e82-4756-9b4f-870bcb92cb93
# ╟─699dd91c-1141-4fb6-88fc-f7f05906d923
# ╟─6e1de0ff-0fef-48b4-ac5b-0279ea8f2d4d
# ╟─284d0a23-a329-4ea7-a069-f387e21ba797
# ╟─bb535c41-48cb-44fd-989b-a6d3e310406f
# ╟─9e7a673e-7cb6-454d-9c57-b6b4f9181b06
# ╠═5046166d-b6d8-4473-8823-5209aac59c84
# ╟─82d0539f-575c-4b98-8679-aefbd11f268e
# ╠═00cb5973-3047-4c57-9213-beae8f116113
# ╟─9e3c0e01-8eb6-4078-bc0f-019466afba5e
# ╟─c0bba3aa-d52c-4192-8eda-32d5d9f49a28
# ╟─ab9195d6-792d-4603-8605-228d479226c6
# ╟─e42c5e18-a647-4281-8a87-1b3c6c2abd33
# ╟─5d6a485d-90c4-4f76-a27e-497e8e12afd8
# ╟─9eaf73e9-0f68-4e8b-9ae1-a42f050f695a
# ╟─828166f7-1a69-4952-9e3b-50a99a99789f
# ╟─24c4d724-5911-4534-a5c6-3ab86999df43
# ╟─11b8b262-32d2-4620-bc6b-afca4a5ce977
# ╟─89f7f633-4f75-4ef5-aa5b-80e318d14ee5
# ╟─11552b20-3407-4d0b-b07d-1488c8e8a759
# ╠═599c2f09-ad5e-4f39-aa7d-c1ba155725d6
# ╟─09ec10d9-a604-480d-8e82-59e84a843749
# ╠═071761f8-a187-47a6-8fee-5fc91e65d04c
# ╠═f001b040-2ae7-4e97-b229-eebaabb537b0
# ╠═9a2d5bdf-9597-40c7-ac18-bb27f187912d
# ╟─f6e6c4bf-9b2f-4047-a6cc-4ab9c3ae1420
# ╟─8382e073-433b-4e42-a6fa-d5a051586457
# ╠═9678396b-d42c-4c7c-821c-08126895efd3
# ╟─0a1d46ed-0295-4000-9e30-3ad838552a7e
# ╟─e5aade9a-4593-4903-bc3a-3a37f9f71c98
# ╠═87db6122-4d28-45bf-b5b0-41189792199d
# ╟─c6e9bb86-dc67-4f42-89da-98581a0c3c98
# ╟─b81924b8-73f6-4b28-899c-ec417d538dd4
# ╟─c3d2ba03-e676-4f0f-bafd-feecd0e4e414
# ╠═97c0c719-fb73-4571-9a6c-629a98cc544d
# ╠═0b3945a8-0ae3-4c18-a9b7-a249eb530bcb
# ╟─4e790ffa-554d-4c46-af68-22ecb461fb7b
# ╟─4b141ffc-4100-47e3-941a-4e72c784ccf0
# ╟─219aafcb-17b1-4f5f-9c2b-9b713ba78b18
# ╟─2833e081-45d6-4f64-8d1e-b3a5895b7952
# ╟─e6a32850-6656-4482-b57b-61f774896514
# ╟─fe8f71b2-4198-4a12-a996-da254d2cc656
# ╟─7e89eee0-dd19-4fec-b7c0-7783a9ffb83c
# ╟─f45eb380-7b43-4fd0-af34-89ffd126a63f
# ╟─4084f646-bce6-4a21-a529-49c7774f5ad1
# ╠═3c49d3e4-3555-4d18-9445-5347247cf639
# ╟─98db344c-2ada-4781-bb4a-f3ec2ea7ccfd
# ╠═f7b158af-537e-4d9f-9c4c-318281097dce
# ╟─69a1f4bb-35f6-42bf-9a2a-e3631bf4e43e
# ╟─b6da2479-1545-4b1d-8d7f-07d6d1f67635
# ╠═d6316b4f-9882-4d25-87d0-31fa3c1f3935
# ╟─c4cc482b-815b-4747-9f5a-5779d69086f7
# ╟─9016cba4-58f0-4b7f-91af-66faaf3fe99c
# ╠═6b1e8fc3-48ee-471b-9c04-7c75cfef156c
# ╠═2eb59993-4ace-4acb-9810-ba064ea1eb3e
# ╠═7c04e47c-eeed-47ec-9c6f-e2b710d0b746
# ╟─79b45389-fa2a-46df-9869-1992c8afb397
# ╟─cb53475c-cc56-46b3-94b0-3ded33eb18d4
# ╟─75ef279d-2c7b-4776-b93f-5b28cbc67f63
# ╟─2844b7a6-002e-4459-9e37-30e3a16c88f0
# ╟─68bb2bfb-6643-4f59-9d2b-c59fd1dc3273
# ╟─5c714d3b-ac72-40dc-ba98-bb2b24435d4c
# ╟─573b8a38-5a9b-4d5f-a9f6-00a5255914f0
# ╟─1ca20976-757f-4e30-94d4-ee1276a614fb
# ╠═aa69d0e8-cbbb-436c-b488-5bb113cdf97f
# ╟─84d7e4dd-23a9-4412-a8de-ab8ee8351770
# ╟─43d563ae-a435-417f-83c6-19b3b7d6e6ee
# ╟─448b4ddf-5be2-4843-8e8e-4afb60aa8843
# ╟─11a5614b-c195-45a8-8be0-b99fda6c60fd
# ╟─f004ec01-1e27-4e30-9a53-23a299208846
# ╟─e7669fea-5aff-4522-81bf-3356ce126b1f
# ╟─a32faf6c-a5bb-4005-ad42-188af732fba5
# ╟─92a4aa17-2e2d-45c2-a9a2-803d389077d5
# ╟─0a3ed179-b60b-4740-ab73-b176bba08d84
# ╟─0259f04b-6739-4803-a3eb-4641a6af8361
# ╠═eeb9d3b0-ab6b-49fd-9d3c-87489ccd7c26
# ╠═54f47150-282e-434c-a588-c7c530c438b9
# ╟─bf1a74e4-cf55-470e-843d-a8e6b90517e9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
