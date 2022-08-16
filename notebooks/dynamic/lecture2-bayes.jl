### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
using Distributions, KernelDensity, LinearAlgebra, Plots, PlutoUI, QuadGK, Random, RCall, StatsBase, Statistics, StatsPlots, Turing

# ╔═╡ 6f157a69-be96-427b-b844-9e76660c4cd6
using BenchmarkTools

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
    max-width: 900px;
  }
</style>
"""

# ╔═╡ aa69729a-0b08-4299-a14c-c9eb2eb65d5c
md" # Introduction "

# ╔═╡ bcba487d-0b1f-4f08-9a20-768d50d67d7f
md""" 

> "When the facts change, I change my mind. What do you do, sir?" -- **John Maynard Keynes**

"""

# ╔═╡ 000021af-87ce-4d6d-a315-153cecce5091
md" In this session we will be looking at the basics of Bayesian econometrics / statistics. We will start with a discussion on probability and Bayes' rule and then we will move on to discuss single parameter models. Some math will be interlaced with the code. I assume some familiarity with linear algebra, probability and calculus for this module. The section is on probability is simply a high level overview that leads us to our derivation of Bayes' theorem / rule. "

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# TableOfContents() # Uncomment to see TOC

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
3. **Normalisation**: Probability of all possible events $A_1, A_2, \ldots$ must add up to one, i.e. $\sum_{n \in \mathbb{N}} P (A_n) = 1$

"

# ╔═╡ 14339a8b-4486-4869-9cf2-6a5f1e4363ac
md" With the axioms established, we are able construct all mathematics pertaining to probability. The first topic of interest to us in the Bayesian school of thought is conditional probability. "

# ╔═╡ 2acbce7f-f528-4047-a2e0-0d7c710e37a1
md" ### Conditional probability "

# ╔═╡ f7592ef8-e966-4113-b60e-559902911e65
md" This is the probability that one event will occur if another has occurred or not. We use the notation $P(A \mid B)$, which can be read as, the probability that we have observe $A$ given that we have already observed $B$."

# ╔═╡ 6d0bb8df-7211-4fca-b021-b1c35da7f7db
md" We can illustrate with an example. Think about a deck of cards, with 52 cards in a deck. The probability that you are dealt a king of hearts is $1/52$. In other words, $P(KH)=\left(\frac{1}{52}\right)$, while the probability of beingn a dealt an ace of hearts is $P(AH)=\left(\frac{1}{52}\right)$. However, the probability that you will be dealt ace of hearts given that you have been dealt king of hearts is

$P(AH \mid KH)=\left(\frac{1}{51}\right)$

This is because we have one less card, since you have already been dealt king of hearts. Next we consider joint probability. "

# ╔═╡ 4dfcf3a2-873b-42be-9f5e-0c93cf7220fc
md" ### Joint probability "

# ╔═╡ b8a11019-e87e-4ccb-b9dd-f369ba51a182
md" Joint probability (in the case of two random variables) is the probability that two events will both occur. Let us extend our card problem to all the kings and aces in the deck. Probability that you will receive an Ace ($A$) and King ($K$) as the two starting cards:

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

**NB note**: Joint probability is commutative, but conditional probability is **NOT**. This means that generally $P(A \mid B) \neq P(B \mid A)$. In our example above we have some nice symmetry with respect to conditional probability, but this doesnt occur often.  "

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

# ╔═╡ 7bbecc2b-8c49-458c-8f2e-8792efa62a32
md"""

> A good discussion on Bayesian thinking is the one by [Cam Davidson-Pilon](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb). You can read the first part of this chapter to get an idea of what the Bayesian way of approaching probability is all about. 

"""

# ╔═╡ 03288125-5ebd-47d3-9e1a-32e2308dbd51
md" Consider an economic model ($M$) that describes an AR($1$) process

$\begin{equation*} y_{t}=\mu+\alpha y_{t-1}+\varepsilon_{t}, \quad \varepsilon_{t} \sim \mathcal{N}\left[0, \sigma^{2}\right] \end{equation*}$ 

where $\mu$, $\alpha$ and $\sigma^{2}$ are parameters in a vector $\theta$. In the usual time series econometrics course one would try and estimate these unkown parameters with methods such as maximum likelihood estimation (MLE), as you did in the first part of the course. 

This means that we want to **estimate** $\theta = \{\mu, \alpha, \sigma^{2}\}$. 

Unobserved variables are usually called **parameters** and can be inferred from other variables. $\theta$ represents the unobservable parameter of interest, where $y$ is the observed data. 

Bayesian conclusions about the parameter $\theta$ is made in terms of **probability statements**. Statements are conditional on the observed values of $y$ and can be written $p(\theta \mid y)$: given the data, what do we know about $\theta$? 

**Notation remark**: You will see that we have switched to a small letter $p$ for probability distribution of a random variable. Previously we have used a capital $P$ to relate probability of events. You will often see probability of events written as $\mathbb{P}$ in textbooks as well. 
"

# ╔═╡ d5d387e4-f7c9-45fd-924c-4afa85ed0a05
md"

### Bayes vs Frequentist
"

# ╔═╡ fc765758-eb48-47c4-b308-a177f2f25766
md"
The Bayesian view is that there may be many possible values for $\theta$ from the population of parameter values $\Theta$. The frequentist view is that only one such a $\theta$ exists and that repeated sampling and greater frequency of observation will reveal this value. In other words, $\theta$ is regarded as a **random variable** in the Bayesian setting. This means that we have some subjective belief about $\theta$ should be in the Bayesian view and there is uncertainty attached as to what the parameter value actually is. 

Bayesians will test initial assertions regarding $\theta$ using data on $y$ to investigate probability of assertions. This provides probability distribution over possible values for $\theta \in \Theta$. 

In the Bayesian view of the world, for our model we start with a numerical formulation of joint beliefs about $y$ and $\theta$ expressed in terms of probability distributions.

1. For each $\theta \in \Theta$ the prior distribution $p(\theta)$ describes belief about true population characteristics
2. For each $\theta \in \Theta$ and $y \in \mathcal{Y}$, our sampling model $p(y \mid \theta)$ describes belief that $y$ would be the outcome of the study if we knew $\theta$ to be true.

We then combine these components through Bayes' theorem to obtain the posterior distribution of interest. We are informed by the data and our prior about the posterior distribution. 
"

# ╔═╡ 0a31e0cf-382e-49f2-bad4-bc4b5a8b1a98
md" 

### Practical Bayesian estimation

"

# ╔═╡ eed2582d-1ba9-48a1-90bc-a6a3bca139ba
md" 

Let us now discuss what this means practically for the estimation process. 

We know that $y$ is the set of observations from our data set and we want to estimate a model with the associated parameter vector $\theta$ and likelihood function $p(y \mid \theta)$. In the world of Bayesian statistics we can estimate $\theta$ in two steps. 

1. Before seeing the data we form a prior belief about $\theta$, which we have labelled the prior distribution $p(\theta)$
2. After seeing the data we form the posterior belief about $\theta$, which is known as the posterior distribution $p(\theta \mid y)$

The question is how do we move from step 1 to step 2? How do we update our beliefs from the prior to posterior? The answer lies with **Bayes' rule**. Let us consider the steps in the construction of Bayes's rule. 

To make probability statements about the parameter given the data ($\theta$ given $y$), we begin with a model providing a **joint probability distribution** for $\theta$ and $y$. Joint probability density can be written as product of two densities: the prior $p(\theta)$ and sampling distribution $p(y \mid \theta)$

$p(\theta, y) = p(\theta)p(y \mid \theta)$

However, using the properties of conditional probability we can also write the joint probability density as

$p(\theta, y) = p(y)p(\theta \mid y)$

Setting these equations equal and rearranging provides us with Bayes' theorem / rule, as discussed before. 

$p(y)p(\theta \mid y) = p(\theta)p(y \mid \theta) \rightarrow p(\theta \mid y) = \frac{p(y \mid \theta)p(\theta)}{p(y)}$

We are then left with the following formulation:

$$\underbrace{p(\theta \mid y)}_{\text {Posterior }}=\frac{\overbrace{p(y \mid \theta)}^{\text {Likelihood }} \cdot \overbrace{p(\theta)}^{\text {Prior }}}{\underbrace{p(y)}_{\text {Normalizing Constant }}}$$ 

The denominator in this expression $p(y)$ is called the marginal likelihood. It is defined by margininalising (integrating) over all possible values of $\theta$. 

$p(y) = \int_{\Theta} p(y \mid \theta) p(\theta) d \theta$

While this quantity will be important later in hypothesis testing, we can safely ignore $p(y)$ it for now since it does not involve the parameter of interest $(\theta)$, which means we can write 

$p(\theta|y)\propto p(\theta)p(y \mid \theta)$

The **posterior density** $p(\theta \mid y)$ summarises all we know about $\theta$ after seeing the data, while the **prior density** $p(\theta)$ does not depend on the data (what you know about $\theta$ prior to seeing data). The **likelihood function** $p(y \mid \theta)$ is the data generating process (density of the data conditional on the parameters in the model). "

# ╔═╡ 0c0d89c0-d70d-42ac-a55c-cd1afbc051ed
md" ### Model vs. likelihood "

# ╔═╡ c6d22671-26e8-4ba3-865f-5cd449a6c9be
md" The following is important to point out, since it can create some confusion (at least I found it confusing at first). The sampling model is shown as $p_{Y}(Y \mid \Theta = \theta) = p(y \mid \theta)$ as a function of $y$ given **fixed** $\theta$ and describes the aleatoric (unknowable) uncertainty. This is the type of uncertainty that cannot be reduced.

On the other hand, likelihood is given as $p_{\Theta}(Y=y \mid \Theta) = p(y \mid \theta) =L(\theta \mid y)$ which is a function of $\theta$ given **fixed** $y$ and provides information about epistemic (knowable) uncertainty, but is **not a probability distribution** 

Bayes' rule combines the **likelihood** with **prior** uncertainty $p(\theta)$ and transforms them to updated **posterior** uncertainty."

# ╔═╡ 34b792cf-65cf-447e-a30e-40ebca992427
md"""

#### What is a likelihood? 

"""

# ╔═╡ b7c5cbe9-a27c-4311-9642-8e0ed80b3d51
md"""

This is a question that has bother me a fair bit since I got started with econometrics. So I will try and give an explanation with some code integrated to give a better idea. This has been mostly inspired by the blog post of [Jim Savage](https://khakieconomics.github.io/2018/07/14/What-is-a-likelihood-anyway.html).

In order to properly fit our Bayesian models we need to construction some type of function that lets us know whether the values of the unknown components of the model are good or not. Imagine that we have some realised values of some random variable and we can form a histogram of those values. Let us quickly construct a dataset. 

"""

# ╔═╡ f364cb91-99c6-4b64-87d0-2cd6e8043142
data_realised = 3.5 .+ 3.5 .* randn(100) # This is a normally distributed dataset that represents our true data. 

# ╔═╡ 6087f997-bcb7-4482-a343-4c7830967e49
histogram(data_realised, color = :black, legend = false, lw = 1.5,  fill = (0, 0.3, :black), bins  = 20)

# ╔═╡ bf5240e8-59fc-4b1f-8b0d-c65c196ab402
md"""

We would like to fit a density function on this histogram so that we can make probabilistic statements about observations that we have not yet observed. Our proposed density should try and match model unknowns (like the location and scale). Let us observe two potential proposed densities. One is bad guess and the other quite good. 

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

A likelihood function would return a higher value for the proposed density in the case of the good guess. 

There are many functions that we could use to determine wheter proposed model unknowns result in a better fit. Likelihood functions are one particular approach. Like we mentioned before, these likelihood functions represent a data generating process. The assumption is that the proposed generative distribution gave rise to the data in question. We will continue this discussion a bit later on.  

"""

# ╔═╡ 699dd91c-1141-4fb6-88fc-f7f05906d923
md" ## Bernoulli and Binomial "

# ╔═╡ 6e1de0ff-0fef-48b4-ac5b-0279ea8f2d4d
md" In this section we will be looking at some single parameter models. In other words, models where we only have a single unknown parameter of interest. This will draw on our knowledge from random variables and their distributions in the previous lecture.  "

# ╔═╡ 284d0a23-a329-4ea7-a069-f387e21ba797
md"""

### Bernoulli random variable 

"""

# ╔═╡ bb535c41-48cb-44fd-989b-a6d3e310406f
md"""

Let us give a general description of the Bernoulli and Binomial random variables and their relation to each other. We will start with data that has a Bernoulli distribution.

Consider an experiment (such as tossing a coin) that is repeated $N$ times. Each time we conduct this experiment / trial we can evaluate the outcome as being a success or failure.

In this case the $y_{i}$'s, for $i = 1, \ldots, N$, are random variables for each repetition of the experiment. A random variable is a function that maps the outcome to a value on the real line. In our example, the realisation of $y_{i}$ can be $0$ or $1$ depending on whether the experiment was a success or failure.  

The probability of success is represented by $\theta$, while the probability of failure is given by $1- \theta$. This is considered an Bernoulli event. Our goal is to gain an estimate for $\theta$.

A binary random variable  $y \in \{0, 1\}$, $0 \leq \theta \leq 1$ follows a Bernoulli distribution if

$p\left(y \mid \theta\right)=\left\{\begin{array}{cl}\theta & \text { if } y=1 \\ 1-\theta & \text { if } y=0\end{array}\right.$

This means that $y$ is a Bernoulli random variable with parameter $\theta$, i.e. $y \sim \text{Ber}(\theta)$. We combine the equations for the probabilities of success and failure into a single expression to give the probability mass function (PMF) as 

$p(y \mid \theta) = \theta^{y} (1 - \theta)^{1-y}$

Now, if we let $\{y_1, \ldots, y_N\}$ denote the set of $N$ independent random variables that each represent the coin showing heads in $N$ independent flips then the join PMF of these flips is given by, 

$\begin{aligned} p(\{y_1, \ldots, y_N\} \mid \theta) &=\prod_{i=1}^{N} p\left(y_{i} \mid \theta\right) \\ 
  =& \prod_{i}^{N} \theta^{y_i}(1-\theta)^{(1-y_i)} \\
  =& \theta^{\sum_{i} {y_i}}(1-\theta)^{\sum_{i}(1-y_i)} \\
  =& \theta^{y}(1-\theta)^{N - y} 
\end{aligned}$

The likelihood has the same form as the joint PMF above. The model above would be the same regardless of whether we take a Bayesian or frequentist view of the world. The main difference is in the way in which the probabilities are interpreted. 

"""

# ╔═╡ fe8f71b2-4198-4a12-a996-da254d2cc656
md" ### Binomial random variable "

# ╔═╡ 7e89eee0-dd19-4fec-b7c0-7783a9ffb83c
md"""

It is worthwhile mentioning the Binomial distribution at this stage. The Bernoulli distribution represents the success or failure of a **single Bernoulli trial**. The Binomial distribution represents the number of successes and failues in $N$ independent Bernoulli trials for some given value of $N$. 

The probability of several events in independent trials is $\theta \cdot \theta(1-\theta)\cdot\theta(1-\theta)(1-\theta)\ldots$ 

If there are $N$ trials then the probability that a success occurs $y$ times is

$$\begin{align*}
      p(y \mid \theta, N) & = \frac{N!}{y!(N-y)!} \theta^y(1-\theta)^{N-y} \\
      &= \binom{N}{y} \theta^y(1-\theta)^{N-y}
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

# ╔═╡ 0520b5e3-cf92-4990-8a65-baf300b19631
# The equivalent code in R. Code is almost exactly the same for this example

R"binomial_rv_r <- function(n, p) {
  # Write the function body here
  y <- c(1:n)
  count <- 0
  for (i in seq_along(y)) {
    if (runif(n)[i] < p) {
      count <- count + 1
    }
  }
  return(count)
}";

# ╔═╡ 98db344c-2ada-4781-bb4a-f3ec2ea7ccfd
md" Given a value of $n = 10000$ indepedent trials, how many times will we observe success? "

# ╔═╡ f7b158af-537e-4d9f-9c4c-318281097dce
PlutoUI.with_terminal() do
	@time binomial_rv(100000, 0.5) # Compare this with the time it takes to run in R. 
end

# ╔═╡ 2cb41330-7ebd-45de-9aa1-632db6f9a140
R"system.time(a <- binomial_rv_r(100000, 0.5))"

# ╔═╡ 69a1f4bb-35f6-42bf-9a2a-e3631bf4e43e
md" Now let us conduct some experiments with our new binomial random variable. "

# ╔═╡ b6da2479-1545-4b1d-8d7f-07d6d1f67635
md"""

!!! note "Interactive sliders for Binomial random variable"
	Shift these sliders around to see what happens to the graph below. Try fixing values for $p$ and increase the number of $N$, what happens to the distribution?
	
``N``: $(@bind binom_n Slider(1:100; show_value=true, default=10))
``p``: $(@bind binom_p Slider(0.01:0.01:0.99; show_value=true, default=0.5))

"""

# ╔═╡ c4cc482b-815b-4747-9f5a-5779d69086f7
Plots.plot(
    Binomial(binom_n, binom_p);
    seriestype=:sticks,
    markershape=:circle,
    xlabel=raw"$k$",
    ylabel=raw"$p_{Y\mid \Theta}(y \mid \theta)$",
    title="\$\\operatorname{Binom}($binom_n, $binom_p)\$",
    label=false,
)

# ╔═╡ 9016cba4-58f0-4b7f-91af-66faaf3fe99c
md" Naturally, one would want to use a pre-packaged solution to sampling with a binomial random variable. The `Distributions.jl` package contains optimised routines that work faster than our code, but is still a good idea to code some things yourself to fully understand the mechanisms. " 

# ╔═╡ 828166f7-1a69-4952-9e3b-50a99a99789f
md" ### Worked example: Estimating bias in a coin  "

# ╔═╡ 24c4d724-5911-4534-a5c6-3ab86999df43
md"""
For a worked example lets look at estimating bias in a coin. We observe the number of heads that result from flipping one coin and we estimate its underlying probability of coming up heads. We want to create a descriptive model with meaningful parameters. The outcome of a flip will be given by $y$, with $y=1$ indicating heads and $y = 0$ tails. 

We need underlying probability of heads as value of parameter $\theta$. This can be written as $p(y = 1 \mid \theta) = \theta$. The probability that the outcome is heads, given a parameter value of $\theta$, is the value $\theta$. We also need the probability of tails, which is the complement of probability of heads $p(y = 0 \mid \theta) = 1 - \theta$. 

Combine the equations for the probability of heads and tails, we have the same as before,  

$$\begin{align*}
  p(y \mid \theta)  = \theta^{y}(1-\theta)^{1-y}
\end{align*}$$

We have established this probability distribution is called the Bernoulli distribution. This is a distribution over two discrete values of $y$ for a fixed value of $\theta$. The sum of the probabilities is $1$ (which must be the case for a probability distribution).

$$\begin{align*}
  \sum_{y} p(y \mid \theta) = p(y = 1 \mid \theta) + p(y = 0 \mid \theta) = \theta + (1-\theta) = 1
\end{align*}$$

If we consider $y$ fixed and the value of $\theta$ as variable, then our equation is a **likelihood function** of $\theta$.

This likelihood function is **NOT** a probability distribution! 

Suppose that $y = 1$ then $\int_{0}^{1}\theta^{y}(1-\theta)^{1-y}\text{d}\theta = \int_{0}^{1}\theta^{y}\text{d}\theta = 1/2$

The formula for the probability of the set of outcomes is given by

$$\begin{align*}
  p(y \mid \theta)  =& \prod_{i}^{N} p(y_i \mid \theta)  \\
  =& \prod_{i}^{N} \theta^{y_i}(1-\theta)^{(1-y_i)} \\
  =& \theta^{\sum_{i} {y_i}}(1-\theta)^{\sum_{i}(1-y_i)} \\
  =& \theta^{\#\text{heads}}(1-\theta)^{\#\text{tails}} \\
	=& \theta^y(1-\theta)^{N - y}
\end{align*}$$

Let us quickly talk about this likelihood, so that we have clear vision of what it looks like. We start with an example where there are $5$ coin flips and $1$ of them is heads (as can be seen below). 

"""

# ╔═╡ 5046166d-b6d8-4473-8823-5209aac59c84
begin
	Random.seed!(1237)
	coin_seq = Int.(rand(Bernoulli(0.4), 5))
end

# ╔═╡ 82d0539f-575c-4b98-8679-aefbd11f268e
md"""

Let us say that we think the probability of heads is $0.3$. Our likelihood can be represented as

$L(\theta \mid y) = p(y = (0, 1, 0, 0, 0) \mid \theta) = \prod_{i=1}^{N} \theta^{y_{i}} \times (1 - \theta)^{1 - y_{i}} = \theta ^ y (1- \theta) ^{N - y}$

Do we think that the proposed probability of heads is a good one? We can use the likelihood function to perhaps determine this. We plot the values of the likelihood function for this data evaluated over the possible values that $\theta$ can take. 
"""

# ╔═╡ 00cb5973-3047-4c57-9213-beae8f116113
begin
	grid_θ = range(0, 1, length = 1001) |> collect; # discretised 
	binom(grid_θ, m, N) = (grid_θ .^ m) .* ((1 .- grid_θ) .^ (N - m))
end

# ╔═╡ c0bba3aa-d52c-4192-8eda-32d5d9f49a28
md"""

!!! note "Coin flippling likelihood"
	Below we have the likelihood function for our coin flipping problem. Shift the sliders to see changes in the likelihood. 

heads = $(@bind m Slider(0:50, show_value = true, default=1));
flips = $(@bind N Slider(1:50, show_value = true, default=5)); 

"""

# ╔═╡ 9e3c0e01-8eb6-4078-bc0f-019466afba5e
binomial  = binom(grid_θ, m, N);

# ╔═╡ ab9195d6-792d-4603-8605-228d479226c6
max_index = argmax(binom(grid_θ, m, N)); # Get argument that maximises this function 

# ╔═╡ e42c5e18-a647-4281-8a87-1b3c6c2abd33
likelihood_max = grid_θ[max_index]; # Value at which the likelihood function is maximised. Makes sense, since we have 3 successes in 5 repetitions. 

# ╔═╡ 5d6a485d-90c4-4f76-a27e-497e8e12afd8
begin
	plot(grid_θ, binomial, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Unnormalised likelihood", legend = false)
	plot!([likelihood_max], seriestype = :vline, lw = 2, color = :black, ls = :dash, alpha =0.7, xticks = [likelihood_max])
end

# ╔═╡ 987aeb82-267a-4ca9-bf21-c75a49edad70
bin_area = sum(binomial)

# ╔═╡ 9eaf73e9-0f68-4e8b-9ae1-a42f050f695a
md"""

What do we notice about the likelihood function? 

1. It is **NOT** a probability distribution, since it doesn't integrate to one (we check this with some code above)
2. We notice that in our particular example the function is maximised at $likelihood_max. This maximum point of the likelihood function is known as the maximum likelihood estimate of our parameter given our data. You have dealt with this quantity before. Formally it is, 

$\hat{\theta}_{M L E}=\operatorname{argmax}_{\theta}(p(y \mid \theta))$

This example shows that it can be dangerous to use maximum likelihood with small samples. The true success rate is $0.4$ but our estimate provided a value of $0.2$.

In this case, our prior information (subjective belief) was that the probability of heads should be $0.3$. This could have helped is in this case get to a better estimate, but unfortunately maximimum likelihood does not reflect this prior belief. This means that we are left with a success rate equal to the frequency of occurence. 

"""

# ╔═╡ 36e6f838-8277-480b-b48d-f70e8fe011eb
md"""

The next step then is to establish the prior, which will be an arbitrary choice here. One assumption could be that the factory producing the coins tends to produce mostly fair coins. Indicate number of heads by $y$ and number of flips by $N$. We need to specify some prior, and we will use the Triangular distribution for our prior in the next section.

Let us code up the likelihood, prior and updated posterior for this example. In order to do this let us implement the grid method. There are many other ways to do this. However, this method is easy to do and gives us some good coding practice.

"""

# ╔═╡ 11b8b262-32d2-4620-bc6b-afca4a5ce977
md" #### The grid method "

# ╔═╡ 89f7f633-4f75-4ef5-aa5b-80e318d14ee5
md" There are four basic steps behind the grid method.

1. Discretise the parameter space if it is not already discrete.
2. Compute prior and likelihood at each “grid point” in the (discretised) parameter space.
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
N₁ = 5 # Number of flips

# ╔═╡ 9a2d5bdf-9597-40c7-ac18-bb27f187912d
triangle_prior = TriangularDist(0, 1) # From the Distributions.jl package

# ╔═╡ f6e6c4bf-9b2f-4047-a6cc-4ab9c3ae1420
plot(triangle_prior, coins_grid, xlab = "theta", ylab = "prior", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))

# ╔═╡ 8382e073-433b-4e42-a6fa-d5a051586457
md" In this small dataset we have $1$ success out of $5$ attempts. Our distribution function will calculate the probability that we want for a given value of $\theta$. We want to do this for each value of $\theta$, but using same values for $m$ and $N$ each time.  "

# ╔═╡ 9678396b-d42c-4c7c-821c-08126895efd3
binomial₁ = binom(grid_θ, m₁, N₁);

# ╔═╡ 0a1d46ed-0295-4000-9e30-3ad838552a7e
begin
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 1.5,  fill = (0, 0.2, :black))
	plot!(coins_grid, binomial₁, color = :steelblue,lw = 1.5,  fill = (0, 0.2, :steelblue), title = "Unnormalised likelihood")
end

# ╔═╡ e5aade9a-4593-4903-bc3a-3a37f9f71c98
md"""
We can normalise the likelihood for the purpose of plotting. We can do this by dividing by the sum of the likelihoods and by the width of the spaces betwen the grid points.
"""

# ╔═╡ 87db6122-4d28-45bf-b5b0-41189792199d
likelihood_norm = binomial₁ / sum(binomial₁) / 0.001 # Normalised

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
prior = pdf(triangle_prior, coins_grid) # Extract the values for the prior at the grid points

# ╔═╡ 0b3945a8-0ae3-4c18-a9b7-a249eb530bcb
posterior = prior .* likelihood_norm # Calculate the posterior

# ╔═╡ 4e790ffa-554d-4c46-af68-22ecb461fb7b
begin
	plot(coins_grid, triangle_prior, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior, color = :black,lw = 2,  fill = (0, 0.4, :green))
end

# ╔═╡ 4b141ffc-4100-47e3-941a-4e72c784ccf0
md" Play around with the sliders here so that you can see what happens to the posterior once it gets updated with new information from the data. In addition, what happens once the size of the dataset increases? What role does the prior play?"

# ╔═╡ 219aafcb-17b1-4f5f-9c2b-9b713ba78b18
md"""
heads = $(@bind y₂ Slider(1:100, show_value = true, default=1));
flips = $(@bind n₂ Slider(1:100, show_value = true, default=5))
"""

# ╔═╡ 2833e081-45d6-4f64-8d1e-b3a5895b7952
begin
	b₂ = Binomial.(n₂, coins_grid)
	likelihood_2 = pdf.(b₂, y₂)
	likelihood_norm_2 = likelihood_2 / sum(likelihood_2) / 0.001
	posterior_2 = prior .* likelihood_norm_2;
	plot(triangle_prior, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm_2, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior_2, color = :black,lw = 2,  fill = (0, 0.4, :green))
end

# ╔═╡ 5c714d3b-ac72-40dc-ba98-bb2b24435d4c
md"""

## Coin flipping example contd. 

"""

# ╔═╡ 573b8a38-5a9b-4d5f-a9f6-00a5255914f0
md"""

In our original coin flipping model we built the model on the foundation of a joint PMF. Now we need to incorporate prior belief about model parameters $\theta$. 

Generally, we need a mathematical description of the **prior probability** for each value of the parameter $\theta$ on interval $[0, 1]$. Any relevant probability density function would work, but there are two desiderata for mathematical tractability.

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

!!! note "Parameters (a, b) of Beta distribution"
	Changing these sliders will illustrate how flexible the Beta distribution really is!

a = $(@bind α Slider(0.1:0.1:10, show_value = true, default=1)); 
b = $(@bind β Slider(0.1:0.1:10, show_value = true, default=1))
"""

# ╔═╡ aa69d0e8-cbbb-436c-b488-5bb113cdf97f
prior_beta = Beta(α, β);

# ╔═╡ dc43d1bc-ea5c-43ca-af0c-fc150756fa76
Plots.plot(
    Beta(α, β);
    xlabel=raw"$\theta$",
    ylabel=raw"$p_{\Theta}(\theta)$",
    title="\$\\mathrm{Beta}\\,($α, $β)\$",
    label=false,
    linewidth=2,
    fill=true,
    fillalpha=0.3,
	color = :black
)

# ╔═╡ 43d563ae-a435-417f-83c6-19b3b7d6e6ee
md"""

!!! note "Beta prior hyperparameters"
	Using different parameterisations of Beta will provide different posteriors.

a1 = $(@bind α₁ Slider(1:0.1:4, show_value = true, default=1));
b1 = $(@bind β₁ Slider(1:1:4, show_value = true, default=1))


"""

# ╔═╡ 11a5614b-c195-45a8-8be0-b99fda6c60fd
begin
	prior_beta₁ = Beta(α₁, β₁)
	prior_beta_pdf = pdf(prior_beta₁, coins_grid); # Beta distribution
	posterior_beta = prior_beta_pdf .* likelihood_norm_2;
	
	plot(prior_beta₁, coins_grid, xlab = "theta", color = :black, legend = false, lw = 0,  fill = (0, 0.2, :black))
	plot!(coins_grid, likelihood_norm_2, color = :steelblue,lw = 0,  fill = (0, 0.2, :steelblue))
	plot!(coins_grid, posterior_beta, color = :black,lw = 2,  fill = (0, 0.4, :green), xlabel=raw"$\theta$",
    ylabel=raw"$p_{\Theta}(\theta)$",
    title="\$\\mathrm{Beta}\\,($α₁, $β₁)\$",)
end

# ╔═╡ f004ec01-1e27-4e30-9a53-23a299208846
md" Initially, with $a = 1$ and $b = 1$ this will be the same as the uniform prior. However, play around with the values on the slider to see how it changes for a different parameterisation of the Beta distribution. "

# ╔═╡ 949ae6bc-315b-48f7-9382-1b0aee462b65
md" ### Incorporating beliefs

"

# ╔═╡ f1b59139-4205-408b-9ba5-f6f9df5d0de0
md""" We can incorporate beliefs about the probability of a coin flip showing heads by choosing values of $a$ and $b$. This is referred to as prior elicitation. This is basically extracting a prior distribution from a person, normally an expert. Common practice is to settle on a distributional family and then elicit **hyperparameters** within the family. 

The Beta distribution, for example, is a two parameter family with great flexibility (as we have observed). For different values of the parameters $a$ and $b$ we get different functional forms for the distribution. We refer to the parameters as hyperparemeters in Bayesian econometrics. One of the things that the researcher might have some information is the expected value of $\theta$.

"""

# ╔═╡ 6a3176dd-587a-4e7d-9649-dff8af789c42
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

# ╔═╡ a33f7f60-389f-4d26-9562-c60380eb1888
md"""
In the case of the Beta distribution the prior mean is given above as 

$\mathbb{E}(\theta) = \frac{a}{a + b}$

The prior mean will, by the fact that the prior is conjugate, also translate to a posterior distribution that has a Beta functional form. Therefore, if you choose the values for $a$ and $b$ properly you are in fact stating something about $\mathbb{E}(\theta)$.

Suppose you believe that $\mathbb{E}(\theta) = 1/2$. This can be obtained by setting $a = b$. 

As an example, set $a = b = 2$, then we have 

$\mathbb{E}(\theta) = \frac{2}{2+2} = 1/2$

We could also choose a completely noninformative prior with $a = b = 1$, which implies that $p(\theta) \propto 1$. This is simply a uniform distribution over the interval $[0, 1]$. Every value for $\theta$ receives the same probability. 

Obviously there are multiple values of $a$ and $b$ will work, play around with the sliders above to see what happens for a choice of different $a$ and $b$ under this restriction for the expected value of $\theta$.

In summary, the Beta distribution is chosen since it has some of the following aspects, 

1. *Natural support*: Support between $0$ and $1$
2. *Flexibility*: The PDF can take on several shapes and therefore express various beliefs. 
3. *Conjugacy*: Results in an analytical posterior distribution. 

We will show the analytical posterior derivation below. 

"""

# ╔═╡ e7669fea-5aff-4522-81bf-3356ce126b1f
md"""

## Analytical derivation

"""

# ╔═╡ a32faf6c-a5bb-4005-ad42-188af732fba5
md"""
We have established the Beta distribution as a convenient prior for the Bernoulli and Binomial likelikood functions. Now we can figure out, mathematically, what the posterior would look like when we apply Bayes' rule. Suppose we have set of data with $N$ flips and $m$ heads, then we can calculate the posterior as,

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

# ╔═╡ 102ca26d-a97a-4f1e-85fa-28d54305afbf
md" ### Alternative derivation "


# ╔═╡ ff3d162a-5325-496f-bb50-c119391729ba
md" An alternative (more compact) derivation from Jamie Cross that uses proportionality and ignores the normalising constant can be found below. 

$$\begin{align*}
        p(\theta | \mathbf{y})  &\propto p(\mathbf{y} | \theta) p(\theta)\\
                                &= \underset{p(\mathbf{y} | \theta)}{\underbrace{\theta^{m} (1-\theta)^{N-m}}} \underset{p(\theta)}{\underbrace{\frac{\theta^{a} (1- \theta)^{b - 1}}{B({a}, {b})}}}\\
                                &\propto \theta^{m} (1-\theta)^{N-m}  \theta^{a-1} (1- \theta)^{b - 1} \\
                                &= \theta^{m + a -1} (1 - \theta)^{N-m+ b - 1 } \\
    \end{align*}$$

It is perhaps easier to see here that the final expression is a Beta distribution with parameters $a + n$ and $b + N - m$. This then means that $\theta|\mathbf{y} \sim \text{Beta}(a + n, b + N - m)$.
"

# ╔═╡ f0de92aa-3bcb-4bad-bb25-9f5c3e4c0b11
md" ## Posterior analysis "

# ╔═╡ 91c8fa46-b9e7-4db2-8e20-0e6c5006483f
md" Given the posterior distribution we can now present the results. Three types of results are generally shown in Bayesian analysis, namely, 

1. Posterior analysis
2. Hypothesis testing
3. Prediction

The notes from Jamie Cross go into detail about posterior analysis. We will not cover this in this course, but you are welcome to look at his notes for answers."

# ╔═╡ 92a4aa17-2e2d-45c2-a9a2-803d389077d5
md" ## Coin flip with `Turing.jl` 🤓"

# ╔═╡ 33b402b9-29c5-43d3-bb77-9b1a172229bb
md""" 

!!! note
	The material from this section comes from a talk by Jose Storopoli

"""

# ╔═╡ 0a3ed179-b60b-4740-ab73-b176bba08d84
md" In this section I will construct a coin tossing model in `Turing.jl`. We can approach the problem in one of two ways. We can see the process as independent Bernoulli trials or use a Binomial model. Don't worry too much about what MCMC methods are at this point, we will spend enough time going through these concepts later in the course. "

# ╔═╡ 47230bb3-de03-4353-9cbe-f974cc25411c
md""" #### How to specify a model """

# ╔═╡ 6f76e32c-32a7-4d77-b1f9-0078807ec103
md"""
**We specify the model inside a macro** `@model` where we can assign variables in two ways:

* using `~`: which means that a variable follows some probability distribution (Normal, Binomial etc.) and its value is random under that distribution

* using `=`: which means that a variable does not follow a probability distribution and its value is deterministic (like the normal `=` assignment in programming languages)

Turing will perform automatic inference on all variables that you specify using `~`.

Just like you would write in mathematical form:

$$\begin{aligned}
\theta &\sim \text{Beta}(1,1) \\
\text{coin flip} &\sim \text{Bernoulli}(\theta)
\end{aligned}$$

In our example we have an unfair coin with $\theta$ = 0.4 being the true value. 
"""

# ╔═╡ c205ff23-f1e7-459f-9339-2c80ab68945f
begin	
	# Set the true probability of heads in a coin.
	θ_true = 0.4	

	# Iterate from having seen 0 observations to 100 observations.
	Ns = 0:5

	# Draw data from a Bernoulli distribution, i.e. draw heads or tails.
	Random.seed!(1237)
	data = rand(Bernoulli(θ_true), last(Ns))

	# Declare our Turing model.
	@model function coin_flip(y; α::Real=1, β::Real=1)
    	# Our prior belief about the probability of heads in a coin.
    	θ ~ Beta(α, β)

    	# The number of observations.
    	N = length(y)
    		for n ∈ 1:N
        	# Heads or tails of a coin are drawn from a Bernoulli distribution.
        	y[n] ~ Bernoulli(θ)
    	end
	end
end

# ╔═╡ 2bfe1d15-210d-43b4-ba4b-dec83f2363cd
md"""

In this example, `coin_flip` is a Julia function. It creates a model of the type `DynamicPPL.Model` which stores the name of the models, the generative function and the arguments of the models and their defaults. 

"""

# ╔═╡ b5a3a660-8928-4097-b1c4-90f045d17444
md"""
#### How to specify a MCMC sampler (`NUTS`, `HMC`, `MH` etc.)
"""

# ╔═╡ 6005e786-d4e8-4eef-8d6c-cc07fe36ea17
md"""
We have [several samplers](https://turing.ml/dev/docs/using-turing/sampler-viz) available:

* `MH()`: **M**etropolis-**H**astings
* `PG()`: **P**article **G**ibbs
* `SMC()`: **S**equential **M**onte **C**arlo
* `HMC()`: **H**amiltonian **M**onte **C**arlo
* `HMCDA()`: **H**amiltonian **M**onte **C**arlo with Nesterov's **D**ual **A**veraging
* `NUTS()`: **N**o-**U**-**T**urn **S**ampling

Just stick your desired `sampler` inside the function `sample(model, sampler, N; kwargs)`.

Play around if you want. Choose your `sampler`:
"""

# ╔═╡ 283fe6c9-6642-4bce-a639-696b92fcabb8
@bind chosen_sampler Radio([
		"MH()",
		"PG()",
		"SMC()",
		"HMC()",
		"HMCDA()",
		"NUTS()"], default = "MH()")

# ╔═╡ 0dc3b4d5-c66e-4fbe-a9fe-67f9212371cf
begin
	your_sampler = nothing
	if chosen_sampler == "MH()"
		your_sampler = MH()
	elseif chosen_sampler == "PG()"
		your_sampler = PG(2)
	elseif chosen_sampler == "SMC()"
		your_sampler = SMC()
	elseif chosen_sampler == "HMC()"
		your_sampler = HMC(0.05, 10)
	elseif chosen_sampler == "HMCDA()"
		your_sampler = HMCDA(10, 0.65, 0.3)
	elseif chosen_sampler == "NUTS()"
		your_sampler = NUTS(10, 0.65)
	end
end

# ╔═╡ 53872a09-db29-4195-801f-54dd5c7f8dc3
begin
	chain_coin = sample(coin_flip(data), your_sampler, 1000)
	summarystats(chain_coin)
end

# ╔═╡ b8053536-0e98-4a72-badd-58d9adbcf5ca
md"""
#### How to inspect chains and plot with `MCMCChains.jl`
"""

# ╔═╡ 97dd43c3-1072-4060-b750-c898ce926861
md"""
We can inspect and plot our model's chains and its underlying parameters with [`MCMCChains.jl`](https://turinglang.github.io/MCMCChains.jl/stable/)

**Inspecting Chains**
   * Summary Statistics: just do `summarystats(chain)`
   * Quantiles (Median, etc.): just do `quantile(chain)`

**Plotting Chains**: Now we have several options. The default `plot()` recipe will plot a `traceplot()` side-by-side with a `mixeddensity()`.

 First, we have to choose either to plot **parameters**(`:parameter`) or **chains**(`:chain`) with the keyword `colordim`.


Second, we have several plots to choose from:
* `traceplot()`: used for inspecting Markov chain **convergence**
* `meanplot()`: running average plots per interaction
* `density()`: **density** plots
* `histogram()`: **histogram** plots
* `mixeddensity()`: **mixed density** plots
* `autcorplot()`: **autocorrelation** plots


"""

# ╔═╡ 927bce0e-e018-4ecb-94e5-09812bf75936
plot(
	traceplot(chain_coin, title="traceplot"),
	meanplot(chain_coin, title="meanplot"),
	density(chain_coin, title="density"),
	histogram(chain_coin, title="histogram"),
	mixeddensity(chain_coin, title="mixeddensity"),
	autocorplot(chain_coin, title="autocorplot"),
	dpi=300, size=(840, 600), 
	alpha = 0.8
)

# ╔═╡ 398da783-5e47-4d39-8048-4541aad6b8b5
#StatsPlots.plot(chain_coin[:θ], lw = 1.75, color = :steelblue, alpha = 0.8, legend = false, dpi = 300)

# ╔═╡ a6ae40e6-ea8c-46f1-b430-961c1185c087
begin
#	StatsPlots.histogram(chain_coin[:θ], lw = 1.75, color = :black, alpha = 0.8, fill = (0, 0.4, :steelblue), legend = false)
end

# ╔═╡ 10beb8b2-0841-44c4-805e-8667da325b01
md""" 

#### Comparison with true posterior 

"""

# ╔═╡ 58f65290-8ba9-4c27-94de-b28a5eac80a4
md" We compare our result from using Turing with the analytical posterior that we derived in the previous section. "

# ╔═╡ c0daa659-f5f6-4e6b-9973-a399cf0ea788
begin
	# Our prior belief about the probability of heads in a coin toss.
	prior_belief = Beta(1, 1);
	
	# Compute the posterior distribution in closed-form.
	M = length(data)
	heads = sum(data)
	updated_belief = Beta(prior_belief.α + heads, prior_belief.β + M - heads)

	# Visualize a blue density plot of the approximate posterior distribution
	p = plot(chain_coin[:θ], seriestype = :density, xlim = (0,1), legend = :best, w = 2, c = :blue, label = "Approximate posterior")
	
	# Visualize a green density plot of posterior distribution in closed-form.
	plot!(p, range(0, stop = 1, length = 100), pdf.(Ref(updated_belief), range(0, stop = 1, length = 100)), xlabel = "probability of heads", ylabel = "", title = "", xlim = (0,1), label = "Closed-form", fill=0, α=0.3, w=3, c = :green)
	
	# Visualize the true probability of heads in red.
	vline!(p, [θ_true], label = "True probability", c = :black, lw = 1.7, style = :dash)
	
end

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
RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Turing = "fce5fe82-541a-59a6-adf8-730c64b5f9a0"

[compat]
BenchmarkTools = "~0.7.0"
Distributions = "~0.24.18"
KernelDensity = "~0.6.3"
Plots = "~1.22.6"
PlutoUI = "~0.7.1"
QuadGK = "~2.4.2"
RCall = "~0.13.12"
StatsBase = "~0.33.12"
StatsPlots = "~0.14.28"
Turing = "~0.15.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
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
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

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
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
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

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

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
git-tree-sha1 = "0c603255764a1fa0b61752d2bec14cfbd18f7fe8"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.5+1"

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
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

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
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

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

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

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
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

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
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

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
git-tree-sha1 = "69fd065725ee69950f3f58eceb6d144ce32d627d"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.2.2"

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
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

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
deps = ["SHA", "Serialization"]
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

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

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
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

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
# ╟─bcba487d-0b1f-4f08-9a20-768d50d67d7f
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═6f157a69-be96-427b-b844-9e76660c4cd6
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
# ╟─7bbecc2b-8c49-458c-8f2e-8792efa62a32
# ╟─03288125-5ebd-47d3-9e1a-32e2308dbd51
# ╟─d5d387e4-f7c9-45fd-924c-4afa85ed0a05
# ╟─fc765758-eb48-47c4-b308-a177f2f25766
# ╟─0a31e0cf-382e-49f2-bad4-bc4b5a8b1a98
# ╟─eed2582d-1ba9-48a1-90bc-a6a3bca139ba
# ╟─0c0d89c0-d70d-42ac-a55c-cd1afbc051ed
# ╟─c6d22671-26e8-4ba3-865f-5cd449a6c9be
# ╟─34b792cf-65cf-447e-a30e-40ebca992427
# ╟─b7c5cbe9-a27c-4311-9642-8e0ed80b3d51
# ╠═f364cb91-99c6-4b64-87d0-2cd6e8043142
# ╟─6087f997-bcb7-4482-a343-4c7830967e49
# ╟─bf5240e8-59fc-4b1f-8b0d-c65c196ab402
# ╠═8165a49f-bd0c-4ad6-8631-cae7425ca4a6
# ╟─5439c858-6ff3-4faa-9145-895390240d76
# ╟─0504929d-207c-4fb7-a8b9-14e21aa0f74b
# ╟─169fbcea-4e82-4756-9b4f-870bcb92cb93
# ╟─699dd91c-1141-4fb6-88fc-f7f05906d923
# ╟─6e1de0ff-0fef-48b4-ac5b-0279ea8f2d4d
# ╟─284d0a23-a329-4ea7-a069-f387e21ba797
# ╟─bb535c41-48cb-44fd-989b-a6d3e310406f
# ╟─fe8f71b2-4198-4a12-a996-da254d2cc656
# ╟─7e89eee0-dd19-4fec-b7c0-7783a9ffb83c
# ╟─f45eb380-7b43-4fd0-af34-89ffd126a63f
# ╟─4084f646-bce6-4a21-a529-49c7774f5ad1
# ╠═3c49d3e4-3555-4d18-9445-5347247cf639
# ╠═0520b5e3-cf92-4990-8a65-baf300b19631
# ╟─98db344c-2ada-4781-bb4a-f3ec2ea7ccfd
# ╠═f7b158af-537e-4d9f-9c4c-318281097dce
# ╠═2cb41330-7ebd-45de-9aa1-632db6f9a140
# ╟─69a1f4bb-35f6-42bf-9a2a-e3631bf4e43e
# ╟─b6da2479-1545-4b1d-8d7f-07d6d1f67635
# ╟─c4cc482b-815b-4747-9f5a-5779d69086f7
# ╟─9016cba4-58f0-4b7f-91af-66faaf3fe99c
# ╟─828166f7-1a69-4952-9e3b-50a99a99789f
# ╟─24c4d724-5911-4534-a5c6-3ab86999df43
# ╠═5046166d-b6d8-4473-8823-5209aac59c84
# ╟─82d0539f-575c-4b98-8679-aefbd11f268e
# ╠═00cb5973-3047-4c57-9213-beae8f116113
# ╠═9e3c0e01-8eb6-4078-bc0f-019466afba5e
# ╟─c0bba3aa-d52c-4192-8eda-32d5d9f49a28
# ╟─ab9195d6-792d-4603-8605-228d479226c6
# ╟─e42c5e18-a647-4281-8a87-1b3c6c2abd33
# ╠═5d6a485d-90c4-4f76-a27e-497e8e12afd8
# ╠═987aeb82-267a-4ca9-bf21-c75a49edad70
# ╟─9eaf73e9-0f68-4e8b-9ae1-a42f050f695a
# ╟─36e6f838-8277-480b-b48d-f70e8fe011eb
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
# ╠═b81924b8-73f6-4b28-899c-ec417d538dd4
# ╟─c3d2ba03-e676-4f0f-bafd-feecd0e4e414
# ╠═97c0c719-fb73-4571-9a6c-629a98cc544d
# ╠═0b3945a8-0ae3-4c18-a9b7-a249eb530bcb
# ╠═4e790ffa-554d-4c46-af68-22ecb461fb7b
# ╟─4b141ffc-4100-47e3-941a-4e72c784ccf0
# ╟─219aafcb-17b1-4f5f-9c2b-9b713ba78b18
# ╟─2833e081-45d6-4f64-8d1e-b3a5895b7952
# ╟─5c714d3b-ac72-40dc-ba98-bb2b24435d4c
# ╟─573b8a38-5a9b-4d5f-a9f6-00a5255914f0
# ╟─1ca20976-757f-4e30-94d4-ee1276a614fb
# ╟─aa69d0e8-cbbb-436c-b488-5bb113cdf97f
# ╟─dc43d1bc-ea5c-43ca-af0c-fc150756fa76
# ╟─43d563ae-a435-417f-83c6-19b3b7d6e6ee
# ╟─11a5614b-c195-45a8-8be0-b99fda6c60fd
# ╟─f004ec01-1e27-4e30-9a53-23a299208846
# ╟─949ae6bc-315b-48f7-9382-1b0aee462b65
# ╟─f1b59139-4205-408b-9ba5-f6f9df5d0de0
# ╟─6a3176dd-587a-4e7d-9649-dff8af789c42
# ╟─a33f7f60-389f-4d26-9562-c60380eb1888
# ╟─e7669fea-5aff-4522-81bf-3356ce126b1f
# ╟─a32faf6c-a5bb-4005-ad42-188af732fba5
# ╟─102ca26d-a97a-4f1e-85fa-28d54305afbf
# ╟─ff3d162a-5325-496f-bb50-c119391729ba
# ╟─f0de92aa-3bcb-4bad-bb25-9f5c3e4c0b11
# ╟─91c8fa46-b9e7-4db2-8e20-0e6c5006483f
# ╟─92a4aa17-2e2d-45c2-a9a2-803d389077d5
# ╟─33b402b9-29c5-43d3-bb77-9b1a172229bb
# ╟─0a3ed179-b60b-4740-ab73-b176bba08d84
# ╟─47230bb3-de03-4353-9cbe-f974cc25411c
# ╟─6f76e32c-32a7-4d77-b1f9-0078807ec103
# ╠═c205ff23-f1e7-459f-9339-2c80ab68945f
# ╟─2bfe1d15-210d-43b4-ba4b-dec83f2363cd
# ╟─b5a3a660-8928-4097-b1c4-90f045d17444
# ╟─6005e786-d4e8-4eef-8d6c-cc07fe36ea17
# ╟─283fe6c9-6642-4bce-a639-696b92fcabb8
# ╟─0dc3b4d5-c66e-4fbe-a9fe-67f9212371cf
# ╠═53872a09-db29-4195-801f-54dd5c7f8dc3
# ╟─b8053536-0e98-4a72-badd-58d9adbcf5ca
# ╟─97dd43c3-1072-4060-b750-c898ce926861
# ╟─927bce0e-e018-4ecb-94e5-09812bf75936
# ╠═398da783-5e47-4d39-8048-4541aad6b8b5
# ╠═a6ae40e6-ea8c-46f1-b430-961c1185c087
# ╟─10beb8b2-0841-44c4-805e-8667da325b01
# ╟─58f65290-8ba9-4c27-94de-b28a5eac80a4
# ╟─c0daa659-f5f6-4e6b-9973-a399cf0ea788
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
