### A Pluto.jl notebook ###
# v0.19.11

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
using BenchmarkTools, Distributions, KernelDensity, LinearAlgebra, MCMCChains, Plots, PlutoUI, QuantEcon, Random, StatsBase, Statistics, StatsPlots

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
">ATS 872: Lecture 4</p>
<p style="text-align: center; font-size: 1.8rem;">
 Markov chain Monte Carlo
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

# ╔═╡ c53220c2-b590-4f40-829b-617d728c7d16
md"""

> Please note, a signicant portion of these notes are directly taken from [QuantEcon](https://julia.quantecon.org/tools_and_techniques/finite_markov.html#equation-fin-mc-fr) and [Bayesian Statistics using Julia](https://storopoli.io/Bayesian-Julia/pages/5_MCMC/). Please visit their respective sites. These individuals offer amazing free resources! 

"""

# ╔═╡ 000021af-87ce-4d6d-a315-153cecce5091
md" In this session we will be covering one of the most important topics of the course. This relates to computational work and we will most likely spend two lectures working through the material.  "

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# TableOfContents() # Uncomment to see TOC

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐ if you want."

# ╔═╡ 49c0eb58-a2b0-4e66-80c7-c030070ca93d
md""" ## General overview """

# ╔═╡ d86c8eae-9bcd-43a7-a9a7-1608af719f98
md""" Today we will cover the following topics

- Quick discussion on Monte Carlo methods
- Why do we need to use MCMC methods?
- The story of King Markov (Metropolis algorithm narrative)
- Discussion on finite state Markov chains
- Technical description of the Metropolis algorithm
- Quick introduction to the Gibbs sampling algorithm

"""

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Monte Carlo methods overview "

# ╔═╡ bd0dc0fd-63bc-492b-bba1-a3746ea4ec22
md"  
Monte Carlo methods were used even before computers, with Buffon, De Forest, Darwin, Galton, Pearson and Gosset working with these methods during the 18th and 19th century. The term Monte Carlo method was a term proposed by Metropolis, von Neumann or Ulam at the end of the 1940s. They worked together on the atomic bomb project. 

Bayesians started to have enough cheap computation time in 1990s so Monte Carlo methods became popular, with the BUGS (Bayesian inference Using Gibbs Sampling) project starting in 1989 with the last OpenBUGS release in 2014. Recent developments have resulted in the initial release of Stan in 2012, a probabilistic programming language. More recently, for Julia, there was the release of the `Turing.jl` package, which shares some of the features of Stan. 

With Monte Carlo methods one simulates draws from the target distribution. These draws can be treated as any observations and a collection of draws is considered a sample. One can use these draws, for example, to compute means, deviations, quantiles to draw histograms to marginalize, etc.

Monte Carlo refers to simulation methods. Evaluation points are selected stochastically (randomly). With deterministic methods (e.g. grid) evaluation points are selected by some deterministic rule. Good deterministic methods converge faster and need fewer function evaluations. 

The problem with grid sampling is that it is prone to the curse of dimensionality. Say we have 10 parameters. If we don't know beforehand where the posterior mass is then we need to choose a wide box for the grid and need to have enough grid points to get some of them where essential mass is (e.g. 50 or 1000 grid points per dimension). This means 50$^{10} \approx$ 1e17 grid points and 1000$^{10} \approx$ 1e30 grid points. Using R and an average computer, we can compute the density of a normal distribution about 20 million times per second. Evaluation in 1e17 grid points would take 150 years. Evaluation in 1e30 grid points would take 1 500 billion years!

"

# ╔═╡ 7460cf95-4fc1-4c04-99d7-b13c2014b37a
md" ## Indirect sampling "

# ╔═╡ d2fcc16f-075e-44d1-83ee-d9ab2372bfc7
md""" From the previous lecture we saw that direct sampling from the posterior is possible in some cases. This method is also preferred if we know the shape of the distribution since draws are independent and identically distributed (i.i.d). However, we are often in the situation where we don't have access to the posterior distribution. In this case we must use **indirect sampling**. The main methods for this approach are

1. Importance sampling
2. Markov chain Monte Carlo (MCMC)

We covered importance sampling last year, so you can view the lecture videos and slides from last year if you want to learn more about that. We will not go into much detail on it in these notes. Our focus is going to be on MCMC methods. Once again, these MCMC methods allow us to sample from unknown distributions and accordingly approximate some of the features of the distributions in question.  """

# ╔═╡ 17382827-e3fc-4038-aa47-9db30f7f45ae
md""" ### Markov chain Monte Carlo """

# ╔═╡ 26de0997-9548-49b5-9642-b401c6c42c41
md"""
We might ask ourselves at this point why getting draws from the posterior is so difficult?

The main computational barrier for Bayesian statistics is the denominator $p(y)$ of the Bayes formula. In the continuous case the denominator $p(y)$ becomes a very large and complicated integral to calculate:

$p(y) = \int_{\Theta} p(y | \theta) \times p(\theta) d\theta$

In many cases this integral becomes intractable (incalculable) and therefore we must find other ways to calculate the posterior probability $p(\theta \mid y)$ without using the denominator $p(y)$.

The purpose of this denominator is to normalize the posterior in order to make it a valid probability distribution. When we remove the denominator we have that the posterior is proportional to the prior multiplied by the likelihood, as discussed in previous lectures.

This is where Markov chain Monte Carlo comes in. As we stated above, MCMC is a broad class of computational tools for approximating integrals and generating samples from a posterior probability. MCMC is used when it is not possible to sample $\theta$ directly from the subsequent probabilistic distribution $p(\theta \mid y)$. Instead, we sample in an iterative manner such that at each step of the process we expect the distribution from which we sample to become increasingly similar to the posterior. All of this is to eliminate the (often impossible) calculation of the denominator of Bayes' rule. 

The basic idea underlying MCMC is to first specify a candidate function from which you obtain random samples. Given this candidate function we generate candidate draws and use those to approximate the target distribution. If certain conditions are met then the approximation will converge in distribution to the target distribution (we will also refer to it as the stationary distribution). There are many different MCMC methods, but we will focus primarily on the Metropolis algorithm and Gibbs sampling. There are newer methods, such as Hamiltonian Monte Carlo, which are slowly making there way into economics. 

Here is a link to an [interactive gallery](https://chi-feng.github.io/mcmc-demo/) of the different types of MCMC algorithms.

In the sections that follow we will first provide a nice narrative that helps establish the idea behind the Metropolis algorithm. In our discussion we will also touch on the idea of Markov chains, which are an essential part of the simulation process.
"""

# ╔═╡ 491b1cbf-bc99-4a31-9c2b-f2a8d0dc37c6
md" ### King Markov 👑 and advisor Metropolis 🧙🏻"

# ╔═╡ 5f8c67ac-c5c7-4999-8d22-417d8199ddac
md" For a good description of the problem, watch the [video](https://www.youtube.com/watch?v=v-j0UmWf3Us) by Richard McElreath. He wrote a book called **Statistical Rethinking**, which is aimed at social scientists and is much less math driven than a lot of the Bayesian textbooks out there. All the code is also available in R. There is also a [project](https://github.com/StatisticalRethinkingJulia) to code up all the material of the book in Julia using Turing and Stan.  "

# ╔═╡ 76853f63-4969-4823-a16b-d1033af26a2c
md" > Narrative for this section taken directly from [this book](https://rpruim.github.io/Kruschke-Notes/markov-chain-monte-carlo-mcmc.html#quick-intro-to-markov-chains). Code was originally written in R. "

# ╔═╡ e9868be9-2c58-45ef-96ad-0a016fdca540
md"""
King Markov is king of a **chain of 10 islands**. Rather than live in a palace, he lives in a royal boat. Each night the royal boat anchors in the harbor of one of the islands. The law declares that the king must harbor at each island in proportion to the population of the island.

The islands have different population sizes, some small and some larger. The contract that King Markov has with his people is that he must visit each island in proportion to its population size. 

King Markov is a peculiar man though, here are **some of his quirks**. 

1. He can’t stand record keeping (partially because he is not numerate). So he doesn’t know the populations on his islands and doesn’t keep track of which islands he has visited when.

2. He can’t stand routine (variety is the spice of his life), so he doesn’t want to know each night where he will be the next night.

He asks Advisor Metropolis to devise a way for him to obey the law but that

1. Randomly picks which island to stay at each night.
2. Doesn’t require him to remember where he has been in the past.
3. Doesn’t require him to remember the populations of all the islands.

Metropolis can ask the clerk on any island what the island’s population is whenever he needs to know. But it takes half a day to sail from one island to another, so he is limited in how much information he can obtain this way each day.
"""

# ╔═╡ 97a92a03-64ce-4f12-be98-f66233b27142
md"""
#### The crafty scheme from Metropolis
"""

# ╔═╡ 452fc619-f315-43ef-af4c-4770808f89fe
md"""
The following scheme is the one that Metropolis came up with...

1. Each morning, have breakfast with the island clerk and inquire about the population of the current island.

2. Randomly pick one of the 2 other islands. Remember that we are in a chain, so that we can only go to adjacent islands. The King doesn't like sailing across wide open bodies of water, since he is afraid of sea monsters 🦑 that live in the deep. Call that island the **proposal island** and travel there in the morning. One might think perhaps to flip a coin to choose the proposal island. 

3. Over lunch at the proposal island, inquire about its population.

4. If the proposal island has more people, stay at the proposal island for the night (since the King should prefer more populated islands).

5. If the proposal island has fewer people, stay at the proposal island with probability $r$, else return to the “current” island (ie, last night’s island).

6. Repeat this process, starting from the first step. 

Let us quickly define some terms. Let  $J(\theta_{b} ∣ \theta_{a})$ be the conditional probability of selecting island $\theta_{b}$ given $\theta_{a}$ is the current island. The value of $J$ does not depend on the populations of the islands (since the King can’t remember them).

Metropolis is convinced that for the right choices of $J$ and $r$, this will satisfy the law (and he is right). It seems like $r$ might need to depend on the populations of the current and proposal islands. But how? If $r$ is too large, the king will visit small islands too often. If  
$r$ is too small, he will visit large islands too often.

Fortunately, Metropolis knows about Markov chains. Unfortunately, some of you may not. So let’s learn a little bit about Markov chains and then figure out how Metropolis should choose $J$ and $r$.

"""

# ╔═╡ 68fab7b3-b3b7-41fa-89ee-cc5d729c5150
md" ### Markov chains "

# ╔═╡ 7354bf26-7530-4a84-ac4d-1dae6b19b623
md" Most of these notes draw from the [QuantEcon page](https://julia.quantecon.org/tools_and_techniques/finite_markov.html) on finite Markov chains. This is an important concept to understand, beyond the application of MCMC algorithms. Markov chains underlie many of the workhorse model of economics and finance. We will not go into too much detail on the theory, but will mention some of it along the way. Read the following section carefully, since it can be difficult on first reading. "

# ╔═╡ 5508d626-591f-4147-99ee-e85160162323
md" #### Fundamental concepts "

# ╔═╡ eb2624dd-5bf2-4a9f-a92e-a3ae7227dede
md"""

Notice that the notation for this section is a bit different from what we had before. Random variables in the section are represented by capital letters and probability distributions are given by $p$, as from before. If you have questions about notation please feel free to ask. I am trying to streamline notation across different sets of notes, but this isn't always easy, since notation is not always standard. 

A **stochastic matrix** is an $n \times n$ square matrix $P$ such that each element of $P$ is nonnegative and each row of $P$ sums to one. **Important**: Each row of $P$ can be regarded as a probability mass function over $n$ possible outcomes.

Stochastic matrices and Markov chains are closely linked.

Let $S$ be a finite set, called the **state space**, with $n$ elements $\{x_1, \ldots, x_{n}\}$. The elements of the set are referred to as the state values. 

A Markov chain $\{X_{t}\}$ on the state space $S$ is a sequence of random variables on $S$ that have the **Markov property**. This Markov property is a defining feature of the Markov chain. 

The Markov property is a memorylessness property. For any date $t$ and any state $x_{j} \in S$ we have that, 

$p(X_{t+1} = x_{j} \mid X_t ) = p(X_{t+1} = x_{j} \mid X_t, X_{t-1}, \ldots)$

The dynamics of the Markov chain are fully determined by the set of values 

$P(x_{i}, x_{j}) = p(X_{t+1} = x_{j} \mid X_{t} = x_{i})$

By construction $P(x_{i}, x_{j})$ is the probability of going from $x_{i}$ to $x_{j}$ in one unit of time. It is important to note that $P(x_{i}, \cdot)$ is the conditional distribution of $X_{t+1}$ given $X_{t} = x_{i}$. We can view $P$ as a stochastic matrix where 

$P_{i j}=P\left(x_{i}, x_{j}\right) \quad 1 \leq i, j \leq n$

If we have the stochastic matrix $P$ then we can generate a Markov chain in the following manner. 

1. Draw $X_{0}$ from some specified distribution
2. For each $t = 0, 1, \ldots$ draw $X_{t+1}$ from $P(X_{t}, \cdot)$

Let us illustrate these concepts with a basic example that relates to our island problem.  

"""


# ╔═╡ b1a9f137-10a3-4940-8948-1c44026ada6c
md" #### Island example"

# ╔═╡ 0d572678-5302-4715-b873-500004dcac78
md"""

Consider a situation where we only have two islands. At any point in time you can be at either island $1$ 🏖️ or island $2$ 🌴. We consider these to be the to state in our Markov model. Suppose that over the period of a month an inhabitant of island $1$ finds himself on island $2$ with probability $\alpha$. While someone who lives on island $2$ might travel to island $1$ with probability $\beta$.

"""

# ╔═╡ 9e907a32-3dfe-4118-9ec1-53593f632745
md"""
!!! note "Travel itinerary for September "
	🏖️ $\rightarrow$ 🌴 with probability $\alpha \in (0, 1)$. 🌴  $\rightarrow$ 🏖️ with probability $\beta \in (0, 1)$

"""

# ╔═╡ ac919514-9cdc-4144-99d0-a1b7ef82a4ed
md" The elements of our Markov model so far are the state space $S = \{1, 2\}$ and the elements of the stochastic matrix $P(1, 2) = \alpha$, $P(2, 1) = \beta$. The transition probabilities can then be written in matrix form as 

$P=\left(\begin{array}{cc}1-\alpha & \alpha \\ \beta & 1-\beta\end{array}\right)$

Once we have the values for $\alpha$ and $\beta$ we can answer questions like, what is the average duration that someone spends on an island. In the long run, what fraction of time does the person on island $1$ find herself there? Conditional on being on island $1$, what is the probability of going to island $2$ next year September? 
"

# ╔═╡ 086dc5e2-032c-4cea-9805-dd44567bc012
md"""

One way in which we can answer these questions is through **simulation**. We have already attempted some simulations, so this should not be completely new to you. To estimate the probability that a certain event will occur, we can simulate many times and simply count the number of times that the event occurs. Let us write our own simulation routine to see how this would work. 

**Note**: The algorithm that we are about to write is for instructive purposes only, it is a much better idea to work with already optimised code, such as the routines found at [QuantEcon](https://quantecon.org/quantecon-jl/).

"""

# ╔═╡ 27153344-0d5e-4bc1-ba11-2d302bf483ae
md" #### Simulating a Markov chain "

# ╔═╡ 4a65adda-688b-4ce5-9a8c-ebc874cfc969
md" Let us try and write something more advanced to generate a Markov chain. We have an idea of the ingredients that we need to generate the chain. In particular, we need a stochastic matrix $P$ and an initial state. If we don't have an initial state then we need a distribution from which we can draw the initial state. We then construct our Markov chain as discussed in one of the previous sections. Our function takes the following three arguments, 

- A stochastic matrix `P`
- An initial state `init`
- A positive integer `sample_size` representing the length of the time series the function should return

"

# ╔═╡ c9a6391e-2903-4cf3-9aa9-b2e20a7c15f1
md" For this example we will be sampling from the `Categorical distribution`. "

# ╔═╡ 8f22fc64-c0fc-4f10-8bac-bc355edc4780
d = Categorical([0.4, 0.6])

# ╔═╡ a081553f-35f8-46e1-925b-bd480cc691e5
[rand(d, 5) for _ = 1:3] # Provides 3 samples with size of 5 for from the Categorical distribution

# ╔═╡ 208d0d6b-f97a-4527-a519-1227c474fdc0
md" We can see that samples from the Categorical distribution align nicely with our state values in the state space. "

# ╔═╡ b531a759-ab64-4777-8b50-580c1f169576
function mc_sample_path(P; init = 1, sample_size = 1000)
    
	@assert size(P)[1] == size(P)[2] # Require that the matrix be square
    N = size(P)[1] # In our example P is 2 x 2 matrix, so this should be 2

    # create vector of discrete RVs for each row
    dists = [Categorical(P[i, :]) for i in 1:N] # N = 2 for this example

    # setup the simulation
    X = fill(0, sample_size) # allocate memory, or zeros(Int64, sample_size)
    X[1] = init # set the initial state

    for t in 2:sample_size
        dist = dists[X[t-1]] # get discrete RV from last state's transition distribution
        X[t] = rand(dist) # draw new value
    end
    return X
end;

# ╔═╡ 0a63d554-c9f0-4df0-91ae-54c59dc312a4
begin
	P = [0.4 0.6; 0.2 0.8]
	X₁ = mc_sample_path(P, init = 1, sample_size = 100_000); # note 100_000 = 100000
	μ₁ = count(X₁ .== 1)/length(X₁) # .== broadcasts test for equality. Could use mean(X .== 1)
end;

# ╔═╡ 14f6cfaf-50e5-4924-8d40-1a2d7d0a9c8a
md" The code illustrates the fraction of the sample that takes the value of $1$. In this case it will be about 25% of the time. Doesnt matter from which island you start, you will be at island $1$ roughly 25% of the time, if we allow for a long series of values to be drawn. " 

# ╔═╡ 48a4323f-db34-40bb-9026-39a4f1131c36
md" We could have also use the QuantEcon routine, with a function called `MarkovChain` that takes a stochastic matrix as input. We can then simulate the a Markov chain with the QuantEcon package. "

# ╔═╡ ddac3546-7949-49be-ac41-3d977d0b99cf
begin
	mc = MarkovChain(P)
	X₂ = simulate(mc, 100_000);
	μ₂ = count(X₂ .== 1)/length(X₂) # or mean(x -> x == 1, X)
end

# ╔═╡ c53eab7a-6849-4a2f-b7d2-4a400728cb11
md" The nice thing about here is that we can add state values in the form of strings (or even emojis if we really want). "

# ╔═╡ 1dcaa4e0-c2a9-4068-af52-39c0215625dc
mc₁ = MarkovChain(P, ["🏖️", "🌴"])

# ╔═╡ fede135f-4a5c-4d40-9341-8e168e186bec
simulate(mc₁, 10, init = 1) # start at 🏖️

# ╔═╡ 5084a86c-58ef-422e-a495-4bf831be3b1a
md" Now we can see how our representative island hopper is jumping between islands. "

# ╔═╡ a2fdceff-88f5-4cf0-b139-729344373d14
md"""
#### Marginal distributions

"""

# ╔═╡ 78793993-f6b1-487e-9891-4755697350b5
md"""

Suppose $\{X_{t}\}$ is a Markov chain with stochastic matrix $P$ where the distribution of the random variable $X_{t}$ is known to be $\psi_{t}$. What is the distribution for $X_{t+1}$ or other random variables in the Markov chain, such as $X_{t+m}$?

Our goal is to find $\psi_{t+1}$ given information on $\psi_t$ and the stochastic matrix $P$.

Pick any $x_{j} \in S$. The law of total probability tells us that,

$p(X_{t+1} = x_{j}) = \sum_{x_{i} \in S} p(X_{t+1} \mid x_{j}) \cdot p(X_{t} = x_{i})$

This shows that to get the probability of being at $x_{j}$ tomorrow, we account for all the ways in which this can happen and then sum those probabilities. 

We can rewrite this in terms of marginal and conditional probabilities, which gives, 

$\psi_{t+1}(x_{j}) = \sum_{x_{i} \in S} P(x_{i}, x_{j})\psi_{t}(x_{i})$

There are $n$ such equations, one for each $x_{j} \in S$. We can then think about $\psi{t+1}$ and $\psi_{t}$ as row vectors, so that we can summarise everything nicely in matrix notation. 

$\psi_{t+1} = \psi_{t} P$

To move the distribution forward one unit in time means simply postmultiplying by P. If we repeat this $m$ times we move forward $m$ steps into the future. This means that,

$\psi_{t+m} = \psi_{t} P^{m}$

We will not go into the notions of irreducibility and aperiodicity here, but it is worthwhile reading. Please refer to the QuantEcon notes for a good treatment. 

"""

# ╔═╡ 44e0f980-e912-4ad3-aa0d-9ab64d5fdfc9
md"""
#### Stationary distributions

"""

# ╔═╡ 018c4a59-3503-4d71-87d2-f150b0c8904b
md" We now know that we can shift probabilities forward one unit of time via postmultiplication by $P$. However, some distributions are invariant under this updating process. "

# ╔═╡ 813fc5a0-ef7c-4395-b2aa-00ecce7b8455
ψ = [0.25, 0.75];

# ╔═╡ e26acec5-ba7a-4448-b01f-519d48adb3ae
ψ' * P # The value of ψ did not change when postmultiplied by P = [0.4 0.6; 0.2 0.8]

# ╔═╡ fb077380-d79d-45cc-96c2-ab44acadd1e1
md"""

Distributions like these are referred to as **stationary** or **invariant**. 

More formally, a distribution $\psi^{\star}$ on $S$ is called stationary for $P$ if, 

$\psi^{\star} = \psi^{\star}P$

If the distribution of $X_{0}$ is a stationary distribution, then $X_{t}$ will have the same distribution for all $t$. Hence stationary distributions have a natural interpretation as stochastic steady states. 

Under some conditions (irreducibility and aperiodicity) we have theorems that tell us that the stochastic matrix has exactly one stationary distribution and that for any initial distribution (postmulitplied by the stochatic matrix) we approach the stationary distribution as time goes to infinity. A stochastic matrix that satsifies the conditions of the theorem is normally called **uniformly ergodic**. We can also say that the Markov chain is **regular** in this case.  

"""

# ╔═╡ dc666834-d602-4800-934b-2c8caae7beb5
md" There is a function in the QuantEcon package that allows us to calculate this stationary distribution. "

# ╔═╡ 70b19d3d-4a6f-450a-8e8e-0e7bdaefaab1
stationary_distributions(mc)

# ╔═╡ ca57a81c-5d42-4415-a979-cec0d9c18391
md"""
The second part of the theorem above tells us that the distribution of $X_{t}$ converges to the stationary distribution regardless of where we start off. The convergence result is illustrated in the next figure.
"""

# ╔═╡ 3f19c62e-e3e2-4650-ba3a-ea8b4f228501
P₂ = [0.971 0.029 0.000
	  0.145 0.778 0.077
      0.000 0.508 0.492]; # stochastic matrix

# ╔═╡ caff2e40-7b71-4a27-b661-1e4c05ec93a2
ψ₀ = [0.0 0.2 0.8]; # initial distribution

# ╔═╡ 433129ff-9c7d-4388-be68-1b06ff26fec0
function dynamics(ψ, P, n)
	
	#t = 20 # path length
	x_vals = zeros(n)
	y_vals = similar(x_vals)
	z_vals = similar(x_vals)
	
	for i in 1:n
    	x_vals[i] = ψ[1]
    	y_vals[i] = ψ[2]
    	z_vals[i] = ψ[3]
    	ψ = ψ * P # update distribution
	end
	return ψ, x_vals, y_vals, z_vals
end

# ╔═╡ ed9890f1-061c-4ada-90e7-a39e08af7af8
n = (@bind n Slider(2:20, show_value = true, default=2))

# ╔═╡ 106087bc-1a39-4e97-b77a-bafe8b692844
begin
	gr()
	colors = [repeat([:red], 20); :black]
	mc₂ = MarkovChain(P₂)
	ψ_star = stationary_distributions(mc₂)[1]
	x_star, y_star, z_star = ψ_star # unpack the stationary dist
	plt = scatter([dynamics(ψ₀, P₂, n)[2]; x_star], [dynamics(ψ₀, P₂, n)[3]; y_star], [dynamics(ψ₀, P₂, n)[4]; z_star], color = colors,
		              gridalpha = 0.5, legend = :none)
	plot!(plt, camera = (45,45))
end

# ╔═╡ 80154dfd-dac5-4579-8cdc-a14b9862df18
md"""  #### Island time series example

Recall our model of island hopping dynamics. Assuming $\alpha \in (0,1)$ and $\beta \in (0,1)$, the uniform ergodicity condition is satisfied. Let $\psi^{\star}=(q, 1−q)$ be the stationary distribution, so that $q$ corresponds to being on island $1$. Using $\psi^{\star}=\psi^{\star}P$ and a bit of algebra yields

$q = \frac{\beta}{\alpha + \beta}$

This is, in some sense, a steady state probability of being on island $1$. Not surprisingly it tends to zero as $\beta \rightarrow 0$, and to one as $\alpha \rightarrow 0$. It represents the long run fraction of time spent on island $1$.

In other words, if $\{X_{t}\}$ represents a Markov chain, then $\bar X_m \to q$ as $m \to \infty$ where

$\bar X_m := \frac{1}{m} \sum_{t = 1}^m \mathbf{1}\{X_t = 1\}$

In this example we want to generate a simulated time series $\{X_{t}\}$ of length $3000$, starting at $X_{0} = 1$. We will then plot $\bar X_m - q$ against $m$, where $q$ is as defined above. Repeat this step, but taking the initial conidtion as $2$, representing a starting point on the second island. The result looks something like the following. 

"""

# ╔═╡ 3742f58c-de0b-40db-bba0-59a4bf9e58ad
begin
	α₃ = 0.4 # probability of staying on island 1
	β₃ = 0.6 # probability of staying on island 2
	N₃ = 3000
	p̄₃ = β₃ / (α₃ + β₃) # steady-state probabilities
	P₃ = [1 - α₃   α₃
	     β₃   1 - β₃] # stochastic matrix
	mc₃ = MarkovChain(P₃)
	labels = ["start at 1", "start at 2"]
	y_vals₃ = Array{Vector}(undef, 2) # sample paths holder
	
	for x0 in 1:2
	    X₃ = simulate_indices(mc₃, N₃; init = x0) # generate the sample path
	    X̄₃ = cumsum(X₃ .== 1) ./ (1:N₃) # compute state fraction. ./ required for precedence
	    y_vals₃[x0] = X̄₃ .- p̄₃ # plot divergence from steady state
	end
	
	plot(y_vals₃, color = [:blue :green], fillrange = 0, fillalpha = 0.3,
	     ylims = (-0.25, 0.25), label = reshape(labels, 1, length(labels)))
end

# ╔═╡ 635cd82c-8fa6-4bf3-b586-fd2ec915c4b7
md" So we have now spent a lot of time gaining some basic understanding on Markov chains. Let us get back to our King Markov example and see how to apply our newfound knowledge. "

# ╔═╡ 411d9644-55c7-4cef-81d1-7ca41181d3fa
md" ### Getting back to King Markov "

# ╔═╡ e9819877-d4c1-4378-8430-04f43d057f1f
md"""

**Note**: Our jumping rule $J$ is very simple right now, we just flip a coin. Our selection of $r$ is going to be the ratio of the size of the population of the islands. 

Metropolis learns from Markov chain theory that for his scheme to work, he must choose $J$ and $r$ such that the algorithm results in a regular Markov chain with uniformly ergodic stochastic matrix $P$. If $\psi$ represents the law-prescribed probabilities for island harboring, then $\psi = \psi P$.  

From our previous example, if $r$ is between $0$ and $1$, and the jumping rule allows us to get to all the islands (eventually), then the Markov chain will be regular, so there will be a stationary (limiting) distribution. But the stationary distribution must be the one the law requires. 

It suffices to show (which we will do later on) that if the law is satisfied at time $t$ it is satisfied at time $t+1$. We will get back to this idea in the section discussing the Metropolis algorithm. This example provides some required background information to make the Metropolis algorithm easier to understand. 

Let us provide a small code snippet to show how King Markov moves around the archipelago. His decision on which island is going to be the proposal island is based on a coin flip in this example, but we will generalise the selection of $J$ in the next section. We want to keep things simple for now.  

"""

# ╔═╡ f5ba6b1c-c6ea-4bed-8dd1-bc35d0f3d75b
# Super simple algorithm, mostly for illustrative purposes. 
function KingMarkovSimple(n, current = 10)

	# initialise the vector
	positions = zeros(n)
	
	for i in 1:n
  		# record current position
  		positions[i] = current
		
  		# flip coin to generate proposal
  		proposal = current + sample([-1, 1])
		
 	 	# now make sure he loops around the archipelago
  		if proposal < 1 proposal = 10 end
		if proposal > 10 proposal = 1 end
		
  		# move?
  		prob_move = proposal / current # Selection of r
  		current   = rand(Uniform()) < prob_move ? proposal : current
	end
	return positions
end;

# ╔═╡ bc1b3e2c-9501-4e3f-a377-02e9c13418c1
md" Below we show a figure indicating the islands that King Markov has visited in his travels. "

# ╔═╡ 039f07f3-dfc8-4337-b807-c1e9f0e2bdf0
weeks_traveled = (@bind k Slider(40:20:400, show_value = true, default=40))

# ╔═╡ 59ebf35d-9d62-4268-b33e-bbcc14e8a23d
travels = KingMarkovSimple(k);

# ╔═╡ 0e73488b-9981-44dd-b7cc-d314e8f123c1
begin
	scatter(travels, color = :steelblue, alpha = 1, markershape = :hexagon)
	plot!(travels, legend = false, alpha = 0.3, lw = 2, color = :black, size = (700, 400), title = "Journey of King Markov", ylab = "Islands", xlab = "Weeks")
end

# ╔═╡ 6791bc6f-ef88-4894-881b-6a89d836efe2
md" We would expect that our posterior distribution would be almost like a staircase, with the islands with smaller population being visited less. If we increase the number of iterations this algorithm is allowed to run, we will eventually get to the correct posterior distribution. However, this algorithm can be quite slow. It will explore the posterior space, but it takes quite a long time to do so. "

# ╔═╡ 0c26c6f2-f96f-4bdc-9379-81a76179bd11
histogram(travels, bins = 10, color = :steelblue, alpha = 0.8, legend = false, normalize = true)

# ╔═╡ ec1270ca-5943-4fe7-8017-6904c72673f0
md" Now let us continue to a more formal / general discussion of the Metropolis algorithm. "

# ╔═╡ f7311d0b-a74c-4a15-be37-5dd8e236cf3d
md" ### Metropolis algorithm "

# ╔═╡ 9e9af560-3898-4bb2-8aa7-0e660f738a72
md" The Metropolis algorithm is one of the most celebrated algorithms of the 20th century and has been influential in many disciplines. From our previous example we know everything we need to know to understand the Metropolis algorithm more formally. In this section we will write out the algorithm in full and also provide the code to show how we can explore the posterior probability distribution space using this method. "

# ╔═╡ c71cc346-e48e-4066-8145-f1aee47c4322
md""" Suppose that we want to sample the parameter vector $\theta$ from the posterior distribution $p(\theta \mid y)$, but it is not possible to draw directly from the distribution. One way to do this is to construct a Markov chain of samples from a transition density (also known as the proposal distribution). This density satisfies the Markov property, in which the current state only depends on the previous state. Under certain conditions, this Markov chain will converge in distribution to the true posterior distribution (see the next section for a discussion) for a large enough set of draws. """

# ╔═╡ c3046dbc-e894-4194-aa54-4f4f28f1066b
md"""

The general Metropolis algorithm can be written as follows,

1. Choose a starting point $\theta^0$
2. For $t=1,2,\ldots$ pick a proposal $\theta^{*}$ from the proposal distribution
          $J_t(\theta^{*} \mid \theta^{t-1})$. The proposal distribution has to be symmetric, i.e.
          $J_t(\theta_a \mid \theta_b)=J_t(\theta_b \mid \theta_a)$, for all $\theta_a,\theta_b$
3. Calculate the acceptance ratio

$\begin{equation*}
            r=\frac{p(\theta^{*} \mid y)}{p(\theta^{t-1} \mid y)}
\end{equation*}$
4. Set the **update probability**

$\theta^{t}= \begin{cases}\theta^{*} & \text { with probability } \min (r, 1) \\ \theta^{t-1} & \text { otherwise }\end{cases}$ 

i.e. if $p(\theta^{*} \mid y)>p(\theta^{t-1})$ accept the proposal always and otherwise reject the proposal with probability $r$.

Rejection of a proposal increments the time $t$ also by one, so the new state is the same as previous.

Step 4 is executed by generating a random number from $\mathcal{U}(0,1)$. 

**Note**: $p(\theta^* \mid y)$ and $p(\theta^{t-1} \mid y)$ have the same normalization terms, and thus instead of $p(\cdot \mid y)$, unnormalized $q(\cdot \mid y)$ can be used, as the normalization terms cancel out!


"""


# ╔═╡ aad23932-c61e-4308-956b-c2d64d85ac93
md"""

#### Why does this algorithm work? 

"""

# ╔═╡ 32b7f7aa-0a56-4cee-87a7-339826fa5c1e
md"""

Intuitively more draws from the higher density areas as jumps to higher density are always accepted and only some of the jumps to the lower density are accepted. 

Theoretically, we need to prove that the simulated series is a Markov chain which has unique stationary distribution. We also need to prove that this stationary distribution is the desired target distribution. 

In order to prove this series is a Markov chain with unique stationary distribution, we have to show that it is irreducible, aperiodic and recurrent. We have mentioned these conditions in the previous section of the Markov chain discussion. 

In short, irreducibilitiy refers to the fact that there is a positive probability of eventually reaching any state from any other state. Aperiodic refers to the fact that return times are not periodic and recurrent reflects the fact that probability to return to certain state is $1$.

Proving that this is the desired target is also relatively straightforward. 

We consider starting algorithm at time $t-1$ with a draw $\theta^{t-1} \sim p(\theta|y)$. Consider any two such points $\theta_a$ and $\theta_b$ drawn from $p(\theta|y)$ and labeled so that $p(\theta_b|y)\geq p(\theta_a|y)$. The unconditional probability density of a transition from $\theta_a$ to $\theta_b$ is

$\begin{equation*}
        p(\theta^{t-1}=\theta_a,\theta^{t}=\theta_b)=
        p(\theta_a|y)J_t(\theta_b|\theta_a),
\end{equation*}$

The unconditional probability density of a transition from $\theta_b$ to $\theta_a$ is

$\begin{align*}
       &p(\theta^{t}=\theta_a,\theta^{t-1}=\theta_b) \\
       & = p(\theta_b|y)J_t(\theta_a|\theta_b)\left(\frac{p(\theta_a|y)}{p(\theta_b|y)}\right)\\
       &  =   p(\theta_a|y)J_t(\theta_a|\theta_b),
\end{align*}$

which is the same as the probability of transition from $\theta_a$ to $\theta_b$, since we have required that $J_t(\cdot|\cdot)$ is symmetric. Since their joint distribution is symmetric, $\theta^t$ and $\theta^{t-1}$ have the same marginal distributions, and so $p(\theta|y)$ is the stationary distribution of the Markov chain of $\theta$.

"""

# ╔═╡ 2799a8c7-a93d-4577-a5ce-9f771538634b
md""" #### Practical implementation # 1 """

# ╔═╡ 2ea2563b-0a28-42f4-ac43-f12f8d3bcfa7
md""" In this section we are using code from [Jamie Cross](https://github.com/Jamie-L-Cross/Bayes/blob/master/4_MCMC.ipynb). For this example we will look at the Metropolis-Hastings algorithm, which is essentially the same as the Metropolis algorithm. The **Metropolis-Hastings** algorithm is a extension of the Metropolis algorithm that incorporates a non-symmertic proposal distribution.  

**Example**: Suppose that we want to sample obtain random samples from
$$f(x) = \frac{1}{2}\exp(-|x|)$$.

Since the support of this function is the entire real line, we can use the standard Normal distribution $N(0,1)$ as a transition density. This is done in the following Julia code. The following example illustrates the independence chain MH algorithm. Another option that is frequently encountered is the Random Walk MH algorithm. """

# ╔═╡ 60bf263c-013c-4e6e-9bbe-2b9683f6eb83
begin
	## Independence chain MH algorithm
	## Create functions
	
	# Target distribution
	p(x) = exp(-abs(x))/(2);
		
	# Proposal distribution
	J(x) = pdf(Normal(0,1),x);
		
	# Acceptance ratio
	r(x,y) = min(1, p(x)*J(y)/(p(y)*J(x)));
end;

# ╔═╡ 155dc645-a2a5-49ec-a836-95f25eed9a88
begin
	
	Random.seed!(7244)
	
	## Independence chain MH sampler
	# Set-up
	D = 10000; # number of draws in Markov chain
	θ = zeros(D); # storage vector
	θ[1] = -5; # initial condition
	
	#Markov chain
	for d in 2:D
	#1. Sample candidate
	    θ_c = rand(Normal(0,1));
	#2. Update
	    alp = r(θ_c, θ[d-1]);
	    u = rand(Uniform(0,1));
	    if alp >= u
	        θ[d] = θ_c;
	    else
	        θ[d] = θ[d-1];
	    end
	end
	
	## Summary
	p1 = plot(θ[2500:10000], title = "Markov chain of draws", legend = false, lc = :steelblue);
	p2 = histogram(θ, title = "Empirical distribution", legend = false, alpha = 0.5, color = :steelblue);
end

# ╔═╡ 7087ac52-c5ae-4398-b4d5-00955b742d84
p1

# ╔═╡ 1e519151-e4e7-46e8-814a-c80ef77ff1e1
md"""

#### Practical implementation # 2

"""

# ╔═╡ 9e7350e2-cd06-4fb5-88c2-1888748dc136
md""" This section still needs to be done, probably for next year. However it implements the Random Walk Metropolis-Hastings algorithm """

# ╔═╡ 3e88a1c5-c1b7-4f1e-b615-a02b6b40de6e
md"""

#### Practical implementation # 3

"""

# ╔═╡ 1970bc03-ff14-4819-86a6-0a8b802a9f8e
md"""

For this course it is more likely that we will be using Gibbs sampling. However, let us show a representation of the Metropolis algorithm. The code is sourced from [José Eduardo Storopoli](https://storopoli.io/Bayesian-Julia/pages/5_MCMC/). It incorporates some good programming principles, so it is worthwhile going through it in more detail. This example is a bit more complicated, so if you are having trouble following the steps it isn't a problem. 

We use an example where we want to explore the multivariate normal distribution of two random variables $\theta_1$ and $\theta_2$, where $\mu_{\theta_1}=\mu_{\theta_2}=0$ and $\sigma_{\theta_1}=\sigma_{\theta_2}=1$. This gives us, 

$$\begin{gathered}
{\left[\begin{array}{c}
\theta_1 \\
\theta_2
\end{array}\right] \sim \text { Multivariate Normal }\left(\left[\begin{array}{l}
0 \\
0
\end{array}\right], \mathbf{\Sigma}\right),} \\
\boldsymbol{\Sigma} \sim\left(\begin{array}{ll}
1 & \rho \\
\rho & 1
\end{array}\right)
\end{gathered}$$

In this example we want to assign a value to $\rho$ which gives the correlation between the two variables. The code below is a Metropolis sampler for the example above. The proposal distributions for $\theta_1$ and $\theta_2$ are given as, 

$$\begin{aligned}
&\theta_1 \sim \text { Uniform }\left(\theta_1-\frac{\text { width }}{2}, \theta_1+\frac{\text { width }}{2}\right) \\
&\theta_2 \sim \text { Uniform }\left(\theta_2-\frac{\text { width }}{2}, \theta_2+\frac{\text { width }}{2}\right)
\end{aligned}$$

It is easier to work with probability logs than absolute values, so we compute $r$ as follows, 

$\begin{aligned} r=& \frac{ \left.\text { PDF ( Multivariate Normal }\left(\left[\begin{array}{l}{\theta_1}_{\text {proposed }} \\ {\theta_2}_{\text {proposed }}\end{array}\right]\right) \mid \text { Multivariate Normal }\left(\left[\begin{array}{c}\mu_{\theta_1} \\ \mu_{\theta_2}\end{array}\right], \mathbf{\Sigma}\right)\right)}{ \left.\text { PDF (Multivariate Normal }\left(\left[\begin{array}{l}{\theta_1}_{\text {current }} \\ {\theta_2}_{\text {current }}\end{array}\right]\right) \mid \text { Multivariate Normal }\left(\left[\begin{array}{c}\mu_{\theta_1} \\ \mu_{\theta_2}\end{array}\right], \mathbf{\Sigma}\right)\right)} \\ &=\frac{P D F_{\text {proposed }}}{P D F_{\text {current }}} \\ &=\exp \left(\log \left(\mathrm{PDF}_{\text {proposed }}\right)-\log \left(\mathrm{PDF}_{\text {current }}\right)\right) \end{aligned}$

"""

# ╔═╡ 24e07f29-a534-4bdc-b26d-353484aad92b
md""" Below is the implementation in Julia """

# ╔═╡ 6b4b1e00-c2cf-4c06-8598-7a16965e73e3
function metropolis(S::Int64, width::Float64, ρ::Float64;
                    μ_x::Float64=0.0, μ_y::Float64=0.0,
                    σ_x::Float64=1.0, σ_y::Float64=1.0,
                    start_x=-2.5, start_y=2.5,
                    seed=123)
	
    rgn = MersenneTwister(seed)
    binormal = MvNormal([μ_x; μ_y], [σ_x ρ; ρ σ_y])
    draws = Matrix{Float64}(undef, S, 2)
    accepted = 0::Int64;
    x = start_x; y = start_y
	
    @inbounds draws[1, :] = [x y] # inbounds will make code run faster, but use sparingly
    
	for s in 2:S
        x_ = rand(rgn, Uniform(x - width, x + width)) # proposal distribution for X
        y_ = rand(rgn, Uniform(y - width, y + width)) # proposal distribution for Y
        r = exp(logpdf(binormal, [x_, y_]) - logpdf(binormal, [x, y])) # acceptance ratio

        if r > rand(rgn, Uniform()) # accept or reject the proposal, if accepted increase count 
            x = x_
            y = y_
            accepted += 1
        end
        @inbounds draws[s, :] = [x y]
    end
    return draws
end;

# ╔═╡ 3453a862-d9ce-4802-ace7-8b825641b4e2
begin
	const S = 5_000
	const width = 2.75
	const ρ = 0.8
	
	X_met = metropolis(S, width, ρ);
end;

# ╔═╡ dbaa28bc-021f-4805-b265-19b9257bbd02
X_met[1:10, :]

# ╔═╡ d7b27485-7659-4bc6-a199-4767739da218
chain_met = Chains(X_met, [:X, :Y]);

# ╔═╡ 85c4a795-98f8-4bc4-89f5-b2a25671b070
summarystats(chain_met)

# ╔═╡ 47063329-5039-4e7f-b8d0-73e991cb3772
mean(summarystats(chain_met)[:, :ess]) / S

# ╔═╡ a19a4458-b167-46ad-97a1-156b789be81a
begin
	const μ = [0, 0]
	const Σ = [1 0.8; 0.8 1]
end;

# ╔═╡ 29298f63-a7d7-4aaa-b888-3e955841f7f5
iterations = (@bind j Slider(1:100, show_value = true, default=1))

# ╔═╡ 364377f1-3528-4e4a-99d1-91b96c546346
begin
	plt₁ = covellipse(μ, Σ,
	    n_std=1.64, # 5% - 95% quantiles
	    xlims=(-3, 3), ylims=(-3, 3),
	    alpha=0.3,
	    c=:steelblue,
	    label="90% HPD",
	    xlabel="θ1", ylabel="θ2")
	
	scatter(plt₁, (X_met[j, 1], X_met[j, 2]), label=false, mc=:red, ma=0.7)
	plot!(X_met[j:j + 1, 1], X_met[j:j + 1, 2], seriestype=:path, lc=:green,
		la=0.5, label=false, lw = 2)
end

# ╔═╡ 492d4b83-e736-4ca4-92d0-2bdb7973b85f
begin
	const warmup = 1_000
	
	scatter((X_met[warmup:warmup + 1_000, 1], X_met[warmup:warmup + 1_000, 2]),
	         label=false, mc=:red, ma=0.3,
	         xlims=(-3, 3), ylims=(-3, 3),
	         xlabel="θ1", ylabel="θ2")
	
	covellipse!(μ, Σ,
	    n_std=1.64, # 5% - 95% quantiles
	    xlims=(-3, 3), ylims=(-3, 3),
	    alpha=0.5,
	    c=:steelblue,
	    label="90% HPD")
end

# ╔═╡ b1bc647a-32f1-4ec8-a60e-f8e43e95be2d
begin
	scatter((X_met[warmup:end, 1], X_met[warmup:end, 2]),
	         label=false, mc=:red, ma=0.3,
	         xlims=(-3, 3), ylims=(-3, 3),
	         xlabel="θ1", ylabel="θ2")
	
	covellipse!(μ, Σ,
	    n_std=1.64, # 5% - 95% quantiles
	    xlims=(-3, 3), ylims=(-3, 3),
	    alpha=0.5,
	    c=:steelblue,
	    label="90% HPD")
end

# ╔═╡ 1234563c-fb5b-42ea-8f5b-abf8adf34e26
md"""

### MCMC and parallel programming
"""

# ╔═╡ a370cec5-904c-43fc-94b0-523100b1fd54
md""" If we have time in class we will have a quick discussion on parallel programming with MCMC methods. In the case of these methods it is often possible to leverage parallel computation to reduce the time it takes to map the posterior distribution. """

# ╔═╡ c22518e2-6cac-451c-bacc-15346dda54a4
md""" ## Gibbs sampling (WIP) """

# ╔═╡ c7401978-aead-42ff-a8ee-333336afde2b
md""" **NB**: Still experimenting with some material in this section. Not ready for class. """

# ╔═╡ 0de3f161-b749-491e-ae32-4b04d5d8f851
md""" We will cover Gibbs sampling in future sessions, so we provide a quick summary of the procedure here. It will be a crucial method for VARs and TVP-VARs. The reason to use Gibbs sampling is because of the low acceptance rate that is often encounted in the Metropolis algorithm. With this method all proposals are accepted. This algorithm excels when we have a multidimensional sample space. 

"""

# ╔═╡ 5373a12c-4732-4d1c-9c71-d113d5c6a5b3
md"""

The basic algorithm is simple and can be illustrated in a few steps. This algorithm entails iterative sampling of parameters conditioned on other parameters. For this example $\theta$ is the parameter of interest and $y$ is the data.

1. Define $p(\theta_{1}), p(\theta_{2}), \ldots, p(\theta_{n})$.
2. Sample a starting point $\theta_{1}^{0}, \theta_{2}^{0}, \ldots, \theta_{n}^{0}$

3. For $t = 1, 2, \ldots$

$\begin{aligned} \theta_{1}^{t} & \sim p\left(\theta_{1} \mid \theta_{2}^{0}, \ldots, \theta_{n}^{0}\right) \\ \theta_{2}^{t} & \sim p\left(\theta_{2} \mid \theta_{1}^{t-1}, \ldots, \theta_{n}^{0}\right) \\ \quad &: \\ \theta_{n}^{t} & \sim p\left(\theta_{n} \mid \theta_{1}^{t-1}, \ldots, \theta_{n-1}^{t-1}\right) \end{aligned}$ 

"""

# ╔═╡ 12819f3d-bed0-4af6-82d8-3be5e6c79b3a
md""" We can showcase the idea behind Gibbs sampling with a simple case. Assume that you have a model that gives a bivariate Normal posterior (similar to the example in the previous section),

$$\begin{equation}
\theta = \left(\begin{array}{l}
\theta_{1} \\
\theta_{2}
\end{array}\right) \sim N\left(\left[\begin{array}{l}
0 \\
0
\end{array}\right],\left[\begin{array}{ll}
1 & \rho \\
\rho & 1
\end{array}\right]\right)
\end{equation}$$ 

where $|\rho| < 1$ is the known posterior correlation between $\theta_1$ and $\theta_2$. 

The first thing we want to write a program that uses Monte Carlo integration to calculate the posterior means and standard deviations of $\theta_1$ and $\theta_2$.

"""

# ╔═╡ d6e39812-f1c2-494e-a48a-04be4a7ba6a1
function monte_carlo(r₁, ρ₁, r₀ = 100)
		
	Random.seed!(1)
	
	Σ₁ = [1.0 ρ₁; ρ₁ 1.0] # variance-covariance matrix
	MC = zeros(2, 1) # initialise Monte Carlo sum to zero
	MC2 = zeros(2, 1)
	
	# Monte Carlo integration
	for i in 1:r₁
		
		MCdraw = rand(MvNormal([0.0; 0.0], Σ₁))
		
		if i > r₀
			MC = MC + MCdraw
			MC2 = MC2 + MCdraw .^ 2
		end
	end
	return MC, MC2
end;

# ╔═╡ 918f3cd0-f6f9-4d23-9f5e-1ef21617e852
md""" Next we will use Gibbs sampling to calculate the posterior means and standard deviations of $\theta_1$ and $\theta_2$ """

# ╔═╡ 7bca38d2-3918-4cc6-9869-983cda421aee
function gibbs_sampler(r₁, ρ₁, r₀ = 100)
		
	Random.seed!(1)
		
	drawθ₂ = 1 # starting value for θ₂
	
	θ_gibbs = zeros(2, 1)
	θ_gibbs2 = zeros(2, 1)
	
	for i in 1:r₁
		
		# first block
		avgθ₁ = ρ₁ * drawθ₂
		varθ₁ = 1 - ρ₁ .^ 2
		drawθ₁ = avgθ₁ - rand(Normal(0, varθ₁))
		
		# second block
		avgθ₂ = ρ₁ * drawθ₁
		varθ₂ = 1 - ρ₁ .^ 2
		drawθ₂ = avgθ₁ - rand(Normal(0, varθ₂))
		
		if i > r₀
			
			θ_draw = [drawθ₁; drawθ₂]
			θ_gibbs = θ_gibbs + θ_draw
			θ_gibbs2 = θ_gibbs + θ_draw .^2
		end
	end
	return θ_gibbs, θ_gibbs2
end;

# ╔═╡ 4ab65a54-d7e9-4949-a494-91886f8ee1e7
replications = (@bind r₁ Slider(2000:100:100000, show_value = true, default=2000))

# ╔═╡ 9ccd3d2d-f178-4dd5-96ab-a63e9f97a3ab
rho = (@bind ρ₁ Slider(0:0.01:0.99, show_value = true, default=0.1))

# ╔═╡ 54c1d150-b8ff-4e95-ad03-6a0f2124e495
md""" Now we can set $\rho$ and compare the results from the above calculations for the posterior means and standard deviations, while increasing the number of draws. This allows us to compare accuracy of the algorithms. """

# ╔═╡ acef097a-7e53-4177-813a-4f69ad83ff42
begin
	# Mean measures for Monte Carlo integration
	
	MC_θ = monte_carlo(r₁, ρ₁)[1] ./ r₁
	MC_θ2 = monte_carlo(r₁, ρ₁)[2] ./ r₁
	MC_θsd = sqrt.(MC_θ2 .- MC_θ .^2)
end

# ╔═╡ 70a1198f-05e5-40f0-8f95-c0056c55cf05
begin
	# Mean measures for Monte Carlo integration
	
	gibbs_θ = gibbs_sampler(r₁, ρ₁)[1] ./ r₁
	gibbs_θ2 = gibbs_sampler(r₁, ρ₁)[2] ./ r₁
	gibbs_θsd = sqrt.(Complex.(gibbs_θ2 .- gibbs_θ .^2))
end

# ╔═╡ 2662a1e1-2135-4d3f-8cb8-65d38f944971
md""" How does the correlation between the two parameters of interest affect the performance of the Gibbs sampler?  """

# ╔═╡ ac6f7203-3758-4f20-bd64-9110e8846326
md""" Finally, plot the results of the Gibbs sampler against Monte Carlo integration to see how an increase inthe number of draws affects the accuracy of the method.  """

# ╔═╡ 2446e3cf-7a7f-4675-99c0-114352d4384d
md""" #### Detailed algorithm """

# ╔═╡ 02422bf4-ec7f-4428-8067-3091e2a70ba4
md"""

The idea can be extended to a more general setting as follows. 

We start this part with an assumption on our ability to sample from **full conditional distributions**. Suppose that there exits a partition $\boldsymbol{\theta}=\{\boldsymbol{\theta}_{1},\dots,\boldsymbol{\theta}_{B}\}$, in which $\boldsymbol{\theta}_{i}$, $i=1,\dots,B$ may be scalars or vectors, and that we can sample from each component of the *full conditional distributions*
1. $p(\theta_{1}|\mathbf{y},\theta_{2},\dots,\theta_{B})$
2. $\ldots$
3. $p(\theta_{B}|\mathbf{y},\theta_{1},\dots,\theta_{B-1})$

Then, we can use these conditional distributions as the transition density in the MH algorithm. In this case the update probability equals one, so each candidate draw is always accepted. This idea is referred to as *Gibbs sampling* as is summarized in the following algorithm.

**Gibbs Sampling**: Let $\boldsymbol{\theta}_{d,b}$ denote the dth draw of the bth block of a partition $\boldsymbol{\theta}=\{\boldsymbol{\theta}_{1},\dots,\boldsymbol{\theta}_{B}\}$. Then, given a set of initial conditions $\boldsymbol{\theta}_{0,1},\dots,\boldsymbol{\theta}_{0,B}$ (typically a guess), the Gibbs sampling algorithm works as follows
1. **Sample** a draw from the full set of conditional distributions: 
    1. $\boldsymbol{\theta}_{1,1}\sim p(\boldsymbol{\theta}_{1,1}|\mathbf{y},\boldsymbol{\theta}_{0,2},\dots,\boldsymbol{\theta}_{0,B})$
    2. $\boldsymbol{\theta}_{1,2}\sim p(\boldsymbol{\theta}_{1,2}|\mathbf{y},\boldsymbol{\theta}_{1,1},\boldsymbol{\theta}_{0,3},\dots,\boldsymbol{\theta}_{0,B})$
    3. $\ldots$
    4. $\boldsymbol{\theta}_{1,B}\sim p(\theta_{1,B}|\mathbf{y},\theta_{1,1},\dots,\theta_{0,B-1})$
2. **Repeat** for $d=2,\dots,D$, i.e. 
    1. $\boldsymbol{\theta}_{d,1}\sim p(\theta_{d,1}|\mathbf{y},\theta_{d-1,2},\dots,\theta_{d-1,B})$
    2. $\boldsymbol{\theta}_{d,2}\sim p(\boldsymbol{\theta}_{d,2}|\mathbf{y},\boldsymbol{\theta}_{d,1},\boldsymbol{\theta}_{d-1,3},\dots,\boldsymbol{\theta}_{0,B})$
    3. $\ldots$
    4. $\boldsymbol{\theta}_{d,B}\sim p(\theta_{d,B}|\mathbf{y},\theta_{d,1},\dots,\theta_{d-1,B-1})$

Following these steps will provide us with a set of $D$ draws $\theta_1,\dots,\theta_D$. Given a large enough set of draws $D$, then the Markov chain will converge in distribution to the true posterior distribution.

"""

# ╔═╡ 56be261f-fc7c-4a27-8d18-54cbb00c149b
md""" ### Random walk with drift model """

# ╔═╡ a6e68438-0bba-4add-8fe9-815ebfbebedb
md"""
>In this section we are using another example from [Jamie Cross](https://github.com/Jamie-L-Cross/Bayes/blob/master/4_MCMC.ipynb).

"""

# ╔═╡ 3d8bf98f-5a0e-4f28-a2fe-c7f832b1850b
md"""

The random walk model used in the previous lecture can be extended to have a non-zero mean by adding a constant term, i.e.

$$Y_t = \mu + Y_{t-1} + e_t, \quad e_t\sim N(0,\sigma^2)$$

in which $\mu$ is an unknown real-valued constant. This is known as the random walk with drift model.

"""

# ╔═╡ 2d1cb903-a130-4d66-a03f-f965f4b12d8d
begin
	## Simulate Data from random walk with drift model
	true_mu = 1; # true mean
	true_sig2 = 1; # true variance
	T = 1001; # no. of dates
	y0 = 0;   # initial condition
	y = zeros(T); # storage vector
	
	y[1] = y0;
	for t = 2:T
	    y[t] = true_mu + y[t-1] + rand(Normal(0,sqrt(true_sig2)));
	end
	
	x = collect(1:1:T);
	plot(x,y, label="Simulated Data")
end

# ╔═╡ 1c165b82-0a76-4897-8c22-31ed5f103cff
md"""

We can estimate this model using Bayesian methods. Since the mean of the above distribution is known, we can take it to the left and side and instead work with the *first-difference* of the data $\Delta Y_t=Y_t-Y_{t-1}$ to estimate the model

$$\Delta Y_t\sim N(\mu,\sigma^2)$$

This shows that estimating the random walk with drift model is the same as estimating the parameters of the normal distribution with unknown mean and unknown variance. 

"""

# ╔═╡ 6cc5a39c-d404-4c41-bb21-6787abc7895f
begin
	# Plot data in first differences
	Dy = y[2:end] - y[1:end-1];
	T₁ = length(Dy);
	x₁ = collect(1:1:T₁);
	plot(x₁,Dy, label="First-difference of simulated data")
end

# ╔═╡ fc921837-304f-4e55-aa45-fbc05717f654
md"""

**Priors**:
When setting the priors we need to make an assumption about the dependence of the unknown parameters. Here we will assume that they are independent, so that $p(\mu,\sigma^2)=p(\mu)p(\sigma^2)$ and set the following independent prior distributions:
1. Since $\mu$ can be any real number, we will assume that $\mu\sim N(m_0,v^2_0)$ 
2. Since $\sigma^2>0$, we will assume that $\sigma^2\sim IG(\nu_0,S_0)$ 

"""

# ╔═╡ b7dbca52-3027-46ff-8a45-e0e430fcb924
md"""

**Likelihood**: Let $\theta = (\mu,\sigma^2)'$ denote the vector of unknown parameters. The likelihood of this model is given by

$$\begin{align}
p(\mathbf{Y}|\theta) &= \prod_{t=1}^{T}p(\Delta Y_t|\theta)\\
&= \prod_{t=1}^{T}(2\pi\sigma^2)^{-\frac{1}{2}}\exp(-\frac{1}{2\sigma^2}(\Delta Y_t-\mu)^2)\\
&= (2\pi\sigma^2)^{-\frac{T}{2}}\exp(-\frac{1}{2\sigma^2}\sum_{t=1}^{T}(\Delta Y_t-\mu)^2)
\end{align}$$

"""

# ╔═╡ 12c4db07-d4c9-43bc-a284-9964f19ce917
md"""

**Posterior**:
To get the posterior, we combine the prior and likelihood

$$\begin{align}
p(\theta|\mathbf{Y}) &\propto p(\mathbf{Y}|\theta)p(\theta)\\
                     &\propto p(\mathbf{Y}|\theta)p(\mu)p(\sigma^2)\\
                     &\propto (\sigma^2)^{-(\frac{T}{2}+\nu_0+1)}\exp(-\frac{1}{2\sigma^2}(S_0 + \sum_{t=1}^{T}(\Delta Y_t-\mu)^2) -\frac{1}{2 v^2}(\mu-m_0)^2)\\
\end{align}$$

The final expression is an **unknown distribution** meaning that we can not apply direct Monte Carlo sampling methods. The trick to estimating it is noticing that the posterior distribution looks like it is Normal in $\mu$ and inverse-Gamma in $\sigma^2$. We can therefore try a two block Gibbs Sampler with the full conditional distributions

$p(\mu|\mathbf{y},\sigma^2)$

$p(\sigma^2|\mathbf{y},\mu)$

"""

# ╔═╡ cdb75867-520c-4527-9d3b-a963f370c900
md"""

**First block:** Let $\bar{Y} = \frac{1}{T}\sum_{t=1}^{T}\Delta Y_t$, then

$$\begin{align} p(\mu|\mathbf{y},\sigma^2)&\propto p(\mathbf{Y}|\theta)p(\mu)\\
                          &\propto\exp(-\frac{1}{2\sigma^2}(\sum_{t=1}^{T}(\Delta Y_t-\mu)^2) -\frac{1}{2 v^2}(\mu-m_0)^2)\\
                          &=\exp(-\frac{1}{2}(\frac{1}{\sigma^2}((T\bar{Y})^2-2 T\bar{Y}\mu+T\mu^2)) -\frac{1}{v^2}(\mu^2-2\mu m_0 + m_0^2))\\
                          &\propto\exp(-\frac{1}{2}(\frac{1}{\sigma^2}((-2 T\bar{Y}\mu+T\mu^2)) -\frac{1}{v^2}(\mu^2-2\mu m_0))\\
                          &=\exp(-\frac{1}{2}((\frac{T}{\sigma^2} + \frac{1}{v^2})\mu^2 -2\mu(\frac{T\bar{Y}}{\sigma^2} + \frac{m_0}{v^2})))                   
\end{align}$$

If we stare at this expression long enough, then we (hopefully) will see that it is the kernel of a Normal distribution. To determine the mean and variance, note that if $\mu\sim N(\hat{\mu},D_\mu)$ then

$$p(\mu|\mathbf{y},\sigma^2) \propto \exp(-\frac{1}{2 D_\mu}(\mu^2-2\mu\hat{\mu}))$$

Thus, $\hat{\mu}=D_\mu(\frac{T\bar{Y}}{\sigma^2} + \frac{m_0}{v^2})$ and $D_{\mu}=(\frac{T}{\sigma^2} + \frac{1}{v^2})^{-1}$.

**Second block:**

$$\begin{align}
p(\sigma^2|\mathbf{Y},\mu) &\propto p(\mathbf{Y}|\theta) p(\sigma^2)\\
                       &= (\sigma^2)^{-(\frac{T}{2}+\nu_0+1)}\exp(-\frac{1}{\sigma^2}(S_0+\frac{1}{2}\sum_{t=1}^{T}(\Delta Y_t-\mu)^2)
\end{align}$$

Thus, the conditional posterior for is an inverse-Gamma distribution with scale parameter $\nu = \frac{T}{2}+\nu_0+1$ and shape parameter $S = S_0+\frac{1}{2}\sum_{t=1}^{T}(\Delta Y_t-\mu)^2$. 

"""

# ╔═╡ 2e1e99e6-fa99-49fa-9ff8-5936afc25227
begin
	## Gibbs Sampler for the random walk with drift model
	## Priors
	# Prior for mu
	pri_m = 0;
	pri_v2 = 10;
	pri_mu = Normal(pri_m,sqrt(pri_v2));
	
	# Prior for sig2
	pri_nu = 3;
	pri_S = 1*(pri_nu-1); # sets E(pri_sig2) = 1
	pri_sig2 = InverseGamma(pri_S,pri_S);
	
	## Gibbs Sampler
	# Controls
	nburn = 1000;
	ndraws = nburn + 10000;
	
	# Storage
	s_mu = zeros(ndraws-nburn,1);
	s_sig2 = zeros(ndraws-nburn,1);
	
	# Deterministic terms
	ybar = mean(Dy);
	post_nu = pri_nu + T/2;
end;

# ╔═╡ 28cfbc9f-4f66-4d3c-b440-2fc01d7324b8
# Initial conditions
let MC_mu = mean(Dy), MC_sig2 = var(Dy)
# Markov chain
    for loop in 1:ndraws
    # Draw mu
        post_v2 = 1/(T/MC_sig2 + 1/pri_v2);
        post_m = post_v2*(T*ybar/MC_sig2 + pri_m/pri_v2);
        MC_mu = rand(Normal(post_m,post_v2));

    # Draw sig2
        post_S = pri_S +0.5*sum((Dy.-MC_mu).^2);
        MC_sig2 = rand(InverseGamma(post_nu,post_S));

    # Store
        if loop > nburn
            count = loop - nburn;
            s_mu[count] = MC_mu;
            s_sig2[count] = MC_sig2;
        end
    end
end

# ╔═╡ f3658270-c1b4-4801-9f46-fd80f288e1b1
begin
	## Summarize results
	# Trace plots
	x₂ = collect(1:(ndraws-nburn))
	p₁ = plot(x₂,s_mu, title = "Markov chain: μ", label="Draws");
	p₂ = plot(x₂,s_sig2, title = "Markov chain: σ2", label="Draws");
	
	# Compute posterior mean using Monte Carlo Integration
	post_mu = mean(s_mu);
	post_sig2 = mean(s_sig2);
	
	true_mu, post_mu, true_sig2, post_sig2
end

# ╔═╡ 402c10b5-eca6-45aa-87c9-e0f7901517e3
begin
	# Plot posterior distribution for mu
	histogram(s_mu, normalize=:pdf, title = "Posterior: μ", label="Empirical distribution")
	p₃ = plot!([post_mu], seriestype="vline", label="MC mean")
	
	# Plot posterior distribution for sig2
	histogram(s_sig2, normalize=:pdf, title = "Posterior: σ2", label="Empirical distribution")
	p₄ = plot!([post_sig2], seriestype="vline", label="MC mean")
	
	plot(p₁,p₂,p₃,p₄,layout = (2,2),legend = false)
end

# ╔═╡ 12b0f50a-dc87-4474-9daf-be30667a357b
md""" We can also do this in `Turing.jl`, but I haven't had time to do it. If you want to play around with Turing, then this might be a nice example to try.  """

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuantEcon = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
BenchmarkTools = "~1.3.1"
Distributions = "~0.25.67"
KernelDensity = "~0.6.5"
MCMCChains = "~5.3.1"
Plots = "~1.31.7"
PlutoUI = "~0.7.39"
QuantEcon = "~0.16.4"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.0-rc4"
manifest_format = "2.0"
project_hash = "d46855a6075df61669f350a0b6558544aa9309f5"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "5c26c7759412ffcaf0dd6e3172e55d783dd7610b"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "4.1.3"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "40debc9f72d0511e12d817c7ca06a721b6423ba3"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.17"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "1dd4d9f5beebac0c03446918741b1a03dc5e5788"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.6"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

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

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

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
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "3fb5d9183b38fdee997151f723da42fb83d1c6f2"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.6"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

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
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "3258d0659f812acde79e8a74b11f17ac06d0ca04"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.7"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "6180800cebb409d7eeef8b2a9a562107b9705be5"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.67"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "90630efff0894f8142308e334473eba54c433549"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.5.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FiniteDiff]]
deps = ["ArrayInterfaceCore", "LinearAlgebra", "Requires", "Setfield", "SparseArrays", "StaticArrays"]
git-tree-sha1 = "5a2cff9b6b77b33b89f3d97a4d367747adce647e"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.15.0"

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
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

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
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2d908286d120c584abbe7621756c341707096ba4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.2+0"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

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

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "db5c7e27c0d46fd824d470a3c32a4fc6c935fa96"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f0956f8d42a92816d2bf062f8a6a6a0ad7f9b937"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.2.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "f366daebdfb079fd1fe4e3d560f99a0c892e15bc"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "64f138f9453a018c8f3562e7bae54edc059af249"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.4"

[[deps.IntervalSets]]
deps = ["Dates", "Random", "Statistics"]
git-tree-sha1 = "57af5939800bce15980bddd2426912c4f83012d8"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.1"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "9816b296736292a80b9a3200eb7fbb57aaa3917a"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.5"

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
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "b864cb409e8e445688bc478ef87c0afe4f6d1f8d"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

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

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "f27132e551e959b3667d8c93eae90973225032dd"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.1.1"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Compat", "Dates", "Distributions", "Formatting", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Serialization", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "8cb9b8fb081afd7728f5de25b9025bff97cb5c7a"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "5.3.1"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "59ac3cc5c08023f58b9cd6a5c447c4407cede6bc"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "e595b205efd49508358f7dc670a940c790204629"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2022.0.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "16fa7c2e14aa5b3854bc77ab5f1dbe2cdc488903"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.6.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "DataStructures", "ForwardDiff", "JSON", "LinearAlgebra", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions", "Test", "Unicode"]
git-tree-sha1 = "b79f525737702ff2a3f2005a0823e3518ce8b04c"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.7.0"

[[deps.MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "6d019f5a0465522bbfdd68ecfad7f86b535d6935"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.9.0"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "4e675d6e9ec02061800d6cfb695812becbd03cdf"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.4"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "50310f934e55e5ca3912fb941dec199b49ca9b68"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.2"

[[deps.NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "5a7e32c569200a8a03c3d55d286254b0321cd262"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.5"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9b1f15a08f9d00cdb2761dcfa6f453f5d0d6f973"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.1+0"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "0e353ed734b1747fc20cd4cba0edd9ac027eff6a"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "dfd8d34871bc3ad08cd16026c1828e271d554db9"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.1"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "1ea784113a6aa054c5ebd95945fa5e52c2f378e7"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.12.7"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "7351d1daa3dad1bcf67c79d1ba34dd3f6136c9aa"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.1"

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
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "a19652399f43938413340b2068e11e55caa46b65"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.7"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "8d1f54886b9037091edf146b517989fc4a09efec"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.39"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3010a6dd6ad4c7384d2f38c58fa8172797d879c1"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.0"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "311a2aa90a64076ea0fac2ad7492e914e6feeb81"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "d7a7aef8f8f2d537104f170139553b14dfe39fe9"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.2"

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

[[deps.QuantEcon]]
deps = ["DSP", "DataStructures", "Distributions", "FFTW", "Graphs", "LinearAlgebra", "Markdown", "NLopt", "Optim", "Pkg", "Primes", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "0069c628273c7a3b793383c7dc5f9744d31dfe28"
uuid = "fcd29c91-0bd7-5a09-975d-7ac3f643a60c"
version = "0.16.4"

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
git-tree-sha1 = "dc84268fe0e3335a62e315a3a7cf2afa7178a734"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.3"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

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
version = "0.7.0"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "db8481cf5d6278a121184809e9eb1628943c7704"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.13"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "85bc4b051546db130aeb1e8a696f1da6d4497200"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.5"

[[deps.StaticArraysCore]]
git-tree-sha1 = "5b413a57dd3cea38497d745ce088ac8592fbb5be"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.1.0"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "30b9236691858e13f167ce829490a68e1a597782"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.2.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "DataValues", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "2b35ba790f1f823872dcf378a6d3c3b520092eac"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.1"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "62846a48a6cd70e63aa29944b8c4ef704360d72f"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.5"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "4ad90ab2bbfdddcae329cba59dab4a8cdfac3832"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.7"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

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
version = "1.2.12+3"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

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
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

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
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─09a9d9f9-fa1a-4192-95cc-81314582488b
# ╟─41eb90d1-9262-42b1-9eb2-d7aa6583da17
# ╟─aa69729a-0b08-4299-a14c-c9eb2eb65d5c
# ╟─c53220c2-b590-4f40-829b-617d728c7d16
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# ╟─d65de56f-a210-4428-9fac-20a7888d3627
# ╟─49c0eb58-a2b0-4e66-80c7-c030070ca93d
# ╟─d86c8eae-9bcd-43a7-a9a7-1608af719f98
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─bd0dc0fd-63bc-492b-bba1-a3746ea4ec22
# ╟─7460cf95-4fc1-4c04-99d7-b13c2014b37a
# ╟─d2fcc16f-075e-44d1-83ee-d9ab2372bfc7
# ╟─17382827-e3fc-4038-aa47-9db30f7f45ae
# ╟─26de0997-9548-49b5-9642-b401c6c42c41
# ╟─491b1cbf-bc99-4a31-9c2b-f2a8d0dc37c6
# ╟─5f8c67ac-c5c7-4999-8d22-417d8199ddac
# ╟─76853f63-4969-4823-a16b-d1033af26a2c
# ╟─e9868be9-2c58-45ef-96ad-0a016fdca540
# ╟─97a92a03-64ce-4f12-be98-f66233b27142
# ╟─452fc619-f315-43ef-af4c-4770808f89fe
# ╟─68fab7b3-b3b7-41fa-89ee-cc5d729c5150
# ╟─7354bf26-7530-4a84-ac4d-1dae6b19b623
# ╟─5508d626-591f-4147-99ee-e85160162323
# ╟─eb2624dd-5bf2-4a9f-a92e-a3ae7227dede
# ╟─b1a9f137-10a3-4940-8948-1c44026ada6c
# ╟─0d572678-5302-4715-b873-500004dcac78
# ╟─9e907a32-3dfe-4118-9ec1-53593f632745
# ╟─ac919514-9cdc-4144-99d0-a1b7ef82a4ed
# ╟─086dc5e2-032c-4cea-9805-dd44567bc012
# ╟─27153344-0d5e-4bc1-ba11-2d302bf483ae
# ╟─4a65adda-688b-4ce5-9a8c-ebc874cfc969
# ╟─c9a6391e-2903-4cf3-9aa9-b2e20a7c15f1
# ╠═8f22fc64-c0fc-4f10-8bac-bc355edc4780
# ╠═a081553f-35f8-46e1-925b-bd480cc691e5
# ╟─208d0d6b-f97a-4527-a519-1227c474fdc0
# ╠═b531a759-ab64-4777-8b50-580c1f169576
# ╠═0a63d554-c9f0-4df0-91ae-54c59dc312a4
# ╟─14f6cfaf-50e5-4924-8d40-1a2d7d0a9c8a
# ╟─48a4323f-db34-40bb-9026-39a4f1131c36
# ╠═ddac3546-7949-49be-ac41-3d977d0b99cf
# ╟─c53eab7a-6849-4a2f-b7d2-4a400728cb11
# ╠═1dcaa4e0-c2a9-4068-af52-39c0215625dc
# ╠═fede135f-4a5c-4d40-9341-8e168e186bec
# ╟─5084a86c-58ef-422e-a495-4bf831be3b1a
# ╟─a2fdceff-88f5-4cf0-b139-729344373d14
# ╟─78793993-f6b1-487e-9891-4755697350b5
# ╟─44e0f980-e912-4ad3-aa0d-9ab64d5fdfc9
# ╟─018c4a59-3503-4d71-87d2-f150b0c8904b
# ╠═813fc5a0-ef7c-4395-b2aa-00ecce7b8455
# ╠═e26acec5-ba7a-4448-b01f-519d48adb3ae
# ╟─fb077380-d79d-45cc-96c2-ab44acadd1e1
# ╟─dc666834-d602-4800-934b-2c8caae7beb5
# ╠═70b19d3d-4a6f-450a-8e8e-0e7bdaefaab1
# ╟─ca57a81c-5d42-4415-a979-cec0d9c18391
# ╠═3f19c62e-e3e2-4650-ba3a-ea8b4f228501
# ╠═caff2e40-7b71-4a27-b661-1e4c05ec93a2
# ╠═433129ff-9c7d-4388-be68-1b06ff26fec0
# ╟─ed9890f1-061c-4ada-90e7-a39e08af7af8
# ╟─106087bc-1a39-4e97-b77a-bafe8b692844
# ╟─80154dfd-dac5-4579-8cdc-a14b9862df18
# ╟─3742f58c-de0b-40db-bba0-59a4bf9e58ad
# ╟─635cd82c-8fa6-4bf3-b586-fd2ec915c4b7
# ╟─411d9644-55c7-4cef-81d1-7ca41181d3fa
# ╟─e9819877-d4c1-4378-8430-04f43d057f1f
# ╠═f5ba6b1c-c6ea-4bed-8dd1-bc35d0f3d75b
# ╟─bc1b3e2c-9501-4e3f-a377-02e9c13418c1
# ╟─039f07f3-dfc8-4337-b807-c1e9f0e2bdf0
# ╠═59ebf35d-9d62-4268-b33e-bbcc14e8a23d
# ╟─0e73488b-9981-44dd-b7cc-d314e8f123c1
# ╟─6791bc6f-ef88-4894-881b-6a89d836efe2
# ╠═0c26c6f2-f96f-4bdc-9379-81a76179bd11
# ╟─ec1270ca-5943-4fe7-8017-6904c72673f0
# ╟─f7311d0b-a74c-4a15-be37-5dd8e236cf3d
# ╟─9e9af560-3898-4bb2-8aa7-0e660f738a72
# ╟─c71cc346-e48e-4066-8145-f1aee47c4322
# ╟─c3046dbc-e894-4194-aa54-4f4f28f1066b
# ╟─aad23932-c61e-4308-956b-c2d64d85ac93
# ╟─32b7f7aa-0a56-4cee-87a7-339826fa5c1e
# ╟─2799a8c7-a93d-4577-a5ce-9f771538634b
# ╟─2ea2563b-0a28-42f4-ac43-f12f8d3bcfa7
# ╠═60bf263c-013c-4e6e-9bbe-2b9683f6eb83
# ╠═155dc645-a2a5-49ec-a836-95f25eed9a88
# ╠═7087ac52-c5ae-4398-b4d5-00955b742d84
# ╟─1e519151-e4e7-46e8-814a-c80ef77ff1e1
# ╟─9e7350e2-cd06-4fb5-88c2-1888748dc136
# ╟─3e88a1c5-c1b7-4f1e-b615-a02b6b40de6e
# ╟─1970bc03-ff14-4819-86a6-0a8b802a9f8e
# ╟─24e07f29-a534-4bdc-b26d-353484aad92b
# ╠═6b4b1e00-c2cf-4c06-8598-7a16965e73e3
# ╠═3453a862-d9ce-4802-ace7-8b825641b4e2
# ╠═dbaa28bc-021f-4805-b265-19b9257bbd02
# ╠═d7b27485-7659-4bc6-a199-4767739da218
# ╠═85c4a795-98f8-4bc4-89f5-b2a25671b070
# ╠═47063329-5039-4e7f-b8d0-73e991cb3772
# ╠═a19a4458-b167-46ad-97a1-156b789be81a
# ╟─29298f63-a7d7-4aaa-b888-3e955841f7f5
# ╟─364377f1-3528-4e4a-99d1-91b96c546346
# ╟─492d4b83-e736-4ca4-92d0-2bdb7973b85f
# ╟─b1bc647a-32f1-4ec8-a60e-f8e43e95be2d
# ╟─1234563c-fb5b-42ea-8f5b-abf8adf34e26
# ╟─a370cec5-904c-43fc-94b0-523100b1fd54
# ╟─c22518e2-6cac-451c-bacc-15346dda54a4
# ╟─c7401978-aead-42ff-a8ee-333336afde2b
# ╟─0de3f161-b749-491e-ae32-4b04d5d8f851
# ╟─5373a12c-4732-4d1c-9c71-d113d5c6a5b3
# ╟─12819f3d-bed0-4af6-82d8-3be5e6c79b3a
# ╠═d6e39812-f1c2-494e-a48a-04be4a7ba6a1
# ╟─918f3cd0-f6f9-4d23-9f5e-1ef21617e852
# ╠═7bca38d2-3918-4cc6-9869-983cda421aee
# ╠═4ab65a54-d7e9-4949-a494-91886f8ee1e7
# ╟─9ccd3d2d-f178-4dd5-96ab-a63e9f97a3ab
# ╟─54c1d150-b8ff-4e95-ad03-6a0f2124e495
# ╠═acef097a-7e53-4177-813a-4f69ad83ff42
# ╠═70a1198f-05e5-40f0-8f95-c0056c55cf05
# ╟─2662a1e1-2135-4d3f-8cb8-65d38f944971
# ╟─ac6f7203-3758-4f20-bd64-9110e8846326
# ╟─2446e3cf-7a7f-4675-99c0-114352d4384d
# ╟─02422bf4-ec7f-4428-8067-3091e2a70ba4
# ╟─56be261f-fc7c-4a27-8d18-54cbb00c149b
# ╠═a6e68438-0bba-4add-8fe9-815ebfbebedb
# ╟─3d8bf98f-5a0e-4f28-a2fe-c7f832b1850b
# ╠═2d1cb903-a130-4d66-a03f-f965f4b12d8d
# ╟─1c165b82-0a76-4897-8c22-31ed5f103cff
# ╠═6cc5a39c-d404-4c41-bb21-6787abc7895f
# ╟─fc921837-304f-4e55-aa45-fbc05717f654
# ╟─b7dbca52-3027-46ff-8a45-e0e430fcb924
# ╟─12c4db07-d4c9-43bc-a284-9964f19ce917
# ╟─cdb75867-520c-4527-9d3b-a963f370c900
# ╠═2e1e99e6-fa99-49fa-9ff8-5936afc25227
# ╠═28cfbc9f-4f66-4d3c-b440-2fc01d7324b8
# ╠═f3658270-c1b4-4801-9f46-fd80f288e1b1
# ╠═402c10b5-eca6-45aa-87c9-e0f7901517e3
# ╟─12b0f50a-dc87-4474-9daf-be30667a357b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
