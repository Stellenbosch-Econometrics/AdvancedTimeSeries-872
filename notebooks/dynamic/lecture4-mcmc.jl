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
TableOfContents() # Uncomment to see TOC

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
