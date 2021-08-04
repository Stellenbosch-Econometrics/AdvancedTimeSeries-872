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
using BenchmarkTools, Distributions, KernelDensity, LaTeXStrings, LinearAlgebra, Plots, PlutoUI, StatsBase, Statistics, StatsPlots

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
">ATS 872: Lecture 1</p>
<p style="text-align: center; font-size: 1.8rem;">
 Sampling and random variables 
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

# ╔═╡ f4a548e6-8949-4528-8b26-d30275b9e2c8
md" # Introduction "

# ╔═╡ c65ae735-b404-4798-97f2-29083e7ae44c
md" > **Note:** A significant portion of the material for this lecture is based on [Computational Thinking](https://computationalthinking.mit.edu), a live online Julia/Pluto textbook. You should check out the course for some amazing notebooks!  "

# ╔═╡ 000021af-87ce-4d6d-a315-153cecce5091
md" In our introductory session (Lecture 0) we had a brief introduction to Julia via the QuantEcon website. In this first tutorial we will be looking at some basic ideas such as sampling and random variables. Julia is an amazing language for computational problems and is much faster than R for most practical applications in Bayesian Econometrics. You are welcome to still code in R if you wish. I will steer you in the right direction with resources from last year. However, I think it is worthwhile to learn Julia since the syntax is similar to Python and Matlab. "

# ╔═╡ 49033e09-fd64-4707-916c-9435d3f0a9d2
md" This notebook we are working with is called a `Pluto` notebook and is useful for educational purposes. If you want to code in an integrated development environment, almost like `Rstudio`, then I recommend `VSCode`. "

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
TableOfContents() # Uncomment for TOC

# ╔═╡ d65de56f-a210-4428-9fac-20a7888d3627
md" Packages used for this notebook are given above. Check them out on **Github** and give a star ⭐ if you want."

# ╔═╡ da871a80-a6d8-4be2-92bb-c3f10e51efe3
md" ## Resources "

# ╔═╡ 98a8dd21-8dc4-4880-8975-265249f816ce
md" Here are some links to useful resources for this course. I have tried to not introduce long textbook treatments. I will strive to provide free resources for the course whenever possible. "

# ╔═╡ 15dcbe6b-b51e-4472-a8e0-08cbd49d1e8c
md"""
!!! note "Some cool links 😎"

1. MIT (2021). [Computational Thinking](https://computationalthinking.mit.edu). -- NB resource! Most of the lecture based on this. 
2. QuantEcon (2021). [Quantitative Economics with Julia](https://julia.quantecon.org/). -- Important for Julia and computational work.
3. Aki Vehtari (2020). [Bayesian Data Analysis](https://avehtari.github.io/BDA_course_Aalto/index.html). -- Important for lectures 2, 3, 4
4. José Eduardo Storopoli (2021). [Bayesian Statistics with Julia and Turing.](https://storopoli.io/Bayesian-Julia/) -- Important for lectures 2, 3, 4, 5
5. Gary Koop (2021). [Bayesian Econometrics](https://sites.google.com/site/garykoop/teaching/sgpe-bayesian-econometrics). -- Important for lectures 6, 7, 8
6. Joshua Chan (2017). [Notes on Bayesian Econometrics](https://joshuachan.org/notes_BayesMacro.html). -- Important for lectures 6, 7, 8

"""

# ╔═╡ be11b67f-d04d-46b1-a935-eedd7e59ede3
md" There is a book by Gary Koop, called Bayesian Econometrics, that accompanies the course above. However, it is not essential to have the book. I will upload some articles that with similar content. Let us get going with the first lecture, we will start with the idea of random sampling. "

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Random sampling "

# ╔═╡ 0d0f08f3-48e2-402c-bbb5-33bd3d09ab06
md" One thing that we will be using quite frequently in Bayesian econometrics is the idea of random sampling. This means sampling from different types of distributions.  "

# ╔═╡ b78745cc-84ef-4bbe-a916-9c87cde47145
rand(1:6) # Choose random value from 1 to 6

# ╔═╡ 08fc15b7-2cc6-4a21-82a6-521f6294ee79
rand([2, 3, 5, 7, 11]) # Choose random value from this set of values

# ╔═╡ 4c48df5c-33c6-4d26-b6e4-80e3c883d389
rand()   # random number between 0 and 1 -- similar to runif() in R

# ╔═╡ 350bb07e-6e88-49fc-ae69-c24b1549650d
dcolours = distinguishable_colors(10)

# ╔═╡ 07b17b42-e968-4a7a-bf56-da1a3a1bef52
rand(dcolours) 

# ╔═╡ dc306e4a-7841-4c36-8c5b-d96acc7714de
md" We can also sample several random objects from the same collection. In this case we use an **array comprehension**. "

# ╔═╡ 71482500-7ca7-4a4e-8635-f29ee6f11ced
[rand(1:6) for i in 1:10]

# ╔═╡ b028662e-a8ac-45c4-86aa-d684bbb864c8
md" An easier way to do this is to simply add another argument to the `rand` function. "

# ╔═╡ f882026e-3393-46a7-b284-f0313386f214
rand(1:6, 10) # This generates an array in the same way as above

# ╔═╡ a69254ad-d7e2-4ddd-9a8f-58eaa9a45ce8
md" We can also generate random matrices in with this function. "

# ╔═╡ 94a2862e-1f6b-4c4e-ad32-4b6c49924454
rand(1:6, 10, 10) # Generates a 10x10 matrix with values that range from 1 to 6

# ╔═╡ fa929cc1-ce08-4c9a-922b-cdf611fa3e2a
rand(dcolours, 10, 10) # Generate matrix of colours

# ╔═╡ 909ea3be-9b65-465a-ab31-0d3d525c021f
md" ## Uniform sampling "

# ╔═╡ b1d3017e-8caf-462f-bb0b-8154202f21e6
md" The `rand` function has performed uniform sampling, which means that each object has the same probability of being selected. For our next example we will be counting heads and tails using the `countmap` function. "

# ╔═╡ 9492ddc5-301d-46d2-bf34-5396563f5d5b
tosses = rand( ["head", "tail"], 10000)

# ╔═╡ 4f5e9c1f-d510-4897-9176-218a1a2f4057
toss_counts = countmap(tosses)

# ╔═╡ 7254eb45-8170-40f2-afd3-e30ec5c26781
md" In this case we have a dictionary that maps keys, such as `heads` and `tails`, to specific values. "

# ╔═╡ caeee82e-b854-4b34-b34f-5899e5a9b952
prob_tail = toss_counts["tail"] / length(tosses) # Determines the probability of a tail. 

# ╔═╡ cdad68ca-dac9-49ad-8149-939d18f00778
md" ## Tossing a weighted coin "

# ╔═╡ ea5a71b7-845b-4310-b1fb-f69ee71ac3fb
md"""
How could we model a coin that is **weighted**, so that it is more likely to come up heads? We want to assign a probability $p = 0.7$ to heads, and $q = 0.3$ to tails.
"""

# ╔═╡ d6aa79f3-45c8-4ff5-84d6-1cd79b845b2f
md"""
One way would be to generate random integers between 1 and 10 and assign heads to a subset of the possible results with the desired probability, e.g. 1:7 get heads, and 8:10 get tails. We will use this same logic later in other examples, so it is important to understand what we are doing here. "
"""

# ╔═╡ 0d46dd99-c614-40a6-9cd0-69b453ec782f
function simple_weighted_coin()
	if rand(1:10) ≤ 7
		"heads"
	else   
		"tails"
	end
end # Quite verbose, but good for pedagogical purposes. 

# ╔═╡ da3b79da-3d14-405f-80af-d58d04b4f801
simple_weighted_coin()

# ╔═╡ 18d07eee-60af-4dad-8f4a-9426f5907ad3
md" Another way to do this might be with a **ternary operator**, see the Julia documentation [here](https://docs.julialang.org/en/v1/manual/control-flow/). "

# ╔═╡ f40f5823-f383-4d6e-a651-91c5a03cbf1e
simple_weighted_coin2() = rand(1:10) ≤ 7 ? "heads" : "tails"

# ╔═╡ 2970a6d2-599a-44ce-ab09-d52db64c0c64
simple_weighted_coin2()

# ╔═╡ 5b54210f-9a7d-447e-9491-f8fbb0892e7f
md" If we generate a uniform number between 0 and 1 and then check if it is less than some probability, this is known as a **Bernoulli trial**. We can construct a simple Bernoulli function that encapsulates this idea.  "

# ╔═╡ e9df057d-3781-4fe1-b0ca-fab08b895ca2
bernoulli(p) = rand() < p # Takes in a value p between 0 and 1 to compare against

# ╔═╡ 7f8b5d7b-25cf-4464-b01a-e9649001b1a1
md"""
p = $(@bind p₁ Slider(0.0:0.01:1.0, show_value=true, default=0.7))
"""

# ╔═╡ 3d1e1190-2ba6-42ad-9c5b-3c3316fd75a0
countmap( [bernoulli(p₁) for _ in 1:1000] ) # 10000 iterations, count how many true and false given the value of p

# ╔═╡ bda26511-d961-413a-8389-ad5be48f79fe
md" **Note**: the output for this function is `true` or `false` instead of `heads` or `tails` in the weighted coin example. "

# ╔═╡ 4c6dd3ba-268e-4fad-b8d1-4bc78f24a46f
md" A Bernoulli random variable model for a weighted coin, for example, will take value 1 with probability $p$ and 0 with probability $(1- p)$. Our Bernoulli function that we wrote provides `true` and `false` values. Let us sample some Bernoulli random variates. "

# ╔═╡ c817e5e6-4cb4-4392-8f7e-e1a4bb009537
flips = [Int(bernoulli(p₁)) for _ in 1:100];

# ╔═╡ 46bb14fb-62b4-402b-8a0b-8096bd2a6289
mean(flips) 

# ╔═╡ b9cdd1c8-2f8f-48c5-846d-e40cedc949b7
md" The calculation for the mean is just the proportion of `true` values, which should be roughly equal to our probability parameter. Accuracy increases with the number of flips. "

# ╔═╡ 370a4ccb-fdb6-4e3f-8004-d6f88f025945
md" # Probability distributions and types "

# ╔═╡ c6d85f60-c820-4269-a6b4-57483de13bd8
md"""
In this section I will provide some basic properties of common probability distributions that are often used in Bayesian econometrics. We will only look at three in this section, and then as we progress we will introduce more. We will also discuss the type system in Julia, which is a key feature of the language. To keep the code clear in the following lectures we won't always use best coding practice, but every now and then we will discuss some core principles. 
"""

# ╔═╡ 601f3dfa-4ea5-4418-aeba-5ab1203f8753
md" ## Bernoulli "

# ╔═╡ ce3b13a8-38e8-449a-8b11-7a61b8632fc9
md" As we have stated, the Bernoulli distribution describes a binary event of a successful experiment. We usually represent 0 as failure and 1 as success, so the result of a Bernoulli distribution is a binary variable. The Bernoulli distribution is widely used to model discrete binary outcomes in which there are only two possible results."

# ╔═╡ c61504df-808a-46f0-b8cc-dcc7197ffb3e
md"""
p = $(@bind p₂ Slider(0.0:0.01:1.0, show_value=true, default=0.7))
"""

# ╔═╡ ed8b654f-f964-4d8f-a998-1032c197f014
begin
	plot(Distributions.Bernoulli(p₂),
	        markershape=:circle,
	        alpha=0.7,
	        xlabel=L"\theta",
	        ylabel="Mass",
	        ylim=(0, 1), 
			lw = 2,
		legend = false
	    )
end

# ╔═╡ c361ae07-61af-44bb-a5ee-be991390fa88
md" We might want to know what the mean (or expected value) of the process is,"

# ╔═╡ 0a98082a-94c3-41d8-a129-4f42e217bcd1
md" ### Make Bernoulli a type "

# ╔═╡ 5b38607d-6cfc-4fa0-b19f-5bea8ad38b39
md" This section is a bit more advanced so you can probably just skim through it on first reading. It relates more to Julia programming than other languages, but it is good knowledge to have. 

Currently we need one function for sampling from a Bernoulli random variable, a different function to calculate the mean and a different function for the standard deviation, etc. 

In mathematical terms we have this Bernoulli random variable and we are calculating properties of the particular concept. We can do the same thing computationally by creating a new object that represents the Bernoulli random variable. "

# ╔═╡ 5aed1914-6960-41c8-91d4-09614766583d
struct Bernoulli_New
	p::Float64
end

# ╔═╡ 5a358aa5-bb4b-4b48-9d46-8628a9722023
md" We want to be able to sample from this using the `rand()` function and also take its mean. In order to do this we will extend the rand function from the `Base` library of Julia and the mean function from the Statistics library. 

Note that we are adding methods to these particular functions. "

# ╔═╡ 2afe4168-640f-4b7e-ab28-7ae22fba16c9
Base.rand(X::Bernoulli_New) = Int( rand() < X.p ) # Add method to the rand function

# ╔═╡ 198663dd-941a-4258-800f-80ad0638f884
B = Bernoulli_New(0.25)

# ╔═╡ 8893ec3a-7b9b-4887-9776-0c9c4f07cf14
md" The object `B` represents a Bernoulli random variable with probability of success $p$. One should note that this type already exists in a package like `Distributions.jl`, so you should be careful about naming conventions. "

# ╔═╡ b186f0b5-721e-4757-9a4d-a839162b22f2
rand(B)

# ╔═╡ 827e960e-057e-40ae-beeb-f3c013d9f883
Statistics.mean(X::Bernoulli_New) = X.p

# ╔═╡ 96787e59-a958-404b-b610-42a28bd0353b
mean(B)

# ╔═╡ 28578d77-1439-49cf-a9f6-120557bce924
md" ## Binomial "

# ╔═╡ b6fc9ad1-5f44-4697-be2e-407e2b9308c0
md" The binomial distribution describes an event of the number of successes in a sequence of $n$ independent experiment(s), each asking a yes-no question with a probability of success $p$. Note that the Bernoulli distribution is a special case of the binomial distribution where the number of experiments is $1$. "

# ╔═╡ 71f12fb3-901d-4feb-9fbc-a5fc6e0f4750
md" The binomial distribution has two parameters and its notation is $\text{Bin}(n, p)$. An example would be the number of heads in 5 coin flips (as illustrated below for different values of $p$). We will deal with the coin flip problem in more detail in the next lecture. "

# ╔═╡ b061d6f2-bcd1-410e-a005-d2e993616b3a
md"""
p = $(@bind p₃ Slider(0.0:0.01:1.0, show_value=true, default=0.7))
"""

# ╔═╡ 1c20116c-339c-453c-b6d1-4ed1477fcf12
begin
	plot(Binomial(5, p₃),
	        markershape=:circle,
	        alpha=0.7,
	        xlabel=L"\theta",
	        ylabel="Mass", 
			lw = 2, 
			legend = false
	    )
end

# ╔═╡ 0a5ed3ea-12d9-46f9-aab8-472eae8a971d
md" We can make the binomial random variable a type, we only require information on $n$ and $p$, so the `struct` is: "

# ╔═╡ 1056e659-b358-451f-85b3-a7ec9a6dac92
struct Binomial_New
	n::Int64
	p::Float64
end

# ╔═╡ 86d8016f-9179-4bb2-be71-3708896ba216
md" Note that this does not require methods at first. We can add the methods later, and other people can add methods too if they are able to load the package. "

# ╔═╡ 3a9c6bbe-5078-4f99-9418-dc22f73706cb
Base.rand(X::Binomial_New) = sum(rand(Bernoulli_New(X.p)) for _ in 1:X.n)

# ╔═╡ 310e07b1-ef44-4588-9fc2-7f70b84e527d
md" We will discuss how to code up the Binomial random variable in the next lecture. For now one can simply take this code as given. If you understand what is written there that is great, but it is not the focus of this section. "

# ╔═╡ e41adcb9-3c78-404b-a501-b359511b9a39
rand(Binomial_New(10, 0.25))

# ╔═╡ d34b5710-2f37-4bb1-9e02-6e95996f7242
md"""
n = $(@bind binomial_n Slider(1:100, show_value=true, default=1)); 
p = $(@bind binomial_p Slider(0.0:0.01:1, show_value=true, default=0))

"""

# ╔═╡ 31675329-f1bd-4752-8da1-af82475fe900
begin
	binomial_data = [rand(Binomial_New(binomial_n, binomial_p)) for _ in 1:10000]
	
	bar(countmap(binomial_data), alpha=0.5, size=(500, 300), leg=false, bin_width=0.5)
end

# ╔═╡ 71128267-d23a-4162-b9b3-52b86ec5f9de
md" We will encounter a similar graph in the next lecture. We will go through the code more methodically there. "

# ╔═╡ 7d74a6be-4aac-4f12-9391-528f9cbf37ba
md" ## Gaussian "

# ╔═╡ 37cbf7a2-6679-40a4-8085-21a4e900c59d
md" While this section is going to be about the Gaussian distribution, we are also going to use it as a platform to discuss software engineering principles. If you don't care much for the programming side of things then you can still learn some things about the Gaussian distribution. In our third lecture we will delve into some further theorethical properties of the Gaussian distribution, so this will definitely not be the last time you encounter it. In fact, this distribution will feature in almost all our lectures so it is a good idea to introduce the concepts early and then reiterate as we move on to other topics. "

# ╔═╡ f99c393f-308f-4821-8f5a-ee8ebcf5b77b
md"""
The two important parameters for the Gaussian distribution are the mean $\mu$ and standard deviation $\sigma$. We can sample from the Gaussian distribution with mean $0$ and variance $1$ with the `randn()` function. 
"""

# ╔═╡ 06f497e4-d1a3-4e99-86f4-f63f69920f53
gauss = randn(10^5)

# ╔═╡ 6747980b-7072-4267-84c5-a352abf4ec25
md"""
A Gaussian random variable is a **continuous** random variable, i.e. it has a continuous range of possible outcomes. The possible range of outcomes is called the **support** of the distribution. For a Gaussian it is the whole real line, $(-\infty, \infty)$.
"""

# ╔═╡ fe0ee6b7-9c42-41b8-929c-2dd7101490a3
md"""
One way to specify a continous random variable $X$ is via its **probability density function**, or **PDF**, $f_X$. The probability that $X$ lies in the interval $[a, b]$ is given by an area under the curve $f_X(x)$ from $a$ to $b$:

$$\mathbb{P}(X \in [a, b]) = \int_{a}^b f_X(x) \, dx.$$
"""

# ╔═╡ 7550ccac-ca63-4f96-b576-595888071c34
md"""
For a Gaussian distribution with mean $\mu$ and variance $\sigma^2$, the PDF is given by

$$f_X(X) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left[ -\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right]$$
"""

# ╔═╡ abe288f3-9a80-4c29-a918-73c57ab16dc2
md" This PDF of the Gaussian is captured in the functions below. It is important to remember the equation for the PDF of the Gaussian since we will need to work with and manipulate it a lot in this course. " 

# ╔═╡ c23627d6-91a2-4b69-9e35-71b8a9578dd6
bell_curve(x) = exp(-x^2 / 2) / √(2π)

# ╔═╡ 24f945be-f3d5-48bd-80e1-12a7cc92d976
bell_curve(x, μ, σ) = bell_curve( (x - μ) / σ ) / σ

# ╔═╡ 62c3c7f1-2839-4c7e-bba7-1741649b3620
md"""
We can shift and scale the Gaussian distribution in the following manner. 
"""

# ╔═╡ b153e5e7-95ba-4425-91aa-ce9986a64392
md"""
μ = $(@bind μ Slider(-3:0.01:3, show_value=true, default=0.0))
σ = $(@bind σ Slider(0.01:0.01:3, show_value=true, default=1))
"""

# ╔═╡ 0f7184f5-a03f-482e-8339-fd12c7391e01
data = μ .+ σ .* randn(10^5)

# ╔═╡ ff9355dc-3e5f-4558-9027-668bd17a7a30
begin
	histogram(data, alpha=0.5, norm=true, bins=100, leg=false, title="μ=$(μ), σ=$(σ)", size=(600, 400))
	
	xlims!(-6, 6)
	ylims!(0, 0.6)
	
	xs = [μ - σ, μ, μ + σ]
	
	plot!(-6:0.01:6, x -> bell_curve(x, μ, σ), lw=2, color = :black)
	
	plot!((μ - σ):0.01:(μ + σ), x -> bell_curve(x, μ, σ), fill=true, alpha=0.6, c=:black)
	
	plot!([μ, μ], [0.05, bell_curve(μ, μ, σ)], ls=:dash, lw=2, c=:white)
	annotate!(μ, 0.03, text("μ", :white))
#	annotate!(μ + σ, 0.03, text("μ+σ", :yellow))
#	annotate!(μ, 0.03, text("μ", :white))

	
end

# ╔═╡ a825e358-1fdc-42cb-87cf-ab0dbd092cb0
md" One fo the nice things about Gaussians is that the sum of two Gaussians is also Gaussian. We will make use of this property of the Gaussian in future, so take note. In order to show this, let us sample from two Gaussian random variables and add the resulting random variable. "

# ╔═╡ 2279f195-a9fa-46ee-925b-f54222d61d9a
begin
	data1 = 4 .+ sqrt(0.3) .* randn(10^5)
	data2 = 6 .+ sqrt(0.7) .* randn(10^5)
	
	total = data1 + data2
end

# ╔═╡ 677da773-6130-41b2-8188-209a8d751f99
begin
	
	histogram(data1, alpha=0.4, norm=true, size=(600, 400), label = "data1", title="Sum of Gaussians")
	histogram!(data2, alpha=0.4, norm=true, size=(600, 400), label = "data2", color = :black)
	histogram!(total, alpha=0.8, norm=true, size=(600, 400), label = "total")
	plot!(2:0.01:14, x -> bell_curve(x, 10, 1), lw=2, color = :black, legend = false)
end

# ╔═╡ c55f846d-d578-4c81-bdc4-ce5d03c62dba
md" Quite interesting, we get a Gaussian again -- the green one at the end is the sum. Gaussians form part of a larger group of distributions called [stable distributions](https://en.wikipedia.org/wiki/Stable_distribution) that share this property. 

Importantly, the sum of two Gaussians with means $\mu_1$ and $\mu_2$ and variances $\sigma^{2}_{1}$ and $\sigma^{2}_{2}$ is Gaussian with mean $\mu_1 + \mu_2$ and variance $\sigma_1^2 + \sigma_2^2$. "

# ╔═╡ 071d902e-a952-480e-9c21-5a3315162a6a
md" ### Let's talk about types "

# ╔═╡ 54bf4640-fa81-4ef4-978a-a87682dd3401
md"""
We have shown how we can represent a random variable in software with Bernoulli and Binomial types that we have defined before. In some other programming languages there are different names for the functions associated to certain random variables, but no specific name for the random variable itself. 

Let us take a look at R as an example. Most of you are comfortable with R, so this should pehaps be more familiar. In R there is a standard naming convention with respect to sampling from random variables, which is best explained by an example. Consider the `norm` function, which allows us to sample from the Normal distribution. There are four functions that we can attach as prefix to `norm`. These indicators are `d` for the density, `p` for the distribution function , `q` for the quantile function and `r` for generating random variates. 

In other words, if you want to generate a random draw from the Normal distribution you should use the `rnorm` function. This seems quite intuitive. However, what is wrong with this? 

All these functions are referring to an underlying random variable (or probability distribution), which you will find in any course on probability. However, there is no way for us to refer to the underlying mathematical object. Think on this point for a second. 

How do we rectify this? We would like to be able to refer to the random variable (or probability distribution) itself. We should be able to provide a type with the name and parameters of a random variable, but not yet specify how to generate random instances. This is an example of thinking ahead by providing **abstraction**.

Once we have established the random variable with our type system we can always provide means for random sampling (and more).

"""

# ╔═╡ 8a01d833-3220-4144-8c2a-dde4c1399795
md" ### Defining abstract types "

# ╔═╡ 760eaee1-0af1-41c1-b38a-c0041559c0ed
md"""
Define an **abstract type** using `abstract type <Name> end`, where `<Name>` is replaced with name of the type.

We can think of an abstract type as being a collection of types that share a particular property. In our case, we want to create a type to represent "any random variable", and also the sub-types "any continuous(-valued) random variable" and "any discrete(-valued) random variable".

This will allow us to specify later on whether a given concrete (i.e. particular) random variable is discrete or continuous.

We use `<:` to denote **sub-type**:
"""

# ╔═╡ bf4df54b-6631-4b59-bf6a-26caea5ab7df
begin
	abstract type RandomVariable end
	
	abstract type DiscreteRandomVariable <: RandomVariable end # Subtype of RandomVariable
	abstract type ContinuousRandomVariable <: RandomVariable end
end

# ╔═╡ 18fbb98d-87a2-4f29-b6f0-3e19ad843b00
md"""
Let's start off by looking at **Gaussian** random variables.
"""

# ╔═╡ fc5709fc-337d-4d42-b023-373089de2c8d
begin
	struct Gaussian <: ContinuousRandomVariable # Subtype of ContinuousRandomVariable
		μ     # mean
		σ²    # variance
	end
	
	Gaussian() = Gaussian(0.0, 1.0)  # normalised Gaussian with mean 0 and variance 1
end

# ╔═╡ e91fe8c0-d13e-401f-b3f1-77e04fe4df34
G = Gaussian(1, 2)

# ╔═╡ ada8809b-8f3a-4298-94d9-e8225df4087d
md" We have now created a Gaussian random variable with given parameter value, without sampling from it. We have not even defined the notion of sampling. However, we will easily be able to apply it to our Gaussian type. More importantly, we will be able to apply it to any random variable that falls under the RandomVariable type that we have defined. 

Important to note that the `Gaussian` type that we have created here is a concrete type.

We now extend the `mean`, `var` and `std` function from the Statistics library to act on our newly created object. "

# ╔═╡ 7580a872-47b1-4efc-9c51-9591d3552c5b
begin
	Statistics.mean(X::Gaussian) = X.μ
	Statistics.var(X::Gaussian) = X.σ²
end

# ╔═╡ a6945e5b-0516-49d1-a978-e4af5090aca3
mean(G)

# ╔═╡ 30b5ae33-c009-4ad5-8950-c75a614acde3
var(G)

# ╔═╡ 265d6bdf-2381-471a-a99a-3d163b96e620
md" Now let us show that we can calculate the standard deviation for any random variable, not just the Gaussian. Calculating the standard deviation is simply going to be the square root of the variance for **any** random variable. 

We can define this to act on any random variable, even ones that we have not created yet!"

# ╔═╡ ec39a8d0-be30-4c7c-9727-f7cffdd117a9
Statistics.std(X::RandomVariable) = sqrt(var(X))

# ╔═╡ d3387ea9-032f-4b62-96fe-3965ad187672
std(G)

# ╔═╡ 9aef8d51-eb5b-4342-8ad5-02e6187b2953
md" #### Sum of two Gaussians (redux) "

# ╔═╡ 9260827c-3262-4870-9e5a-ac49bfa1dbae
md" Gaussians have the special property that we mentioned before that the sum of two Gaussians is always a Gaussian. We can code this property up with our type specification as follows. " 

# ╔═╡ fbe80edb-7964-4103-bffa-c01a89904bd1
Base.:+(X::Gaussian, Y::Gaussian) = Gaussian(X.μ + Y.μ, X.σ² + Y.σ²)

# ╔═╡ 2c91f496-3780-4257-8da1-8fbf8eeca908
md" We are essentially saying that we can extend the $+$ operator from the `Base` Julia package to include a summation over two Gaussian distributions. " 

# ╔═╡ 215e2f59-0541-46d5-8d48-fa381139fd54
begin
	G1 = Gaussian(0, 1)
	G2 = Gaussian(5, 6)
end

# ╔═╡ 5f537343-2d7d-433f-a3aa-b075425fc9e2
G1 + G2

# ╔═╡ c28e6273-bad5-4688-af21-484c5de2bdf0
mean(G1 + G2) == mean(G1) + mean(G2)

# ╔═╡ df7f90b7-c989-4856-9adf-41be3f4e6444
md" #### Probability distribution of Gaussian "

# ╔═╡ de83e0a2-6655-469e-8ab9-6e00b60e245c
md" We have already provided a mathematical description of the PDF of the Gaussian, which is provided again below as a reminder. 

For a Gaussian distribution with mean $\mu$ and variance $\sigma^2$, the PDF is given by

$$f_X(X) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp \left[ -\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right]$$

Now let us define the pdf function with our Gaussian type. 
"

# ╔═╡ 7d92b70f-fc40-47e4-97c4-0703181e2322
pdf(X::Gaussian) = x -> exp(-0.5 * ( (x - X.μ)^2 / X.σ²) ) / √(2π * X.σ²)

# ╔═╡ f4f723ea-3574-419a-84a4-655d09375c3a
pdf(G)

# ╔═╡ 544822a0-329e-48f7-9a16-1e18053ea9f0
pdf(Gaussian())(0.0)

# ╔═╡ 321b215c-ca89-4fe5-8f15-9bf37fe10064
md"""
μ = $(@bind μμ Slider(-3:0.01:3, show_value=true, default=0.0))
σ = $(@bind σσ Slider(0.01:0.01:3, show_value=true, default=1.0))
"""

# ╔═╡ 118957fc-e85a-4829-b3ae-9aa8fccc0a33
begin
	plot(pdf(Gaussian(μμ, σσ)), leg=false, lw = 2)
	xlims!(-6, 6)
	ylims!(0, 0.5)
end

# ╔═╡ bf84e5cc-8617-4a29-b5c9-ff7de784e29b
md"""
#### Sampling from a Gaussian distribution
"""

# ╔═╡ 7255af4d-611d-489d-b607-70eefb858170
md"""
We can also specify how to sample from a Gaussian distribution. We can re-purpose `rand` for this!
"""

# ╔═╡ 0fcbe952-87af-4c56-a6e8-cf80ada41497
Base.rand(X::Gaussian) = X.μ + √(X.σ²) * randn()

# ╔═╡ 75afbded-6b3b-4a7e-ad53-a5f34808c056
histogram!([rand(Gaussian(μμ, σσ)) for i in 1:10^4], alpha=0.5, norm=true)

# ╔═╡ 248e90a9-6c28-4ef2-bd51-112077f93f9c
md" #### General application "

# ╔═╡ aba1a15f-15a8-495a-a585-77fc18ccb7dd
md"""
Let's recall the Bernoulli distribution. This represents a weighted coin with probability $p$ to come up "heads" (1), and probability $1-p$ to come up "tails" (0).

Note that this is a **discrete** random variable: the possible outcomes are the discrete values $0$ and $1$.
"""

# ╔═╡ efb934f0-2b02-42c6-9d5d-b0243bf889bd
struct Bernoulli <: DiscreteRandomVariable
	p::Float64
end

# ╔═╡ 24a2dc89-51db-40d2-8990-832ac2c65fe2
B1 = Bernoulli(0.25)

# ╔═╡ 29d1ca09-60f4-4786-9623-3e09584aeece
begin
	Statistics.mean(X::Bernoulli) = X.p
	Statistics.var(X::Bernoulli) = X.p * (1 - X.p)
end

# ╔═╡ 46e74b96-7f94-408c-a63d-d898e538bd59
md" Now for the amazing part... The `std` function can be called, even though we did not write it directly for the Bernoulli random variable. "

# ╔═╡ 6dd8ebb3-a23f-4071-becd-94a8de5fd4f7
mean(B1), var(B1), std(B1)

# ╔═╡ 686e2146-46ae-4799-bab7-51b42a3074eb
md"""
Finally we specify how to sample:
"""

# ╔═╡ 5415b2b7-9c7c-4420-a664-a96e4bd23199
Base.rand(X::Bernoulli) = Int(rand() < X.p)

# ╔═╡ ea8e9149-7c8f-492d-9577-19284fe50238
md"""
#### Adding two random variables
"""

# ╔═╡ 8d22a8ff-3bf5-4226-b605-1a496a03d667
md"""
What happens if we add two Bernoulli random variables? There are two routes we could go: We could use the known theoretical sum, or we could write a general-purpose tool. Let's do the latter.
"""

# ╔═╡ 3cbe3970-098e-4476-9062-87ce8dbf747c
md"""
When we add two Bernoulli random variables we do *not* get a Bernoulli back. To see this it's enough to observe that the sum can have the outcome 2, which is impossible for a Bernoulli. 
		
So the result is just the random variable "the sum of these two given random variables". In general it won't even have a common name. 
		
So we actually need to *define a new type* to represent the "sum of two given random variables", which itself will be a random variable!:
		
		
"""

# ╔═╡ f51e0a4a-3d2c-47ee-8eee-dfc6e5c522ac
struct SumOfTwoRandomVariables <: RandomVariable
	X1::RandomVariable
	X2::RandomVariable
end

# ╔═╡ 70b112cc-3b6e-4c9e-a2fb-5282f2cc5605
begin
	B2 = Bernoulli(0.25)
	B3 = Bernoulli(0.6)
end

# ╔═╡ 5acc26ff-fe68-4b6e-a67c-fd753efc949b
md"""
Now we can define the sum of two random variables of *any* type:
"""

# ╔═╡ b4c5f5e3-4975-4fa6-91ea-c4d634825c0e
Base.:+(X1::RandomVariable, X2::RandomVariable) = SumOfTwoRandomVariables(X1, X2)

# ╔═╡ 6031286a-85c9-4770-b03e-3bf6ddd12451
md"""
For example, let's sum two Bernoullis:
"""

# ╔═╡ 34908b18-2277-4d80-b615-2ddaf7d12d85
B2 + B3

# ╔═╡ 9ec9be96-c127-4124-ae7a-3eaddf21dd49
md"""
For the special case of Gaussians we still get the correct result (we have *not* overwritten the previous definition):
"""

# ╔═╡ dc66f1b5-f04a-4c2b-8ed2-295394c10a79
G1 + G2

# ╔═╡ 4e86f431-8737-4055-be4d-51d8fb1250aa
md"""
Now we need to define the various functions on this type representing a sum
"""

# ╔═╡ 7469e43a-f963-4228-bbe3-4cffd113cb2b
Statistics.mean(S::SumOfTwoRandomVariables) = mean(S.X1) + mean(S.X2)

# ╔═╡ a2228c42-03d6-45e0-b3b1-18db8737e848
mean(B2 + B3)

# ╔═╡ d9eafc22-71c3-48e8-88ec-c332148ea98d
md"""
To have a simple equation for the variance, we need to assume that the two random variables are **independent**. Perhaps the name should have been `SumOfTwoIndependentRandomVariables`, but it seems too long.
"""

# ╔═╡ 43e5b217-965a-4312-bd1b-ffda74253653
Statistics.var(S::SumOfTwoRandomVariables) = var(S.X1) + var(S.X2)

# ╔═╡ b76070e9-ef7a-4cb5-ade7-625073173c5c
md"""
How can we sample from the sum? It's actually easy!
"""

# ╔═╡ 3d11987d-5afb-4bc6-b0be-a355d804b6c6
Base.rand(S::SumOfTwoRandomVariables) = rand(S.X1) + rand(S.X2)

# ╔═╡ f180e914-af0c-4d8a-afe4-bdc54ee988f6
md"""
Now it's easy to look at the sum of a Bernoulli and a Gaussian. This is an example of a [**mixture distribution**](https://en.wikipedia.org/wiki/Mixture_distribution).
"""

# ╔═╡ bf20d3e0-64b5-49d4-aabc-059aa6a390ad
md"""
Let's extend the `histogram` function to easily draw the histogram of a random variable:
"""

# ╔═╡ d93dfb88-0602-4913-a59e-587803a9b5a3
Plots.histogram(X::RandomVariable; kw...) = histogram([rand(X) for i in 1:10^6], norm=true, leg=false, alpha=0.5, size=(500, 300), kw...)

# ╔═╡ b2265d0c-9f85-4eff-8872-2fa968474e3f
histogram(Bernoulli(0.25) + Bernoulli(0.75))

# ╔═╡ 485c79a6-2c47-4320-81c8-2d2ac2b5d5a2
histogram(Bernoulli(0.25) + Gaussian(0, 0.1))

# ╔═╡ 0244cc32-6371-42c0-8564-570d3424460d
mixture = Bernoulli(0.25) + Bernoulli(0.75) + Gaussian(0, 0.1)

# ╔═╡ 913c8389-3501-4a0c-abe3-4faa42ef9a04
rand( mixture )

# ╔═╡ 8d29975c-5adc-458b-95b0-f4369c5c2f3a
histogram( mixture )

# ╔═╡ b0ccc2ee-0355-4591-b243-5f56715a01b8
md" #### Generic programming "

# ╔═╡ c9cd7943-7b2a-4387-a4fb-766d9ee00594
md"""
Now we have defined `+`, Julia's generic definition of `sum` can kick in to define the sum of many random variables!
"""

# ╔═╡ d2a12c25-4277-4f19-9811-9d371d91022c
S = sum(Bernoulli(0.25) for i in 1:30)

# ╔═╡ 3f3b7906-7bd0-4a9f-8a49-14b8a8924218
md"""
Note that we do not need the `[...]` in the following expression. There is no need to actually create an array of random variables; instead we are using an **iterator** or **generator expression**:
"""

# ╔═╡ 25e39a6b-5d8e-419f-9a44-8d72a8fde502
histogram(S)

# ╔═╡ 6560fe42-daa9-47ce-8a88-dbddcc2d3a1c
mean(S)

# ╔═╡ ea7b83b9-2e95-43da-b089-cdf4d6d5247d
var(S)

# ╔═╡ 3245d573-ffe6-44b2-9734-753a011ab10c
rand(S)

# ╔═╡ 5c294ee4-1d99-4d7d-a94d-afc3a643614e
md"""
This is a big deal! Everything just works. That is it for today, next time we will move on to Bayesian statistics.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
KernelDensity = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
BenchmarkTools = "~1.1.1"
Distributions = "~0.25.11"
KernelDensity = "~0.6.3"
LaTeXStrings = "~1.2.1"
Plots = "~1.19.4"
PlutoUI = "~0.7.9"
StatsBase = "~0.33.9"
StatsPlots = "~0.14.26"
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
git-tree-sha1 = "c31ebabde28d102b602bada60ce8922c266d205b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.1"

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

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

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
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

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
git-tree-sha1 = "1470c80592cf1f0a35566ee5e93c5f8221ebc33a"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.3"

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

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

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

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

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

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "885838778bb6f0136f8317757d7803e0d81201e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.9"

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
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

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
# ╟─f4a548e6-8949-4528-8b26-d30275b9e2c8
# ╟─c65ae735-b404-4798-97f2-29083e7ae44c
# ╟─000021af-87ce-4d6d-a315-153cecce5091
# ╟─49033e09-fd64-4707-916c-9435d3f0a9d2
# ╠═c4cccb7a-7d16-4dca-95d9-45c4115cfbf0
# ╠═2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# ╟─d65de56f-a210-4428-9fac-20a7888d3627
# ╟─da871a80-a6d8-4be2-92bb-c3f10e51efe3
# ╟─98a8dd21-8dc4-4880-8975-265249f816ce
# ╟─15dcbe6b-b51e-4472-a8e0-08cbd49d1e8c
# ╟─be11b67f-d04d-46b1-a935-eedd7e59ede3
# ╟─040c011f-1653-446d-8641-824dc82162eb
# ╟─0d0f08f3-48e2-402c-bbb5-33bd3d09ab06
# ╠═b78745cc-84ef-4bbe-a916-9c87cde47145
# ╠═08fc15b7-2cc6-4a21-82a6-521f6294ee79
# ╠═4c48df5c-33c6-4d26-b6e4-80e3c883d389
# ╠═350bb07e-6e88-49fc-ae69-c24b1549650d
# ╠═07b17b42-e968-4a7a-bf56-da1a3a1bef52
# ╟─dc306e4a-7841-4c36-8c5b-d96acc7714de
# ╠═71482500-7ca7-4a4e-8635-f29ee6f11ced
# ╟─b028662e-a8ac-45c4-86aa-d684bbb864c8
# ╠═f882026e-3393-46a7-b284-f0313386f214
# ╟─a69254ad-d7e2-4ddd-9a8f-58eaa9a45ce8
# ╠═94a2862e-1f6b-4c4e-ad32-4b6c49924454
# ╠═fa929cc1-ce08-4c9a-922b-cdf611fa3e2a
# ╟─909ea3be-9b65-465a-ab31-0d3d525c021f
# ╟─b1d3017e-8caf-462f-bb0b-8154202f21e6
# ╠═9492ddc5-301d-46d2-bf34-5396563f5d5b
# ╠═4f5e9c1f-d510-4897-9176-218a1a2f4057
# ╟─7254eb45-8170-40f2-afd3-e30ec5c26781
# ╠═caeee82e-b854-4b34-b34f-5899e5a9b952
# ╟─cdad68ca-dac9-49ad-8149-939d18f00778
# ╟─ea5a71b7-845b-4310-b1fb-f69ee71ac3fb
# ╟─d6aa79f3-45c8-4ff5-84d6-1cd79b845b2f
# ╠═0d46dd99-c614-40a6-9cd0-69b453ec782f
# ╠═da3b79da-3d14-405f-80af-d58d04b4f801
# ╟─18d07eee-60af-4dad-8f4a-9426f5907ad3
# ╠═f40f5823-f383-4d6e-a651-91c5a03cbf1e
# ╠═2970a6d2-599a-44ce-ab09-d52db64c0c64
# ╟─5b54210f-9a7d-447e-9491-f8fbb0892e7f
# ╠═e9df057d-3781-4fe1-b0ca-fab08b895ca2
# ╟─7f8b5d7b-25cf-4464-b01a-e9649001b1a1
# ╠═3d1e1190-2ba6-42ad-9c5b-3c3316fd75a0
# ╟─bda26511-d961-413a-8389-ad5be48f79fe
# ╟─4c6dd3ba-268e-4fad-b8d1-4bc78f24a46f
# ╠═c817e5e6-4cb4-4392-8f7e-e1a4bb009537
# ╠═46bb14fb-62b4-402b-8a0b-8096bd2a6289
# ╟─b9cdd1c8-2f8f-48c5-846d-e40cedc949b7
# ╟─370a4ccb-fdb6-4e3f-8004-d6f88f025945
# ╟─c6d85f60-c820-4269-a6b4-57483de13bd8
# ╟─601f3dfa-4ea5-4418-aeba-5ab1203f8753
# ╟─ce3b13a8-38e8-449a-8b11-7a61b8632fc9
# ╟─c61504df-808a-46f0-b8cc-dcc7197ffb3e
# ╟─ed8b654f-f964-4d8f-a998-1032c197f014
# ╟─c361ae07-61af-44bb-a5ee-be991390fa88
# ╟─0a98082a-94c3-41d8-a129-4f42e217bcd1
# ╟─5b38607d-6cfc-4fa0-b19f-5bea8ad38b39
# ╠═5aed1914-6960-41c8-91d4-09614766583d
# ╟─5a358aa5-bb4b-4b48-9d46-8628a9722023
# ╠═2afe4168-640f-4b7e-ab28-7ae22fba16c9
# ╠═198663dd-941a-4258-800f-80ad0638f884
# ╟─8893ec3a-7b9b-4887-9776-0c9c4f07cf14
# ╠═b186f0b5-721e-4757-9a4d-a839162b22f2
# ╠═827e960e-057e-40ae-beeb-f3c013d9f883
# ╠═96787e59-a958-404b-b610-42a28bd0353b
# ╟─28578d77-1439-49cf-a9f6-120557bce924
# ╟─b6fc9ad1-5f44-4697-be2e-407e2b9308c0
# ╟─71f12fb3-901d-4feb-9fbc-a5fc6e0f4750
# ╟─b061d6f2-bcd1-410e-a005-d2e993616b3a
# ╟─1c20116c-339c-453c-b6d1-4ed1477fcf12
# ╟─0a5ed3ea-12d9-46f9-aab8-472eae8a971d
# ╠═1056e659-b358-451f-85b3-a7ec9a6dac92
# ╟─86d8016f-9179-4bb2-be71-3708896ba216
# ╠═3a9c6bbe-5078-4f99-9418-dc22f73706cb
# ╟─310e07b1-ef44-4588-9fc2-7f70b84e527d
# ╠═e41adcb9-3c78-404b-a501-b359511b9a39
# ╟─d34b5710-2f37-4bb1-9e02-6e95996f7242
# ╠═31675329-f1bd-4752-8da1-af82475fe900
# ╟─71128267-d23a-4162-b9b3-52b86ec5f9de
# ╟─7d74a6be-4aac-4f12-9391-528f9cbf37ba
# ╟─37cbf7a2-6679-40a4-8085-21a4e900c59d
# ╟─f99c393f-308f-4821-8f5a-ee8ebcf5b77b
# ╠═06f497e4-d1a3-4e99-86f4-f63f69920f53
# ╟─6747980b-7072-4267-84c5-a352abf4ec25
# ╟─fe0ee6b7-9c42-41b8-929c-2dd7101490a3
# ╟─7550ccac-ca63-4f96-b576-595888071c34
# ╟─abe288f3-9a80-4c29-a918-73c57ab16dc2
# ╠═c23627d6-91a2-4b69-9e35-71b8a9578dd6
# ╠═24f945be-f3d5-48bd-80e1-12a7cc92d976
# ╟─62c3c7f1-2839-4c7e-bba7-1741649b3620
# ╟─b153e5e7-95ba-4425-91aa-ce9986a64392
# ╠═0f7184f5-a03f-482e-8339-fd12c7391e01
# ╟─ff9355dc-3e5f-4558-9027-668bd17a7a30
# ╟─a825e358-1fdc-42cb-87cf-ab0dbd092cb0
# ╠═2279f195-a9fa-46ee-925b-f54222d61d9a
# ╟─677da773-6130-41b2-8188-209a8d751f99
# ╟─c55f846d-d578-4c81-bdc4-ce5d03c62dba
# ╟─071d902e-a952-480e-9c21-5a3315162a6a
# ╟─54bf4640-fa81-4ef4-978a-a87682dd3401
# ╟─8a01d833-3220-4144-8c2a-dde4c1399795
# ╟─760eaee1-0af1-41c1-b38a-c0041559c0ed
# ╠═bf4df54b-6631-4b59-bf6a-26caea5ab7df
# ╟─18fbb98d-87a2-4f29-b6f0-3e19ad843b00
# ╠═fc5709fc-337d-4d42-b023-373089de2c8d
# ╠═e91fe8c0-d13e-401f-b3f1-77e04fe4df34
# ╟─ada8809b-8f3a-4298-94d9-e8225df4087d
# ╠═7580a872-47b1-4efc-9c51-9591d3552c5b
# ╠═a6945e5b-0516-49d1-a978-e4af5090aca3
# ╠═30b5ae33-c009-4ad5-8950-c75a614acde3
# ╟─265d6bdf-2381-471a-a99a-3d163b96e620
# ╠═ec39a8d0-be30-4c7c-9727-f7cffdd117a9
# ╠═d3387ea9-032f-4b62-96fe-3965ad187672
# ╟─9aef8d51-eb5b-4342-8ad5-02e6187b2953
# ╟─9260827c-3262-4870-9e5a-ac49bfa1dbae
# ╠═fbe80edb-7964-4103-bffa-c01a89904bd1
# ╟─2c91f496-3780-4257-8da1-8fbf8eeca908
# ╠═215e2f59-0541-46d5-8d48-fa381139fd54
# ╠═5f537343-2d7d-433f-a3aa-b075425fc9e2
# ╠═c28e6273-bad5-4688-af21-484c5de2bdf0
# ╟─df7f90b7-c989-4856-9adf-41be3f4e6444
# ╟─de83e0a2-6655-469e-8ab9-6e00b60e245c
# ╠═7d92b70f-fc40-47e4-97c4-0703181e2322
# ╠═f4f723ea-3574-419a-84a4-655d09375c3a
# ╠═544822a0-329e-48f7-9a16-1e18053ea9f0
# ╟─321b215c-ca89-4fe5-8f15-9bf37fe10064
# ╠═118957fc-e85a-4829-b3ae-9aa8fccc0a33
# ╟─bf84e5cc-8617-4a29-b5c9-ff7de784e29b
# ╟─7255af4d-611d-489d-b607-70eefb858170
# ╠═0fcbe952-87af-4c56-a6e8-cf80ada41497
# ╠═75afbded-6b3b-4a7e-ad53-a5f34808c056
# ╟─248e90a9-6c28-4ef2-bd51-112077f93f9c
# ╟─aba1a15f-15a8-495a-a585-77fc18ccb7dd
# ╠═efb934f0-2b02-42c6-9d5d-b0243bf889bd
# ╠═24a2dc89-51db-40d2-8990-832ac2c65fe2
# ╠═29d1ca09-60f4-4786-9623-3e09584aeece
# ╟─46e74b96-7f94-408c-a63d-d898e538bd59
# ╠═6dd8ebb3-a23f-4071-becd-94a8de5fd4f7
# ╟─686e2146-46ae-4799-bab7-51b42a3074eb
# ╠═5415b2b7-9c7c-4420-a664-a96e4bd23199
# ╟─ea8e9149-7c8f-492d-9577-19284fe50238
# ╟─8d22a8ff-3bf5-4226-b605-1a496a03d667
# ╟─3cbe3970-098e-4476-9062-87ce8dbf747c
# ╠═f51e0a4a-3d2c-47ee-8eee-dfc6e5c522ac
# ╠═70b112cc-3b6e-4c9e-a2fb-5282f2cc5605
# ╟─5acc26ff-fe68-4b6e-a67c-fd753efc949b
# ╠═b4c5f5e3-4975-4fa6-91ea-c4d634825c0e
# ╟─6031286a-85c9-4770-b03e-3bf6ddd12451
# ╠═34908b18-2277-4d80-b615-2ddaf7d12d85
# ╟─9ec9be96-c127-4124-ae7a-3eaddf21dd49
# ╠═dc66f1b5-f04a-4c2b-8ed2-295394c10a79
# ╟─4e86f431-8737-4055-be4d-51d8fb1250aa
# ╠═7469e43a-f963-4228-bbe3-4cffd113cb2b
# ╠═a2228c42-03d6-45e0-b3b1-18db8737e848
# ╟─d9eafc22-71c3-48e8-88ec-c332148ea98d
# ╠═43e5b217-965a-4312-bd1b-ffda74253653
# ╟─b76070e9-ef7a-4cb5-ade7-625073173c5c
# ╠═3d11987d-5afb-4bc6-b0be-a355d804b6c6
# ╟─f180e914-af0c-4d8a-afe4-bdc54ee988f6
# ╟─bf20d3e0-64b5-49d4-aabc-059aa6a390ad
# ╠═d93dfb88-0602-4913-a59e-587803a9b5a3
# ╠═b2265d0c-9f85-4eff-8872-2fa968474e3f
# ╠═485c79a6-2c47-4320-81c8-2d2ac2b5d5a2
# ╠═0244cc32-6371-42c0-8564-570d3424460d
# ╠═913c8389-3501-4a0c-abe3-4faa42ef9a04
# ╠═8d29975c-5adc-458b-95b0-f4369c5c2f3a
# ╟─b0ccc2ee-0355-4591-b243-5f56715a01b8
# ╟─c9cd7943-7b2a-4387-a4fb-766d9ee00594
# ╠═d2a12c25-4277-4f19-9811-9d371d91022c
# ╟─3f3b7906-7bd0-4a9f-8a49-14b8a8924218
# ╟─25e39a6b-5d8e-419f-9a44-8d72a8fde502
# ╠═6560fe42-daa9-47ce-8a88-dbddcc2d3a1c
# ╠═ea7b83b9-2e95-43da-b089-cdf4d6d5247d
# ╠═3245d573-ffe6-44b2-9734-753a011ab10c
# ╟─5c294ee4-1d99-4d7d-a94d-afc3a643614e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
