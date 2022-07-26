### A Pluto.jl notebook ###
# v0.19.10

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
begin
    import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="BenchmarkTools", version="1"),
        Pkg.PackageSpec(name="Distributions", version="0.25"),
        Pkg.PackageSpec(name="KernelDensity", version="0.6"),
        Pkg.PackageSpec(name="LaTeXStrings", version="1"),
        Pkg.PackageSpec(name="Plots", version="1"),
        Pkg.PackageSpec(name="PlutoUI", version="0.7"),
        Pkg.PackageSpec(name="StatsBase", version="0.33"),
        Pkg.PackageSpec(name="StatsPlots", version="0.14"),
    ])
    using BenchmarkTools, Distributions, KernelDensity, LaTeXStrings, LinearAlgebra, Plots, PlutoUI, StatsBase, Statistics, StatsPlots
end

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
md" In our introductory session (Lecture 0) we had a brief introduction to Julia via the QuantEcon website. In this first tutorial we will be looking at some basic ideas such as sampling and random variables. Julia is an amazing language for computational problems and is much faster than R for most practical applications in Bayesian Econometrics. You are welcome to still code in R if you wish. I will steer you in the right direction with resources from last year. However, I think it is worthwhile to learn Julia since the syntax is similar to Python and Matlab. 

**Note**: In terms of the project that you have to do at the end of the semester, most of you might want to use packages that are available in R, since the ecosystem is more mature and more packages are available. If you want to code up your own routines for your project using Julia, this will be more difficult. However, if you follow the notebooks carefully it should be within your reach. "

# ╔═╡ 49033e09-fd64-4707-916c-9435d3f0a9d2
md" This notebook we are working with is called a `Pluto` notebook and is useful for educational purposes. If you want to code in an integrated development environment, almost like `Rstudio`, then I recommend `VSCode`. "

# ╔═╡ 2eb626bc-43c5-4d73-bd71-0de45f9a3ca1
# TableOfContents() # Uncomment for TOC

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
2. QuantEcon (2021). [Quantitative Economics with Julia](https://julia.quantecon.org/). -- Lectures 0, 4, 8.
3. Aki Vehtari (2020). [Bayesian Data Analysis](https://avehtari.github.io/BDA_course_Aalto/index.html). -- Lectures 2, 3, 4, 5
5. Mattias Villani (2021). [Bayesian Learning](https://github.com/mattiasvillani/BayesianLearningBook) -- Lectures 2, 3, 4, 5
4. José Eduardo Storopoli (2021). [Bayesian Statistics with Julia and Turing.](https://storopoli.io/Bayesian-Julia/) -- Lectures 2, 3, 4, 9
5. Gary Koop (2021). [Bayesian Econometrics](https://sites.google.com/site/garykoop/teaching/sgpe-bayesian-econometrics). -- Lectures 5, 6, 7
6. Joshua Chan (2017). [Notes on Bayesian Econometrics](https://joshuachan.org/notes_BayesMacro.html). -- Lectures 5, 6, 7

"""

# ╔═╡ be11b67f-d04d-46b1-a935-eedd7e59ede3
md" There is a book by Gary Koop, called Bayesian Econometrics, that accompanies the course above. However, it is not essential to have the book. I will upload some articles that with similar content. Let us get going with the first lecture, we will start with the idea of random sampling. "

# ╔═╡ 040c011f-1653-446d-8641-824dc82162eb
md" ## Random sampling "

# ╔═╡ 0d0f08f3-48e2-402c-bbb5-33bd3d09ab06
md" One thing that we will be using quite frequently in Bayesian econometrics is the idea of random sampling. This means sampling from different types of distributions.  "

# ╔═╡ 350bb07e-6e88-49fc-ae69-c24b1549650d
dcolours = distinguishable_colors(10)

# ╔═╡ dc306e4a-7841-4c36-8c5b-d96acc7714de
md" We can also sample several random objects from the same collection. In this case we use an **array comprehension**. "

# ╔═╡ b028662e-a8ac-45c4-86aa-d684bbb864c8
md" An easier way to do this is to simply add another argument to the `rand` function. "

# ╔═╡ a69254ad-d7e2-4ddd-9a8f-58eaa9a45ce8
md" We can also generate random matrices in with this function. "

# ╔═╡ 909ea3be-9b65-465a-ab31-0d3d525c021f
md" ## Uniform sampling "

# ╔═╡ b1d3017e-8caf-462f-bb0b-8154202f21e6
md" The `rand` function has performed uniform sampling, which means that each object has the same probability of being selected. For our next example we will be counting heads and tails using the `countmap` function. "

# ╔═╡ 7254eb45-8170-40f2-afd3-e30ec5c26781
md" In this case we have a dictionary that maps keys, such as `heads` and `tails`, to specific values. "

# ╔═╡ cdad68ca-dac9-49ad-8149-939d18f00778
md" ## Tossing a weighted coin "

# ╔═╡ ea5a71b7-845b-4310-b1fb-f69ee71ac3fb
md"""
How could we model a coin that is **weighted**, so that it is more likely to come up heads? We want to assign a probability $p = 0.7$ to heads, and $q = 0.3$ to tails. 
"""

# ╔═╡ d6aa79f3-45c8-4ff5-84d6-1cd79b845b2f
md"""
One way would be to generate random integers between $1$ and $10$ and assign heads to a subset of the possible results with the desired probability, e.g. $1:7$ get heads, and $8:10$ get tails. We will use this same logic later in other examples, so it is important to understand what we are doing here. 
"""

# ╔═╡ 18d07eee-60af-4dad-8f4a-9426f5907ad3
md" Another way to do this might be with a **ternary operator**, see the Julia documentation [here](https://docs.julialang.org/en/v1/manual/control-flow/). "

# ╔═╡ 5b54210f-9a7d-447e-9491-f8fbb0892e7f
md" If we generate a uniform number between $0$ and $1$ and then check if it is less than some probability, this is known as one **Bernoulli trial**. Binomial and Bernoulli random variables will be covered in more detail later in the lecture and also defined more formally in the next lecture. For now you simply need to understand the process. We can construct a simple Bernoulli function that encapsulates this idea.  "

# ╔═╡ 7f8b5d7b-25cf-4464-b01a-e9649001b1a1
md"""
p = $(@bind p₁ PlutoUI.Slider(0.0:0.01:1.0, show_value=true, default=0.7))
"""

# ╔═╡ bda26511-d961-413a-8389-ad5be48f79fe
md" **Note**: the output for this function is `true` or `false` instead of `heads` or `tails` in the weighted coin example. "

# ╔═╡ 4c6dd3ba-268e-4fad-b8d1-4bc78f24a46f
md" A Bernoulli random variable model for a weighted coin, for example, will take value $1$ with probability $p$ and $0$ with probability $(1- p)$. Our Bernoulli function that we wrote provides `true` and `false` values. Let us sample some Bernoulli random variates. "

# ╔═╡ b9cdd1c8-2f8f-48c5-846d-e40cedc949b7
md" The calculation for the mean is just the proportion of `true` values, which should be roughly equal to our probability parameter. Accuracy increases with the number of flips. How would you increase the number of flips in the code above? Play around with the code to see if you can do it. "

# ╔═╡ 370a4ccb-fdb6-4e3f-8004-d6f88f025945
md" # Probability distributions and types "

# ╔═╡ c6d85f60-c820-4269-a6b4-57483de13bd8
md"""
In this section I will provide some basic properties of common probability distributions that are often used in Bayesian econometrics. We will brielfy introduce three distributions in this section, and then as we progress we will introduce more. We will also discuss the type system in Julia, which is a key feature of the language. To keep the code clear in the following lectures we won't always use best coding practice, but every now and then we will discuss some core principles. 
"""

# ╔═╡ 601f3dfa-4ea5-4418-aeba-5ab1203f8753
md" ## Bernoulli "

# ╔═╡ ce3b13a8-38e8-449a-8b11-7a61b8632fc9
md" As we have stated, the Bernoulli distribution describes a binary event of a successful experiment. We usually represent $0$ as failure and $1$ as success, so the result of a Bernoulli distribution is a binary variable. The Bernoulli distribution is widely used to model discrete binary outcomes in which there are only two possible results. The value of $p$ represents the probability of success. "

# ╔═╡ c61504df-808a-46f0-b8cc-dcc7197ffb3e
md"""
p = $(@bind p₂ PlutoUI.Slider(0.0:0.01:1.0, show_value=true, default=0.7))
"""

# ╔═╡ ed8b654f-f964-4d8f-a998-1032c197f014
begin
	Plots.plot(Distributions.Bernoulli(p₂),
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
md" We might want to know what the mean (or expected value) of the process is. We can do this easily by constructing a Bernoulli `type` with certain properties. If you are new to programming, the idea of types will be strange. However, try and follow along with the example to see what the benefit is in creating types.  "

# ╔═╡ 0a98082a-94c3-41d8-a129-4f42e217bcd1
md" ### Make Bernoulli a type "

# ╔═╡ 5b38607d-6cfc-4fa0-b19f-5bea8ad38b39
md"
Currently we need one function for sampling from a Bernoulli random variable, a different function to calculate the mean and a different function for the standard deviation. So many different functions! 

In mathematical terms we have this Bernoulli random variable and we are calculating properties of the particular concept. We can do the same thing computationally by creating a new object that represents the Bernoulli random variable. "

# ╔═╡ 5aed1914-6960-41c8-91d4-09614766583d
struct Bernoulli_New
	p::Float64
end

# ╔═╡ 5a358aa5-bb4b-4b48-9d46-8628a9722023
md" We want to be able to sample from this using the `rand()` function and also take its mean. In order to do this we will extend the rand function from the `Base` library of Julia and the `mean` function from the `Statistics.jl` library. 

Note that we are adding [methods](https://docs.julialang.org/en/v1/manual/methods/) to these particular functions. "

# ╔═╡ 2afe4168-640f-4b7e-ab28-7ae22fba16c9
Base.rand(X::Bernoulli_New) = Int( rand() < X.p ) # Add method to the rand function

# ╔═╡ cc4578f7-358c-4635-9a16-816e0b0f9d4e
md" Adding a method to a function in Julia means that depending on the `type` of the input received, the function will output something different. This is the idea behind multiple dispatch in Julia. You can read more about multiple dispatch [here](https://opensourc.es/blog/basics-multiple-dispatch/)."

# ╔═╡ 198663dd-941a-4258-800f-80ad0638f884
B = Bernoulli_New(0.25)

# ╔═╡ 8893ec3a-7b9b-4887-9776-0c9c4f07cf14
md" The object `B` represents a Bernoulli random variable with probability of success $p$. One should note that this type already exists in a package like `Distributions.jl`, so you should be careful about naming conventions. "

# ╔═╡ ad2cadea-d982-4b4c-939d-7c8c4b587539
md" Next we can extend the `mean` function to accept our Bernoulli `type`. This means that whenever we input a variable of the `Bernoulli_New` type the mean will be calculated in the way specified. If we were for example to calculate the mean of a sum of floating point values, Julia would recognise that we are inputting a different type and therefore look for the associated method. "

# ╔═╡ 827e960e-057e-40ae-beeb-f3c013d9f883
Statistics.mean(X::Bernoulli_New) = X.p

# ╔═╡ 55bb47ce-558c-451d-a752-aa56b8640832
typeof(B) # You can see that this is an instance of our created type!

# ╔═╡ 28578d77-1439-49cf-a9f6-120557bce924
md" ## Binomial "

# ╔═╡ b6fc9ad1-5f44-4697-be2e-407e2b9308c0
md" The binomial distribution describes an event of the number of successes in a sequence of $n$ independent experiment(s), each asking a yes-no question with a probability of success $p$. Note that the Bernoulli distribution is a special case of the binomial distribution where the number of experiments is $1$. "

# ╔═╡ 71f12fb3-901d-4feb-9fbc-a5fc6e0f4750
md" The binomial distribution has two parameters and its notation is $\text{Bin}(n, p)$. An example would be the number of heads in $5$ coin flips (as illustrated below for different values of $p$). We will deal with the coin flip problem in more detail in the next lecture. "

# ╔═╡ b061d6f2-bcd1-410e-a005-d2e993616b3a
md"""
p = $(@bind p₃ PlutoUI.Slider(0.0:0.01:1.0, show_value=true, default=0.7))
"""

# ╔═╡ 1c20116c-339c-453c-b6d1-4ed1477fcf12
begin
	Plots.plot(Binomial(5, p₃),
	        markershape=:circle,
	        alpha=0.7,
	        xlabel=L"\theta",
	        ylabel="Mass", 
			lw = 2, 
			legend = false
	    )
end

# ╔═╡ 0a5ed3ea-12d9-46f9-aab8-472eae8a971d
md" We can make the binomial random variable a type. We only require information on $n$ and $p$, so the `struct` is: "

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

# ╔═╡ d34b5710-2f37-4bb1-9e02-6e95996f7242
md"""
n = $(@bind binomial_n PlutoUI.Slider(1:100, show_value=true, default=1)); 
p = $(@bind binomial_p PlutoUI.Slider(0.0:0.01:1, show_value=true, default=0.5))

"""

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

$$\mathbb{P}(X \in [a, b]) = \int_{a}^b f_X(y) \, dx.$$

**Notation remark**: The tradition in statistics is to use capital letters for random variables and then lowe case letters for realisation of that random variable. Our notation will change from the second lecture onward. Please make a note of this so that you do not get confused by notation. I will mention this again in another lecture. 

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

# ╔═╡ 9c93fdce-c678-4c53-986d-e870c53a50e4
methods(bell_curve);

# ╔═╡ 62c3c7f1-2839-4c7e-bba7-1741649b3620
md"""
We can shift and scale the Gaussian distribution in the following manner. 
"""

# ╔═╡ b153e5e7-95ba-4425-91aa-ce9986a64392
md"""
μ = $(@bind μ PlutoUI.Slider(-3:0.01:3, show_value=true, default=0.0))
σ = $(@bind σ PlutoUI.Slider(0.01:0.01:3, show_value=true, default=1))
"""

# ╔═╡ a825e358-1fdc-42cb-87cf-ab0dbd092cb0
md" One fo the nice things about Gaussians is that the sum of two Gaussians is also Gaussian. We will make use of this property of the Gaussian in future, so take note. In order to show this, let us sample from two Gaussian random variables and add the resulting random variable. "

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
Thus far we have only defined **concrete types**. Think about concrete types as the types with specific data attached. Now we will define an **abstract type** using `abstract type <Name> end`, where `<Name>` is replaced with name of the type.

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

Important to note that the `Gaussian` type that we have created here is a **concrete type**.

We now extend the `mean`, `var` and `std` function from the Statistics library to act on our newly created object. "

# ╔═╡ 7580a872-47b1-4efc-9c51-9591d3552c5b
begin
	Statistics.mean(X::Gaussian) = X.μ
	Statistics.var(X::Gaussian) = X.σ²
end

# ╔═╡ 265d6bdf-2381-471a-a99a-3d163b96e620
md" Now let us show that we can calculate the standard deviation for any random variable, not just the Gaussian. Calculating the standard deviation is simply going to be the square root of the variance for **any** random variable. 

We can define this to act on any random variable, even ones that we have not created yet!"

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

# ╔═╡ 797f952a-abac-4dde-9834-0e46a06bfa96
sum_of = 2.0 + 3 + 4 + 5 # If we have one floating point value then the sum gets `promoted` to a floating point value, even if the other values are integers. 

# ╔═╡ 8aa2ed56-95e4-48ee-8f7e-3da02e7c51c6
typeof(sum_of) # In this case the type relates to the sum that we have taken

# ╔═╡ 0f7184f5-a03f-482e-8339-fd12c7391e01
data = μ .+ σ .* randn(10^5) # Take note of the broadcasting performed by the dot operator

# ╔═╡ 2279f195-a9fa-46ee-925b-f54222d61d9a
begin
	data1 = 4 .+ sqrt(0.3) .* randn(10^5)
	data2 = 6 .+ sqrt(0.7) .* randn(10^5)
	
	total = data1 + data2
end

# ╔═╡ 5f537343-2d7d-433f-a3aa-b075425fc9e2
G1 + G2

# ╔═╡ 0fcbe952-87af-4c56-a6e8-cf80ada41497
Base.rand(X::Gaussian) = X.μ + √(X.σ²) * randn()

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

# ╔═╡ 96787e59-a958-404b-b610-42a28bd0353b
mean(B)

# ╔═╡ 16db4c10-cac0-4cc4-a473-9c5ccf488e92
mean(sum_of)

# ╔═╡ a6945e5b-0516-49d1-a978-e4af5090aca3
mean(G)

# ╔═╡ c28e6273-bad5-4688-af21-484c5de2bdf0
mean(G1 + G2) == mean(G1) + mean(G2)

# ╔═╡ a2228c42-03d6-45e0-b3b1-18db8737e848
mean(B2 + B3)

# ╔═╡ d9eafc22-71c3-48e8-88ec-c332148ea98d
md"""
To have a simple equation for the variance, we need to assume that the two random variables are **independent**. Perhaps the name should have been `SumOfTwoIndependentRandomVariables`, but it seems too long.
"""

# ╔═╡ 43e5b217-965a-4312-bd1b-ffda74253653
Statistics.var(S::SumOfTwoRandomVariables) = var(S.X1) + var(S.X2)

# ╔═╡ 30b5ae33-c009-4ad5-8950-c75a614acde3
var(G)

# ╔═╡ ec39a8d0-be30-4c7c-9727-f7cffdd117a9
Statistics.std(X::RandomVariable) = sqrt(var(X))

# ╔═╡ d3387ea9-032f-4b62-96fe-3965ad187672
std(G)

# ╔═╡ 6dd8ebb3-a23f-4071-becd-94a8de5fd4f7
mean(B1), var(B1), std(B1)

# ╔═╡ b76070e9-ef7a-4cb5-ade7-625073173c5c
md"""
How can we sample from the sum? It's actually easy!
"""

# ╔═╡ 3d11987d-5afb-4bc6-b0be-a355d804b6c6
Base.rand(S::SumOfTwoRandomVariables) = rand(S.X1) + rand(S.X2)

# ╔═╡ b78745cc-84ef-4bbe-a916-9c87cde47145
rand(1:6) # Choose random value from 1 to 6 

# ╔═╡ 08fc15b7-2cc6-4a21-82a6-521f6294ee79
rand([2, 3, 5, 7, 11]) # Choose random value from this set of values

# ╔═╡ 4c48df5c-33c6-4d26-b6e4-80e3c883d389
rand()   # random number between 0 and 1 -- similar to runif() in R

# ╔═╡ 07b17b42-e968-4a7a-bf56-da1a3a1bef52
rand(dcolours) 

# ╔═╡ 71482500-7ca7-4a4e-8635-f29ee6f11ced
[rand(1:6) for i in 1:10]

# ╔═╡ f882026e-3393-46a7-b284-f0313386f214
rand(1:6, 10) # This generates an array in the same way as above

# ╔═╡ 94a2862e-1f6b-4c4e-ad32-4b6c49924454
rand(1:6, 10, 10) # Generates a 10x10 matrix with values that range from 1 to 6

# ╔═╡ fa929cc1-ce08-4c9a-922b-cdf611fa3e2a
rand(dcolours, 10, 10) # Generate matrix of colours

# ╔═╡ 9492ddc5-301d-46d2-bf34-5396563f5d5b
tosses = rand( ["head", "tail"], 10000)

# ╔═╡ 4f5e9c1f-d510-4897-9176-218a1a2f4057
toss_counts = countmap(tosses)

# ╔═╡ caeee82e-b854-4b34-b34f-5899e5a9b952
prob_tail = toss_counts["tail"] / length(tosses) # Determines the probability of a tail. 

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

# ╔═╡ f40f5823-f383-4d6e-a651-91c5a03cbf1e
simple_weighted_coin2() = rand(1:10) ≤ 7 ? "heads" : "tails" 

# ╔═╡ 2970a6d2-599a-44ce-ab09-d52db64c0c64
simple_weighted_coin2()

# ╔═╡ e9df057d-3781-4fe1-b0ca-fab08b895ca2
bernoulli(p) = rand() < p # Takes in a value p between 0 and 1 to compare against

# ╔═╡ 3d1e1190-2ba6-42ad-9c5b-3c3316fd75a0
countmap( [bernoulli(p₁) for _ in 1:1000] ) # 10000 iterations, count how many true and false given the value of p

# ╔═╡ c817e5e6-4cb4-4392-8f7e-e1a4bb009537
flips = [Int(bernoulli(p₁)) for _ in 1:100];

# ╔═╡ 46bb14fb-62b4-402b-8a0b-8096bd2a6289
mean(flips) 

# ╔═╡ b186f0b5-721e-4757-9a4d-a839162b22f2
rand(B)

# ╔═╡ e41adcb9-3c78-404b-a501-b359511b9a39
rand(Binomial_New(10, 0.25))

# ╔═╡ 31675329-f1bd-4752-8da1-af82475fe900
begin
	binomial_data = [rand(Binomial_New(binomial_n, binomial_p)) for _ in 1:10000]
	
	bar(countmap(binomial_data), alpha=0.5, size=(500, 300), leg=false, bin_width=0.5)
end

# ╔═╡ 75afbded-6b3b-4a7e-ad53-a5f34808c056
histogram!([rand(Gaussian(μμ, σσ)) for i in 1:10^4], alpha=0.5, norm=true)

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

# ╔═╡ 677da773-6130-41b2-8188-209a8d751f99
begin
	
	histogram(data1, alpha=0.4, norm=true, size=(600, 400), label = "data1", title="Sum of Gaussians")
	histogram!(data2, alpha=0.4, norm=true, size=(600, 400), label = "data2", color = :black)
	histogram!(total, alpha=0.8, norm=true, size=(600, 400), label = "total")
	plot!(2:0.01:14, x -> bell_curve(x, 10, 1), lw=2, color = :black, legend = false)
end

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
# ╟─cc4578f7-358c-4635-9a16-816e0b0f9d4e
# ╠═198663dd-941a-4258-800f-80ad0638f884
# ╟─8893ec3a-7b9b-4887-9776-0c9c4f07cf14
# ╠═b186f0b5-721e-4757-9a4d-a839162b22f2
# ╟─ad2cadea-d982-4b4c-939d-7c8c4b587539
# ╠═827e960e-057e-40ae-beeb-f3c013d9f883
# ╠═96787e59-a958-404b-b610-42a28bd0353b
# ╠═797f952a-abac-4dde-9834-0e46a06bfa96
# ╠═16db4c10-cac0-4cc4-a473-9c5ccf488e92
# ╠═8aa2ed56-95e4-48ee-8f7e-3da02e7c51c6
# ╠═55bb47ce-558c-451d-a752-aa56b8640832
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
# ╟─31675329-f1bd-4752-8da1-af82475fe900
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
# ╠═9c93fdce-c678-4c53-986d-e870c53a50e4
# ╟─62c3c7f1-2839-4c7e-bba7-1741649b3620
# ╟─b153e5e7-95ba-4425-91aa-ce9986a64392
# ╠═0f7184f5-a03f-482e-8339-fd12c7391e01
# ╠═ff9355dc-3e5f-4558-9027-668bd17a7a30
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
# ╠═25e39a6b-5d8e-419f-9a44-8d72a8fde502
# ╠═6560fe42-daa9-47ce-8a88-dbddcc2d3a1c
# ╠═ea7b83b9-2e95-43da-b089-cdf4d6d5247d
# ╠═3245d573-ffe6-44b2-9734-753a011ab10c
# ╟─5c294ee4-1d99-4d7d-a94d-afc3a643614e
