### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ 12068ed6-10bf-4e85-8ccc-9daf51828ea2
using LinearAlgebra, Statistics, Distributions

# ╔═╡ edd72740-ff27-11eb-3135-d9daa7fdf047
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
">ATS 872: Lecture 0</p>
<p style="text-align: center; font-size: 1.8rem;">
 Introduction to Julia (live coding)
</p>

<style>
body {
overflow-x: hidden;
}
</style>"""

# ╔═╡ 180a5bc1-ce8f-40e5-b718-ed00499841c4
md"""

This is our first Julia session. We can add math 

$x + y = 5$

"""

# ╔═╡ 18821b21-5880-4083-a4ab-31f2c66e9379
x = 9 + 1; # Semicolon supresses output

# ╔═╡ d0b77d0e-69e5-4646-a3ed-b02a02bc4aee
y = 5;

# ╔═╡ b16dfe67-1799-4c77-a667-2d4882d7655b
x + y

# ╔═╡ 5aeaa45d-e7ab-4064-9c5e-cf9317585425
md""" #### Let's load some packages """

# ╔═╡ ed2be8f9-34bc-4990-afae-dfe67dce497e
md""" ### Data types """

# ╔═╡ 56428b1f-b188-4236-b034-0c94194769a5
typeof(x)

# ╔═╡ a4cba836-9866-4444-8898-c7ce7193b5c8
z = 1 > 2

# ╔═╡ ce9cad0f-7d96-4453-a474-7ddd1669499f
typeof(z)

# ╔═╡ 0fbafe9d-660f-43c8-9d20-088e84bbe701
typeof(1.0)

# ╔═╡ 9d24bfd3-3d13-4e39-8318-e49df12a953d
x * y

# ╔═╡ e1a8086e-ee54-4846-8474-7da39d652e76
x ^ y 	# x ** y

# ╔═╡ 92e06373-0ea3-4368-b6a9-348adff7b975
y / x

# ╔═╡ 8996789a-1533-479a-a152-a8805f182613
2x + 3y

# ╔═╡ aa5d67e7-0f84-4bec-b5eb-830aebccf42e
t = "foobar"

# ╔═╡ 8d5b7366-7361-4537-883e-a1bb3f646a6a
typeof(t)

# ╔═╡ f627d687-4d35-495c-a8c7-d9131057e74a
"My favourite number is $x"

# ╔═╡ 15023be0-7c88-4201-a295-2a8891ec607b
"foo" * "bar"

# ╔═╡ 1f049c43-fa4b-4745-8f56-019868925dc1
s = "Charlie don't surf"

# ╔═╡ 64b5adb0-dffa-4002-9b51-5bea4c6fb2b7
split(s)

# ╔═╡ 6d05d9a4-4fc5-4a64-96ad-26f1740d3ec6
replace(s, "surf" => "ski")

# ╔═╡ e544680e-b620-4a23-9397-fca55023f68b
strip(" foobar ")  # remove whitespace

# ╔═╡ 4b56b96e-aaa9-4257-b3c1-d6e8b8dda033
md"""

### Containers

"""

# ╔═╡ ad4bbc48-9170-4366-abf5-3b6486a1679d
t₁ = ("foo", "bar") # Immutable 

# ╔═╡ f0157a91-aa7d-4d0a-b7dd-a8ff24739726
t₂ = ("foo", 2) # Can contain different data types

# ╔═╡ 8b0a85f6-36b2-45ff-9af3-3c0b29642b6d
typeof(t₂)

# ╔═╡ 7e4ad5a2-89b8-4f79-918d-27ea45689748
t₃ = ["foo", 2] # Arrays

# ╔═╡ 0f2fad19-7b6d-4953-b12f-9283048d8a77
typeof(t₃)

# ╔═╡ 3eed2489-04d1-4169-a695-9b0927274034
t₄ = [2, 4]

# ╔═╡ 086626fb-272f-4a2d-8fc6-7e1bd34699d4
typeof(t₄) # One dimensional array

# ╔═╡ 94342ed2-6376-4c9f-b0a7-cc1939d53426
m₁ = [2 3 4; 2 5 6.0; 3 5 8]

# ╔═╡ 6496a671-4229-4fed-87b9-49029437c28f
t₅ = "foo", 1

# ╔═╡ 06ffd4ab-4d53-4c06-85fa-ac7b020bee64
word, val = t₅ # Unpacking tuples into variables

# ╔═╡ 63344f9b-8625-48e4-af1c-80259acd164b
word

# ╔═╡ 4d111c25-d08d-497d-92ab-e602eef9438f
val

# ╔═╡ 99db77f6-9dbc-4136-b647-ce1bcd53fa54
"word is $word, and val is $val"

# ╔═╡ 8823d272-1536-451c-b55b-437c7c52eb76
md""" ### Referencing items """

# ╔═╡ 47d2165a-c6e8-4fa7-b155-7dbcfaa7c94c
a₁ = [10, 20, 30, 40] # Array / vector

# ╔═╡ 9b50adbd-1bde-43da-b034-88a53fbb6130
a₁

# ╔═╡ 74d39962-8f7b-4965-ab7b-ab571e6a7619
push!(a₁, 50) # ! mutation

# ╔═╡ b76c0285-4eb8-4246-8041-40e7589862e8
a₁[1:3] # Julia uses 1 based indexing, MATLAB and R are similar. However, python uses 0 based indexing. 

# ╔═╡ deeb26df-0209-4534-a89f-88921be176dc
a₁[2:end]

# ╔═╡ 94f2fb7f-1b7e-4d17-8dce-24db50f07ebc
"foobar"[3:end]

# ╔═╡ 0fbe4644-fb8e-4f3a-b9bf-c952040214f1
d = Dict("name" => "Frodo", "age" => 33)

# ╔═╡ 0920d0c7-ed6f-4a43-9fd6-6f392609998e
keys(d)

# ╔═╡ d574201f-6b1b-40d8-b755-adfadedd71ba
values(d)

# ╔═╡ 5e8f3186-5a81-4bab-b57a-a86f6652a4e4
md""" ### Iteration """

# ╔═╡ 1695265f-441f-41ba-9597-e6dea6a52c04
begin
	actions = ["surf", "ski"] # Iterable
	for action in actions 
		println("Charlie doesn't $action")
	end
end

# ╔═╡ 21d77e55-d2ae-4738-a72f-1b5aa2903cde
x_values = 1:5

# ╔═╡ b7ef0cb7-318f-4c12-860c-2f7584c0fd95
for x in x_values
    println(x * x)
end

# ╔═╡ aadbf0bd-9d30-412b-935e-e06d606da334
doubles = [ 2i for i in 1:4 ] # Array comprehension (list comprehension)

# ╔═╡ 29d4b074-cf3f-45f4-b47d-4c3fcc860230
animals = ["dog", "cat", "bird"]  

# ╔═╡ f4144452-eec8-44f8-b7d9-e8c9235f1424
plurals = [ animal * "s" for animal in animals ]

# ╔═╡ 3ff56a1f-e3e6-45f5-906b-a0e34a06e302
[ i + j for i in 1:3, j in 4:6 ]

# ╔═╡ 6cb975ac-d82a-40ac-be4e-a38eca2c109f
[ i + j + k for i in 1:3, j in 4:6, k in 7:9 ]

# ╔═╡ bf1c00cf-b6d1-4a95-b25d-a108cd9d3b6c
[ (i, j) for i in 1:2, j in animals]

# ╔═╡ 4775256a-f7b1-4819-b0ea-e23e89d00d43
md""" ### Functions """

# ╔═╡ d64d5b04-d65a-4d09-88ef-055c45336b44
function f1(a, b) # a, b positional arguments
	return a * b
end

# ╔═╡ 20349b96-2ba7-4923-b62f-f24b91976177
f1(2, 3) == f1(3, 2)

# ╔═╡ de7b58df-066f-4eb3-81e2-ea3a24987f2b
f(x) = sin(1/x)

# ╔═╡ 67bab112-611e-45d2-ba46-987f28ec3010
f(1/ pi)

# ╔═╡ 7fc85825-96b4-4000-95ae-c8272465471d
map(x -> sin(1/x), randn(3)) 

# ╔═╡ bec09f5c-6fa3-476a-8f23-1afaaae6893f
randn(3)

# ╔═╡ 1f4dfccc-97f9-425f-b79d-875447ba2176
g(x; a = 1) = exp(cos(a * x)) # a is a keyword argument

# ╔═╡ 7b2bda6a-3701-4fa8-b002-87eb2f9b30ba
g₂(x, a = 1) = exp(cos(a * x))

# ╔═╡ 2287afcc-e7fd-4dfe-bcb5-3aed4041b478
g₂(1, 2)

# ╔═╡ 31400c16-5add-4ff7-a43b-9d7887da026a
g(1, a = 2) # positional and keyword arguments

# ╔═╡ 2cb70fff-4509-46b2-bd1c-0d21d61961e9


# ╔═╡ 4e18cfb3-1f46-45d6-91d6-7857dd77836e


# ╔═╡ aeb41add-1dbe-4214-9bf8-1ff807e5ce75


# ╔═╡ 6563768a-6e36-45d6-b686-9734f3603d22
begin
	countries = ("Japan", "Korea", "China")
	cities = ("Tokyo", "Seoul", "Beijing")
	for (i, country) in enumerate(countries)
	    city = cities[i]
	    println("The capital of $hello is $city")
	end
end

# ╔═╡ b75a03b7-53fd-4e13-ae01-abc54e2e9dc3
begin
	countries = ("Japan", "Korea", "China")
	cities = ("Tokyo", "Seoul", "Beijing")
	for (country, city) in zip(countries, cities)
	    println("The capital of $country is $city")
	end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distributions = "~0.25.11"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "bdc0937269321858ab2a4f288486cb258b9a0af7"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.3.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "79b9563ef3f2cc5fc6d3046a5ee1a57c9de52495"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.33.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "7d9d316f04214f7efdbb6398d545446e246eff02"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.10"

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

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IrrationalConstants]]
git-tree-sha1 = "f76424439413893a832026ca355fe273e93bce94"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

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

[[LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "3d682c07e6dd250ed082f883dc88aee7996bf2cc"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.0"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

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

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

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

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

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

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─edd72740-ff27-11eb-3135-d9daa7fdf047
# ╟─180a5bc1-ce8f-40e5-b718-ed00499841c4
# ╠═18821b21-5880-4083-a4ab-31f2c66e9379
# ╠═d0b77d0e-69e5-4646-a3ed-b02a02bc4aee
# ╠═b16dfe67-1799-4c77-a667-2d4882d7655b
# ╟─5aeaa45d-e7ab-4064-9c5e-cf9317585425
# ╠═12068ed6-10bf-4e85-8ccc-9daf51828ea2
# ╟─ed2be8f9-34bc-4990-afae-dfe67dce497e
# ╠═56428b1f-b188-4236-b034-0c94194769a5
# ╠═a4cba836-9866-4444-8898-c7ce7193b5c8
# ╠═ce9cad0f-7d96-4453-a474-7ddd1669499f
# ╠═0fbafe9d-660f-43c8-9d20-088e84bbe701
# ╠═9d24bfd3-3d13-4e39-8318-e49df12a953d
# ╠═e1a8086e-ee54-4846-8474-7da39d652e76
# ╠═92e06373-0ea3-4368-b6a9-348adff7b975
# ╠═8996789a-1533-479a-a152-a8805f182613
# ╠═aa5d67e7-0f84-4bec-b5eb-830aebccf42e
# ╠═8d5b7366-7361-4537-883e-a1bb3f646a6a
# ╠═f627d687-4d35-495c-a8c7-d9131057e74a
# ╠═15023be0-7c88-4201-a295-2a8891ec607b
# ╠═1f049c43-fa4b-4745-8f56-019868925dc1
# ╠═64b5adb0-dffa-4002-9b51-5bea4c6fb2b7
# ╠═6d05d9a4-4fc5-4a64-96ad-26f1740d3ec6
# ╠═e544680e-b620-4a23-9397-fca55023f68b
# ╟─4b56b96e-aaa9-4257-b3c1-d6e8b8dda033
# ╠═ad4bbc48-9170-4366-abf5-3b6486a1679d
# ╠═f0157a91-aa7d-4d0a-b7dd-a8ff24739726
# ╠═8b0a85f6-36b2-45ff-9af3-3c0b29642b6d
# ╠═7e4ad5a2-89b8-4f79-918d-27ea45689748
# ╠═0f2fad19-7b6d-4953-b12f-9283048d8a77
# ╠═3eed2489-04d1-4169-a695-9b0927274034
# ╠═086626fb-272f-4a2d-8fc6-7e1bd34699d4
# ╠═94342ed2-6376-4c9f-b0a7-cc1939d53426
# ╠═6496a671-4229-4fed-87b9-49029437c28f
# ╠═06ffd4ab-4d53-4c06-85fa-ac7b020bee64
# ╠═63344f9b-8625-48e4-af1c-80259acd164b
# ╠═4d111c25-d08d-497d-92ab-e602eef9438f
# ╠═99db77f6-9dbc-4136-b647-ce1bcd53fa54
# ╟─8823d272-1536-451c-b55b-437c7c52eb76
# ╠═47d2165a-c6e8-4fa7-b155-7dbcfaa7c94c
# ╠═9b50adbd-1bde-43da-b034-88a53fbb6130
# ╠═74d39962-8f7b-4965-ab7b-ab571e6a7619
# ╠═b76c0285-4eb8-4246-8041-40e7589862e8
# ╠═deeb26df-0209-4534-a89f-88921be176dc
# ╠═94f2fb7f-1b7e-4d17-8dce-24db50f07ebc
# ╠═0fbe4644-fb8e-4f3a-b9bf-c952040214f1
# ╠═0920d0c7-ed6f-4a43-9fd6-6f392609998e
# ╠═d574201f-6b1b-40d8-b755-adfadedd71ba
# ╟─5e8f3186-5a81-4bab-b57a-a86f6652a4e4
# ╠═1695265f-441f-41ba-9597-e6dea6a52c04
# ╠═21d77e55-d2ae-4738-a72f-1b5aa2903cde
# ╠═b7ef0cb7-318f-4c12-860c-2f7584c0fd95
# ╠═b75a03b7-53fd-4e13-ae01-abc54e2e9dc3
# ╠═6563768a-6e36-45d6-b686-9734f3603d22
# ╠═aadbf0bd-9d30-412b-935e-e06d606da334
# ╠═29d4b074-cf3f-45f4-b47d-4c3fcc860230
# ╠═f4144452-eec8-44f8-b7d9-e8c9235f1424
# ╠═3ff56a1f-e3e6-45f5-906b-a0e34a06e302
# ╠═6cb975ac-d82a-40ac-be4e-a38eca2c109f
# ╠═bf1c00cf-b6d1-4a95-b25d-a108cd9d3b6c
# ╟─4775256a-f7b1-4819-b0ea-e23e89d00d43
# ╠═d64d5b04-d65a-4d09-88ef-055c45336b44
# ╠═20349b96-2ba7-4923-b62f-f24b91976177
# ╠═de7b58df-066f-4eb3-81e2-ea3a24987f2b
# ╠═67bab112-611e-45d2-ba46-987f28ec3010
# ╠═7fc85825-96b4-4000-95ae-c8272465471d
# ╠═bec09f5c-6fa3-476a-8f23-1afaaae6893f
# ╠═1f4dfccc-97f9-425f-b79d-875447ba2176
# ╠═7b2bda6a-3701-4fa8-b002-87eb2f9b30ba
# ╠═2287afcc-e7fd-4dfe-bcb5-3aed4041b478
# ╠═31400c16-5add-4ff7-a43b-9d7887da026a
# ╠═2cb70fff-4509-46b2-bd1c-0d21d61961e9
# ╠═4e18cfb3-1f46-45d6-91d6-7857dd77836e
# ╠═aeb41add-1dbe-4214-9bf8-1ff807e5ce75
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
