

# Here is an example of a cool plot that can be done with Makie. Code is from https://lazarusa.github.io/BeautifulMakie/

using GLMakie

let
    x = y =  LinRange(-2, 2, 100)
    z = (-x .* exp.(-x .^ 2 .- (y') .^ 2)) .* 4
    zmin, zmax = minimum(z), maximum(z)
    cmap = :ice
    fig = Figure(resolution = (900,900))
    ax = Axis3(fig, aspect = :data, perspectiveness = 0.5, elevation = π/9,
        xzpanelcolor= (:black, 0.75), yzpanelcolor= (:black,0.75),
        zgridcolor = :grey, ygridcolor = :grey, xgridcolor = :grey)
    surface!(ax, x, y, z, colormap = cmap, colorrange = (zmin, zmax))
    xm, ym, zm = minimum(ax.finallimits[])
    contour!(ax, x, y, z, levels = 20, colormap = cmap, linewidth = 2,
        colorrange=(zmin, zmax), transformation = (:xy, zm))
    wireframe!(ax, x, y, z, overdraw = true, transparency = true,
        color = (:black, 0.1))
    fig[1,1] = ax
    fig
end