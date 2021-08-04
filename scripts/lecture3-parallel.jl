
# Start the Julia shell with : julia -p 8

using Distributed

@everywhere function compute_pi(N::Int)
    """
    Compute pi with a Monte Carlo simulation of N darts thrown in [-1,1]^2
    Returns estimate of pi
    """
    n_landed_in_circle = 0  # counts number of points that have radial coordinate < 1, i.e. in circle
    for i = 1:N
        x = rand() * 2 - 1  # uniformly distributed number on x-axis
        y = rand() * 2 - 1  # uniformly distributed number on y-axis

        r2 = x*x + y*y  # radius squared, in radial coordinates
        if r2 < 1.0
            n_landed_in_circle += 1
        end
    end

    return n_landed_in_circle / N * 4.0    
end

job = @spawn compute_pi(1000000000)
fetch(job)

function parallel_pi_computation(N::Int; ncores::Int=8)
    """
    Compute pi in parallel, over ncores cores, with a Monte Carlo simulation throwing N total darts
    """

    # compute sum of pi's estimated among all cores in parallel
    sum_of_pis = @distributed (+) for i=1:ncores
        compute_pi(ceil(Int, N / ncores))
    end

    return sum_of_pis / ncores  # average value
end

@time parallel_pi_computation(8000000000)

