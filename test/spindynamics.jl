using Plots
using LinearAlgebra
using DifferentialEquations

# System parameters
struct SpinSystemParams
    S::Int
    v::Float64
    J::Float64
    B::Float64
    K::Float64
    delta::Float64
end

function default_params()
    S = 25
    v = 1.0
    J = 1.0
    B = 1.0
    K = 0.5*B*J
    delta = 0.0
    return SpinSystemParams(S, v, J, B, K, delta)
end

# Core transition rate functions
function transition_rates(params::SpinSystemParams)
    f(s) = params.delta + (params.K/params.S)*s
    pup(s) = params.v*exp(f(s))
    pdown(s) = params.v*exp(-f(s))
    wup(s) = (params.S-s)*pup(s)
    wdown(s) = (params.S+s)*pdown(s)
    return wup, wdown
end

# Matrix construction
function build_transition_matrix(params::SpinSystemParams)
    wup, wdown = transition_rates(params)
    return Tridiagonal(
        [wdown(s) for s in -params.S+1:params.S],
        [-(wup(s) + wdown(s)) for s in -params.S:params.S],
        [wup(s) for s in -params.S:params.S-1]
    )
end

# Stationary distribution calculation
function calculate_stationary_distribution(W)
    eigvals, eigvecs = eigen(Matrix(W)')
    zero_indices = findall(x -> abs(x) < 1e-10, eigvals)
    
    stationary_dist = real.(eigvecs[:, zero_indices[1]])
    stationary_dist = stationary_dist / sum(stationary_dist)
    if any(x -> x < 0, stationary_dist)
        stationary_dist = -stationary_dist
    end
    return stationary_dist
end

# Analytical solution
function analytical_dist(s, N, params::SpinSystemParams)
    n = Int(s)
    binom = binomial(2N, N+n)
    f(s) = params.delta + (params.K/params.S)*s
    exp_factor = exp(f(s)*s + params.delta*s)
    prefactor = factorial(big(N))^2 / factorial(big(2N))
    return Float64(prefactor * binom * exp_factor)
end

# Theoretical drift calculation
function theoretical_drift(s, params::SpinSystemParams)
    f(s) = params.delta + (params.K/params.S)*s
    fs = f(s)
    return 2*params.v*(sinh(fs) - (s/params.S)*cosh(fs))
end

# Plotting functions
function plot_free_energy(stationary_dist, params::SpinSystemParams)
    free_energy = -(1/params.S) .* log.(stationary_dist)
    center_index = params.S + 1
    free_energy = free_energy .- free_energy[center_index]
    
    p = plot((-params.S:params.S), free_energy, 
            label="Free Energy",
            xlabel="Spin", 
            ylabel="Free Energy", 
            marker=:circle)
    savefig(p, "free_energy.png")
    return free_energy
end

function plot_drift_comparison(params::SpinSystemParams)
    wup, wdown = transition_rates(params)
    s_range = range(-params.S, params.S, length=1000)
    
    drift_actual = [wup(s) - wdown(s) for s in s_range]
    drift_theory = [params.S*theoretical_drift(s, params) for s in s_range]
    
    p = plot(s_range, drift_actual, 
            label="Numerical: wup-wdown",
            xlabel="s", 
            ylabel="Drift", 
            linewidth=2)
    plot!(p, s_range, drift_theory, 
          label="Theory: 2v[sinh(f) - (s/S)cosh(f)]",
          linestyle=:dash, 
          linewidth=2)
    savefig(p, "drift_comparison.png")
end

function plot_free_energy_comparison(B_values, params::SpinSystemParams)
    p = plot(xlabel="Spin", 
            ylabel="Free Energy", 
            title="Free Energy Landscape for Different K")
    
    for B_test in B_values
        local_params = SpinSystemParams(
            params.S, params.v, params.J, B_test, 
            0.5*B_test*params.J, params.delta
        )
        
        W = build_transition_matrix(local_params)
        stationary_dist = calculate_stationary_distribution(W)
        free_energy = -(1/local_params.S) .* log.(stationary_dist)
        center_index = local_params.S + 1
        free_energy = free_energy .- free_energy[center_index]
        
        plot!(p, -params.S:params.S, free_energy, 
              label="K = $(round(local_params.K, digits=2))", 
              linewidth=2)
    end
    savefig(p, "free_energy_comparison.png")
end

# Initial condition generators
function gaussian_initial(params::SpinSystemParams, sigma::Float64=2.0, mu::Float64=0.0)
    s_values = -params.S:params.S
    dist = @. exp(-(s_values - mu)^2 / (2sigma^2))
    return dist / sum(dist)
end

# Time evolution with DifferentialEquations.jl
function propagate_dynamics(W::Matrix, p0::Vector, times::Vector{Float64})
    # Define the master equation dp/dt = Wp
    function master_equation!(dp, p, W, t)
        dp .= W * p
    end
    
    # Create ODE problem
    prob = ODEProblem(master_equation!, p0, (times[1], times[end]), W)
    
    # Solve using an appropriate method
    # Choosing Tsit5() as it's good for non-stiff problems
    # Could use TRBDF2() for stiffer problems if needed
    sol = solve(prob, Tsit5(), saveat=times)
    
    # Convert solution to matrix form
    p_t = hcat(sol.u...)
    
    return p_t
end

# Visualization
function plot_dynamics_3D(p_t::Matrix, times::Vector{Float64}, params::SpinSystemParams)
    s_values = -params.S:params.S
    time_indices = round.(Int, range(1, length(times), length=20))
    plot_times = times[time_indices]
    
    # Calculate moments for all times
    means = Float64[]
    variances = Float64[]
    for i in 1:size(p_t, 2)
        mean_s, var_s = calculate_moments(p_t[:, i], s_values)
        push!(means, mean_s)
        push!(variances, var_s)
    end
    
    # Create color gradient from maroon to navy blue - intermediate brightness
    colors = [RGB(0.55*(1-t) + 0.125, 0.125*(1-t), 0.45*t + 0.225) for t in range(0, 1, length=length(plot_times))]
    
    p = plot(
        xlabel="Spin",
        ylabel="Time",
        zlabel="Population",
        camera=(45, -30),
        legend=:topright,
        grid=false,
        zlims=(0, maximum(p_t) * 1.1),
        margin=-5Plots.mm,
        guidefontsize=10,
        tickfontsize=8,
        title="",
        bottom_margin=-5Plots.mm,
        left_margin=-5Plots.mm,
        right_margin=-5Plots.mm,
        top_margin=-5Plots.mm
    )
    
    # Plot first time slice with label for legend
    plot!(p, s_values, fill(plot_times[1], length(s_values)), p_t[:, time_indices[1]],
          seriestype=:line,
          linewidth=2,
          linecolor=colors[1],
          fillrange=fill(minimum(p_t), length(s_values)),
          fillalpha=0.1,
          fillcolor=colors[1],
          label="Sol")
    
    # Plot remaining time slices without labels
    for (i, t) in enumerate(plot_times[2:end])
        plot!(p, s_values, fill(t, length(s_values)), p_t[:, time_indices[i+1]],
              seriestype=:line,
              linewidth=2,
              linecolor=colors[i+1],
              fillrange=fill(minimum(p_t), length(s_values)),
              fillalpha=0.1,
              fillcolor=colors[i+1],
              label=false)
    end
    
    # Add mean trajectory
    plot!(p, means, times, zeros(length(times)),
          linewidth=2,
          linecolor=:black,
          label="Mean")
    
    # Add variance bands (mean ± sqrt(variance))
    plot!(p, means .+ sqrt.(variances), times, zeros(length(times)),
          linewidth=2,
          linecolor=:black,
          linestyle=:dash,
          label="±σ")
    plot!(p, means .- sqrt.(variances), times, zeros(length(times)),
          linewidth=2,
          linecolor=:black,
          linestyle=:dash,
          label=false)
    
    savefig(p, "dynamics_3D.png")
    return p
end

# Add these functions before the run_dynamics_example function:

function calculate_moments(p::Vector, s_values::AbstractRange)
    mean_s = sum(s_values .* p)
    variance_s = sum((s_values .- mean_s).^2 .* p)
    return mean_s, variance_s
end

function plot_moments(times::Vector{Float64}, p_t::Matrix, params::SpinSystemParams)
    s_values = -params.S:params.S
    means = Float64[]
    variances = Float64[]
    
    for i in 1:size(p_t, 2)
        mean_s, var_s = calculate_moments(p_t[:, i], s_values)
        push!(means, mean_s)
        push!(variances, var_s)
    end
    
    p1 = plot(times, means,
        label="Mean",
        xlabel="Time",
        ylabel="⟨s⟩",
        linewidth=2)
    
    p2 = plot(times, variances,
        label="Variance",
        xlabel="Time",
        ylabel="⟨(s-⟨s⟩)²⟩",
        linewidth=2)
    
    p_combined = plot(p1, p2, layout=(2,1))
    savefig(p_combined, "moments.png")
    return means, variances
end

function build_vandermonde(s_values::AbstractRange, max_order::Int)
    # Normalize s_values to [-1, 1] range
    s_norm = s_values ./ maximum(abs.(s_values))
    
    # Create Vandermonde matrix for moment transformation
    # Note: We want moments from 0 to max_order-1 to get max_order total moments
    V = zeros(max_order, length(s_values))  # Changed from max_order + 1
    for i in 0:max_order-1                  # Changed from max_order
        for j in 1:length(s_values)
            V[i+1, j] = s_norm[j]^i
        end
    end
    return V
end

function truncation_closure(W::Matrix, max_order::Int, params::SpinSystemParams)
    s_values = -params.S:params.S
    V = build_vandermonde(s_values, max_order)
    V_pinv = pinv(V; rtol=1e-12)
    W_trunc = V * W * V_pinv
    # Return the full truncated matrix (removed the +1 indexing)
    return W_trunc
end

function analyze_matrix_properties(W::Matrix, name::String="Matrix")
    # Calculate eigenvalues using eigen() instead of eigvals()
    evals = eigen(W).values
    
    # Calculate condition number
    cond_num = cond(W)
    
    # Check if matrix is nearly singular
    is_singular = cond_num > 1e15
    
    # Check if any eigenvalues have large positive real parts
    max_real_part = maximum(real.(evals))
    
    # Check matrix norm
    matrix_norm = norm(W)
    
    println("\n=== $name Analysis ===")
    println("Condition number: ", cond_num)
    println("Nearly singular: ", is_singular)
    println("Maximum eigenvalue real part: ", max_real_part)
    println("Matrix norm: ", matrix_norm)
    println("Eigenvalue range: [$(minimum(abs.(evals))), $(maximum(abs.(evals)))]")
    
    # Return results as a dictionary with properly typed values
    return Dict{String, Any}(
        "condition_number" => cond_num,
        "is_singular" => is_singular,
        "max_real_part" => max_real_part,
        "matrix_norm" => matrix_norm,
        "eigenvalues" => collect(evals)  # Convert eigenvalues to a vector
    )
end

function qss_closure(W::Matrix, n_active::Int, n_total::Int, params::SpinSystemParams)
    s_values = -params.S:params.S
    V = build_vandermonde(s_values, n_total)
    
    W_moments = V * W * pinv(V)
    
    # Split into active and QSS blocks
    W11 = W_moments[1:n_active, 1:n_active]
    W12 = W_moments[1:n_active, n_active+1:end]
    W21 = W_moments[n_active+1:end, 1:n_active]
    W22 = W_moments[n_active+1:end, n_active+1:end]
    
    # Calculate effective rate matrix
    W_eff = W11 - W12 * (W22 \ W21)
    
    analyze_matrix_properties(W_eff, "W_eff")
    return W_eff
end

function mean_field_dynamics(mean0::Float64, var0::Float64, times::Vector{Float64}, params::SpinSystemParams)
    # Helper functions for transition rates
    f(s) = params.delta + (params.K/params.S)*s
    pup(s) = params.v*exp(f(s))
    pdown(s) = params.v*exp(-f(s))
    wup(s) = (params.S-s)*pup(s)
    wdown(s) = (params.S+s)*pdown(s)
    
    # K(<n>) = wup - wdown
    K(n) = wup(n) - wdown(n)
    
    # Q(<n>) = wup + wdown
    Q(n) = wup(n) + wdown(n)
    
    # K'(<n>) = d/dn(wup - wdown)
    function Kprime(n)
        h = 1e-6  # Small step for numerical derivative
        return (K(n + h) - K(n)) / h
    end
    
    # Define the coupled ODEs
    function mean_field_eom!(du, u, p, t)
        mean_n, var_n = u
        
        # d<n>/dt = K(<n>)
        du[1] = K(mean_n)
        
        # dvar/dt = K'(<n>)var + Q(<n>)
        du[2] = 2*Kprime(mean_n)*var_n + Q(mean_n)
    end
    
    # Initial conditions and solve
    u0 = [mean0, var0]
    prob = ODEProblem(mean_field_eom!, u0, (times[1], times[end]))
    sol = solve(prob, Tsit5(), saveat=times)
    
    # Extract solutions
    means = [s[1] for s in sol.u]
    variances = [s[2] for s in sol.u]
    
    return means, variances
end

function compare_closures(p0::Vector, times::Vector{Float64}, W::Matrix, p_t::Matrix, params::SpinSystemParams)
    # Reduce number of active moments for better stability
    n_active = 16  # Reduced from 10
    n_trunc = 16   # Reduced from 16
    n_total = 17  # Reduced from 51
    
    s_values = -params.S:params.S
    V_trunc = build_vandermonde(s_values, n_trunc)
    #V_qss = build_vandermonde(s_values, n_active)
    m0_trunc = V_trunc * p0
    #m0_qss = V_qss * p0

    # Truncation closure (using regular propagator)
    W_trunc = truncation_closure(W, n_trunc, params)
    trunc_sol = propagate_dynamics(W_trunc, m0_trunc, times)

    # QSS closure (using specialized propagator)
    #W_qss = qss_closure(W, n_active, n_total, params)
    #qss_sol = propagate_dynamics(W_qss, m0_qss, times)

    # Calculate exact moments
    exact_means = Float64[]
    exact_variances = Float64[]
    for i in 1:size(p_t, 2)
        mean_s, var_s = calculate_moments(p_t[:, i], s_values)
        push!(exact_means, mean_s)
        push!(exact_variances, var_s)
    end
    
    # Calculate initial mean and variance for mean-field
    s_values = -params.S:params.S
    mean0, var0 = calculate_moments(p0, s_values)
    
    # Get mean-field solutions
    mf_means, mf_vars = mean_field_dynamics(mean0, var0, times, params)
    
    # Calculate scaling factor based on system size
    scaling = params.S  # Scale by system size
    
    # Update plotting to include mean-field results
    p1 = plot(times, exact_means, label="Full", xlabel="Time", ylabel="Mean")
    plot!(p1, times, scaling .* trunc_sol[2, :], label="Truncation")
    plot!(p1, times, mf_means, label="Mean Field")
    
    p2 = plot(times, exact_variances, label="Full", xlabel="Time", ylabel="Second Moment")
    plot!(p2, times, scaling^2 .* (trunc_sol[3, :] .- trunc_sol[2, :].^2), label="Truncation")
    plot!(p2, times, mf_vars, label="Mean Field")
    
    p = plot(p1, p2, layout=(2,1))
    savefig(p, "closure_comparison.png")
    return p
end

function analyze_moment_timescales(W::Matrix, n_active::Int, n_total::Int, params::SpinSystemParams)
    s_values = -params.S:params.S
    V = build_vandermonde(s_values, n_total)
    
    # Transform to moment space
    V_scaled = V ./ sqrt.(sum(abs2, V, dims=2))
    W_moments = V_scaled * W * pinv(V_scaled; rtol=1e-6)
    
    # Get blocks
    W11 = W_moments[1:n_active, 1:n_active]
    W22 = W_moments[n_active+1:end, n_active+1:end]
    
    # Calculate eigenvalues for both blocks
    evals_active = eigvals(W11)
    evals_qss = eigvals(W22)
    
    # Plot eigenvalue spectrum
    p = scatter(real.(evals_active), imag.(evals_active),
                label="Active moments",
                xlabel="Re(λ)",
                ylabel="Im(λ)",
                title="Eigenvalue Spectrum of Moment Blocks")
    scatter!(real.(evals_qss), imag.(evals_qss),
             label="QSS moments")
    
    # Calculate and print timescales
    τ_active = -1 ./ real.(evals_active)
    τ_qss = -1 ./ real.(evals_qss)
    
    println("\nTimescales (1/|Re(λ)|):")
    println("Active moments: ", sort(filter(isfinite, abs.(τ_active))))
    println("QSS moments: ", sort(filter(isfinite, abs.(τ_qss))))
    
    savefig(p, "moment_timescales.png")
    return p
end

# Modify run_dynamics_example to include moment analysis:
function run_dynamics_example()
    params = default_params()
    W = Matrix(build_transition_matrix(params)')
    
    p0 = gaussian_initial(params, 1.0, -5.0)
    times = range(0.0, 50.0, length=100)
    
    p_t = propagate_dynamics(W, p0, collect(times))
    plot_dynamics_3D(p_t, collect(times), params)
    
    # Pass p_t to compare_closures
    compare_closures(p0, collect(times), W, p_t, params)
    
end

# Main execution
function main()
    params = default_params()
    
    # Build transition matrix - Convert to Matrix when needed
    W = build_transition_matrix(params)
    
    # Calculate stationary distribution
    stationary_dist = calculate_stationary_distribution(W)
    
    # Generate plots
    #plot_drift_comparison(params)
    
    # Compare with analytical solution
    s_values = -params.S:params.S
    analytical = [analytical_dist(s, params.S, params) for s in s_values]
    analytical = analytical / sum(analytical)
    
    # Multiple B values comparison
    #B_values = [0.67, 0.83, 1.0, 1.17, 1.33]
    #plot_free_energy_comparison(B_values, params)
    
    # Run dynamics example
    run_dynamics_example()
end

# Run the analysis
main()
