using quantumdynamics
using Plots
using QuadGK
using HCubature

struct VPolHamiltonian
    # Scaled coupling D*K
    DK::Float64
    # Spectral density function
    J::Function
    # System Hamiltonian in eigenbasis (already diagonal)
    Hs::Matrix{ComplexF64}
    # System-bath coupling operators in eigenbasis
    SBx::Matrix{ComplexF64}  # becomes σz in eigenbasis
    SBy::Matrix{ComplexF64}  # becomes -σx in eigenbasis
    SBz::Matrix{ComplexF64}  # becomes σy in eigenbasis
    # Eigenvalues of system Hamiltonian
    spectrum::Vector{Float64}

    # Inner constructor
    function VPolHamiltonian(DK::Float64, J::Function)
        # Define matrices directly in eigenbasis
        Hs = DK * ComplexF64[1 0; 0 -1]  # diagonalized σx
        SBx = ComplexF64[1 0; 0 -1]  # σz
        SBy = ComplexF64[0 -1; -1 0]  # -σx
        SBz = ComplexF64[0 -im; im 0]  # σy
        spectrum = [DK, -DK]  # eigenvalues
        
        new(DK, J, Hs, SBx, SBy, SBz, spectrum)
    end
end

struct SysParams
    D::Float64  # coupling strength
    B::Float64  # inverse temperature
    L::Float64  # intensity
    O::Float64  # cutoff frequency
end

function superohmic_J(w, params::SysParams)
    return (params.L * w / params.O)^3 * exp(-w / params.O)
end

function F(w, K, params::SysParams)
    B = params.B
    D = params.D

    f = 1 + 2*D*K*coth(B*w/2)*tanh(B*D*K)/w
    return f^(-1) 
end

function Kappa(k, params::SysParams)
    B = params.B

    DebyeWaller = quadgk(w -> F(w, k, params)^2*coth(B*w/2)*J(w)/w^2, 0, Inf, rtol=1e-8)[1]

    return exp(-0.5*DebyeWaller)
end

function variation(params::SysParams, tol=1e-8, max_iter=1000)
    K_prev = 1.0
    K = 0.0
    iter = 0
    
    while abs(K - K_prev) > tol && iter < max_iter
        K_prev = K
        K = Kappa(K_prev, params)
        iter += 1
    end

    if iter == max_iter
        @warn "Maximum iterations reached without convergence"
    end

    return K
end

function plot_transformed_density(params::SysParams)
    # Get K value for current parameters
    K = variation(params)
    println("K = $K")
    
    # Create plot with 2 subplots
    fig = plot(layout=(2,1), size=(800,600))
    
    # Plot range
    w_range = 0:0.1:10.0
    
    # Plot F(ω) for current parameters
    plot!(fig[1], w_range, w -> F(w, K, params), 
          label="F(ω) at B=$(params.B)", 
          title="Polaron Transform", 
          xlabel="Frequency", 
          ylabel="F(ω)",
          linewidth=2)
    
    # Plot spectral densities
    plot!(fig[2], w_range, w -> J(w), 
          label="J(ω)", 
          title="Spectral Density", 
          xlabel="Frequency",
          ylabel="Magnitude",
          linewidth=2)
    
    plot!(fig[2], w_range, w -> J(w) * F(w,K,params)^2, 
          label="J(ω)F(ω)² at B=$(params.B)", 
          linewidth=2, 
          linestyle=:dash)
    
    display(fig)
end

# Replace xx_integrand and yy_integrand with a single function
function phi_integrand(w::Float64, t::Float64, K::Float64, params::SysParams)
    B = params.B
    return (J(w)*(F(w, K, params)/w)^2) * 
           (coth(B*w/2)*(cos(w*t)-1) - im*sin(w*t))
end

function xx_correlation(t::Float64, K::Float64, params::SysParams)
    D = params.D
    phi = quadgk(w -> phi_integrand(w, t, K, params), 0, 100.0, rtol=1e-6)[1]
    return (D^2/2) * (exp(phi) + exp(-phi) - 2*K^2)
end

function yy_correlation(t::Float64, K::Float64, params::SysParams)
    D = params.D
    phi = quadgk(w -> phi_integrand(w, t, K, params), 0, 100.0, rtol=1e-6)[1]
    return (D^2/2) * (exp(phi) - exp(-phi))
end

# Pre-define 2D integrand functions with explicit types
function zz_integrand(x::AbstractVector{Float64}, f::Float64, K::Float64, params::SysParams)
    t, omega = x[1], x[2]
    B = params.B
    corr_term = J(omega) * (1-F(omega, K, params)^2) * 
                (coth(omega*B/2)*cos(omega*t) - im*sin(omega*t))
    return corr_term * exp(-im*f*t)
end

function yz_integrand(x::AbstractVector{Float64}, f::Float64, K::Float64, params::SysParams)
    t, omega = x[1], x[2]
    B = params.B
    corr_term = J(omega) * (F(omega, K, params)/omega)*(1-F(omega, K, params)) * 
                (coth(omega*B/2)*sin(omega*t) + im*cos(omega*t))
    return corr_term * exp(-im*f*t)
end

# Force compilation by calling with dummy values
function precompile_integrands()
    dummy_params = SysParams(1.0, 1.0, 1.0, 1.0)
    dummy_x = [1.0, 1.0]
    
    # Precompile basic integrands
    phi_integrand(1.0, 1.0, 1.0, dummy_params)  # Replace xx and yy with phi
    zz_integrand(dummy_x, 1.0, 1.0, dummy_params)
    yz_integrand(dummy_x, 1.0, 1.0, dummy_params)
    
    # Precompile correlations
    xx_correlation(1.0, 1.0, dummy_params)
    yy_correlation(1.0, 1.0, dummy_params)
    
    # Precompile theta calculations
    for a in 1:3, b in 1:3
        theta(0.0, a, b, 1.0, dummy_params)
    end
end

function theta(f::Float64, a::Int, b::Int, K::Float64, params::SysParams)
    # Define fixed integration bounds
    t_max = 20.0
    ω_max = 100.0
    lower = [0.0, 0.0]
    upper = [t_max, ω_max]

    if (a == 1 && b == 1)  # xx term
        return quadgk(t -> xx_correlation(t, K, params)*exp(-im*f*t), 
                     0, t_max, rtol=1e-6)[1]
    elseif (a == 2 && b == 2)  # yy term
        return quadgk(t -> yy_correlation(t, K, params)*exp(-im*f*t), 
                     0, t_max, rtol=1e-6)[1]
    elseif (a == 3 && b == 3)  # zz term
        result, _ = hcubature(x -> zz_integrand(x, f, K, params), 
                            lower, upper; rtol=1e-6, atol=1e-6)
        return result

    elseif (a == 2 && b == 3)  # yz term
        result, _ = hcubature(x -> yz_integrand(x, f, K, params), 
                            lower, upper; rtol=1e-6, atol=1e-6)
        return result

    elseif (a == 3 && b == 2)  # zy term
        return -theta(f, 2, 3, K, params)

    else
        return 0.0
    end
end

function check_convergence(f::Float64, a::Int, b::Int, K::Float64, params::SysParams)
    # Standard bounds
    t_max = 5.0
    ω_max = 100.0
    
    # Test temporal bound convergence for xx and yy
    if (a == 1 && b == 1) || (a == 2 && b == 2)
        # Compare with extended t bound
        standard = theta(f, a, b, K, params)
        extended_t = quadgk(t -> (a == 1 ? xx_correlation(t, K, params) : yy_correlation(t, K, params))*exp(-im*f*t), 
                          0, 2*t_max, rtol=1e-6)[1]
        
        t_error = abs((extended_t - standard)/standard)
        if t_error > 1e-4
            @warn "Time bound may not be converged for θ_{$(a)$(b)}(Ω=$(f))" relative_error=t_error
        end
    
    # Test 2D bound convergence for yz and zz
    elseif (a == 2 && b == 3) || (a == 3 && b == 3)
        lower = [0.0, 0.0]
        upper = [t_max, ω_max]
        standard, _ = hcubature(x -> (a == 2 ? yz_integrand(x, f, K, params) : zz_integrand(x, f, K, params)), 
                              lower, upper; rtol=1e-6)
        
        # Test temporal bound
        upper_t = [2*t_max, ω_max]
        extended_t, _ = hcubature(x -> (a == 2 ? yz_integrand(x, f, K, params) : zz_integrand(x, f, K, params)), 
                                lower, upper_t; rtol=1e-6)
        t_error = abs((extended_t - standard)/standard)
        
        # Test frequency bound
        upper_ω = [t_max, 2*ω_max]
        extended_ω, _ = hcubature(x -> (a == 2 ? yz_integrand(x, f, K, params) : zz_integrand(x, f, K, params)), 
                                lower, upper_ω; rtol=1e-6)
        ω_error = abs((extended_ω - standard)/standard)
        
        if t_error > 1e-4
            @warn "Time bound may not be converged for θ_{$(a)$(b)}(Ω=$(f))" relative_error=t_error
        end
        if ω_error > 1e-4
            @warn "Frequency bound may not be converged for θ_{$(a)$(b)}(Ω=$(f))" relative_error=ω_error
        end
    end
end

function construct_theta_cache(Ham::VPolHamiltonian, K::Float64, params::SysParams; check_bounds::Bool=false)
    theta_cache = Dict{Float64, Dict{Tuple{Int,Int}, ComplexF64}}()
    frequencies = unique([Ham.spectrum[i] - Ham.spectrum[j] for i in 1:2, j in 1:2])
    nonzero_pairs = [(1,1), (2,2), (2,3), (3,2), (3,3)]
    
    for Ω in frequencies
        theta_cache[Ω] = Dict{Tuple{Int,Int}, ComplexF64}()
        
        for (a,b) in nonzero_pairs
            if haskey(theta_cache[Ω], (a,b))
                continue
            elseif a == 3 && b == 2 && haskey(theta_cache[Ω], (2,3))
                theta_cache[Ω][(3,2)] = -theta_cache[Ω][(2,3)]
            else
                theta_cache[Ω][(a,b)] = theta(Ω, a, b, K, params)
                
                # Check convergence if requested
                if check_bounds && (b != 2 || a != 3)  # Skip zy as it uses yz
                    check_convergence(Ω, a, b, K, params)
                end
            end
        end
    end
    
    return theta_cache
end

function construct_redfield_tensor(Ham::VPolHamiltonian, theta_cache::Dict{Float64, Dict{Tuple{Int,Int}, ComplexF64}}; secular::Bool=false)
    N = 2  # 2-level system
    frequencies = zeros(N, N)
    tensor = zeros(ComplexF64, N, N, N, N)
    
    # Helper function to safely get theta value
    function get_theta(freq, a, b)
        haskey(theta_cache, freq) || return 0.0im
        haskey(theta_cache[freq], (a,b)) || return 0.0im
        return theta_cache[freq][(a,b)]
    end
    
    # Compute frequencies
    for a in 1:N, b in 1:N
        frequencies[a,b] = Ham.spectrum[a] - Ham.spectrum[b]
    end
    
    # Build Redfield tensor
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        # Check secular approximation if enabled
        if !secular || abs(frequencies[i,j] - frequencies[k,l]) < 1e-6
            # Sum over all components (x,y,z)
            for (a, SBa) in [(1, Ham.SBx), (2, Ham.SBy), (3, Ham.SBz)]
                for (b, SBb) in [(1, Ham.SBx), (2, Ham.SBy), (3, Ham.SBz)]
                    # Both Γ terms with correct frequency indices
                    tensor[i,j,k,l] += SBb[l,j] * SBa[i,k] * (
                        get_theta(frequencies[i,k], a, b) + 
                        conj(get_theta(frequencies[j,l], a, b))
                    )
                    
                    # Both sum terms with their respective r,k and r,l frequencies
                    l == j && (tensor[i,j,k,l] -= sum(
                        SBa[i,r] * SBb[r,k] * get_theta(frequencies[r,k], a, b) 
                        for r in 1:N))
                    i == k && (tensor[i,j,k,l] -= sum(
                        SBb[l,r] * SBa[r,j] * conj(get_theta(frequencies[r,l], a, b))
                        for r in 1:N))
                end
            end
        end
    end
    
    return tensor, frequencies
end

function display_theta_cache(theta_cache)
    println("Theta Cache Contents:")
    println("====================")
    
    # Sort frequencies for consistent display
    frequencies = sort(collect(keys(theta_cache)))
    
    for Ω in frequencies
        println("\nFrequency Ω = $Ω:")
        
        # Get all operator pairs for this frequency
        pairs = sort(collect(keys(theta_cache[Ω])))
        
        for (a,b) in pairs
            op_names = ["x", "y", "z"]
            θ_val = theta_cache[Ω][(a,b)]
            if abs(θ_val) > 1e-10  # Only print non-zero values
                println("θ_{$(op_names[a]),$(op_names[b])}(Ω) = $θ_val")
            end
        end
    end
end

# Create and display the cache
#= theta_cache = construct_theta_cache(Ham, K, params, check_bounds=true)
display_theta_cache(theta_cache) =#

function display_redfield_tensor(tensor; threshold=1e-10)
    println("\nRedfield Tensor Elements:")
    println("========================")
    
    N = size(tensor, 1)
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        if abs(tensor[i,j,k,l]) > threshold
            println("R[$i,$j,$k,$l] = $(tensor[i,j,k,l])")
        end
    end
end

# Create Hamiltonian and get optimal K
params = SysParams(12, 0.005, 0.98, 1.37)
J(w) = superohmic_J(w, params)

K = variation(params)
DK = params.D * K
Ham = VPolHamiltonian(DK, J)

plot_transformed_density(params)

# Call precompilation
precompile_integrands()
theta_cache = construct_theta_cache(Ham, K, params, check_bounds=true)

# Construct the tensor
tensor, frequencies = construct_redfield_tensor(Ham, theta_cache)

# Display non-zero elements
display_redfield_tensor(tensor)

# After constructing the tensor:
println("\nDiagnostic Information:")
println("=====================")
println("K value: ", K)
println("Maximum tensor magnitude: ", maximum(abs.(tensor)))
println("Minimum non-zero magnitude: ", minimum(filter(x -> abs(x) > 0, abs.(tensor))))
println("\nTensor elements above 1e-20:")
display_redfield_tensor(tensor, threshold=1e-20)

# Optional: print a few theta cache values
println("\nSample theta values:")
for (Ω, dict) in theta_cache
    for ((a,b), val) in dict
        if abs(val) > 1e-20
            println("θ_$(a)$(b)(Ω=$(Ω)) = $(val)")
        end
    end
end