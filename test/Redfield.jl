using quantumdynamics
using LinearAlgebra
using Plots
using QuadGK
using DifferentialEquations

# Organize like Pyrho
# Write Hamiltonian Struct, will store System, System Part of Sys-Bath, and Spec Dens

struct Hamiltonian
    # System Hamiltonian (matrix form)
    Hs::Matrix{ComplexF64}
    # System part of system-bath coupling operators (matrix form)
    SB::Matrix{ComplexF64}
    # Spectral density function
    J::Function
    # Eigenvalues of system Hamiltonian
    spectrum::Vector{Float64}
    # Eigenvectors of system Hamiltonian
    eigenstates::Matrix{ComplexF64}
    # System-bath coupling in eigenbasis
    SB_eigen::Matrix{ComplexF64}
    # Inner constructor to compute eigendecomposition
    function Hamiltonian(Hs::Matrix{ComplexF64}, SB::Matrix{ComplexF64}, J::Function)
        # Compute eigendecomposition
        eigen_decomp = eigen(Hermitian(Hs))
        spectrum = eigen_decomp.values
        eigenstates = eigen_decomp.vectors
        # Transform system-bath coupling to eigenbasis
        SB_eigen = eigenstates' * SB * eigenstates
        new(Hs, SB, J, spectrum, eigenstates, SB_eigen)
    end
end

struct PauliMatrices
    X::Matrix{ComplexF64}
    Y::Matrix{ComplexF64}
    Z::Matrix{ComplexF64}
    
    # Inner constructor that creates the standard Pauli matrices
    function PauliMatrices()
        X = ComplexF64[0 1; 1 0]
        Y = ComplexF64[0 -im; im 0]
        Z = ComplexF64[1 0; 0 -1]
        new(X, Y, Z)
    end
end

s = PauliMatrices()

Tinv = 1000.0;
bias = 0.5;
tunn = 0.25*bias;
cut = 5.0;
specparam_ohm = SpectralParams([0.25*cut/2.0], [cut], [0.0], [0.0], [0.0])

H = bias*s.Z + tunn*s.X;
SB = s.X;
J(ω) = total_spectral_density(specparam_ohm, ω);
Ham = Hamiltonian(H, SB, J)

# Define an abstract type that our specific dynamics types will inherit from
abstract type DynamicsType end
# Define two concrete types for different dynamics
# The "<:" means "is a subtype of"
struct Markovian <: DynamicsType end
struct NonMarkovian <: DynamicsType end

# T<:DynamicsType means this struct is parameterized by a type T 
# that must be a subtype of DynamicsType
struct RedfieldTensor{T<:DynamicsType}
    frequencies::Matrix{Float64}
    dynamics_type::T
    secular::Bool
    
    corr::Union{Nothing, Matrix{ComplexF64}}
    tensor::Union{Nothing, Array{ComplexF64, 4}}
    
    corr_t::Union{Nothing, Function}
end

function build_redfield_tensor!(tensor, ham::Hamiltonian, frequencies, g, secular::Bool)
    N = size(ham.Hs, 1)
    S = ham.SB_eigen
    corr = g
    
    fill!(tensor, zero(ComplexF64))
    
    for i in 1:N, j in 1:N, k in 1:N, l in 1:N
        # Check secular approximation if enabled
        if !secular || abs(frequencies[i,j] - frequencies[k,l]) < 1e-6
            # Both Γ terms
            tensor[i,j,k,l] = S[l,j]*S[i,k]*corr[i,k] + 
                             S[l,j]*S[i,k]*conj(corr[j,l])
            
            # Both sum terms
            l == j && (tensor[i,j,k,l] -= sum(S[i,r]*S[r,k]*corr[r,k] for r in 1:N))
            i == k && (tensor[i,j,k,l] -= sum(S[l,r]*S[r,j]*conj(corr[r,l]) for r in 1:N))
        end
    end
end

# Outer Constructor for Markovian case
function RedfieldTensor(ham::Hamiltonian, ::Markovian; secular::Bool=false)
    N = size(ham.Hs, 1)
    frequencies = zeros(N, N)
    tensor = zeros(ComplexF64, N, N, N, N)
    corr = zeros(ComplexF64, N, N)
    
    # Compute frequencies and correlation function
    for a in 1:N, b in 1:N
        frequencies[a,b] = ham.spectrum[a] - ham.spectrum[b]
        corr[a,b] = principal_value(frequencies[a,b], Tinv, Inf, ham.J)
    end
    
    build_redfield_tensor!(tensor, ham, frequencies, corr, secular)
    
    return RedfieldTensor{Markovian}(
        frequencies,
        Markovian(),
        secular,
        corr,
        tensor,
        nothing
    )
end

RT = RedfieldTensor(Ham, Markovian(), secular=true)

# Outer Constructor for Non-Markovian case
function RedfieldTensor(ham::Hamiltonian, ::NonMarkovian)
    N = size(ham.Hs, 1)
    frequencies = zeros(N, N)
    
    # Compute frequencies
    for a in 1:N, b in 1:N
        frequencies[a,b] = ham.spectrum[a] - ham.spectrum[b]
    end
    
    # Define time-dependent correlation function
    function corr_t(t::Float64, ω::Float64)
        # Return complex number: γ(t,ω) + i*S(t,ω)
        # Implementation depends on your specific non-Markovian theory
    end
    
    return RedfieldTensor{NonMarkovian}(
        frequencies,
        NonMarkovian(),
        nothing,
        nothing,
        corr_t
    )
end

# Method for Markovian case - uses pre-computed tensor
function apply_tensor(R::RedfieldTensor{Markovian}, ρ::Matrix{ComplexF64})
    N = size(ρ, 1)
    result = zeros(ComplexF64, N, N)
    
    # First add the coherent evolution term -i[H,ρ]
    # Note: in eigenbasis, H is diagonal with eigenvalues
    for a in 1:N, b in 1:N
        result[a,b] -= im * (R.frequencies[a,b]) * ρ[a,b]
    end
    
    # Add the dissipative Redfield tensor terms
    for a in 1:N, b in 1:N
        for c in 1:N, d in 1:N
            result[a,b] += R.tensor[a,b,c,d] * ρ[c,d]
        end
    end
    
    return result
end

# Method for NonMarkovian case - computes tensor elements as needed
function apply_tensor(R::RedfieldTensor{NonMarkovian}, 
                     ρ::Matrix{ComplexF64}, 
                     t::Float64)
    N = size(ρ, 1)
    result = zeros(ComplexF64, N, N)
    
    # Compute tensor elements using time-dependent correlations
    for a in 1:N, b in 1:N
        for c in 1:N, d in 1:N
            # Compute tensor element at time t
            tensor_element = compute_tensor_element(R, a, b, c, d, t)
            result[a,b] += tensor_element * ρ[c,d]
        end
    end
    return result
end

# Modified propagation code
struct RedfieldDynamics{T<:DynamicsType}
    ham::Hamiltonian
    redfield::RedfieldTensor{T}
    rho0::Matrix{ComplexF64}
    times::Vector{Float64}
end

function propagate!(dynamics::RedfieldDynamics{Markovian})
    N = size(dynamics.ham.Hs, 1)
    tspan = (dynamics.times[1], dynamics.times[end])
    
    # Transform initial density matrix to eigenstate basis
    ρ0_eigen = dynamics.ham.eigenstates' * dynamics.rho0 * dynamics.ham.eigenstates
    println("\nInitial density matrix in eigenstate basis:")
    display(ρ0_eigen)
    
    println("\nRedfield tensor elements:")
    for a in 1:N, b in 1:N, c in 1:N, d in 1:N
        if abs(dynamics.redfield.tensor[a,b,c,d]) > 1e-10
            println("R[$a,$b,$c,$d] = $(dynamics.redfield.tensor[a,b,c,d])")
        end
    end
    
    # Define the ODE function in the eigenstate basis
    function redfield_ode!(dr, r, p, t)
        ρ = Matrix(reshape(reinterpret(ComplexF64, r), (N, N)))
        
        # Check if t is close to any of the first three requested times
        if any(abs(t - requested_t) < 1e-10 for requested_t in dynamics.times[1:3])
            println("\nAt time t = $t:")
            println("Current density matrix in eigenstate basis:")
            display(ρ)
            
            dρ = apply_tensor(dynamics.redfield, ρ)
            println("Derivative (dρ/dt):")
            display(dρ)
        end
        
        dρ = apply_tensor(dynamics.redfield, ρ)
        dr .= reinterpret(Float64, vec(dρ))
    end
    
    # Convert initial eigenstate density matrix to vector form
    r0 = reinterpret(Float64, vec(ρ0_eigen))
    
    # Solve the ODE
    prob = ODEProblem(redfield_ode!, r0, tspan)
    sol = solve(prob, Tsit5(), saveat=dynamics.times)
    
    # Convert solution back to density matrices and transform back to site basis
    results = Vector{Matrix{ComplexF64}}(undef, length(dynamics.times))
    for (i, r) in enumerate(sol.u)
        # First reshape to density matrix in eigenstate basis
        ρ_eigen = reshape(reinterpret(ComplexF64, r), (N, N))
        # Transform back to site basis
        results[i] = dynamics.ham.eigenstates * ρ_eigen * dynamics.ham.eigenstates'
    end
    
    return results
end



function propagate!(dynamics::RedfieldDynamics{NonMarkovian})
    N = size(dynamics.ham.Hs, 1)
    results = Vector{Matrix{ComplexF64}}(undef, length(dynamics.times))
    
    rho = copy(dynamics.rho0)
    results[1] = copy(rho)
    
    for (i, t) in enumerate(dynamics.times[2:end])
        # Compute tensor action at current time t
        drho = apply_tensor(dynamics.redfield, rho, t)
        # Update rho using computed derivatives
        # ...
        results[i+1] = copy(rho)
    end
    
    return results
end

# Set up initial state (for example, spin up)
ρ0 = ComplexF64[1.0 0.0; 0.0 0.0]
# Set up time grid (adjust range and number of points as needed)
times = collect(range(0.0, 15.0, length=150))
# Create dynamics object
dynamics = RedfieldDynamics{Markovian}(Ham, RT, ρ0, times)
# Propagate
results = propagate!(dynamics)
# Plot real parts
p1 = plot(times, [real(result[1,1]) for result in results], 
    label="ρ11", 
    xlabel="Time", 
    ylabel="Real Part",
    title="Population Dynamics")
plot!(times, [real(result[2,2]) for result in results], label="ρ22")

# Plot imaginary parts of coherences and <SB>
p2 = plot(times, [imag(result[1,2]) for result in results], 
    label="Im[ρ12]", 
    xlabel="Time", 
    ylabel="Imaginary Part",
    title="Coherence Dynamics")
    plot!(times, [real(result[1,2]) for result in results], label="Re[ρ12]")
plot!(times, [real(tr(result * SB)) for result in results], 
    label="⟨SB⟩", linestyle=:dash)

# Display plots side by side
plot(p1, p2, layout=(2,1), size=(800,600))