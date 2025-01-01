module quantumdynamics

using Plots
using QuadGK
using SpecialFunctions
using DifferentialEquations

export Glf, Ghf, total_spectral_density, plot_spectral_density, SpectralParams, spectral_moment, principal_value

Glf(L,O,x) = (L*x/O)^1*exp(-x/O)
#Ghf(L,S,W,x) = (L*x)*(((x-S)^2 + (W)^2)^(-1))*((2*W)/(pi+2*atan(S/W)))
Ghf(L,S,W,x) = (L*x)*exp(-(((x-S)/W)^2)/2)*(W*sqrt(pi/2)*(erf(S/(sqrt(2)*W))+1))^(-1)

H = 1.239*10^(-4) #\hbar 2 pi c [=] eV cm

struct SpectralParams
    # Low frequency parameters
    Llf::Vector{Float64}  # Coupling strengths
    Olf::Vector{Float64}  # Frequency cutoffs

    # High frequency parameters  
    Lhf::Vector{Float64}  # Coupling strengths
    Shf::Vector{Float64}  # Peak positions
    Whf::Vector{Float64}  # Peak widths
end

function total_spectral_density(params::SpectralParams, x)
    # Sum up low frequency contributions
    lf_sum = sum(Glf(params.Llf[i], params.Olf[i], x) 
                 for i in 1:length(params.Llf))
                 
    # Return early if no high frequency contributions
    if all(x -> x == 0.0, params.Lhf)
        return lf_sum
    end
    
    # Sum up high frequency contributions
    hf_sum = sum(Ghf(params.Lhf[i], params.Shf[i], params.Whf[i], x)
                 for i in 1:length(params.Lhf))
    
    return lf_sum + (1-exp(-x^2/params.Whf[2]))*hf_sum
end

function plot_spectral_density(params1::SpectralParams, x_range, params2::Union{SpectralParams,Nothing}=nothing)
    fig = plot(xlabel="Frequency (cm⁻¹)", ylabel="J(ω) (cm⁻¹)", 
               title="Total Spectral Density")
    
    y_values1 = [total_spectral_density(params1, x) for x in x_range]
    plot!(fig, x_range, y_values1, label="Spectral Density 1")
    
    if params2 !== nothing
        y_values2 = [total_spectral_density(params2, x) for x in x_range]
        plot!(fig, x_range, y_values2, label="Spectral Density 2")
    end
    
    return fig
end

function spectral_moment(params::SpectralParams,n)
    return quadgk(x -> total_spectral_density(params, x)*x^n, 0, Inf)[1]
end

function principal_value(w_0::Float64, B::Float64, t::Union{Float64,Type{Inf}}, J::Function)
    # Define antisymmetrized spectral density
    J_anti(w) = sign(w) * J(abs(w))
    
    # Define integrand with principal value consideration
    function base_integrand(w)
        if abs(w - w_0) < 1e-10  # Avoid exact singularity
            return 0.0
        end
        return 0.5 * J_anti(w) * (coth(w*B/2) - 1) / (w - w_0)
    end
    
    if t === Inf
        # Handle zero frequency case using the limit
        if abs(w_0) < 1e-10
            # Use finite difference to approximate J'(0)
            h = 1e-6
            J_prime_0 = (J(h) - J(-h)) / (2h)
            result1 = π/B * J_prime_0
        else
            result1 = pi/2 * sign(w_0) * J(abs(w_0)) * (coth(B*w_0/2) - 1)
        end
        
        epsilon = 1e-3
        result2 = quadgk(base_integrand, -100.0, w_0-epsilon, w_0+epsilon, 100.0, 
                     rtol=1e-4, atol=1e-4)[1]
        return result1 + im*result2
    else
        # First integral with im*(1-cos((w-w_0)t))
        function integrand1(w)
            return base_integrand(w) * im * (1 - cos((w-w_0)*t))
        end
        
        # Second integral with -sin((w-w_0)t)
        function integrand2(w)
            return base_integrand(w) * (sin((w-w_0)*t))
        end
        
        epsilon = 1e-3
        result1 = quadgk(integrand1, -100.0, w_0-epsilon, w_0+epsilon, 100.0, 
                        rtol=1e-4, atol=1e-4)[1]
        result2 = quadgk(integrand2, -100.0, w_0-epsilon, w_0+epsilon, 100.0, 
                        rtol=1e-4, atol=1e-4)[1]
        
        return result1 + result2
    end
end

end