using quantumdynamics
using QuadGK
using Plots

Glf(L,O,x) = (L*x/O)^3*exp(-x/O)
Ghf(L,S,W,x) = (L*x)*(((x-S)^2 + (W)^2)^(-1))*((2*W)/(pi+2*atan(S/W)))

H = 1.239*10^(-4) #\hbar 2 pi c [=] eV cm

function adrenorm(D,L,S,W)
    p = 1
    adD = D
    adD0 = 0

    while abs(adD - adD0) > 1e-8
        adD0 = adD
        integral = getfield(quadgk(x -> Ghf(L,S,W,x)/x^2, p*adD0, Inf, rtol=1e-8),1)
        println(integral)
        adD = D*exp(-(1/(2*H))*integral)
        println(adD)
    end

    return adD

end

struct SpectralParams
    # Low frequency parameters
    Llf::Vector{Float64}  # Coupling strengths
    Olf::Vector{Float64}  # Frequency cutoffs

    # High frequency parameters  
    Lhf::Vector{Float64}  # Coupling strengths
    Shf::Vector{Float64}  # Peak positions
    Whf::Vector{Float64}  # Peak widths
end

specparam = SpectralParams([0.68], [21], [0.42,0.30,0.42], [150,200,311], [15,13,22])

function total_spectral_density(params::SpectralParams, x)
    # Sum up low frequency contributions
    lf_sum = sum(Glf(params.Llf[i], params.Olf[i], x) 
                 for i in 1:length(params.Llf))
    
    # Sum up high frequency contributions
    hf_sum = sum(Ghf(params.Lhf[i], params.Shf[i], params.Whf[i], x)
                 for i in 1:length(params.Lhf))
    
    return lf_sum + (1-exp(-x/params.Olf[1]))*hf_sum
end

function plot_spectral_density(params::SpectralParams, x_range)
    fig = plot(xlabel="Frequency (cm⁻¹)", ylabel="J(ω) (cm⁻¹)", 
               title="Total Spectral Density")
    
    y_values = [total_spectral_density(params, x) for x in x_range]
    plot!(fig, x_range, y_values)
    
    return fig
end

display(plot_spectral_density(specparam, 0:1000))

function poltransf(D,specparam::SpectralParams)
 
    integral = getfield(quadgk(x -> coth(x/400)*total_spectral_density(specparam,x)/x^2, 0, Inf, rtol=1e-8),1)
    println(integral)
    return D*exp(-(1/(2*H))*integral)

end

# println(poltransf(400,specparam))

specparamhf = SpectralParams([0], [21], [0.42,0.30,0.42], [150,200,311], [15,13,22])

function splittest(specparam::SpectralParams)

    integral = getfield(quadgk(x -> (1-exp(-x/specparam.Olf[1]))*total_spectral_density(specparam,x)/x, 0, Inf, rtol=1e-8),1)
    return integral
    
end

println(splittest(specparamhf))