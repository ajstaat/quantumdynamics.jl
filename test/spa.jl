using quantumdynamics
using QuadGK
using Plots
using SpecialFunctions

H = 1.239*10^(-4) #\hbar 2 pi c [=] eV cm

specparam1 = SpectralParams([0.68], [21], [0.42, 0.30, 0.42], [150,200,311], [15,13,22])
specparam_superohm = SpectralParams([0.98], [45.21], [0.0], [0.0], [0.0])

#println(total_spectral_density(specparam_superohm, 100))

display(plot_spectral_density(specparam_superohm, 0:1000))

function DWintegral(specparam::SpectralParams, B)
    return quadgk(x -> (1/H)*(total_spectral_density(specparam, x)/x^2)*coth(B*H*x/2), 0, Inf)[1]
end

#= function DWanalytic(L, O, B)
    z = 1/(B*H*O)
    return (L^3/(H*O))*(2*z^2*polygamma(1,z)-1)
end =#

function plot_DW(specparam::SpectralParams,specparam_superohm::SpectralParams, B_range)
    fig = plot(xlabel="1/T (K⁻¹)", ylabel="DW", 
               title="Full vs Superohmic Debye-Waller Argument")
    Dnum_values = [DWintegral(specparam, x) for x in B_range]
    Danaly_values = [DWintegral(specparam_superohm, x) for x in B_range]
    plot!(fig, B_range, Dnum_values)
    plot!(fig, B_range, Danaly_values)
    return fig
end

#display(plot_DW(specparam1, specparam_superohm, 20.0:160.0))

#= function spa_lf(L, O, B)
    # Compute first integral with tanh term
    function int1_integrand(x)
        (Glf(L, O, x)/x^2)*tanh(B*H*x/4)
    end
    int1 = quadgk(int1_integrand, 0, Inf, rtol=1e-8)[1]
    num = exp(-int1/H)
    
    # Compute second integral with csch term
    function int2_integrand(x) 
        Glf(L, O, x)*csch(B*H*x/2)
    end
    int2 = quadgk(int2_integrand, 0, Inf, rtol=1e-8)[1]
    denom = sqrt(2*pi*H*int2)
    
    j = 0.055 # eV
    prefactor = 30*(4*(6.75/1.0)^2)*(2*pi*j^2/H) # (cm^-1 -> ns^-1)*(4a^2/z^2)*k

    return log(prefactor*num/denom)
end =#



#= function plot_spa(L, O, B_range)
    fig = plot(xlabel="1/T (K⁻¹)", ylabel="log(D)", 
               title="Stat Phase Diffusion")
    
    D_values = [spa_lf(L, O, x) for x in B_range]
    plot!(fig, B_range, D_values)
    
    return fig
end =#

#println(spa_lf(0.98, 45.21, 40.0))
#display(plot_spa(0.98, 45.21, 40.0:80.0))