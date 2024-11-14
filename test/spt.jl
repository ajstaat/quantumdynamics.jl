using quantumdynamics
using QuadGK
using Plots

function spt(specparam::SpectralParams, B)

    function I1(x, t, specparam, B)
    
        total_spectral_density(specparam,x)*[csch.(B*H*x/2)*cos(x*t) - coth.(B*H*x/2)]/x^2  
    
    end

    function I2(x, t, specparam, B)
    
        total_spectral_density(specparam,x)*[coth.(B*H*x/2)]/x^2  

    end
    
    f(t) = getfield(quadgk(x->I(x, t, specparam, B), 0, Inf; rtol=10^(-8), atol=10^(-8)), 1)
    g(t) = getfield(quadgk(x->I2(x, t, specparam, B), 0, Inf; rtol=10^(-8), atol=10^(-8)), 1)
    
    K(t) = exp.(f(t)[1]/H) - exp.(g(t)[1]/H)
    k = getfield(quadgk(t->K(t), -Inf, Inf; rtol=10^(-8), atol=10^(-8)), 1)

    j = 0.055
    prefactor = 4*1000*(6.75/10.0)^2*j^2/H^2

    return log.(prefactor*k)

end

function plot_spt(params::SpectralParams, B_range)
    fig = plot(xlabel="1/T (K⁻¹)", ylabel="log(D)", 
               title="Small Polaron Diffusion")
    
    D_values = [spt(params, x) for x in B_range]
    plot!(fig, B_range, D_values)
    
    return fig
end

specparam = SpectralParams([0.68], [21], [0.42,0.30,0.42], [150,200,311], [15,13,22])

display(plot_spt(specparam, 20.0:1.0:200.0))