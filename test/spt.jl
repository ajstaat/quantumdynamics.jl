using quantumdynamics
using QuadGK
using Plots

H = 1.239*10^(-4) #\hbar 2 pi c [=] eV cm

function spt(specparam::SpectralParams, B, ϵ)

    function I1(x, t, specparam, B)
    
        total_spectral_density(specparam,x)*(csch.(B*H*x/2)*cos(x*t) - coth.(B*H*x/2))/x^2  
    
    end

    function I2(x, t, specparam, B)
    
        -total_spectral_density(specparam,x)*coth.(B*H*x/2)/x^2  

    end
    
    f(t) = quadgk(x->I1(x, t, specparam, B), 0, Inf; rtol=10^(-ϵ), atol=10^(-ϵ))
    g(t) = quadgk(x->I2(x, t, specparam, B), 0, Inf; rtol=10^(-ϵ), atol=10^(-ϵ))
    
    K(t) = exp.(f(t)[1]/H) - exp.(g(t)[1]/H)
    k = quadgk(t->K(t), -Inf, Inf; rtol=10^(-ϵ+1.0), atol=10^(-ϵ+1.0))

    j = 0.055
    prefactor = 4*1000*(6.75/10.0)^2*j^2/H^2

    return log.(prefactor*k[1])

end

function plot_spt(params::SpectralParams, B_range, ϵ)
    fig = plot(xlabel="1/T (K⁻¹)", ylabel="log(D)", 
               title="Small Polaron Diffusion")
    
    D_values = [spt(params, x, ϵ) for x in B_range]
    plot!(fig, B_range, D_values)
    
    return fig
end

specparam = SpectralParams([0.68], [21], [0.42,0.30,0.42], [150,200,311], [15,13,22])

println(spt(specparam, 100, 8.0))

#display(plot_spt(specparam, 60.0:80.0, 8.0))