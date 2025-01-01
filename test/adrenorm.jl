using quantumdynamics
using QuadGK
using Plots

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

specparam = SpectralParams([0.68], [21], [0.42,0.30,0.42], [150,200,311], [15,13,22])

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

println(poltransf(400,specparam))

#specparamhf = SpectralParams([0], [21], [0.42,0.30,0.42], [150,200,311], [15,13,22])

#function splittest(specparam::SpectralParams)

#    integral = getfield(quadgk(x -> (1-exp(-x/specparam.Olf[1]))*total_spectral_density(specparam,x)/x, 0, Inf, rtol=1e-8),1)
#    return integral
    
#end

#println(splittest(specparamhf))

function var_poltransf(D,specparam::SpectralParams,T)

    #Optical Mode Coupling Strengths
    x = [150,200,311]
    g = total_spectral_density(specparam,x)

    #Initialize f0 and choose large initial guess for f
    f = 10*ones(length(x))
    f0 = g

    while f - f0 > 1e-8
        f0 = f
        d = D*exp(-(1/H^2)*sum((f0.^(2.0))*coth(x/(2*T))/x.^(2.0)))
        f = g*(1+2*D*coth(x/(2*T))*tanh(D/T))^(-1)
    end

    return d

end

println(var_poltransf(400,specparam,400))

