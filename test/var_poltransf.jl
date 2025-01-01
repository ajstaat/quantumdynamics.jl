using quantumdynamics
using QuadGK
using Plots

H = 1.239*10^(-4) #\hbar 2 pi c [=] eV cm

specparam_superohm = SpectralParams([0.98], [45.21], [0.0], [0.0], [0.0])
specparam1 = SpectralParams([0.69], [22], [0.42, 0.30, 0.42], [150,200,311], [17,14,27])

function F(T, D, x, k)
    return (1 + (k*D*coth(x/(2*T))*tanh(k*D/2))/x)^(-1)
end

fig = plot(xlabel="Frequency (cm⁻¹)")
#display(plot!(fig, x -> F(200, 400, x, 1)^2, 0, 1000))
display(plot!(fig, x -> total_spectral_density(specparam1, x), 0, 1000, label="J(ω)"))
display(plot!(fig, x -> total_spectral_density(specparam1, x)*F(200, 400, x, 0.9994934575275614)^2, 0, 1000, label="J(ω)F^2(ω)"))

#0.999585

#println(spectral_moment(specparam_superohm, -1))
#println(quadgk(x -> (F(200, 400, x, 1)^(2))*total_spectral_density(specparam_superohm, x)*x^(-1), 0, Inf)[1])

function kappa(specparam::SpectralParams, T, D, k)

    varDW = getfield(quadgk(x -> F(T, D, x, k)^2*coth(x/(2*T))*total_spectral_density(specparam, x)/x^2, 0, Inf, rtol=1e-8),1)

    return exp(-varDW)
end

#println(kappa(specparam_superohm, 200, 400, 0.5))

function solve_self_consistent(specparam::SpectralParams, B, D, tol=1e-8, max_iter=1000)
    k = 0.5  # Initial guess
    k_prev = -1.0
    iter = 0
    
    while abs(k - k_prev) > tol && iter < max_iter
        k_prev = k
        k = kappa(specparam, B, D, k_prev)
        #println(k)
        iter += 1
    end
    
    if iter == max_iter
        @warn "Maximum iterations reached without convergence"
    end
    
    return k
end

#println(solve_self_consistent(specparam1, 200, 400))