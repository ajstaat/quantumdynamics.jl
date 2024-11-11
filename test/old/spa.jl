using Integrals
using QuadGK
using Optim
using Plots 

G(ω, λ1, λ2, λ3, Ω1, Ω2, Ω3) = (λ1*ω/((ω-Ω1)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω1/0.75))) + (λ2*ω/((ω-Ω2)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω2/0.75))) + (λ3/Ω3)*ω*exp(-(ω/Ω3))

# G(ω, λ, Ω) = (λ/Ω)*ω*exp(-(ω/Ω))

fig = plot()
#plot!(fig, 0:0.05:20, ω-> G(ω, 0.5, 1.4, 10.0, 1.5))
H = 6.57*10^(-4)

function lkspa(λ1, λ2, Ω1, Ω2, β)

    p = [λ1, λ2, Ω1, Ω2, β]
    f(ω, p) = (G(ω, p[1], p[2], p[3], p[4])/ω^2)*tanh.(p[5]*H*ω/4)
    numprob = IntegralProblem(f, 0.0, Inf, p)
    num = solve(numprob, HCubatureJL(); reltol = 1e-10, abstol = 1e-10)

    g(ω, p) = G(ω, p[1], p[2], p[3], p[4])*csch.(p[5]*H*ω/2)
    denomprob = IntegralProblem(g, 0.0, Inf, p)
    denom = solve(denomprob, HCubatureJL(); reltol = 1e-10, abstol = 1e-10)

    knum = exp.(-num.u/H)
    kdenom = sqrt.(2*pi*H*denom.u)
    j = 0.3
    prefactor = 4*1000*(6.75/10.0)^2*2*pi*j^2/(H)

    lk = log.(prefactor*knum./kdenom)

end

lkspa(0.0, 1.9, 10.0, 1.65, 80.0)

function lkspa2(λ2, Ω2, β)

    p = [λ2, Ω2, β]
    f(ω, p) = (G(ω, p[1], p[2])/ω^2)*tanh.(p[3]*H*ω/4)
    numprob = IntegralProblem(f, 0.0, Inf, p)
    num = solve(numprob, HCubatureJL(); reltol = 1e-14, abstol = 1e-14)

    g(ω, p) = G(ω, p[1], p[2])*csch.(p[3]*H*ω/2)
    denomprob = IntegralProblem(g, 0.0, Inf, p)
    denom = solve(denomprob, HCubatureJL(); reltol = 1e-14, abstol = 1e-14)

    knum = exp.(-num.u/H)
    kdenom = sqrt.(2*pi*H*denom.u)
    j = 0.055
    prefactor = 4*1000*(6.75/10.0)^2*2*pi*j^2/(H)

    lk = log.(prefactor*knum./kdenom)

end

function lkspa3(λ1, λ2, λ3, Ω1, Ω2, Ω3, β)

    p = [λ1, λ2, λ3, Ω1, Ω2, Ω3, β]
    f(ω, p) = (G(ω, p[1], p[2], p[3], p[4], p[5], p[6])/ω^2)*tanh.(p[7]*H*ω/4)
    numprob = IntegralProblem(f, 0.0, Inf, p)
    num = solve(numprob, HCubatureJL(); reltol = 1e-14, abstol = 1e-14)

    g(ω, p) = G(ω, p[1], p[2], p[3], p[4], p[5], p[6])*csch.(p[7]*H*ω/2)
    denomprob = IntegralProblem(g, 0.0, Inf, p)
    denom = solve(denomprob, HCubatureJL(); reltol = 1e-14, abstol = 1e-14)

    knum = exp.(-num.u/H)
    kdenom = sqrt.(2*pi*H*denom.u)
    j = 0.3
    prefactor = 4*1000*(6.75/10.0)^2*2*pi*j^2/(H)

    lk = log.(prefactor*knum./kdenom)

end

fig = plot()
plot!(fig, big(40.0):big(1.0):big(80.0), β->lkspa3(big(0.68), big(0.3), big(0.94), big(5.0), big(9.0), big(0.62), β))
#plot!(fig, big(40.0):big(1.0):big(80.0), β->lkspa3(big(0.71), big(0.42), big(0.63), big(5.0), big(9.0), big(0.62), β))

data = lkspa3(big(0.2), big(0.3), big(1.4), big(9.0), big(12.0), big(1.65), big(40.0):big(1.0):big(2000.0));

data = lkspa2(big(1.9),big(1.65),big(40.0):big(1.0):big(80.0))

experiment = [-3.75, -6.25, -8.90, -10.10, -10.95]
#experiment = [-8.90, -10.10, -10.95]

function lkspaopt(expt, β, x0)

    lkdist(x) = (lkspa(x[1], x[2], x[3], x[4], β) - expt)
    lksqr(x) = lkdist(x)'lkdist(x)
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [5.0, 2.0, 200.0, 2.0]
    res = optimize(lksqr, lower, upper, x0, Fminbox(LBFGS()), autodiff = :forward)
    return Optim.minimizer(res)

end

function lkspa2opt(expt, β, x0)

    lkdist(x) = (lkspa2(x[1], x[2], β) - expt)
    lksqr(x) = lkdist(x)'lkdist(x)
    lower = [0.0, 0.0]
    upper = [10.0, 500.0]
    res = optimize(lksqr, lower, upper, x0, Fminbox(LBFGS()), autodiff = :forward)
    return Optim.minimizer(res)

end

β = [39.5, 45.2, 51.0, 58.8, 72.5]
#β = [51.0, 58.8, 72.5]
fit = lkspaopt(experiment, β, [2.0, 0.0, 50.0, 1.65]) #[2.0, 0.0, 50.0, 1.65])
fig2 = plot()
plot!(fig2, β, experiment)
plot!(fig2, 20.0:1.0:100.0, β -> lkspa(fit[1], fit[2], fit[3], fit[4], β))