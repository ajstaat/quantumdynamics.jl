using Integrals
using QuadGK
using Optim
using Plots 

G(ω, λ1, λ2, λ3, Ω1, Ω2, Ω3) = (λ1*ω/((ω-Ω1)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω1/0.75))) + (λ2*ω/((ω-Ω2)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω2/0.75))) + (λ3/Ω3)*ω*exp(-(ω/Ω3))

# G(ω, λ, Ω) = (λ/Ω)*ω*exp(-(ω/Ω))

fig = plot()
#plot!(fig, 0:0.05:20, ω-> G(ω, 0.5, 1.4, 10.0, 1.5))
H = 6.57*10^(-4)


function lkmlj(λs, λ1, Ω1, β)

    ω1 = 0.0041*Ω1
    A = β/(4*π*λs)

    knum = 0.0
    for ν in big(0):big(50)
        knum = knum .+ sqrt.(A)*exp(-λ1/ω1).*(λ1/ω1)^(ν)/factorial(ν).*exp.(-π*A*(λs + ω1*ν)^2)
    end

    j = 0.3
    prefactor = 4*1000*(6.75/10.0)^2*2*pi*j^2/(H)

    log.(prefactor*knum)
end

function twomodemlj(λs, λ1, λ2, Ω1, Ω2, β)

    ω1 = 0.0041*Ω1
    ω2 = 0.0041*Ω2
    A = β/(4*π*λs)

    knum = 0.0
    for ν in big(1):big(100)
        for μ in big(1):big(100)
            knum = knum .+ sqrt.(A)*exp(-λ1/ω1-λ2/ω2).*((λ1/ω1)^(ν)*(λ2/ω2)^(μ)/(factorial(ν)*factorial(μ))).*exp.(-π*A*(λs + ω1*ν + ω2*μ)^2)
        end
    end

    j = 0.3
    prefactor = 4*1000*(5.0/0.1)^2*2*pi*j^2/(H)

    log.(prefactor*knum)

end

#data2 = twomodemlj(1.4, 0.3, 0.2, 10.0, 12.0, 40.0:1.0:2000.0);
figmlj = plot()
plot!(fig, 40:1:80, β->twomodemlj(0.95, 0.68, 0.3, 5.0, 9.0, β)) #0.63, 0.72, 0.42, 5.0, 9.0, β))

data = twomodemlj(big(0.95), big(0.68), big(0.3), big(5.0), big(9.0), big(20.0):big(0.5):big(80.0))

function lkmljopt(expt, β, x0)

    lkdist(x) = (lkmlj(x[1], x[2], x[3], β) - expt)
    lksqr(x) = lkdist(x)'lkdist(x)
    lower = [1.9, 0.01, 0.01]
    upper = [10.0, 10.0, 200.0]
    res = optimize(lksqr, lower, upper, x0, Fminbox(LBFGS()), autodiff = :forward)
    return Optim.minimizer(res)

end

function twomodemljopt(expt, β, x0)

    lkdist(x) = (twomodemlj(x[1], x[2], x[3], 9.0, 12.0, β) - expt)
    lksqr(x) = lkdist(x)'lkdist(x)
    lower = [0.01, 0.01, 0.01]#, 0.01, 0.01]
    upper = [10.0, 10.0, 10.0]#, 200.0, 200.0]
    res = optimize(lksqr, lower, upper, x0, Fminbox(LBFGS()), autodiff = :forward)
    return Optim.minimizer(res)

end

fit = lkmljopt(experiment, β, [big(5.0), big(5.0), big(150.0)])
fig3 = plot()
plot!(fig3, β, experiment)
plot!(fig3, 40.0:1.0:200.0, β -> lkmlj(fit[1], fit[2], fit[3], β))

fit = twomodemljopt(experiment, β, [big(5.0), big(5.0), big(5.0)]) #, big(150.0), big(150.0)])
fig11 = plot()
plot!(fig11, β, experiment)
plot!(fig11, 20.0:1.0:200.0, β -> twomodemlj(fit[1], fit[2], fit[3], 9.0, 12.0, β))#, fit[4], fit[5], β))