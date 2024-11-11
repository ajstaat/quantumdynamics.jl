using Integrals
using QuadGK
using Optim
using Plots 

G(ω, λ1, λ2, λ3, Ω1, Ω2, Ω3) = (λ1*ω/((ω-Ω1)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω1/0.75))) + (λ2*ω/((ω-Ω2)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω2/0.75))) + (λ3/Ω3)*ω*exp(-(ω/Ω3))

# G(ω, λ, Ω) = (λ/Ω)*ω*exp(-(ω/Ω))

fig = plot()
#plot!(fig, 0:0.05:20, ω-> G(ω, 0.5, 1.4, 10.0, 1.5))
H = 6.57*10^(-4)



function spt(λ1, λ2, Ω1, Ω2, β)

    function I(ω, t, λ1, λ2, Ω1, Ω2, β)
    
        G(ω, λ1, λ2, Ω1, Ω2)*[csch.(β*H*ω/2)*cos(ω*t) - coth.(β*H*ω/2)]/ω^2  
    
    end

    f(t) = getfield(quadgk(ω->I(ω, t, λ1, λ2, Ω1, Ω2, β), 0, Inf; rtol=10^(-9), atol=10^(-9)), 1)

    K(t) = exp.(f(t)[1]/H)
    k = getfield(quadgk(t->K(t), -Inf, Inf; rtol=10^(-14), atol=10^(-14)), 1)

    j = 0.3
    prefactor = 4*1000*(6.75/10.0)^2*j^2/H^2

    return log.(prefactor*k)

end

fig10 = plot()
plot!(fig10, 20.0:1.0:200.0, β->spt(0.0, 1.9, 10.0, 1.65, β))

end