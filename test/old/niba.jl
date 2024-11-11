using Integrals
using QuadGK
using Optim
using Plots 

G(ω, λ1, λ2, λ3, Ω1, Ω2, Ω3) = (λ1*ω/((ω-Ω1)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω1/0.75))) + (λ2*ω/((ω-Ω2)^2+(0.75)^2))*(2*0.75/(pi + 2*atan(Ω2/0.75))) + (λ3/Ω3)*ω*exp(-(ω/Ω3))

# G(ω, λ, Ω) = (λ/Ω)*ω*exp(-(ω/Ω))

fig = plot()
#plot!(fig, 0:0.05:20, ω-> G(ω, 0.5, 1.4, 10.0, 1.5))
H = 6.57*10^(-4)


#function niba(λ1, λ2, Ω1, Ω2, β)

    p = [λ1, λ2, Ω1, Ω2, β]
    
    function Q1(t)
        
        q1(ω,p) = G(ω, p[1], p[2], p[3], p[4])*sin.(ω*t)/ω^2
        Q1prob = IntegralProblem(q1, 10^(-10), Inf, p)
        Q1 = solve(Q1prob, HCubatureJL(); reltol = 1e-10, abstol = 1e-10)
    
    end

    function Q2(t,β)

        q2(ω, p) = G(ω, p[1], p[2], p[3], p[4])*coth(β*H*ω/2)*(1 .- cos.(ω*t))/ω^2
        Q2prob = IntegralProblem(q2, 10^(-10), Inf, p)
        Q2 = solve(Q2prob, HCubatureJL(); reltol = 1e-10, abstol = 1e-10)

    end

    function K(s)

        Kt(t,p) = cos.(Q1(t)/H)*exp.(-Q2(t,40)/H .- s*t)
        Kprob = IntegralProblem(Kt, 10^(-10), Inf, p)
        K = solve(Kprob, HCubatureJL(); reltol = 1e-10, abstol = 1e-10)
    
    end

    t = range(0, 0.05, length=100)
    plot(t, exp.(-Q2(t,40)/H))

    function ksolve()

        kdist(x) = x[1] - K(-x[1])
        res = optimize(kdist, [0.0], Fminbox(LBFGS()), autodiff = :forward)
        return Optim.minimizer(res)

    end

    j = 0.3
    prefactor = 4*1000*(6.75/10.0)^2*j^2/H^2

    lk = log.(prefactor*ksolve())

#end

#function niba(λ1, λ2, Ω1, Ω2, β)

    function Q1Integrand(ω, t, λ1, λ2, Ω1, Ω2)

        G(ω, λ1, λ2, Ω1, Ω2)*sin(ω*t)/ω^2

    end

    function Q2Integrand(ω, t, λ1, λ2, Ω1, Ω2, β)

        G(ω, λ1, λ2, Ω1, Ω2)*coth.(β*H*ω/2)*(1 - cos(ω*t))/ω^2

    end

    function K(s, λ1, λ2, Ω1, Ω2, β)

        Q1(t) = getfield(quadgk(ω->Q1Integrand(ω, t, λ1, λ2, Ω1, Ω2), 0, Inf; rtol=10^(-12), atol=10^(-12)), 1)
        Q2(t) = getfield(quadgk(ω->Q2Integrand(ω, t, λ1, λ2, Ω1, Ω2, β), 0, Inf; rtol=10^(-12), atol=10^(-12)), 1)

        Kt(t,s) = cos.(Q1(t)[1]/H)*exp.(-Q2(t)[1]/H - s*t)

        getfield(quadgk(t->Kt(t,s), 0.0, Inf; rtol=10^(-12), atol=10^(-12)), 1)

    end

    figx = plot()
    plot!(figx, 40:1:80, β->K(0.0, 0.0, 1.9, 10.0, 1.65, β))

    function Ksolve(λ1, λ2, Ω1, Ω2, β)

        kdist(x) = x[1] - K(-x[1], λ1, λ2, Ω1, Ω2, β)
        lower = [0.0]
        upper = [1.0]
        res = optimize(kdist, lower, upper, [10^(-6)], Fminbox(LBFGS())) #, autodiff = :forward)
        return Optim.minimizer(res)

    end

    Ksolve(0.0, 1.9, 10.0, 1.65, 40)

    j = 0.3
    prefactor = 4*1000*(6.75/10.0)^2*j^2/H^2

    #lk = log.(prefactor*Ksolve(λ1, λ2, Ω1, Ω2, β))
    lk = log.(prefactor*Ksolve(0.0, 1.9, 10.0, 1.65, 40))

#end

niba(0.0, 1.9, 10.0, 1.65, 40)