using quantumdynamics
using SpecialFunctions
using DifferentialEquations
using QuadGK
using Plots

H = 1.239*10^(-4) #\hbar 2 pi c [=] eV cm
specparam_superohm = SpectralParams([0.98], [45.21], [0.0], [0.0], [0.0])
specparam1 = SpectralParams([0.69], [22], [0.42, 0.30, 0.42], [150,200,311], [17,14,27])

I2(x, t, specparam, B) = total_spectral_density(specparam,x)*coth.(B*H*x/2)*(1-cos(x*t))/x^2  
I1(x, t, specparam) = total_spectral_density(specparam,x)*sin(x*t)/x^2
Q2(specparam, B, t) = quadgk(x->(1/H)*I2(x, t, specparam, B), 0, Inf; rtol=10^(-8), atol=10^(-8))[1]
Q1(specparam, t) = quadgk(x->(1/H)*I1(x, t, specparam), 0, Inf; rtol=10^(-6), atol=10^(-6))[1]

function Q2analytic(specparam, B, t)
    lambda = specparam.Llf[1]
    omega = specparam.Olf[1]
    z = 1/(B*H*omega)
    return ((lambda^3)/(H*omega))*(z^2*real(2*polygamma(1,z) - polygamma(1,z*(1+im*omega*t)) - polygamma(1,z*(1-im*omega*t))) - (omega^2*t^2)*(3+omega^2*t^2)/(1+omega^2*t^2)^2)
end

function Q1analytic(specparam, t) 

    lambda = specparam.Llf[1]
    omega = specparam.Olf[1]
    return ((2*lambda^3)/(H*omega))*(omega*t)/(1+omega^2*t^2)^2

end

function plot_Q1_comparison(specparam, t_range)
    fig = plot(xlabel="Time (ps)", ylabel="Q1", 
               title="Numerical vs Analytic Q1")
    
    Q1_values = [cos(Q1(specparam, t)) for t in t_range]
    Q1c_values = [Q1(specparam, t)/100 for t in t_range]
    #Q1a_values = [cos(Q1analytic(specparam, t)) for t in t_range]
    
    t_scale = 33.3564 # THz

    plot!(fig, t_scale*t_range, Q1_values, label="Q1")
    plot!(fig, t_scale*t_range, Q1c_values, label="cos(Q1)")
    
    return fig
end

#display(plot_Q1_comparison(specparam1, 0:10^(-5):0.3))

function plot_kernel_comparison(specparam1, specparam2, B, t_range)
    fig = plot(xlabel="Time (ps)", ylabel="cos(Q1)exp(-Q2)", 
               title="Memory Kernel")

    Q2_values = [cos(Q1(specparam1, t))*exp(-Q2(specparam1, B, t)) for t in t_range]
    Q2c_values = [cos(Q1(specparam2, t))*exp(-Q2(specparam2, B, t)) for t in t_range]

    lambda = specparam1.Llf[1]
    kernel_values = [cos(((2*lambda^3)/H)*t)*exp(-((2*lambda^3)/(B*H^2))*t^2) for t in t_range]
    
    t_scale = 33.3564 # THz

    plot!(fig, t_scale*t_range, Q2_values, label="superohm")
    plot!(fig, t_scale*t_range, Q2c_values, label="structured")
    plot!(fig, t_scale*t_range, kernel_values, label="slow-bath")
    
    return fig
end

#display(plot_kernel_comparison(specparam_superohm, specparam1, 110, 0:10^(-5):0.003))

function POP_factor(specparam, B, t)
    Hb = H/(2*pi*30) # divide by 2 pi c, with c in units of cm/ns.
    J = 0.05 # eV
    return quadgk(s->(J^2/(H*Hb))*cos(Q1analytic(specparam, s))*exp(-Q2analytic(specparam, B, s)), 0, t, rtol=10^(-12), atol=10^(-12))[1]
end

#println([POP_factor(specparam_superohm, 40, t) for t in 0.001:0.001:0.005])
#t_range = 0.0:10^(-5):0.002
#y_values = [POP_factor(specparam1, 40, t) for t in t_range]
#display(plot(t_range, y_values, linewidth = 2, title = "POP Factor", xaxis = "Time (t)", yaxis = "POP Factor"))

function slow_bath_factor(specparam, B, t)
    lambda = big(specparam.Llf[1])
    Hb = big(H/(2*pi*30)) # divide by 2 pi c, with c in units of cm/ns.
    J = big(0.05) # eV
    #return quadgk(s->c2s*(J/H)^2*cos(((2*lambda^3)/H)*s)*exp(-((2*lambda^3)/(B*H^2))*s^2), 0, t, rtol=10^(-10), atol=10^(-10))[1]
    return quadgk(s->(J^2/(H*Hb))*cos(((2*lambda^3)/H)*s)*exp(-((2*lambda^3)/(B*H^2))*s^2), 0, t, rtol=10^(-10), atol=10^(-10))[1]
end

#t_range = 0.0:10^(-5):0.0035
#y_values = [slow_bath_factor(specparam_superohm, 40, t) for t in t_range]
#display(plot(t_range, y_values, linewidth = 2, title = "Slow-Bath Factor", xaxis = "Time (t)", yaxis = "Slow-Bath Factor"))

function plot_factors_vs_temperature(specparam, B_range, t)
    fig = plot(xlabel="Temperature (meV^(-1))", ylabel="Factor Value", 
               title="POP and Slow-Bath Factors vs Temperature")
    
    pop_values = [log(POP_factor(specparam, B, t)) for B in B_range]
    slow_bath_values = [log(slow_bath_factor(specparam, B, t)) for B in B_range]
    
    plot!(fig, B_range, pop_values, label="POP factor")
    plot!(fig, B_range, slow_bath_values, label="Slow-bath factor")
    
    return fig
end

display(plot_factors_vs_temperature(specparam_superohm, 40:1:80, 0.02))


function POP(specparam, B, t_range)

    intK(t) = quadgk(s->(J^2/(H*Hb))*cos(Q1analytic(specparam, s))*exp(-Q2analytic(specparam, B, s)), 0, t, rtol=10^(-8), atol=10^(-8))[1]
    f(u, p, t) = -intK(t) * u
    u0 = 1.0
    #t_range = (0.0, 1.0)
    prob = ODEProblem(f, u0, t_range)
    sol = solve(prob, Tsit5(), reltol = 1e-10, abstol = 1e-10)

    return sol
end

function plot_pop_comparison(specparam, B, t_range)
    fig = plot(linewidth = 2, title = "Population", xaxis = "Time (t)", yaxis = "Population")
    
    t_span = (t_range[1], t_range[end])
    pop_values = POP(specparam_superohm, 40, t_span)
    
    #t_scale = 33.3564 # THz

    plot!(fig, pop_values, label="superohm")
    
    return fig
end

#println(POP(specparam_superohm, 40, (0.0 , 0.1)))
#display(plot_pop_comparison(specparam_superohm, 40, 0:10^(-5):0.0035))
