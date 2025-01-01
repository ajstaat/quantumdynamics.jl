# This is code to evaluate the NIBA master equation for spin-boson with ohmic spectral density

using quantumdynamics
using QuadGK
using Plots
using SpecialFunctions
using DifferentialEquations
using Statistics
using MittagLeffler

V = 1.0
s = 1.0
O = 100.0 
L = 0.1*O
J(w) = (L*w/O)^s*exp(-w/O)

# Numerical integrals for polaron transformed bath correlator
function NQ1(t)
    integrand(x) = J(x) * sin(x*t) / x^2
    return quadgk(x -> integrand(x), 0, Inf, rtol=1e-6, atol=1e-6)[1]
end
function NQ2(B, t)
    integrand(x) = J(x) * coth(B*x/2) * (1 - cos(x*t)) / x^2
    return quadgk(x -> integrand(x), 0, Inf, rtol=1e-8, atol=1e-8)[1]
end

# Escher-Ankerhold 2010 expression: Strictly Ohmic
function EA_Q1analytic(t)
    return (L/O)*atan(O*t)
end
function EA_Q2analytic(B, t)
    u = 1 + 1/(B*O)
    return (L/O)*log( ((1+im*O*t)*gamma(u)^2)/(gamma(u+im*t/B)*gamma(u-im*t/B)) )
end

# Gorlich-Weiss 1988 expressions: Arbitrary s Ohmic
function Qanalytic(B,t)
    u = 1 + 1/(B*O)
    if s == 1
        return (L/O)*log( B*O*gamma(u)^2/(gamma(u-1 + im*t/B)*gamma(u - im*t/B)) )
    else 
        return (L/O)^s*gamma(s-1)*((1-(1+im*O*t)^(1-s)) + (B*O)^(1-s)*(2*zeta(s-1+10^(-12),u) - zeta(s-1+10^(-12),u+im*t/B) - zeta(s-1+10^(-12),u-im*t/B)))
    end
end

function plot_Q_functions(t_range, B)
    fig = plot(layout=(2,1), 
              xlabel=["" "Time"], 
              ylabel=["Q1" "Q2"],
              title=["Q1 Comparison" "Q2 Comparison"])
    
    # Plot Q1 functions
    Q1_numeric = [Q1(t,1) for t in t_range]
    Q1_analytic = [imag(Qanalytic(B,t,1)) for t in t_range]
    
    plot!(fig[1], t_range, Q1_numeric, label="Q1 Numeric", linestyle=:solid)
    plot!(fig[1], t_range, Q1_analytic, label="Q1 Analytic", linestyle=:dash)
    
    # Plot Q2 functions
    Q2_numeric = [Q2(B, t, 1) for t in t_range]
    Q2_analytic = [real(Qanalytic(B, t, 1)) for t in t_range]
    
    plot!(fig[2], t_range, Q2_numeric, label="Q2 Numeric", linestyle=:solid)
    plot!(fig[2], t_range, Q2_analytic, label="Q2 Analytic", linestyle=:dash)
    
    return fig
end

function Kernel(t,B)
    Q1 = imag(Qanalytic(B,t))
    Q2 = real(Qanalytic(B,t))
    return cos(Q1)*exp(-Q2)
end

function plot_Kernel(t_range, B)
    fig = plot(xlabel="Time", 
              ylabel="K(t)",
              title="Kernel")
    
    Kernel_numeric = [Kernel(t, B) for t in t_range]
    
    plot!(fig, t_range, Kernel_numeric, label="Kernel Numeric", linestyle=:solid)
    
    return fig
end

#display(plot_Kernel(0.0:1*10^(-2):2.0, 1.0))

function propagate(t_range,B)
    K(t) = quadgk(x -> Kernel(x, B), 0, t, rtol=10^(-8), atol=10^(-8))[1]
    f(u, p, t) = -K(t) * u
    u0 = 1.0
    prob = ODEProblem(f, u0, t_range)
    sol = solve(prob, Tsit5(), reltol = 1e-4, abstol = 1e-4)
    return sol
end

function propagate_nonlocal(t_range, B)
    K(t) = V^2*Kernel(t, B)
    
    function f(du, u, h, p, t)
        if t <= 0
            du[1] = 0.0
        else
            # Increase accuracy of quadgk
            integral = quadgk(s -> K(t-s) * h(p,s)[1], 0, t, 
                            rtol=10^(-6), atol=10^(-6))[1]
            du[1] = -integral
        end
    end
    
    # History function only needs to define t <= 0
    function h(p, t)
        [1.0]  # Initial condition for t <= 0
    end
    
    u0 = [1.0]
    prob = DDEProblem(f, u0, h, t_range)
    sol = solve(prob, MethodOfSteps(Tsit5()), reltol=1e-7, abstol=1e-7)
    return sol
end

function plot_dynamics(t_range, B; nonlocal=false)
    if nonlocal
        sol = propagate_nonlocal(t_range, B)
        # Create dense output points for smooth plotting
        t_dense = range(t_range[1], t_range[2], length=500)
        populations = [sol(t)[1] for t in t_dense]
        t_plot = t_dense
    else
        sol = propagate(t_range, B)
        populations = sol.u
        t_plot = sol.t
    end
    
    display(plot(t_plot, 0.5*(populations .+ 1.0), linewidth = 5, title = "Dynamics",
        xaxis = "Time (t)", yaxis = "Population", 
        label = nonlocal ? "Nonlocal Solution" : "Local Solution"))
end

#plot_dynamics((0.0,20.0), 1000.0, nonlocal=true)

function lifetime(t_range, B, tstart)
    sol = propagate(t_range, B)
    t_full = sol.t
    p_full = sol.u
    
    # Use interpolation to get exact population at tstart
    p_start = sol(tstart)
    
    # Find index of starting time
    tstart_idx = findfirst(x -> x >= tstart, t_full)
    
    # Get time points and solution values after tstart (for fitting)
    t_fit = t_full[tstart_idx:end]
    p_fit = p_full[tstart_idx:end]
    
    # Calculate first moment (mean lifetime) using data after tstart
    dt = t_fit[2] - t_fit[1]
    τ = sum((t_fit .- tstart) .* p_fit) * dt / sum(p_fit * dt)
    
    # Plot original solution and exponential approximation
    γ = 1/τ # Decay rate
    
    # Create exponential fit data from tstart onwards
    exp_approx = p_start * exp.(-γ * (t_full[tstart_idx:end] .- tstart))
    
    plt = plot(t_full, p_full, label="Numerical", linewidth=2)
    # Plot forward fit with dashed line
    plot!(plt, t_full[tstart_idx:end], exp_approx, 
          label="Exponential fit", linestyle=:dash, linewidth=2, color=:orange)
    vline!([tstart], label="Fit Start", linestyle=:dot, color=:black)
    title!("Population Decay (τ = $(round(τ, digits=3)))")
    xlabel!("Time")
    ylabel!("Population")
    display(plt)
    
    return τ
end

#lifetime((0.0,20.0), 1.0, 4.1)

function decay_rate_from_log(t_range, B, tstart)
    sol = propagate(t_range, B)
    t_full = sol.t
    p_full = sol.u
    
    # Find index of starting time
    tstart_idx = findfirst(x -> x >= tstart, t_full)
    
    # Get time points and solution values after tstart
    t_fit = t_full[tstart_idx:end]
    p_fit = p_full[tstart_idx:end]
    
    # Calculate log of population
    log_p = log.(p_fit)
    
    # Calculate numerical derivative of log population using central differences
    deriv = zeros(length(t_fit)-1)
    for i in 1:length(deriv)
        deriv[i] = (log_p[i+1] - log_p[i]) / (t_fit[i+1] - t_fit[i])
    end
    
    # Find where derivative stabilizes (use moving average to smooth)
    window = min(20, div(length(deriv), 4))  # Ensure window isn't too large
    moving_avg = zeros(length(deriv))
    
    for i in 1:length(deriv)
        start_idx = max(1, i - window)
        end_idx = min(length(deriv), i + window)
        moving_avg[i] = mean(deriv[start_idx:end_idx])
    end
    
    # Get average rate from latter half of the data
    start_idx = div(length(moving_avg), 2)  # Start from halfway point
    γ = -mean(deriv[start_idx:end])  # Use raw derivative instead of moving average
    τ = 1/γ
    
    # Create plots
    p1 = plot(t_full, p_full, label="Population", linewidth=2)
    vline!([tstart], label="Analysis Start", linestyle=:dot, color=:black)
    ylabel!("Population")
    
    p2 = plot(t_fit, log_p, label="Log Population", linewidth=2)
    plot!(t_fit[1:end-1], -γ .* (t_fit[1:end-1] .- t_fit[1]) .+ log_p[1],
          label="Fitted Slope", linestyle=:dash, linewidth=2)
    ylabel!("Log Population")
    
    p3 = plot(t_fit[1:end-1], -deriv, label="Instantaneous Rate", linewidth=1, alpha=0.5)
    plot!(t_fit[1:end-1], -moving_avg, label="Moving Average", linewidth=2)
    hline!([γ], label="Mean Rate", linestyle=:dash, color=:black)
    ylabel!("Decay Rate")
    
    plot(p1, p2, p3, layout=(3,1), size=(800,1000),
         xlabel="Time", title=["Population" "Log Population" "Decay Rate"])
    display(plot!(title="Decay Rate Analysis (τ = $(round(τ, digits=3)))"))
    
    return τ
end

#decay_rate_from_log((0.0,20.0), 1000.0, 1.0)

function FGR_rate(B)
    # Integrate the kernel function over time to get the rate
    function kernel(t)
        Q = Qanalytic(B, t)
        return exp(-real(Q)) * cos(imag(Q))
    end
    
    # Find appropriate cutoff time where kernel effectively vanishes
    function find_cutoff()
        t = 0.0
        step = 0.1
        threshold = 1e-6
        
        while abs(kernel(t)) > threshold && t < 1000.0
            t += step
        end
        
        return min(t, 1000.0)  # Cap at 1000 if not converged
    end
    
    cutoff = find_cutoff()
    rate, err = quadgk(kernel, 0, cutoff, rtol=1e-6, atol=1e-6)
    
    return rate  # Factor of 2 from NIBA rate expression
end

#println("Fermi Golden Rule rate: ", FGR_rate(1.0))

# Simple exponential kernel
function M(t, N=1.0)
    t ≥ 0 ? exp(-N*t) : 0.0
end

# Test propagation with simple kernel
function propagate_test(t_range, N=1.0)
    function f(du, u, h, p, t)
        if t ≤ 0
            du[1] = 0.0
        else
            integral = quadgk(s -> M(t-s, N) * h(p,s)[1], 0, t, rtol=1e-8)[1]
            du[1] = -integral
        end
    end
    
    tspan = (t_range[1], t_range[2])
    h(p,t) = [1.0]  # Initial condition is constant 1.0
    
    solve(DDEProblem(f, [1.0], h, tspan), MethodOfSteps(Tsit5()))
end

# Analytical solution for comparison
function analytical_solution(t, N=1.0)
    if N == 1.0
        exp(-t/2) * (cos(sqrt(3)/2 * t) + 1/sqrt(3) * sin(sqrt(3)/2 * t))
    else
        # For other values of N, would need different solution
        error("Analytical solution only implemented for N=1.0")
    end
end

#= t_range = (0.0, 20.0)
sol = propagate_test(t_range)
t = 0:0.1:20
plot(t, [analytical_solution.(t) getindex.(sol.(t), 1)],
     label=["Analytical" "Numerical"],
     xlabel="Time",
     ylabel="Population",
     title="Comparison of Analytical and Numerical Solutions") =#

## EXAMPLE 1: Toulouse Hamiltonian -> Validated for O = 10.0, L = 1.0, gives correct slope by log/FGR
## EXAMPLE 2: Zero Temp Dynamics -> Validated for O = 100.0, L = 0.1, agrees with Mittag-Leffler

function zero_temp_dynamics(t)
    l = L/O
    Deff = V*(V/O)^(l/(2-l))*(gamma(1-l)*cos(pi*l/2))^(1/(2-l))
    y = Deff*t
    
    return mittleff(2-l, -y^(2-l))
end

function plot_zero_temp_dynamics(t_range)
    # Dense time points for both solutions
    t_dense = range(t_range[1], t_range[2], length=1000)
    
    # Calculate analytical solution
    zero_temp = zero_temp_dynamics.(t_dense)
    
    # Get numerical solution
    sol = propagate_nonlocal(t_range, 1000.0)
    
    # Use the solution's interpolation capability
    numerical_sol = [sol(t)[1] for t in t_dense]
    
    plt = plot(t_dense, zero_temp, 
              label="Zero Temperature Analytical",
              linewidth=2,
              linestyle=:dash)
    plot!(plt, t_dense, numerical_sol, 
          label="Numerical Solution",
          linewidth=2)
    
    xlabel!("Time")
    ylabel!("Population")
    title!("Zero Temperature Dynamics Comparison")
    
    display(plt)
end

plot_zero_temp_dynamics((0.0, 20.0))

## EXAMPLE 3: Dube Benchmarks