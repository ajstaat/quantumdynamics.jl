using quantumdynamics
using QuadGK
using Plots
using HCubature

reorg = 1.0;
cut = 10.0;
J(ω) = reorg*ω*exp(-ω/cut);

function bath_correlation(t::Float64, B::Float64)
    return quadgk(w ->J(w)*(coth(w*B/2)*cos(w*t)-im*sin(w*t)), 0, 100.0, rtol=1e-2, atol=1e-2)[1]
end

function plot_correlation(t_range, B)
    fig = plot(xlabel="time", ylabel="B(t)", 
               title="Bath Correlation Function")
    
    y_values = [bath_correlation(t, B) for t in t_range]
    plot!(fig, t_range, real.(y_values), label="Real part")
    plot!(fig, t_range, imag.(y_values), label="Imaginary part")
    
    return fig
end

#display(plot_correlation(0.0:10^(-2):1.0, 1.0))

#= function fourier_transform(w::Float64, B::Float64)
    return quadgk(t -> bath_correlation(t, B)*exp(-im*w*t), 0, 2.0, rtol=1e-2, atol=1e-2)[1]
end =#

function fourier_transform(w::Float64, B::Float64)
    # Integrand for 2D integral combining bath correlation and Fourier transform
    function integrand(x)
        t, omega = x[1], x[2]
        # Inner part of bath correlation
        corr_term = J(omega) * (coth(omega*B/2)*cos(omega*t) - im*sin(omega*t))
        # Multiply by Fourier transform factor
        return corr_term * exp(-im*w*t)
    end
    
    # Integration bounds
    lower = [0.0, 0.0]  # t_min, omega_min
    upper = [2.0, 100.0]  # t_max, omega_max
    
    result, _ = hcubature(integrand, lower, upper; 
                         rtol=1e-6, atol=1e-6)
    return result
end

function principal_value(w_0::Float64, B::Float64, t::Union{Float64,Type{Inf}})
    # Define antisymmetrized spectral density
    J_anti(w) = sign(w) * J(abs(w))
    
    # Define integrand with principal value consideration
    function base_integrand(w)
        if abs(w - w_0) < 1e-10  # Avoid exact singularity
            return 0.0
        end
        return 0.5 * J_anti(w) * (coth(w*B/2) - 1) / (w - w_0)
    end
    
    if t === Inf
        # Original calculation
        epsilon = 1e-3
        result1 = pi/2 * sign(w_0) * J(abs(w_0)) * (coth(B*w_0/2) - 1)
        result2 = quadgk(base_integrand, -100.0, w_0-epsilon, w_0+epsilon, 100.0, 
                     rtol=1e-4, atol=1e-4)[1]
        return result1 + im*result2
    else
        # First integral with im*(1-cos((w-w_0)t))
        function integrand1(w)
            return base_integrand(w) * im * (1 - cos((w-w_0)*t))
        end
        
        # Second integral with -sin((w-w_0)t)
        function integrand2(w)
            return base_integrand(w) * (sin((w-w_0)*t))
        end
        
        epsilon = 1e-3
        result1 = quadgk(integrand1, -100.0, w_0-epsilon, w_0+epsilon, 100.0, 
                        rtol=1e-4, atol=1e-4)[1]
        result2 = quadgk(integrand2, -100.0, w_0-epsilon, w_0+epsilon, 100.0, 
                        rtol=1e-4, atol=1e-4)[1]
        
        return result1 + result2
    end
end

#= function principal_value(w_0::Float64, B::Float64)
    # Define antisymmetrized spectral density
    J_anti(w) = sign(w) * J(abs(w))
    
    # Define integrand with principal value consideration
    function integrand(w)
        if abs(w - w_0) < 1e-10  # Avoid exact singularity
            return 0.0
        end
        return 0.5 * J_anti(w) * (coth(w*B/2) - 1) / (w - w_0)
    end
    
    # Compute integral avoiding the singularity point
    epsilon = 1e-3  # Increased epsilon to better avoid singularity
    result = quadgk(integrand, -100.0, w_0-epsilon, w_0+epsilon, 100.0, 
                   rtol=1e-4, atol=1e-4)[1]
    
    return result
end =#

#println(fourier_transform(10.0, 1.0))

function plot_transform(w_range, B)
    fig = plot(xlabel="freq", ylabel="B(w)", 
               title="Bath Transform at t = Inf")
    
    y_values = [fourier_transform(w, B) for w in w_range]
    plot!(fig, w_range, real.(y_values), label="Real part")
    plot!(fig, w_range, imag.(y_values), label="Imaginary part")
    #analytical_values = [pi/2 * sign(w) * J(abs(w)) * (coth(B*w/2) - 1) for w in w_range]
    #plot!(fig, w_range, real.(analytical_values), label="Analytical", linestyle=:dash)
    pv_values = [principal_value(w, B, Inf) for w in w_range]
    plot!(fig, w_range, real.(pv_values), label="Real Principal Value", linestyle=:dashdot)
    plot!(fig, w_range, imag.(pv_values), label="Imaginary Principal Value", linestyle=:dashdot)
    return fig
end

#display(plot_transform(0.0:0.1:4.0, 1.0))

## Meier-Tannor High-Frequency Integrals

jtest(w) = (pi/2)*2*(w/(((w+1.5)^2 + 0.5^2)*((w-1.5)^2 + 0.5^2)))
q(t) = quadgk(w -> (1/(2*pi))*jtest(w)*sin(w*t), -Inf, Inf, rtol=1e-4, atol=1e-4)[1]
qanalytic(t) = (pi/8)*(2/(0.5*1.5))*exp(-0.5*t)*sin(1.5*t)

#display(plot(q, 0.0:0.01:10.0))
#display(plot(qanalytic, 0.0:0.01:10.0))

function safe_coth(x)
    if abs(x) < 1e-6
        return 1/x  # First-order approximation of coth near 0
    else
        return coth(x)
    end
end

function p(t,B)
    epsilon = 1e-6  # Small region around w=0 for asymptotic behavior
    
    # jtest without the w in numerator
    jsmall(w) = (pi/2)*2*(1/(((w+1.5)^2 + 0.5^2)*((w-1.5)^2 + 0.5^2)))
    
    function integrand(w)
        if abs(w) < epsilon
            # Near w=0:
            # - use jsmall (no w in numerator)
            # - coth(Bw/2) ≈ 2/(Bw), so w cancels but 2/B remains
            return (1/(pi*B)) * jsmall(w) * cos(w*t)
        else
            # Away from w=0, use full expressions
            term = (1/(2*pi)) * jtest(w) * coth(B*w/2) * cos(w*t)
            return isfinite(term) ? term : 0.0
        end
    end
    
    return quadgk(integrand, -Inf, Inf, rtol=1e-5, atol=1e-5)[1]
end

jMT(w,pk,ok,gk) = (pi/2)*pk*w/(((w+ok)^2+gk^2)*((w-ok)^2+gk^2))

cut = 7.5
lam = 0.1

JMTohm(w) = jMT(w,lam*(cut^4)*12.0677,cut*0.2378,cut*2.2593) + jMT(w,lam*(cut^4)*(-19.9762),cut*0.0888,cut*5.4377) + jMT(w,lam*(cut^4)*0.1834,cut*0.0482,cut*0.8099)
Johm(w) = lam*(pi/2)*w*exp(-w/cut)

# Calculate normalization factors
#norm_factor = Johm(cut)

# Create combined plot with normalized functions
#= w_range = 0.0:0.1:75.0
fig = plot(xlabel="ω/ωc", ylabel="J(ω)/J(ωc)", 
          title="Comparison of Spectral Densities")
plot!(fig, w_range, JMTohm.(w_range) ./ norm_factor, 
      label="Meier-Tannor", linewidth=2, linestyle=:dash)
plot!(fig, w_range, Johm.(w_range) ./ norm_factor, 
      label="Ohmic", linewidth=2)
display(fig) =#

# Q functions (imaginary part of correlation function)
function q_ohm(t)
    return quadgk(w -> (1/(2*pi))*sign(w)*Johm(abs(w))*sin(w*t), -Inf, Inf, rtol=1e-4, atol=1e-4)[1]
end

function p_ohm(t, B)
    epsilon = 1e-6
    
    function integrand(w)
        if abs(w) < epsilon
            # Near w=0, use careful treatment
            return (1/(pi*B)) * sign(w)*lam*(pi/2)*exp(-abs(w)/cut)*cos(w*t)
        else
            term = (1/(2*pi)) * sign(w)*Johm(abs(w)) * coth(B*w/2) * cos(w*t)
            return isfinite(term) ? term : 0.0
        end
    end
    
    return quadgk(integrand, -Inf, Inf, rtol=1e-5, atol=1e-5)[1]
end

function q_MT(t)
    return quadgk(w -> (1/(2*pi))*JMTohm(w)*sin(w*t), -Inf, Inf, rtol=1e-4, atol=1e-4)[1]
end

# First define jMT without the ω in numerator
function jMTsmall(w,pk,ok,gk)
    return (pi/2)*pk/(((w+ok)^2+gk^2)*((w-ok)^2+gk^2))
end

# Create jMTsmallohm combining the three terms without ω
function jMTsmallohm(w)
    return jMTsmall(w,lam*(cut^4)*12.0677,cut*0.2378,cut*2.2593) + 
           jMTsmall(w,lam*(cut^4)*(-19.9762),cut*0.0888,cut*5.4377) + 
           jMTsmall(w,lam*(cut^4)*0.1834,cut*0.0482,cut*0.8099)
end

function p_MT(t, B)
    epsilon = 1e-6
    
    function integrand(w)
        if abs(w) < epsilon
            # Near w=0:
            # - use jMTsmallohm (no w in numerator)
            # - coth(Bw/2) ≈ 2/(Bw), so w cancels but 2/B remains
            return (1/(pi*B)) * jMTsmallohm(w) * cos(w*t)
        else
            term = (1/(2*pi)) * JMTohm(w) * coth(B*w/2) * cos(w*t)
            return isfinite(term) ? term : 0.0
        end
    end
    
    return quadgk(integrand, -Inf, Inf, rtol=1e-5, atol=1e-5)[1]
end

nu(n,B) = 2*pi*n/B
h(t,B,pk,ok,gk) = (pi*pk/(16*ok*gk))*exp(-gk*t)*((sin(pk*t)*sin(B*gk) + cos(ok*t)*sinh(B*ok))/(cosh(B*ok)-cos(B*gk)))
g(t,B,pk,ok,gk) = (pi*pk/(8*ok*gk*(ok^2+gk^2)*B))*exp(-gk*t)*(ok*cos(ok*t) + gk*sin(ok*t))
panalytic(t,B,pk,ok,gk) = h(t,B,pk,ok,gk) + g(t,B,pk,ok,gk) + (im/B)*sum(jMT(im*nu(n,B),pk,ok,gk)*exp(-nu(n,B)*t) for n in 1:20)

qanalytic(t,pk,ok,gk) = (pi/8)*(pk/(gk*ok))*exp(-gk*t)*sin(ok*t)

qohmanalytic(t) = qanalytic(t,lam*(cut^4)*12.0677,cut*0.2378,cut*2.2593) + qanalytic(t,lam*(cut^4)*(-19.9762),cut*0.0888,cut*5.4377) + qanalytic(t,lam*(cut^4)*0.1834,cut*0.0482,cut*0.8099)
pohmanalytic(t,B) = panalytic(t,B,lam*(cut^4)*12.0677,cut*0.2378,cut*2.2593) + panalytic(t,B,lam*(cut^4)*(-19.9762),cut*0.0888,cut*5.4377) + panalytic(t,B,lam*(cut^4)*0.1834,cut*0.0482,cut*0.8099)

# Plot comparisons
function compare_correlations(t_range, B)
    # Q function comparison
    fig1 = plot(xlabel="Time", ylabel="q(t)", 
                title="Comparison of q(t)")
    plot!(fig1, t_range, q_ohm.(t_range), label="Ohmic (numerical)", linewidth=2)
    plot!(fig1, t_range, q_MT.(t_range), label="Meier-Tannor (numerical)", 
          linewidth=2, linestyle=:dash)
    
    # P function comparison
    fig2 = plot(xlabel="Time", ylabel="p(t)", 
                title="Comparison of p(t)")
    plot!(fig2, t_range, p_ohm.(t_range, B), label="Ohmic (numerical)", linewidth=2)
    plot!(fig2, t_range, p_MT.(t_range, B), label="Meier-Tannor (numerical)", 
          linewidth=2, linestyle=:dash)
    
    # Add analytical solutions
    analytic_values = pohmanalytic.(t_range, B)
    qanalytic_values = qohmanalytic.(t_range)
    plot!(fig2, t_range, real.(analytic_values), 
          label="Ohmic (analytic)", linewidth=2, linestyle=:dot)
    plot!(fig2, t_range, imag.(analytic_values), 
          label="Ohmic (analytic imag)", linewidth=2, linestyle=:dot)
    plot!(fig1, t_range, real.(qanalytic_values), 
          label="Ohmic (analytic q)", linewidth=2, linestyle=:dot)
    
    return plot(fig1, fig2, layout=(2,1))
end

# Display the comparison plots
t_range = 0.0:0.01:1.0  # Reduced time range for better detail
B = 0.1  # Increased B to see more oscillations
display(compare_correlations(t_range, B))

# Test each component at t=0 for different B values
for B in [0.1, 1.0, 10.0]
    println("\nB = $B:")
    
    # Parameters for all three terms
    terms = [
        (lam*(cut^4)*12.0677, cut*0.2378, cut*2.2593, "Term 1"),
        (lam*(cut^4)*(-19.9762), cut*0.0888, cut*5.4377, "Term 2"),
        (lam*(cut^4)*0.1834, cut*0.0482, cut*0.8099, "Term 3")
    ]
    
    total = 0.0
    for (pk, ok, gk, name) in terms
        h0 = h(0,B,pk,ok,gk)
        g0 = g(0,B,pk,ok,gk)
        matsubara = (im/B)*sum(jMT(im*nu(n,B),pk,ok,gk) for n in 1:20)
        term_total = h0 + g0 + matsubara
        total += term_total
        
        println("\n  $name:")
        println("    h(0): ", h0)
        println("    g(0): ", g0)
        println("    matsubara: ", matsubara)
        println("    term total: ", term_total)
    end
    
    println("\n  Sum of all terms: ", total)
    println("  Numerical p(0): ", p_ohm(0,B))
end
