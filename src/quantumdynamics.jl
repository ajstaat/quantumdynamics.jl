module quantumdynamics

export Glf, Ghf, total_spectral_density, SpectralParams

Glf(L,O,x) = (L*x/O)^3*exp(-x/O)
Ghf(L,S,W,x) = (L*x)*(((x-S)^2 + (W)^2)^(-1))*((2*W)/(pi+2*atan(S/W)))

H = 1.239*10^(-4) #\hbar 2 pi c [=] eV cm

struct SpectralParams
    # Low frequency parameters
    Llf::Vector{Float64}  # Coupling strengths
    Olf::Vector{Float64}  # Frequency cutoffs

    # High frequency parameters  
    Lhf::Vector{Float64}  # Coupling strengths
    Shf::Vector{Float64}  # Peak positions
    Whf::Vector{Float64}  # Peak widths
end

function total_spectral_density(params::SpectralParams, x)
    # Sum up low frequency contributions
    lf_sum = sum(Glf(params.Llf[i], params.Olf[i], x) 
                 for i in 1:length(params.Llf))
                 
    # Return early if no high frequency contributions
    if all(x -> x == 0.0, params.Lhf)
        return lf_sum
    end
    
    # Sum up high frequency contributions
    hf_sum = sum(Ghf(params.Lhf[i], params.Shf[i], params.Whf[i], x)
                 for i in 1:length(params.Lhf))
    
    return lf_sum + (1-exp(-x/params.Olf[1]))*hf_sum
end

function plot_spectral_density(params1::SpectralParams, x_range)#, params2::Union{SpectralParams,Nothing}=nothing)
    fig = plot(xlabel="Frequency (cm⁻¹)", ylabel="J(ω) (cm⁻¹)", 
               title="Total Spectral Density")
    
    y_values1 = [total_spectral_density(params1, x) for x in x_range]
    plot!(fig, x_range, y_values1, label="Spectral Density 1")
    
    #= if params2 !== nothing
        y_values2 = [total_spectral_density(params2, x) for x in x_range]
        plot!(fig, x_range, y_values2, label="Spectral Density 2")
    end =#
    
    return fig
end

end