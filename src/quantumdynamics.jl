module quantumdynamics

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
    
    # Sum up high frequency contributions
    hf_sum = sum(Ghf(params.Lhf[i], params.Shf[i], params.Whf[i], x)
                 for i in 1:length(params.Lhf))
    
    return lf_sum + (1-exp(-x/params.Olf[1]))*hf_sum
end



end