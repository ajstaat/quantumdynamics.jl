module quantumdynamics

using QuadGk

Glf(L,O,x) = (L*x/O)^3*exp(-x/O)
Ghf(L,S,W,x) = (L*x)*(((x-S)^2 + (W)^2)^(-1))*((2*W)/(pi+2*atan(S/W)))

function adrenorm(D,L,S,W)
    
    integral = getfield(quadgk(x -> Ghf(L,S,W,x)/x^2, 1, 10, rtol=1e-8),1)
    adD = D*exp(-0.5*integral)
    return adD

end

adrenorm(0.055,0.42,150,15)

end