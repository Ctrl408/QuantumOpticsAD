using QuantumOpticsBase
using FFTW

"""
    correlation_ad(tspan, rho0, H_func, J_func, A, B, p; kwargs...)

Calculate the two-time correlation <A(t)B(0)> using the AD-native master equation.
"""
function correlation_ad(tspan, rho0::Operator, H_func, J_func, A, B, p; kwargs...)
    rho_start = B * rho0
    
    rhos = master_ad(tspan, rho_start, H_func, J_func, p; kwargs...)
    
    return [expect(A, r) for r in rhos]
end

"""
    spectrum_ad(omega, tspan, rho0, H_func, J_func, A, B, p; kwargs...)

Calculate the spectrum as the Fourier transform of the correlation function.
"""
function spectrum_ad(tspan, rho0, H_func, J_func, A, B, p; kwargs...)
    corr = correlation_ad(tspan, rho0, H_func, J_func, A, B, p; kwargs...)
    
    dt = tspan[2] - tspan[1]
    return 2 * dt .* fftshift(real(fft(corr)))
end