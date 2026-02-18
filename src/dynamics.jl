# src/dynamics.jl
using LinearAlgebra
using SciMLBase
using SciMLSensitivity

"""
    master_derivative(rho, p, t, H_func, J_func)

Calculates the Lindbladian L(ρ) for density matrix evolution. 
This is a standalone function used by both `master_ad` and `steadystate_ad` 
to maintain consistency across solvers.
"""
function master_derivative(rho::Union{Operator, ADOperator}, p, t, H_func, J_func)
    H = H_func(t, p).data
    Js, Jds = J_func(t, p)
    rho_mat = rho.data
    
    drho = -1.0im * (H * rho_mat - rho_mat * H)
    
    for i in 1:length(Js)
        J = Js[i].data
        Jd = Jds[i].data
        drho += J * rho_mat * Jd - 0.5 * (Jd * J * rho_mat + rho_mat * Jd * J)
    end
    
    return ADOperator(rho.basis_l, rho.basis_r, drho)
end

"""
    schroedinger_ad(tspan, psi0, H_func, p; alg=DP5(), kwargs...)

Differentiable Schrödinger equation solver. Returns a vector of `ADKet` 
objects to ensure `ForwardDiff.Dual` numbers are preserved.
"""
function schroedinger_ad(tspan, psi0::Ket, H_func, p; alg=DP5(), kwargs...)
    u0 = psi0.data
    
    f(u, p, t) = -1.0im * (H_func(t, p).data * u)
    
    prob = ODEProblem(f, u0, (tspan[1], tspan[end]), p)
    sol = solve(prob, alg; sensealg=InterpolatingAdjoint(), kwargs...)
    
    return [ADKet(psi0.basis, u) for u in sol.u]
end

"""
    master_ad(tspan, rho0, H_func, J_func, p; alg=DP5(), kwargs...)

Differentiable Master equation solver. Returns a vector of `ADOperator` 
objects to maintain compatibility with Zygote and ForwardDiff.
"""
function master_ad(tspan, rho0::Operator, H_func, J_func, p; alg=DP5(), kwargs...)
    u0 = rho0.data
    sz = size(u0)
    bl, br = rho0.basis_l, rho0.basis_r

    function f(u, p, t)
        rho = ADOperator(bl, br, reshape(u, sz))
        drho = master_derivative(rho, p, t, H_func, J_func)
        return vec(drho.data)
    end
    
    prob = ODEProblem(f, vec(u0), (tspan[1], tspan[end]), p)
    sol = solve(prob, alg; sensealg=InterpolatingAdjoint(), kwargs...)
    
    return [ADOperator(bl, br, reshape(u, sz)) for u in sol.u]
end