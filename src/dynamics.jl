using LinearAlgebra
using SciMLBase
using SciMLSensitivity

# Schrödinger Evolution 

function schroedinger_derivative(u, p, t, H_func)
    H = H_func(t, p)
    return -1.0im * (H.data * u)
end

"""
    schroedinger_ad(tspan, psi0, H_func, p; alg=DP5(), kwargs...)

Differentiable Schrödinger equation solver for closed systems.
"""
function schroedinger_ad(tspan, psi0::Ket, H_func, p; alg=DP5(), kwargs...)
    u0 = psi0.data
    
    f(u, p, t) = schroedinger_derivative(u, p, t, H_func)
    
    prob = ODEProblem(f, u0, (tspan[1], tspan[end]), p)
    sol = solve(prob, alg; sensealg=InterpolatingAdjoint(), kwargs...)
    
    return [Ket(psi0.basis, u) for u in sol.u]
end


"""
    master_derivative(rho, p, t, H_func, J_func)

Functional Lindbladian for density matrix evolution. 
Used by both master_ad and steadystate_ad.
"""
function master_derivative(rho::Operator, p, t, H_func, J_func)
    H = H_func(t, p).data
    Js_all, Jds_all = J_func(t, p)
    rho_mat = rho.data
    
    drho = -1.0im * (H * rho_mat - rho_mat * H)
    
    for i in 1:length(Js_all)
        J = Js_all[i].data
        Jd = Jds_all[i].data
        drho += J * rho_mat * Jd - 0.5 * (Jd * J * rho_mat + rho_mat * Jd * J)
    end
    return Operator(rho.basis_l, rho.basis_r, drho)
end


function master_ad(tspan, rho0::Operator, H_func, J_func, p; alg=DP5(), kwargs...)
    u0 = rho0.data
    sz = size(u0)
    bl, br = rho0.basis_l, rho0.basis_r

    function f(u, p, t)
        rho_mat = reshape(u, sz)
        H = H_func(t, p).data
        Js_all, Jds_all = J_func(t, p)
        
        drho = -1.0im * (H * rho_mat - rho_mat * H)
        for i in 1:length(Js_all)
            J = Js_all[i].data
            Jd = Jds_all[i].data
            drho += J * rho_mat * Jd - 0.5 * (Jd * J * rho_mat + rho_mat * Jd * J)
        end
        return vec(drho)
    end
    
    prob = ODEProblem(f, vec(u0), (tspan[1], tspan[end]), p)
    sol = solve(prob, alg; sensealg=InterpolatingAdjoint(), kwargs...)
    
    return [Operator(bl, br, reshape(u, sz)) for u in sol.u]
end