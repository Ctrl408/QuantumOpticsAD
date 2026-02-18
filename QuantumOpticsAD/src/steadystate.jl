using NLsolve
using LinearAlgebra

"""
    steadystate_ad(H_func, J_func, p; guess=nothing)

Finds the steady state ρ_ss such that L(ρ_ss) = 0 using a root-finding algorithm.
This method supports implicit differentiation for optimization.
"""
function steadystate_ad(H_func, J_func, p; guess=nothing)
    H0 = H_func(0.0, p)
    sz = size(H0.data)
    n = sz[1]
    
    function f!(fvec, u_vec)
        rho_mat = reshape(u_vec, sz)
=        drho = master_derivative(Operator(H0.basis_l, H0.basis_r, rho_mat), p, 0.0, H_func, J_func)
        fvec .= vec(drho.data)
    end
    
=    if guess === nothing
        u0 = vec(Matrix(1.0I, n, n) ./ n)
    else
        u0 = vec(guess.data)
    end
    
    res = nlsolve(f!, u0; autodiff=:forward)
    
    return Operator(H0.basis_l, H0.basis_r, reshape(res.zero, sz))
end