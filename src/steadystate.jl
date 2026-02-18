using NLsolve, LinearAlgebra

function steadystate_ad(H_func, J_func, p; guess=nothing)
    H0 = H_func(0.0, p)
    sz = size(H0.data)
    n2 = sz[1] * sz[2]
    
    function f!(fvec, u_vec)
        u_complex = u_vec[1:n2] + 1.0im * u_vec[n2+1:end]
        rho_mat = reshape(u_complex, sz)
        
        rho_tmp = ADOperator(H0.basis_l, H0.basis_r, rho_mat)
        drho = master_derivative(rho_tmp, p, 0.0, H_func, J_func)
        res_flat = vec(drho.data)
        

        res_flat[1] = tr(rho_mat) - 1.0
        
        fvec[1:n2] .= real(res_flat)
        fvec[n2+1:end] .= imag(res_flat)
    end
    
    if guess === nothing
        # Initial guess: Identity matrix normalized to Trace=1
        u0_mat = Matrix(1.0I, sz...) ./ sz[1]
        u0 = [real(vec(u0_mat)); imag(vec(u0_mat))]
    else
        u0 = [real(vec(guess.data)); imag(vec(guess.data))]
    end
    
    res = nlsolve(f!, u0; autodiff=:forward)
    
    final_complex = res.zero[1:n2] + 1.0im * res.zero[n2+1:end]
    return ADOperator(H0.basis_l, H0.basis_r, reshape(final_complex, sz))
end