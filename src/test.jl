using QuantumOpticsAD
using Zygote

b = FockBasis(5)
a = destroy(b)
ad = dagger(a)
rho0 = dm(coherentstate(b, 0.5))

function H_params(t, p)
    return p[1] * ad * a
end

function J_params(t, p)
    return ([sqrt(p[2]) * a], [sqrt(p[2]) * ad])
end

function objective(p)
    tspan = range(0.0, 10.0, length=100)
    corr = correlation_ad(tspan, rho0, H_params, J_params, ad, a, p)    
    return -real(sum(corr))
end

p_init = [0.1, 0.05]
val, grad = Zygote.withgradient(objective, p_init)

println("Objective (Integrated Corr): ", val)
println("Gradients w.r.t [detuning, decay]: ", grad[1])