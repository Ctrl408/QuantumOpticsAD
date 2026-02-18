# src/sensitivity.jl
using ChainRulesCore
using QuantumOpticsBase
using LinearAlgebra

# 1. Custom Rule for expect(A, rho::ADOperator)
# This handles the reverse-mode gradient for density matrix observables.
function ChainRulesCore.rrule(::typeof(QuantumOpticsBase.expect), A::AbstractOperator, rho::ADOperator)
    val = expect(A, rho)
    
    function expect_pullback(Δ)

        ∂rho_data = A.data' .* Δ
        ∂rho = Tangent{typeof(rho)}(data = ∂rho_data)
        
        return NoTangent(), NoTangent(), ∂rho
    end
    
    return val, expect_pullback
end

# 2. Custom Rule for expect(A, psi::ADKet)
# This handles the reverse-mode gradient for wavefunction observables.
function ChainRulesCore.rrule(::typeof(QuantumOpticsBase.expect), A::AbstractOperator, psi::ADKet)
    val = expect(A, psi)
    
    function expect_pullback(Δ)

        ∂psi_data = (A.data * psi.data) .* Δ
        ∂psi = Tangent{typeof(psi)}(data = ∂psi_data)
        
        return NoTangent(), NoTangent(), ∂psi
    end
    
    return val, expect_pullback
end

# 3. Implicit Rule for steadystate_ad
# This allows Zygote to skip differentiating through the NLsolve iterations,
# which is much faster and more stable.
function ChainRulesCore.rrule(::typeof(steadystate_ad), H_func, J_func, p; kwargs...)
    rho_ss = steadystate_ad(H_func, J_func, p; kwargs...)
    
    function steadystate_pullback(Δrho)

        return NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end
    
    return rho_ss, steadystate_pullback
end