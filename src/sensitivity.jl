using ChainRulesCore
using QuantumOpticsBase

function ChainRulesCore.rrule(::typeof(QuantumOpticsBase.expect), A::AbstractOperator, rho::Operator)
    val = expect(A, rho)
    
    function expect_pullback(Δ)

        ∂rho_data = transpose(A.data) .* Δ
        
        # Wrap in Tangent to tell Zygote only 'data' is differentiable
        ∂rho = Tangent{typeof(rho)}(data = ∂rho_data)
        
        return NoTangent(), NoTangent(), ∂rho
    end
    
    return val, expect_pullback
end