using QuantumOpticsBase

struct ADKet{B<:Basis, T<:AbstractVector}
    basis::B
    data::T
end

function wrap_state(psi::Ket)
    return ADKet(psi.basis, psi.data)
end

function unwrap_state(psi::ADKet)
    return Ket(psi.basis, psi.data)
end

import Base: +, *, -
+(a::ADKet, b::ADKet) = ADKet(a.basis, a.data + b.data)
*(alpha::Number, a::ADKet) = ADKet(a.basis, alpha * a.data)

import Base: +, *
+(a::Operator{B,B,T}, b::Operator{B,B,T}) where {B,T} = Operator(a.basis_l, a.basis_r, a.data + b.data)
*(alpha::Number, a::Operator{B,B,T}) where {B,T} = Operator(a.basis_l, a.basis_r, alpha * a.data)