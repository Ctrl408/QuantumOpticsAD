# src/types.jl
using QuantumOpticsBase
using LinearAlgebra

struct ADKet{B<:Basis, T<:AbstractVector}
    basis::B
    data::T
end

struct ADOperator{BL<:Basis, BR<:Basis, T<:AbstractMatrix}
    basis_l::BL
    basis_r::BR
    data::T
end

wrap_state(psi::Ket) = ADKet(psi.basis, psi.data)
wrap_state(op::Operator) = ADOperator(op.basis_l, op.basis_r, op.data)

import Base: +, *, -
+(a::ADKet, b::ADKet) = ADKet(a.basis, a.data + b.data)
-(a::ADKet, b::ADKet) = ADKet(a.basis, a.data - b.data)
*(alpha::Number, a::ADKet) = ADKet(a.basis, alpha * a.data)

+(a::ADOperator, b::ADOperator) = ADOperator(a.basis_l, a.basis_r, a.data + b.data)
-(a::ADOperator, b::ADOperator) = ADOperator(a.basis_l, a.basis_r, a.data - b.data)
*(alpha::Number, a::ADOperator) = ADOperator(a.basis_l, a.basis_r, alpha * a.data)

import QuantumOpticsBase: expect, tr, dagger

expect(A::AbstractOperator, psi::ADKet) = dot(psi.data, A.data * psi.data)
expect(A::AbstractOperator, rho::ADOperator) = tr(A.data * rho.data)
expect(A::ADOperator, rho::ADOperator) = tr(A.data * rho.data)
expect(A::ADOperator, rho::Operator) = tr(A.data * rho.data)

tr(op::ADOperator) = tr(op.data)
dagger(op::ADOperator) = ADOperator(op.basis_r, op.basis_l, dagger(op.data))