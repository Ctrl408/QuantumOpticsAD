using QuantumOpticsAD
using ForwardDiff
using Zygote
using Test

@testset "QuantumOpticsAD Full Suite" begin

    b = FockBasis(4)
    a = destroy(b)
    ad = dagger(a)
    psi0 = coherentstate(b, 1.0)
    rho0 = dm(psi0)
    tspan = [0.0, 1.0]

    @testset "Algebra" begin
        k1 = wrap_state(psi0)
        k2 = 0.5 * k1
        k_diff = k1 - k2
        @test k_diff.data ≈ 0.5 * psi0.data
    end

    @testset "ForwardDiff (Schroedinger)" begin
        function obj_forward(p)
            H_f(t, p) = p[1] * (a + ad)
            states = schroedinger_ad(tspan, psi0, H_f, p)
            return real(expect(ad * a, states[end]))
        end
        
        p_val = [0.5]
        grad = ForwardDiff.gradient(obj_forward, p_val)
        @test length(grad) == 1
        @test !isnan(grad[1])
        println("ForwardDiff Gradient: ", grad[1])
    end

    @testset "Zygote (Master + Correlation)" begin
        function obj_reverse(p)
            H_f(t, p) = p[1] * ad * a
            J_f(t, p) = ([sqrt(p[2]) * a], [sqrt(p[2]) * ad])

            corr = correlation_ad(tspan, rho0, H_f, J_f, ad, a, p)
            return real(sum(corr))
        end

        p_init = [0.1, 0.05]
        val, grad = Zygote.withgradient(obj_reverse, p_init)
        @test !isnothing(grad[1])
        println("Zygote Gradient: ", grad[1])
    end

    @testset "Steady State" begin
        H_f(t, p) = 0.1 * ad * a
        J_f(t, p) = ([sqrt(0.05) * a], [sqrt(0.05) * ad])
        rho_ss = steadystate_ad(H_f, J_f, [0.1])
        
        @test tr(Operator(rho_ss.basis_l, rho_ss.basis_r, rho_ss.data)) ≈ 1.0
    end
end