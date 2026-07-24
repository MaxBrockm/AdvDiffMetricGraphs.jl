"""
    sysof_equations(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real}, 
                    HVV::AbstractMatrix{<:Real}, RHS_e::AbstractVector{<:Real},
                    RHS_v::AbstractVector{<:Real}, D::AbstractArray{<:Real}, 
                    tol::Float64; maxiter=50)

Solves the coupled block system of linear equations arising from a graph discretization
by reducing it to a vertex-based Schur complement system and solving it via a Preconditioned 
Conjugate Gradient (PCG) method.

# Arguments
- `HEE::AbstractMatrix{<:Real}`: Edge-edge block submatrix `H_EE`.
- `HEV::AbstractMatrix{<:Real}`: Edge-vertex block submatrix `H_EV`.
- `HVV::AbstractMatrix{<:Real}`: Vertex-vertex block submatrix `H_VV`.
- `RHS_e::AbstractVector{<:Real}`: Right-hand side vector corresponding to edge degrees of freedom.
- `RHS_v::AbstractVector{<:Real}`: Right-hand side vector corresponding to vertex degrees of freedom.
- `D::AbstractArray{<:Real}`: Preconditioning operator/matrix for the Schur complement system.
- `tol::Float64`: Convergence tolerance for the iterative PCG solver.
- `maxiter::Int=50`: Maximum allowable iterations for the PCG solver (default: `50`).

# Output
- `sol_e::AbstractVector{<:Real}`: Solution vector corresponding to edge degrees of freedom.
- `sol_v::AbstractVector{<:Real}`: Solution vector corresponding to vertex degrees of freedom.
"""
function sysof_equations(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real}, 
    HVV::AbstractMatrix{<:Real},RHS_e::AbstractVector{<:Real},
    RHS_v::AbstractVector{<:Real}, D::AbstractArray{<:Real}, 
    tol::Float64; maxiter=50)

    RHS_mod_1 = RHS_v - transpose(HEV) * HEEInv(HEE, RHS_e)
    sol_v, _ = PCG_mod(HEE, HEV, HVV, D*RHS_mod_1, zeros(size(RHS_mod_1)), tol, maxiter=maxiter, Dinv=D)
    sol_v = D*sol_v
    RHS_mod_2 = RHS_e - HEV * sol_v
    sol_e = HEEInv(HEE, RHS_mod_2)
    return sol_e, sol_v
end

"""
    HEEInv(HEE::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})

Computes the solution to the linear system `HEE * x = v` by solving for `x` using a direct solver.

# Arguments
- `HEE::AbstractMatrix{<:Real}`: Edge-edge block submatrix `H_EE`.
- `v::AbstractVector{<:Real}`: Right-hand side vector for the system

# Output
- `x::AbstractVector{<:Real}`: Solution vector satisfying `HEE * x = v`.
"""
@inline function HEEInv(HEE::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})
    return HEE\v
end


"""
    Schur_Mult(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real}, 
               HVV::AbstractMatrix{<:Real}, v::AbstractVector{<:Real})

Computes the matrix-vector product of the Schur complement operator 
`S = H_VV - H_EV^T H_EE^{-1} H_EV` with a vector `v`.

# Arguments
- `HEE::AbstractMatrix{<:Real}`: Edge-edge block submatrix `H_EE`.
- `HEV::AbstractMatrix{<:Real}`: Edge-vertex block submatrix `H_EV`.
- `HVV::AbstractMatrix{<:Real}`: Vertex-vertex block submatrix `H_VV`.
- `v::AbstractVector{<:Real}`: Vector to be multiplied with the Schur complement matrix.

# Output
- `sol::AbstractVector{<:Real}`: Result of `S v = (H_VV - H_EV^T H_EE^{-1} H_EV) v`.
"""
function Schur_Mult(HEE::AbstractMatrix{<:Real}, 
    HEV::AbstractMatrix{<:Real}, HVV::AbstractMatrix{<:Real},
    v::AbstractVector{<:Real})
    sol = HVV*v - transpose(HEV) * HEEInv(HEE, HEV * v)
    return sol 
end


"""
    PCG_mod(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real},
            HVV::AbstractMatrix{<:Real}, f::AbstractVector{<:Real}, 
            x0::AbstractVector{<:Real}, abstol::Float64; maxiter=1000, Dinv=I)

Solves the Schur complement system using a modified Preconditioned Conjugate Gradient (PCG) algorithm.

# Arguments
- `HEE::AbstractMatrix{<:Real}`: Edge-edge block submatrix `H_EE`.
- `HEV::AbstractMatrix{<:Real}`: Edge-vertex block submatrix `H_EV`.
- `HVV::AbstractMatrix{<:Real}`: Vertex-vertex block submatrix `H_VV`.
- `f::AbstractVector{<:Real}`: Right-hand side vector for the Schur system.
- `x0::AbstractVector{<:Real}`: Initial guess vector.
- `abstol::Float64`: Absolute convergence tolerance for the residual norm.
- `maxiter::Int=1000`: Maximum number of iterations (default: `1000`).
- `Dinv`: Preconditioning matrix or operator (default: `I`).

# Output
- `x::AbstractVector{<:Real}`: Computed solution vector.
- `anz_iter::Int`: Total number of iterations performed.
"""
function PCG_mod(HEE::AbstractMatrix{<:Real}, HEV::AbstractMatrix{<:Real},
    HVV::AbstractMatrix{<:Real}, f::AbstractVector{<:Real}, 
    x0::AbstractVector{<:Real}, abstol::Float64; maxiter=1000, Dinv = I)

    r = f- Dinv*Schur_Mult(HEE, HEV, HVV, Dinv*x0)
    r_alt = copy(r)

    d = copy(r)
    x = copy(x0)

    anz_iter = 0
    norm_r = norm(r,2)

    while (norm_r > abstol) && anz_iter < maxiter
        z = Dinv*Schur_Mult(HEE, HEV, HVV, Dinv*d)
        alpha = dot(r,r)/dot(d,z)
        x = x + alpha * d
        r_alt = copy(r)
        r = r-alpha*z
        beta = dot(r,r)/dot(r_alt, r_alt)
        d = r + beta*d
        norm_r = norm(r,2)
        anz_iter += 1
    end
    return x, anz_iter
end