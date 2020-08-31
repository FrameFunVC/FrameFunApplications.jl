module FrameFunApplications

using Reexport
@reexport using FrameFunDerivativeDicts.PDEs
using FrameFunDerivativeDicts.PDEs: PDE, PDERule, PDEBlock
using FrameFun.FrameFunInterface: directsolver
using BasisFunctions: expansion

module Util
    using GridArrays
    using GridArrays: element
    using SymbolicDifferentialOperators: δx, δy

    export heatmap_matrix
    function heatmap_matrix(u,D,g;fill=NaN)
        M = u(g)
        M[.!(g .∈ Ref(D))] .= fill
        (element(g,1),element(g,2)),M'
    end
    export velocity_matrices
    function velocity_matrices(u,D,g;fill=NaN)
        (x,y),Vx = heatmap_matrix((δx^1*δy^0)(u),D, g;fill=fill)
        (x,y),Vy = heatmap_matrix((δx^0*δy^1)(u),D, g;fill=fill)
        V = sqrt.(abs2.(Vx) .+ abs2.(Vy))
        (x,y),((Vx,Vy),V)
    end
    export velocity_vectors
    function velocity_vectors(u,g)
        Vx = ((δx^1*δy^0)(u))(g)
        Vy = ((δx^0*δy^1)(u))(g)
        V = sqrt.(abs2.(Vx) .+ abs2.(Vy))
        ([x[1] for x in g],[x[2] for x in g]),((Vx,Vy),V)
    end
end # module Util
@reexport using .Util

# module Plots
#     using Plots, GridArrays, ..Util
#
#     export heatmap_plot
#     heatmap_plot(N,u,D;opts...) =
#         heatmap(EquispacedGrid(N,0,1),EquispacedGrid(N,0,1),real.(heatmap_matrix(u,D,EquispacedGrid(N,0,1)^2));opts...)
#     export quiver_plot
#     function quiver_plot(N,u,D;γ=2,a=.2,b=.8,c=.2,d=.8,opts...)
#         _pgγ = EquispacedGrid(γ*N,a,b)×EquispacedGrid(γ*N,c,d)
#         xγ,yγ = elements(_pgγ)
#         heatmap(xγ,yγ,heatmap_matrix(u,D,_pgγ);color=:blues)
#         contour!(xγ,yγ,heatmap_matrix(u,D,_pgγ);color=:black)
#
#         _pg = EquispacedGrid(N,a,b)×EquispacedGrid(N,c,d)
#         x,y = elements(_pgγ)
#         Vx,Vy,V = velocity_matrices(u,D,_pg)
#         Vmax = norm(Base.filter(x->!isnan(x),V),Inf)
#         @show Vmax
#         _pg_vec = _pg[:]
#         quiver!([x[1] for x in _pg_vec],[x[2] for x in _pg_vec];quiver=(Vx[:]./(Vmax*N),Vy[:]./(Vmax*N)),color="white")
#     end
# end # module Plots
# @reepoxrt using .Plots

export pdesolve
pdesolve(pde::PDE; opts...) =
    expansion(pde.blocks[1].dict,_solve((pde.lhs,pde.rhs); opts...))
pdesolve(rules::PDERule...; opts...) =
    pdesolve(PDE(rules...); opts...)
pdesolve(blocks::PDEBlock...; opts...) =
    pdesolve(PDE(blocks...); opts...)

function _solve(pde; options...)
    A = pde[1]
    b = pde[2]
    S = directsolver(A; options...)
    c = S*b
    # @info "Residual is $(maximum(abs.(A*c-b)))"
    # @info "Coefficient norm is $(maximum(abs.(c)))"
    c
end

include("SplineCircleExample.jl")
@reexport using .SplineCircleExample
include("SplineSquareExample.jl")
@reexport using .SplineSquareExample

end # module
