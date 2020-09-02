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



module PDEPlots
    function initialize_plots()
        @eval Main begin
            using Plots
            export Plots
        end
    end
    using GridArrays, ..Util, BasisFunctions

    export heatmap_plot
    function heatmap_plot(N,u,D;opts...)
        initialize_plots()
        (x,y),M = heatmap_matrix(u,D,EquispacedGrid(N,0,1)^2)
        Main.Plots.heatmap(x,y,real.(M);opts...)
    end

    export quiver_plot
    function quiver_plot(N,u,D;γ=2,a=.2,b=.8,c=.2,d=.8,v_scaling=1,v_uniform = false,opts...)
        initialize_plots()
        _pgγ = EquispacedGrid(γ*N,a,b)×EquispacedGrid(γ*N,c,d)
        ((xγ,yγ),M) = heatmap_matrix(u,D,_pgγ)
        Main.Plots.heatmap(xγ,yγ,real.(M);color=:blues)
        Main.Plots.contour!(xγ,yγ,real.(M);color=:black)

        _pg = EquispacedGrid(N,a,b)×EquispacedGrid(N,c,d)
        (x,y),((Vx,Vy),V) = velocity_vectors(u,subgrid(_pg,D))

        Vmax = norm(V,Inf)*2
        Vx ./= (Vmax*N)
        Vy ./= (Vmax*N)

        Main.Plots.quiver!(x,y;quiver=(real.(Vx),real.(Vy)),color="white",opts...)
    end
end # module PDEPlots
@reexport using .PDEPlots

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
