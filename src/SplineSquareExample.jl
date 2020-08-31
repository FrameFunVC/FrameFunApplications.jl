module SplineSquareExample
using FrameFunApplications, BasisFunctions, CompactTranslatesDict, DomainSets, StaticArrays, FrameFun, FrameFunTranslates
using SymbolicDifferentialOperators: Δ, I, δx, δy

using DelimitedFiles

function _run_square_example(;N=20,γ=4,v=1,r=.05,a=.2,b=.8,c=.2,d=.8,p=3,verbose=false,directsolver=sparseQR_solver)
    # basis = Fourier(N)^2
    basis = BSplineTranslatesBasis(N,p)^2
    innerbox = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))
    centre = (.5,.5)

    L = (nextfloat(centre[1]-r)..prevfloat(centre[2]+r))^2
    domain = setdiff(innerbox,L)

    outergrid = PeriodicEquispacedGrid(γ*N,0,1)^2
    innergrid = subgrid(outergrid,innerbox)

    collocation_grid = subgrid(outergrid,domain)
    laplace_rule = PDERule(Δ,basis,collocation_grid,(x,y)->0.)

    L_boundary_x =  EquispacedGrid(round(Int,2r*γ*N),centre[1]-r,centre[1]+r)×EquispacedGrid(2,centre[2]-r,centre[2]+r)
    L_boundary_y =  EquispacedGrid(2,centre[1]-r,centre[1]+r)×EquispacedGrid(round(Int,2r*γ*N),centre[2]-r,centre[2]+r)

    @assert all(L_boundary_x .∈ Ref(domain))
    @assert all(L_boundary_y .∈ Ref(domain))

    L_rule_x = PDERule(δx^0*δy^1, basis, L_boundary_x, (x,y)->0)
    L_rule_y = PDERule(δx^1*δy^0, basis, L_boundary_y, (x,y)->0)

    homogeneous_grid = EquispacedGrid(2,nextfloat(a),prevfloat(b))×EquispacedGrid(size(innergrid,2),nextfloat(c),prevfloat(d))
    @assert all(homogeneous_grid .∈ Ref(domain))
    homogeneous_rule = PDERule(δx^1*δy^0,basis,homogeneous_grid,(x,y)->0.)


    flow_grid = EquispacedGrid(size(innergrid,1),nextfloat(a),prevfloat(b))×EquispacedGrid(2,nextfloat(c),prevfloat(d))
    @assert all(flow_grid .∈ Ref(domain))
    flow_rule = PDERule(δy^1*δx^0,basis,flow_grid,(x,y)->-v)
    flow_rule_tan = PDERule(δx^1*δy^0,basis,flow_grid,(x,y)->0)


    semi_flow_grid = element(innergrid,1)×EquispacedGrid(1,nextfloat(c),nextfloat(c))
    @assert all(semi_flow_grid .∈ Ref(domain))
    flow_rule_homo = PDERule(I,basis,semi_flow_grid,(x,y)->0)

    pde = PDE(laplace_rule,
        flow_rule,flow_rule_tan,
        L_rule_x,
        L_rule_y,
        homogeneous_rule,
        flow_rule_homo,
        )

    u = pdesolve(pde;directsolver=directsolver)
    if verbose
        @info "============================="
        @info "testing solution"
        @info "Norm coefficients" norm(coefficients(u),Inf)
        @info "Residual" norm(pde.lhs*coefficients(u)-pde.rhs,Inf) norm(pde.lhs*coefficients(u)-pde.rhs)/norm(pde.rhs)
        @show norm(Δ(u)(collocation_grid),Inf)
        @show norm((δx^0*δy)(u)(flow_grid) .+v,Inf)
        @show norm((δx*δy^0)(u)(flow_grid),Inf)

        @show norm((δx^0*δy)(u)(L_boundary_x),Inf)
        @show norm((δx*δy^0)(u)(L_boundary_y),Inf)

        @show norm((δx*δy^0)(u)(homogeneous_grid),Inf)
        @info "============================="
    end
    #, pde, basis, innerbox,
        # outergrid, , collocation_grid, , flow_grid,
        # semi_flow_grid, homogeneous_grid)
    (;u, domain,innergrid,L_boundary_x, L_boundary_y,flow_grid)
end

module CornerSingDicts
    import FrameFunDerivativeDicts.SummationDicts: laplace
    import BasisFunctions: length, size, support, unsafe_eval_element, diff,
        hasderivative, ℝ, Dictionary, evaluation, AbstractSubGrid, GridBasis,
        dense_evaluation, plotgrid, EquispacedGrid
    using StaticArrays: SVector

    # Bisection of corner along negative x axis
    corner_singI(z::Complex) = (z)^(2/3)
    corner_singΔ(z::Complex) = 0.
    corner_singδx(z::Complex) = 2/3*z^(-1/3)
    corner_singδy(z::Complex) = 2im/2*z^(-1/3)

    for fun in (:corner_singI ,:corner_singΔ, :corner_singδx, :corner_singδy)
        @eval $fun((x,y), (x0,y0)=(0.,0.), angle=0.) =
            imag(cis(angle)^(2/3)*$fun( complex(x-x0,y-y0)))
    end

    for D in (:I,:Δ,:δx,:δy)
        DICT = Symbol("CornerSingDict", D)
        eval = Symbol("corner_sing", D)
        @eval begin
            struct $DICT <: Dictionary{SVector{2,Float64},Float64}
                loc::Vector{NTuple{2,Float64}}
                angle::Vector{Float64}
                function $DICT(loc,angle)
                    @assert length(loc)==length(angle)
                    new(loc,angle)
                end
            end
            length(dict::$DICT) = length(dict.loc)
            size(dict::$DICT) = (length(dict.loc),)
            support(::$DICT) = ℝ^2
            unsafe_eval_element(dict::$DICT, i, x) =
                $eval(x, dict.loc[i], dict.angle[i])
            evaluation(::Type{T}, Φ::$DICT, grid::AbstractSubGrid; options...) where {T} =
                dense_evaluation(T, Φ, GridBasis{T}(grid); options...)
            plotgrid(::$DICT,n) = EquispacedGrid(n,.2,.8)^2

        end
    end


    laplace(dict::CornerSingDictI) = CornerSingDictΔ(dict.loc,dict.angle)
    function diff(dict::CornerSingDictI,order::NTuple{2,Integer})
        order==(1,0) && (return CornerSingDictδx(dict.loc,dict.angle))
        order==(0,1) && (return CornerSingDictδy(dict.loc,dict.angle))
        error("Derivative not known")
    end
    hasderivative(::CornerSingDictI) = true

    export CornerSingDict
    const CornerSingDict = CornerSingDictI
end # module CornerSingDicts
using .CornerSingDicts
using LinearAlgebra: norm_sqr

_check_grid(g, domain, corners) =
    all(g .∈ Ref(domain)) && all( _dist(x,corner) > sqrt(eps(1.))  for x in g, corner in corners  )
_dist(x,corner) =
    norm_sqr(x.-corner)



function _run_square_example_sing(;N=20,γ=4,v=1,r=.05,a=.2,b=.8,c=.2,d=.8,p=3,verbose=false,directsolver=sparseQR_solver)
    # basis = Fourier(N)^2

    centre = (.5,.5)
    _corners = collect(Iterators.product(ntuple(k->(centre[1]-r,centre[2]+r),2)...))[:]
    basis = vcat(
        BSplineTranslatesBasis(N,p)^2,
        FrameFunApplications.SplineSquareExample.CornerSingDict(_corners, [3pi/4,pi/4,-3pi/4,-pi/4] )
        )
# plot(layout=(2,2));[contour!(dict[i],subplot=i) for i in 1:4];plot!()
    innerbox = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))

    L = (nextfloat(centre[1]-r)..prevfloat(centre[2]+r))^2
    domain = setdiff(innerbox,L)

    outergrid = PeriodicEquispacedGrid(γ*N,0,1)^2
    innergrid = subgrid(outergrid,innerbox)

    collocation_grid = subgrid(outergrid,domain)
    laplace_rule = PDERule(Δ,basis,collocation_grid,(x,y)->0.)

    full_x = EquispacedGrid(round(Int,2r*γ*N),centre[1]-r,centre[1]+r)
    full_y = EquispacedGrid(round(Int,2r*γ*N),centre[2]-r,centre[2]+r)
    L_boundary_x =  EquispacedGrid(length(full_x)-2, full_x[2],full_x[end-1])×EquispacedGrid(2,centre[2]-r,centre[2]+r)
    L_boundary_y =  EquispacedGrid(2,centre[1]-r,centre[1]+r)×EquispacedGrid(length(full_y)-2, full_y[2],full_y[end-1])

    # L_boundary_x =  EquispacedGrid(round(Int,2r*γ*N),centre[1]-r,centre[1]+r)×EquispacedGrid(2,centre[2]-r,centre[2]+r)
    # L_boundary_y =  EquispacedGrid(2,centre[1]-r,centre[1]+r)×EquispacedGrid(round(Int,2r*γ*N),centre[2]-r,centre[2]+r)

    @assert _check_grid(L_boundary_x, domain, _corners)
    @assert _check_grid(L_boundary_y, domain, _corners)

    L_rule_x = PDERule(δx^0*δy^1, basis, L_boundary_x, (x,y)->0)
    L_rule_y = PDERule(δx^1*δy^0, basis, L_boundary_y, (x,y)->0)

    homogeneous_grid = EquispacedGrid(2,nextfloat(a),prevfloat(b))×EquispacedGrid(size(innergrid,2),nextfloat(c),prevfloat(d))
    @assert _check_grid(homogeneous_grid, domain, _corners)
    homogeneous_rule = PDERule(δx^1*δy^0,basis,homogeneous_grid,(x,y)->0.)


    flow_grid = EquispacedGrid(size(innergrid,1),nextfloat(a),prevfloat(b))×EquispacedGrid(2,nextfloat(c),prevfloat(d))
    @assert _check_grid(flow_grid, domain, _corners)
    flow_rule = PDERule(δy^1*δx^0,basis,flow_grid,(x,y)->-v)
    flow_rule_tan = PDERule(δx^1*δy^0,basis,flow_grid,(x,y)->0)


    semi_flow_grid = element(innergrid,1)×EquispacedGrid(1,nextfloat(c),nextfloat(c))
    @assert _check_grid(semi_flow_grid, domain, _corners)
    flow_rule_homo = PDERule(I,basis,semi_flow_grid,(x,y)->0)

    pde = PDE(laplace_rule,
        flow_rule,flow_rule_tan,
        L_rule_x,
        L_rule_y,
        homogeneous_rule,
        flow_rule_homo,
        )

    u = pdesolve(pde;directsolver=directsolver)
    if verbose
        @info "============================="
        @info "testing solution"
        @info "Norm coefficients" norm(coefficients(u),Inf)
        @info "Residual" norm(pde.lhs*coefficients(u)-pde.rhs,Inf) norm(pde.lhs*coefficients(u)-pde.rhs)/norm(pde.rhs)
        @show norm(Δ(u)(collocation_grid),Inf)
        @show norm((δx^0*δy)(u)(flow_grid) .+v,Inf)
        @show norm((δx*δy^0)(u)(flow_grid),Inf)

        @show norm((δx^0*δy)(u)(L_boundary_x),Inf)
        @show norm((δx*δy^0)(u)(L_boundary_y),Inf)

        @show norm((δx*δy^0)(u)(homogeneous_grid),Inf)
        @info "============================="
    end
    #, pde, basis, innerbox,
        # outergrid, , collocation_grid, , flow_grid,
        # semi_flow_grid, homogeneous_grid)
    (;u, domain,innergrid,L_boundary_x, L_boundary_y,flow_grid,pde)
end







function _run_quarter_square_example(;N=20,γ=4,v=1,r=.05,a=.2,b=.8,c=.2,d=.8,p=3,verbose=false,directsolver=sparseQR_solver)
    # basis = Fourier(N)^2
    basis = BSplineTranslatesBasis(N,p)^2
    innerbox = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))

    L = (prevfloat(a)..prevfloat(a+r))×(prevfloat(c)..prevfloat(c+r))
    domain = setdiff(innerbox,L)

    outergrid = PeriodicEquispacedGrid(γ*N,0,1)^2
    innergrid = subgrid(outergrid,innerbox)

    collocation_grid = subgrid(outergrid,domain)
    laplace_rule = PDERule(Δ,basis,collocation_grid,(x,y)->0.)

    L_boundary_x =  EquispacedGrid(round(Int,r*γ*N),nextfloat(a),nextfloat(a+r))×
                        EquispacedGrid(1,nextfloat(c+r),nextfloat(c+r))
    L_boundary_y =  EquispacedGrid(1,nextfloat(a+r),nextfloat(a+r))×
                        EquispacedGrid(round(Int,r*γ*N),nextfloat(c),nextfloat(c+r))

    @assert all(L_boundary_x .∈ Ref(domain))
    @assert all(L_boundary_y .∈ Ref(domain))

    L_rule_x = PDERule(δx^0*δy^1, basis, L_boundary_x, (x,y)->0)
    L_rule_y = PDERule(δx^1*δy^0, basis, L_boundary_y, (x,y)->0)

    homogeneous_grid_a = EquispacedGrid(1,prevfloat(b),prevfloat(b))×EquispacedGrid(size(innergrid,2),nextfloat(c),prevfloat(d))
    homogeneous_grid_b = EquispacedGrid(1,nextfloat(a),nextfloat(a))×EquispacedGrid(size(innergrid,2),nextfloat(c+r),prevfloat(d))
    @assert all(homogeneous_grid_a .∈ Ref(domain))
    @assert all(homogeneous_grid_b .∈ Ref(domain))
    homogeneous_rule_a = PDERule(δx^1*δy^0,basis,homogeneous_grid_a,(x,y)->0.)
    homogeneous_rule_b = PDERule(δx^1*δy^0,basis,homogeneous_grid_b,(x,y)->0.)


    flow_grid = EquispacedGrid(size(innergrid,1),nextfloat(a),prevfloat(b))×EquispacedGrid(1,prevfloat(d),prevfloat(d))
    @assert all(flow_grid .∈ Ref(domain))
    flow_rule = PDERule(δy^1*δx^0,basis,flow_grid,(x,y)->-v)
    flow_rule_tan = PDERule(δx^1*δy^0,basis,flow_grid,(x,y)->0)

    flow_grid_mid = EquispacedGrid(size(innergrid,1),nextfloat(a+r),prevfloat(b))×EquispacedGrid(1,nextfloat(c),nextfloat(c))
    @assert all(flow_grid_mid .∈ Ref(domain))
    flow_rule_mid_tan = PDERule(δx^1*δy^0,basis,flow_grid_mid,(x,y)->0)

    semi_flow_grid = element(innergrid,1)×EquispacedGrid(1,prevfloat(d),prevfloat(d))
    @assert all(semi_flow_grid .∈ Ref(domain))
    flow_rule_homo = PDERule(I,basis,semi_flow_grid,(x,y)->0)

    pde = PDE(laplace_rule,
        flow_rule,
        flow_rule_tan,
        flow_rule_mid_tan,
        L_rule_x,
        L_rule_y,
        homogeneous_rule_a,
        homogeneous_rule_b,
        flow_rule_homo,
        )

    u = pdesolve(pde;directsolver=directsolver)
    if verbose
        @info "============================="
        @info "testing solution"
        @info "Norm coefficients" norm(coefficients(u),Inf)
        @info "Residual" norm(pde.lhs*coefficients(u)-pde.rhs,Inf) norm(pde.lhs*coefficients(u)-pde.rhs)/norm(pde.rhs)
        @show norm(Δ(u)(collocation_grid),Inf)
        @show norm((δx^0*δy)(u)(flow_grid) .+v,Inf)
        @show norm((δx*δy^0)(u)(flow_grid),Inf)

        @show norm((δx^0*δy)(u)(L_boundary_x),Inf)
        @show norm((δx*δy^0)(u)(L_boundary_y),Inf)

        @show norm((δx*δy^0)(u)(homogeneous_grid_a),Inf)
        @info "============================="
    end
    #, pde, basis, innerbox,
        # outergrid, , collocation_grid, , flow_grid,
        # semi_flow_grid, homogeneous_grid)
    (;u, domain,innergrid,L_boundary_x, L_boundary_y,flow_grid)
end

function _run_quarter_square_example_sing(;N=20,γ=4,v=1,r=.05,a=.2,b=.8,c=.2,d=.8,p=3,sing=true,verbose=false,directsolver=sparseQR_solver)
    # basis = Fourier(N)^2


    innerbox = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))

    L = (prevfloat(a)..prevfloat(a+r))×(prevfloat(c)..prevfloat(c+r))
    domain = setdiff(innerbox,L)

    _corners = [(prevfloat(a+r),prevfloat(c+r)),]
    _angles = [-pi/4,]
    basis = sing ? vcat(
        BSplineTranslatesBasis(N,p)^2,
        FrameFunApplications.SplineSquareExample.CornerSingDict(_corners, _angles )
        ) :
        BSplineTranslatesBasis(N,p)^2

    outergrid = PeriodicEquispacedGrid(γ*N,0,1)^2
    innergrid = subgrid(outergrid,innerbox)

    collocation_grid = subgrid(outergrid,domain)
    laplace_rule = PDERule(Δ,basis,collocation_grid,(x,y)->0.)

    gridx = EquispacedGrid(round(Int,r*γ*N),nextfloat(a),nextfloat(a+r))
    gridy = EquispacedGrid(round(Int,r*γ*N),nextfloat(c),nextfloat(c+r))
    L_boundary_x =  EquispacedGrid(length(gridx)-1,first(gridx),gridx[end-1])×
                        EquispacedGrid(1,nextfloat(c+r),nextfloat(c+r))
    L_boundary_y =  EquispacedGrid(1,nextfloat(a+r),nextfloat(a+r))×
                        EquispacedGrid(length(gridy)-1,first(gridy),gridy[end-1])

    @assert _check_grid(L_boundary_x,domain,_corners)

    @assert _check_grid(L_boundary_y,domain,_corners)


    L_rule_x = PDERule(δx^0*δy^1, basis, L_boundary_x, (x,y)->0)
    L_rule_y = PDERule(δx^1*δy^0, basis, L_boundary_y, (x,y)->0)

    homogeneous_grid_a = EquispacedGrid(1,prevfloat(b),prevfloat(b))×EquispacedGrid(size(innergrid,2),nextfloat(c),prevfloat(d))
    homogeneous_grid_b = EquispacedGrid(1,nextfloat(a),nextfloat(a))×EquispacedGrid(size(innergrid,2),nextfloat(c+r),prevfloat(d))
    @assert _check_grid(homogeneous_grid_a,domain,_corners)
    @assert _check_grid(homogeneous_grid_b,domain,_corners)

    homogeneous_rule_a = PDERule(δx^1*δy^0,basis,homogeneous_grid_a,(x,y)->0.)
    homogeneous_rule_b = PDERule(δx^1*δy^0,basis,homogeneous_grid_b,(x,y)->0.)


    flow_grid = EquispacedGrid(size(innergrid,1),nextfloat(a),prevfloat(b))×EquispacedGrid(1,prevfloat(d),prevfloat(d))
    @assert _check_grid(flow_grid,domain,_corners)
    flow_rule = PDERule(δy^1*δx^0,basis,flow_grid,(x,y)->-v)
    flow_rule_tan = PDERule(δx^1*δy^0,basis,flow_grid,(x,y)->0)
    flow_grid_mid = EquispacedGrid(size(innergrid,1),nextfloat(a+r),prevfloat(b))×EquispacedGrid(1,nextfloat(c),nextfloat(c))
    @assert _check_grid(flow_grid_mid,domain,_corners)
    flow_rule_mid_tan = PDERule(δx^1*δy^0,basis,flow_grid_mid,(x,y)->0)
    semi_flow_grid = element(innergrid,1)×EquispacedGrid(1,prevfloat(d),prevfloat(d))
    @assert _check_grid(semi_flow_grid,domain,_corners)

    flow_rule_homo = PDERule(I,basis,semi_flow_grid,(x,y)->0)

    pde = PDE(laplace_rule,
        flow_rule,
        flow_rule_tan,
        flow_rule_mid_tan,
        L_rule_x,
        L_rule_y,
        homogeneous_rule_a,
        homogeneous_rule_b,
        flow_rule_homo,
        )

    u = pdesolve(pde;directsolver=directsolver)
    if verbose
        @info "============================="
        @info "testing solution"
        @info "Norm coefficients" norm(coefficients(u),Inf)
        @info "Residual" norm(pde.lhs*coefficients(u)-pde.rhs,Inf) norm(pde.lhs*coefficients(u)-pde.rhs)/norm(pde.rhs)
        @show norm(Δ(u)(collocation_grid),Inf)
        @show norm((δx^0*δy)(u)(flow_grid) .+v,Inf)
        @show norm((δx*δy^0)(u)(flow_grid),Inf)

        @show norm((δx^0*δy)(u)(L_boundary_x),Inf)
        @show norm((δx*δy^0)(u)(L_boundary_y),Inf)

        @show norm((δx*δy^0)(u)(homogeneous_grid_a),Inf)
        @info "============================="
    end
    #, pde, basis, innerbox,
        # outergrid, , collocation_grid, , flow_grid,
        # semi_flow_grid, homogeneous_grid)
    (;u, domain,innergrid,L_boundary_x, L_boundary_y,flow_grid, pde)
end

end # module SplineSquareExample
