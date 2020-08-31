module SplineCircleExample
using FrameFunApplications, BasisFunctions, CompactTranslatesDict, DomainSets, StaticArrays, FrameFun, FrameFunTranslates
using SymbolicDifferentialOperators: Δ, I, δx, δy

using DelimitedFiles

export write_circle_example
function write_circle_example(;opts...)
    u,D,pde,_ = _run_circle_example(;opts...)

    name = "write_circle_example"
    opts_string = ""
    for s in keys(opts)
        opts_string *= "_$(string(s))$(opts[s])"
    end
    open(name*"coefs"*opts_string,"w") do io
        writedlm(io,coefficients(u)[:])
    end
    open(name*"residual"*opts_string,"w") do io
        writedlm(io,[norm(pde.lhs*coefficients(u)-pde.rhs),])
    end
end

function _run_circle_example(;N=20,γ=4,v=1,r=.05,a=.2,b=.8,c=.2,d=.8,p=3)
    # basis = Fourier(N)^2
    basis = BSplineTranslatesBasis(N,p)^2
    innerbox = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))
    _disk = r*disk().+[.5,.5]
    domain = setdiff(innerbox,_disk)

    outergrid = PeriodicEquispacedGrid(γ*N,0,1)^2
    innergrid = subgrid(outergrid,innerbox)

    collocation_grid = subgrid(outergrid,domain)
    laplace_rule = PDERule(Δ,basis,collocation_grid,(x,y)->0.)

    disk_grid  = boundary(outergrid,_disk)
    disk_rule = PDENormalRule(_disk, basis, disk_grid, (x,y)->0)

    flow_grid = EquispacedGrid(size(innergrid,1),a,b)×EquispacedGrid(2,c,d)
    flow_rule = PDERule(δy^1*δx^0,basis,flow_grid,(x,y)->v)
    flow_rule_tan = PDERule(δy^0*δx^1,basis,flow_grid,(x,y)->0)

    semi_flow_grid = EquispacedGrid(size(innergrid,1),a,b)×EquispacedGrid(1,c,c)
    flow_rule_homo = PDERule(I,basis,semi_flow_grid,(x,y)->0)

    homogeneous_grid = EquispacedGrid(2,a,b)×EquispacedGrid(size(innergrid,2),c,d)
    homogeneous_rule = PDERule(δy^0*δx,basis,homogeneous_grid,(x,y)->0.)

    pde = PDE(laplace_rule,disk_rule,flow_rule,flow_rule_tan,homogeneous_rule,flow_rule_homo)

    u = pdesolve(pde;directsolver=sparseQR_solver)
    u, domain, pde, basis, innerbox, _disk,
        outergrid, innergrid, collocation_grid, disk_grid, flow_grid,
        semi_flow_grid, homogeneous_grid
end

end
