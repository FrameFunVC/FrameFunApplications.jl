module SplineCircleExample
using FrameFunApplications, BasisFunctions, CompactTranslatesDict, DomainSets, StaticArrays, FrameFun, FrameFunTranslates
using SymbolicDifferentialOperators: Δ, I, δx, δy

using DelimitedFiles

_check_grid(g, domain) =
    @assert(all(g .∈ Ref(domain)))

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

const DEFAULT_OPTS = (;N=20,γ=4,v=1,r=.05,a=.2,b=.8,c=.2,d=.8,p=3,verbose=false)
export arg
arg(a, opts) =
    get(opts,a,DEFAULT_OPTS[a])

export spline_basis
spline_basis(;opts...) =
    BSplineTranslatesBasis(arg(:N,opts),arg(:p,opts))^2
export inwendige_rechthoek
inwendige_rechthoek(;opts...) =
    innerbox = (prevfloat(arg(:a,opts))..nextfloat(arg(:b,opts)))×(prevfloat(arg(:c,opts))..nextfloat(arg(:d,opts)))
export schijf
schijf(;opts...) =
    prevfloat(arg(:r,opts))*disk().+[.5,.5]
export domein
domein(;opts...) =
    domain = setdiff(inwendige_rechthoek(;opts...),schijf(;opts...))
export uitwendig_rooster
uitwendig_rooster(;opts...) =
    PeriodicEquispacedGrid(arg(:γ,opts)*arg(:N,opts),0,1)^2
export inwendig_rooster
inwendig_rooster(;opts...) =
    subgrid(uitwendig_rooster(;opts...),inwendige_rechthoek(;opts...))
export collocatie_rooster
collocatie_rooster(;opts...) =
    subgrid(uitwendig_rooster(;opts...),domein(;opts...))
export schijf_rooster
schijf_rooster(;opts...) =
    ScatteredGrid(map(x->SVector(real(x),imag(x)), (.5+.5im) .+ (arg(:r,opts)+eps())*cis.(PeriodicEquispacedGrid(arg(:N,opts),0,2pi))))
export stroming_randvoorwaarde_rooster
stroming_randvoorwaarde_rooster(;opts...) =
    EquispacedGrid(size(inwendig_rooster(;opts...),1),arg(:a,opts),arg(:b,opts))×EquispacedGrid(2,arg(:c,opts),arg(:d,opts))
export wand_randvoorwaarde_rooster
wand_randvoorwaarde_rooster(;opts...) =
    EquispacedGrid(2,arg(:a,opts),arg(:b,opts))×EquispacedGrid(size(inwendig_rooster(;opts...),2),arg(:c,opts),arg(:d,opts))


function _run_circle_example(;opts...)
    # basis = Fourier(N)^2
    basis = spline_basis(;opts...)
    # innerbox = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))
    # _disk = prevfloat(r)*disk().+[.5,.5]
    # domain = setdiff(innerbox,_disk)

    domain = domein(;opts...)
    _disk = schijf(;opts...)

    # outergrid = PeriodicEquispacedGrid(γ*N,0,1)^2
    # innergrid = subgrid(outergrid,innerbox)

    # collocation_grid = subgrid(outergrid,domain)
    collocation_grid = collocatie_rooster(;opts...)
    _check_grid(collocation_grid, domain)
    laplace_rule = PDERule(Δ,basis,collocation_grid,(x,y)->0.)

    disk_grid  = schijf_rooster(;opts...)#boundary(outergrid,_disk)
    _check_grid(disk_grid, domain)
    disk_rule = PDENormalRule(_disk, basis, disk_grid, (x,y)->0)

    # flow_grid = EquispacedGrid(size(innergrid,1),a,b)×EquispacedGrid(2,c,d)
    flow_grid = stroming_randvoorwaarde_rooster(;opts...)
    _check_grid(flow_grid, domain)
    flow_rule = PDERule(δy^1*δx^0,basis,flow_grid,(x,y)->arg(:v,opts))
    flow_rule_tan = PDERule(δy^0*δx^1,basis,flow_grid,(x,y)->0)

    semi_flow_grid = EquispacedGrid(size(inwendig_rooster(;opts...),1),arg(:a,opts),arg(:b,opts))×EquispacedGrid(1,arg(:c,opts),arg(:c,opts))
    _check_grid(semi_flow_grid, domain)
    flow_rule_homo = PDERule(I,basis,semi_flow_grid,(x,y)->0)

    homogeneous_grid = wand_randvoorwaarde_rooster(;opts) # EquispacedGrid(2,a,b)×EquispacedGrid(size(innergrid,2),c,d)
    _check_grid(homogeneous_grid, domain)
    homogeneous_rule = PDERule(δy^0*δx,basis,homogeneous_grid,(x,y)->0.)

    pde = PDE(laplace_rule,disk_rule,flow_rule,flow_rule_tan,homogeneous_rule,flow_rule_homo)

    u = pdesolve(pde;directsolver=sparseQR_solver)

    residuals = _residuals(pde,u)

    if arg(:verbose,opts)
        @info "============================="
        @info "testing solution"
        @info "Norm coefficients" residuals.coefficient_norm_inf
        @info "Residual" residuals.abs_residual_inf, residuals.rel_residual_l2
        @info "============================="
    end

    (;u, domain, pde, residuals...)
end

function _residuals(pde,u)
    abs_residual_inf = norm(pde.lhs*coefficients(u)-pde.rhs,Inf)
    abs_residual_l2 = norm(pde.lhs*coefficients(u)-pde.rhs)
    coefficient_norm_inf = norm(coefficients(u),Inf)
    coefficient_norm_l2 = norm(coefficients(u))
    rel_residual_inf = abs_residual_inf / coefficient_norm_inf
    rel_residual_l2 = abs_residual_l2 / coefficient_norm_l2
    (;abs_residual_inf,abs_residual_l2,coefficient_norm_inf,coefficient_norm_l2,rel_residual_inf,rel_residual_l2)
end


export run_circle_example
function run_circle_example(;opts...)
    u = _run_circle_example(;opts...)
    (;u...)
end


end
