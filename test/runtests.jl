using FrameFunApplications, BasisFunctions, Test, CompactTranslatesDict, DomainSets, StaticArrays, FrameFun, FrameFunTranslates
using SymbolicDifferentialOperators: Δ, I, δx, δy

# Exact x derivative is 10
@testset begin
    B = BSplineTranslatesBasis(10,2)^2
    D = (0.2..0.8)×(.4..0.8)
    g = interpolation_grid(B)
    x = subgrid(g,D)
    C_e = (0.0:0.1:0.9).*ones(10)'
    u_e = expansion(B,C_e)
    @test all((δx*δy^0)(u_e)(x) .≈ 10)
    @test all((δy^0*δx)(u_e)(x) .≈ 10)
    @test all((δx)(u_e)(x) .≈ 10)
    @test all((δy*δx^0)(u_e)(x).+1 .≈ 1)
    @test all((δx^0*δy)(u_e)(x).+1 .≈ 1)
    @test all(Δ(u_e)(x).+1 .≈ 1)
    # heatmap(u_e,xlabel="x",ylabel="y") # Gradient from left to right
    r = PDERule(Δ,B,x,(x,y)->0.)
    b = PDEBlock(r)
    @test norm(b.lhs*C_e).+1≈1
    rb1 = PDERule(δx^1*δy^0,B,x,(x,y)->10.)
    b = PDEBlock(rb1)
    @test all(b.lhs*C_e .≈ 10)
    rb2 = PDERule(δx^0*δy,B,x,(x,y)->0.)
    b = PDEBlock(rb2)
    @test all(b.lhs*C_e.+1 .≈ 1)

    pde = PDE(r,rb1,rb2)
    @test norm(pde.lhs*C_e-pde.rhs).+1≈1

    pde = PDE(r,rb1,rb2)
    u = pdesolve(pde)

    @test all((δx*δy^0)(u)(x) .≈ 10)
    @test all((δy^0*δx)(u)(x) .≈ 10)
    @test all((δx)(u)(x) .≈ 10)
    @test all((δy*δx^0)(u)(x).+1 .≈ 1)
    @test all((δx^0*δy)(u)(x).+1 .≈ 1)
    @test all(Δ(u)(x).+1 .≈ 1)
end


# Exact y derivative is 10
@testset begin
    B = BSplineTranslatesBasis(10,2)^2
    D = (0.2..0.8)×(.4..0.8)
    g = interpolation_grid(B)
    x = subgrid(g,D)
    C_e = ((0.0:0.1:0.9).*ones(10)')'
    u_e = expansion(B,C_e)
    @test all((δx^0*δy)(u_e)(x) .≈ 10)
    @test all((δy*δx^0)(u_e)(x) .≈ 10)
    @test all((δy)(u_e)(x) .≈ 0)
    @test all((δx*δy^0)(u_e)(x).+1 .≈ 1)
    @test all((δy^0*δx)(u_e)(x).+1 .≈ 1)
    @test all(Δ(u_e)(x).+1 .≈ 1)
    # heatmap(u_e,xlabel="x",ylabel="y") # Gradient from left to right
    r = PDERule(Δ,B,x,(x,y)->0.)
    b = PDEBlock(r)
    @test norm(b.lhs*C_e).+1≈1
    rb1 = PDERule(δy^1*δx^0,B,x,(x,y)->10.)
    b = PDEBlock(rb1)
    @test all(b.lhs*C_e .≈ 10)
    rb2 = PDERule(δy^0*δx,B,x,(x,y)->0.)
    b = PDEBlock(rb2)
    @test all(b.lhs*C_e.+1 .≈ 1)

    pde = PDE(r,rb1,rb2)
    @test norm(pde.lhs*C_e-pde.rhs).+1≈1

    pde = PDE(r,rb1,rb2)
    u = pdesolve(pde)

    @test all((δy*δx^0)(u)(x) .≈ 10)
    @test all((δx^0*δy)(u)(x) .≈ 10)
    @test all((δx)(u)(x).+1 .≈ 1)
    @test all((δx*δy^0)(u)(x).+1 .≈ 1)
    @test all((δy^0*δx)(u)(x).+1 .≈ 1)
    @test all(Δ(u)(x).+1 .≈ 1)
end

# Exact y derivative is 10 with boundary conditions
@testset begin
    B = BSplineTranslatesBasis(20,2)^2
    D = (prevfloat(0.2)..nextfloat(0.8))×(prevfloat(.4)..nextfloat(0.8))
    g = PeriodicEquispacedGrid(20,0,1)^2
    x = subgrid(g,D)
    C_e = ((0.0:0.05:0.95).*ones(20)')'./20
    u_e = expansion(B,C_e)
    @test all((δx^0*δy)(u_e)(x) .≈ 1)
    @test all((δy*δx^0)(u_e)(x) .≈ 1)
    @test all((δy)(u_e)(x) .≈ 0)
    @test all((δx*δy^0)(u_e)(x).+1 .≈ 1)
    @test all((δy^0*δx)(u_e)(x).+1 .≈ 1)
    @test all(Δ(u_e)(x).+1 .≈ 1)
    # heatmap(u_e,xlabel="x",ylabel="y") # Gradient from left to right
    r = PDERule(Δ,B,x,(x,y)->0.)
    b = PDEBlock(r)
    @test norm(b.lhs*C_e).+1≈1
    g1 = EquispacedGrid(size(x,1),.2,.8)×EquispacedGrid(2,.4,.8)
    rb1 = PDERule(δy^1*δx^0,B,g1,(x,y)->1.)
    b = PDEBlock(rb1)
    @test all(b.lhs*C_e .≈ 1)
    g2 = EquispacedGrid(2,.2,.8)×EquispacedGrid(size(x,1),.4,.8 )
    rb2 = PDERule(δy^0*δx,B,g2,(x,y)->0.)
    b = PDEBlock(rb2)
    @test all(b.lhs*C_e.+1 .≈ 1)

    pde = PDE(r,rb1,rb2)
    @test norm(pde.lhs*C_e-pde.rhs).+1≈1

    pde = PDE(r,rb1,rb2)
    u = pdesolve(pde)

    @test all((δy*δx^0)(u)(x) .≈ 1)
    @test all((δx^0*δy)(u)(x) .≈ 1)
    @test all((δx)(u)(x).+1 .≈ 1)
    @test all((δx*δy^0)(u)(x).+1 .≈ 1)
    @test all((δy^0*δx)(u)(x).+1 .≈ 1)
    @test all(Δ(u)(x).+1 .≈ 1)
end

# Exact y derivative is 10 with boundary conditions and oversampling
@testset begin
    B = BSplineTranslatesBasis(20,3)^2
    D = (prevfloat(0.2)..nextfloat(0.8))×(prevfloat(.4)..nextfloat(0.8))
    g = PeriodicEquispacedGrid(40,0,1)^2
    x = subgrid(g,D)
    C_e = ((0.0:0.05:0.95).*ones(20)')'./20
    u_e = expansion(B,C_e)
    @test all((δx^0*δy)(u_e)(x) .≈ 1)
    @test all((δy*δx^0)(u_e)(x) .≈ 1)
    @test all((δy)(u_e)(x).+1 .≈ 1)
    @test all((δx*δy^0)(u_e)(x).+1 .≈ 1)
    @test all((δy^0*δx)(u_e)(x).+1 .≈ 1)
    @test all(Δ(u_e)(x).+1 .≈ 1)


    # heatmap(u_e,xlabel="x",ylabel="y") # Gradient from left to right
    r = PDERule(Δ,B,x,(x,y)->0.)
    b = PDEBlock(r)
    @test norm(b.lhs*C_e).+1≈1
    g1 = EquispacedGrid(size(x,1),.2,.8)×EquispacedGrid(2,.4,.8)
    rb1 = PDERule(δy^1*δx^0,B,g1,(x,y)->1.)
    b = PDEBlock(rb1)
    @test all(b.lhs*C_e .≈ 1)
    g2 = EquispacedGrid(2,.2,.8)×EquispacedGrid(size(x,1),.4,.8 )
    rb2 = PDERule(δy^0*δx,B,g2,(x,y)->0.)
    b = PDEBlock(rb2)
    @test all(b.lhs*C_e.+1 .≈ 1)

    pde = PDE(r,rb1,rb2)
    @test norm(pde.lhs*C_e-pde.rhs).+1≈1

    pde = PDE(r,rb1,rb2)
    u = pdesolve(pde)

    @test all((δy*δx^0)(u)(x) .≈ 1)
    @test all((δx^0*δy)(u)(x) .≈ 1)
    @test all((δx)(u)(x).+1 .≈ 1)
    @test all((δx*δy^0)(u)(x).+1 .≈ 1)
    @test all((δy^0*δx)(u)(x).+1 .≈ 1)
    @test all(Δ(u)(x).+1 .≈ 1)
end


# function
@testset begin
    N = 20
    γ = 2
    a,b,c,d = .2,.8,.2,.8
    B = Fourier(N)^2#
    # B = BSplineTranslatesBasis(N,3)^2
    DΞ = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))
    C = .2disk().+[.5,.5]
    D = setdiff(DΞ,C)
    g = PeriodicEquispacedGrid(γ*N,0,1)^2
    gΞ = subgrid(g,DΞ)
    x = subgrid(g,D)
    r = PDERule(Δ,B,x,(x,y)->0.)
    gc  = boundary(g,C)
    rc = PDENormalRule(C, B, gc, (x,y)->0)
    g_flow = EquispacedGrid(size(gΞ,1),a,b)×EquispacedGrid(2,c,d)
    v = 1
    # Flow along y direction
    rb1 = PDERule(δy^1*δx^0,B,g_flow,(x,y)->v)
    g_homo = EquispacedGrid(2,a,b)×EquispacedGrid(size(gΞ,1),c,d)
    rb2 = PDERule(δy^0*δx,B,g_homo,(x,y)->0.)
    pde = PDE(r,rc,rb1,rb2)
    u = pdesolve(pde;directsolver=:qr)
    coefs = coefficients(u)
    # u = pdesolve(pde)
    @show norm((δx^0*δy)(u)(g_flow) .- v,Inf)
    @show norm((δx*δy^0)(u)(g_homo),Inf)
    @show norm(Δ(u)(x),Inf)
    @test norm((δy*δx^0)(u)(g_flow) .- v,Inf) < .02
    @test norm((δx*δy^0)(u)(g_homo),Inf) < .02
    @test norm(Δ(u)(x),Inf) < .02
end




function _run_L_example(;N=20,γ=4,v=1,r=.05,a=.2,b=.8,c=.2,d=.8,p=3,verbose=false,directsolver=sparseQR_solver)
    # basis = Fourier(N)^2
    basis = BSplineTranslatesBasis(N,p)^2
    innerbox = (prevfloat(a)..nextfloat(b))×(prevfloat(c)..nextfloat(d))
    centre = (.5,.5)
    # Leave out left (x) top (y) part
    L = setdiff((nextfloat(centre[1]-r)..prevfloat(centre[2]+r))^2,
        (prevfloat(centre[1]-r)..nextfloat(centre[1]))×(prevfloat(centre[2])..nextfloat(centre[2]+r))   )
    domain = setdiff(innerbox,L)

    outergrid = PeriodicEquispacedGrid(γ*N,0,1)^2
    innergrid = subgrid(outergrid,innerbox)

    collocation_grid = subgrid(outergrid,domain)
    laplace_rule = PDERule(Δ,basis,collocation_grid,(x,y)->0.)

    L_boundary_xa =  EquispacedGrid(round(Int,r*γ*N),prevfloat(centre[1]),nextfloat(centre[1]+r))×
                        EquispacedGrid(1,nextfloat(centre[2]+r),nextfloat(centre[2]+r))
    L_boundary_xb =  EquispacedGrid(round(Int,r*γ*N),prevfloat(centre[1]-r),prevfloat(centre[1]))×
                        EquispacedGrid(1,nextfloat(centre[2]),nextfloat(centre[2]))
    L_boundary_xc =  EquispacedGrid(round(Int,2r*γ*N),prevfloat(centre[1]-r),nextfloat(centre[1]+r))×
                        EquispacedGrid(1,prevfloat(centre[2]-r),prevfloat(centre[2]-r))

    L_boundary_x = ScatteredGrid(vcat(L_boundary_xa[:],L_boundary_xb[:],L_boundary_xc[:]))
    @assert all(L_boundary_x .∈ Ref(domain))

    L_boundary_ya =  EquispacedGrid(1,prevfloat(centre[1]-r),prevfloat(centre[1]-r))×
                        EquispacedGrid(round(Int,r*γ*N),prevfloat(centre[2]-r),nextfloat(centre[2]))
    L_boundary_yb =  EquispacedGrid(1,prevfloat(centre[1]),prevfloat(centre[1]))×
                        EquispacedGrid(round(Int,r*γ*N),nextfloat(centre[2]),nextfloat(centre[2]+r))
    L_boundary_yc =  EquispacedGrid(1,nextfloat(centre[1]+r),nextfloat(centre[1]+r))×
                        EquispacedGrid(round(Int,2r*γ*N),prevfloat(centre[2]-r),nextfloat(centre[2]+r))

    L_boundary_y = ScatteredGrid(vcat(L_boundary_ya[:],L_boundary_yb[:],L_boundary_yc[:]))
    @assert all(L_boundary_y .∈ Ref(domain))

    L_rule_x = PDERule(δx^0*δy^1, basis, L_boundary_x, (x,y)->0)
    L_rule_y = PDERule(δx^1*δy^0, basis, L_boundary_y, (x,y)->0)

    homogeneous_grid = EquispacedGrid(2,nextfloat(a),prevfloat(b))×element(innergrid,2)
    @assert all(homogeneous_grid .∈ Ref(domain))
    homogeneous_rule = PDERule(δy^0*δx,basis,homogeneous_grid,(x,y)->0.)

    flow_grid = element(innergrid,1)×EquispacedGrid(2,nextfloat(c),prevfloat(d))
    @assert all(flow_grid .∈ Ref(domain))
    flow_rule = PDERule(δy^1*δx^0,basis,flow_grid,(x,y)->-v)
    flow_rule_tan = PDERule(δy^0*δx^1,basis,flow_grid,(x,y)->0)

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
    # (;u, domain, pde, basis, innerbox,
    #     outergrid, innergrid, collocation_grid, L_boundary_x, L_boundary_y, flow_grid,
    #     semi_flow_grid, homogeneous_grid)
    (;u, domain,innergrid,L_boundary_x, L_boundary_y,flow_grid)
end


sol = FrameFunApplications.SplineSquareExample._run_square_example(;
    N=100,γ=2,r=sqrt(2)/20,verbose=true)
sol = FrameFunApplications.SplineSquareExample._run_square_example_sing(;
    N=100,γ=2,r=sqrt(2)/20,verbose=true)
sol = FrameFunApplications.SplineSquareExample._run_quarter_square_example(;
    N=100,γ=2,r=.3,verbose=true)
sol = FrameFunApplications.SplineSquareExample._run_quarter_square_example_sing(;
    N=100,γ=2,r=.17,verbose=true,sing=false)
