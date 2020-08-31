using FrameFunApplications, ProgressMeter
prNγ_itr = Iterators.product(2:5,.05:.05:.25, 50:5:300, 1:5)
progress = Progres(prNγ_itr)

Threads.@threads for (p,r,N,γ) in prNγ_itr
    write_circle_example(N=N,γ=γ,r=r,p=p)
    next!(progress)
end
