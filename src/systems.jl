#################
##### Systems
#################
# The systems are all defined here, because it makes saving and loading results much easier (with e.g. JLD2) if they are part of the library as well.
# However, one can always define the equations and parameters outside the library as well.
# TO-DO: also defining all the jacobians would improve the performance of some of the solvers used
#################

using Parameters

abstract type DEParameters end

@with_kw struct logistic_parameters <: DEParameters  # with_kw enables default values and keyword initializing and more importantly a very convinient reconstruct routine!
    r::Float64
end

function logistic(u_next, u, p::logistic_parameters, t)
    u_next[1] = p.r * u[1] * ( 1. - u[1])
end

@with_kw struct henon_parameters <: DEParameters
    a::Float64
    b::Float64
end

function henon(u_next, u, p::henon_parameters, t)
    xnext[1] = 1.0 - p.a*x[1]^2 + x[2]
    xnext[2] = p.b*x[1]
end

@with_kw struct kuramoto_parameters <: DEParameters
    K::Number # coupling strength
    w::Array{Float64} # eigenfrequencies
    N::Int  # Number of Oscillators
end

# fully connected kuramoto model
function kuramoto(du, u, p::kuramoto_parameters, t)
    for i=1:p.N
        du[i] = 0.
        for j=1:p.N
            du[i] += sin.(u[j] - u[i])
        end
        du[i] *= p.K/p.N
        du[i] += p.w[i]
    end
end

@with_kw struct kuramoto_network_parameters <: DEParameters # unfortunatly one can only subtype from abstract types in Julia
    K::Number # coupling strength
    w::Array{Float64} # eigenfrequencies
    N::Int  # Number of Oscillators
    A # adjacency_matrix, either sparse or dense
end

# kuramoto model on a network
function kuramoto_network(du, u, p::kuramoto_network_parameters, t)
    for i=1:p.N
        du[i] = 0.
        for j=1:p.N
            if p.A[i,j]!=0
                du[i] += p.A[i,j]*sin.(u[j] - u[i])
            end
        end
        du[i] *= p.K/p.N
        du[i] += p.w[i]
    end
end

@with_kw struct second_order_kuramoto_chain_parameters <: DEParameters
    systemsize::Int
    damping::Float64
    coupling::Float64
    drive::Array{Float64}
    perturbation::Array{Float64}

    function parameters(sys_size::Int)
        new(sys_size,
            0.1,
            8.,
            [isodd(i) ? +1. : -1. for i = 1:sys_size],
            [i < sys_size + 1 ? 0. : 1. for i = 1:2*sys_size])
    end

    function parameters(p::second_order_kuramoto_chain_parameters; sigma::Float64=1.)
        new(p.systemsize,
            p.damping,
            p.coupling,
            p.drive,
            [i < p.systemsize + 1 ? 0. : sigma for i = 1:2*p.systemsize])
    end
end

function second_order_kuramoto_chain(du, u, p::second_order_kuramoto_chain_parameters, t)
    du[1:p.systemsize] .= u[1 + p.systemsize:2*p.systemsize]
    du[p.systemsize+1] = p.drive[1] - p.damping * u[p.systemsize+1] - p.coupling * (sin(u[1] - u[2]))
    for i in 2:p.systemsize-1
        du[p.systemsize + i] = p.drive[i] - p.damping * u[p.systemsize + i] - p.coupling * (sin(u[i] - u[i-1]) + sin(u[i] - u[i+1]))
    end
    du[2*p.systemsize] = p.drive[p.systemsize] - p.damping * u[2*p.systemsize] - p.coupling * (sin(u[p.systemsize] - u[p.systemsize-1]))
end

# order_parameter
# Kuratomo Order Parameter
function order_parameter(u::AbstractArray, N::Int)
    1./N*sqrt((sum(sin.(u)))^2+(sum(cos.(u)))^2)
end

# roessler parameters for Roessler Network
@with_kw struct roessler_parameters <: DEParameters
    a::Array{Float64}
    b::Array{Float64}
    c::Array{Float64}
    K::Float64 # coupling
    L::AbstractSparseMatrix # Laplacian
    N::Int # number of nodes
end

function roessler_parameters(a, b, c, K, k::Int64, p::Float64, N)
    G = watts_strogatz(N, k, p)
    L = laplacian_matrix(G)
    roessler_parameters(a,b,c,K,L,N)
end

function roessler_network(du, u, p::roessler_parameters, t)
    for i=1:p.N
        ii = 3*i
        du[ii-2] = - u[ii-1] - u[ii]
        for j=1:p.N
            if p.L[i,j]!=0
                du[ii-2] -= p.K*p.L[i,j]*du[ii-2]
            end
        end
        du[ii-1] = u[ii-2] + p.a[i]*u[ii-1]
        du[ii] = p.b[i] + u[ii]*(du[ii-2] - p.c[i])
    end
end
"""
function roessler_network(::Type{Val{:jac}},J,u,p::roessler_parameters,t)
    for i=1:p.N
        ii=3*i
        for j=1:p.N
            jj=3*i
            if i==j
                J[ii-2,jj-2] = -p.K*p.L[i,j]
                J[ii-2,jj-1] = -1.
                J[ii-2,jj] = -1.
                J[ii-1,jj-2] = 1.
                J[ii-1,jj-1] = p.a[i]
                J[ii-1,jj] = 0
                J[ii,jj-2] = u[ii]
                J[ii,jj-1] = 0
                J[ii,jj] = u[ii-2] - p.c[i]
            else
                J[ii-2,jj-2] = -p.K*p.L[i,j]
                J[ii-2,jj-1] = 0
                J[ii-2,jj] = 0
                J[ii-1,jj-2] = 0
                J[ii-1,jj-1] = 0
                J[ii-1,jj] = 0
                J[ii,jj-2] = 0
                J[ii,jj-1] = 0
                J[ii,jj] = 0
            end
        end
    end
    nothing
end
"""

@with_kw struct lotka_volterra_parameters <: DEParameters
    a::Matrix
    b::AbstractArray
    N::Int
end

function lotka_volterra(du, u, p::lotka_volterra_parameters, t)
    for i=1:p.N
        du[i] = 1.
        for j=1:p.N
            du[i] -= p.a[i,j]*p.N
        end
        du[i] *= p.b[i]*p.N
    end
end
