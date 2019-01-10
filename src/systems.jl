#################
##### Systems
#################
# The systems are all defined here, because it makes saving and loading results much easier (with e.g. JLD2) if they are part of the library as well.
# However, one can always define the equations and parameters outside the library as well.
# TO-DO: also defining all the jacobians would improve the performance of some of the solvers used
#################

using Parameters
"""
    DEParameters

Abstract type for all parameter types. If you define your own systems the parameters have to have this as a supertype.
"""
abstract type DEParameters end

"""
     logistic_parameters

Parameters of the logistic map. Its function is [`logistic`](@ref).

Fields:
*`r::Float64`
"""
@with_kw struct logistic_parameters <: DEParameters  # with_kw enables default values and keyword initializing and more importantly a very convinient reconstruct routine!
    r::Float64
end

"""
    logistic(u_next, u, p::logistic_parameters, t)

Logistic Map: ``x_{n+1} = rx_n (1 - x_n)``
"""
function logistic(u_next, u, p::logistic_parameters, t)
    u_next[1] = p.r * u[1] * ( 1. - u[1])
end

"""
    henon_parameters

Parameters of the Henon map. Its function is [`henon`](@ref).

Fields:
*`a::Float64`
*`b::Float64`
"""
@with_kw struct henon_parameters <: DEParameters
    a::Float64
    b::Float64
end

"""
    henon(u_next, u, p::henon_parameters, t)

Henon map.
``x_{n+1} = 1 - a x_{n}^2 + y_{n}``
``y_{n+1} = b x_{n}``
"""
function henon(u_next, u, p::henon_parameters, t)
    xnext[1] = 1.0 - p.a*x[1]^2 + x[2]
    xnext[2] = p.b*x[1]
end

"""
    kuramoto_parameters

Parameters of a first order Kuramoto system with all-to-all coupling.
Fields:
* `K::Number`: Coupling Strength
* `w::Array{Float64}`: Eigenfrequencies
* `N::Int`: Number of Oscillators
"""
@with_kw struct kuramoto_parameters <: DEParameters
    K::Number # coupling strength
    w::Array{Float64} # eigenfrequencies
    N::Int  # Number of Oscillators
end

"""
    kuramoto(du, u, p::kuramoto_parameters, t)

First order Kuramoto system with all-to-all coupling.
``\\dot{\\theta}_i = w_i + K/N \\sum_{i} \\sin(\\theta_j - \\theta_i)``
"""
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

"""
    kuramoto_network_parameters

Parameters of a first order Kuramoto system on a network.
Fields:
* `K::Number`: Coupling Strength
* `w::Array{Float64}`: Eigenfrequencies
* `N::Int`: Number of Oscillators
* `A`: Adjacency matrix
"""
@with_kw struct kuramoto_network_parameters <: DEParameters # unfortunatly one can only subtype from abstract types in Julia
    K::Number # coupling strength
    w::Array{Float64} # eigenfrequencies
    N::Int  # Number of Oscillators
    A # adjacency_matrix, either sparse or dense
end

"""
    kuramoto_network(du, u, p::kuramoto_parameters, t)

First order Kuramoto system on a network.
``\\dot{\\theta}_i = w_i + K/N \\sum_{i} A_{ij}\\sin(\\theta_j - \\theta_i)``
"""
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

"""
    second_order_kuramoto_chain_parameters

Fields:
* `systemsize::Int`, number of oscillators
* `damping::Float64`, Damping, also referred to as ``\\alpha``
* `coupling::Float64`, Coupling strength, also referred to as ``\\lambda``
* `drive::Array{Float64}`, external driving, also referred to as ``\\Omega``
"""
@with_kw struct second_order_kuramoto_chain_parameters <: DEParameters
    systemsize::Int
    damping::Float64 = 0.1
    coupling::Float64 = 8.
    drive::Array{Float64}
end

"""
    second_order_kuramoto_chain(du, u, p::kuramoto_parameters, t)

Second order Kuramoto system on a chain.

``\\dot{\\theta}_i = w_i``
``\\dot{\\omega} = \\Omega_i - \\alpha\\omega + \\lambda\\sum_{j=1}^N A_{ij} sin(\\theta_j - \\theta_i)``

``A_{ij} = 1`` if and only if ``|i-j|=1``, otherwise ``A_{ij}=0``
"""
function second_order_kuramoto_chain(du, u, p::second_order_kuramoto_chain_parameters, t)
    du[1:p.systemsize] .= u[1 + p.systemsize:2*p.systemsize]
    du[p.systemsize+1] = p.drive[1] - p.damping * u[p.systemsize+1] - p.coupling * (sin(u[1] - u[2]))
    for i in 2:p.systemsize-1
        du[p.systemsize + i] = p.drive[i] - p.damping * u[p.systemsize + i] - p.coupling * (sin(u[i] - u[i-1]) + sin(u[i] - u[i+1]))
    end
    du[2*p.systemsize] = p.drive[p.systemsize] - p.damping * u[2*p.systemsize] - p.coupling * (sin(u[p.systemsize] - u[p.systemsize-1]))
end

"""
    non_local_kuramoto_ring_parameters <: DEParameters

* `N::Integer`: Number of Oscillators
* `fak::Float64`: ``\\frac{2\\pi}{n}``
* `omega_0::Number`: eigenfrequency of all oscillators
* `phase_delay`: Phase Delay ``\\alpha``
* `coupling_function`: Coupling function ``G(x)``, signature (x::Number)-> value::Number
"""
@with_kw struct non_local_kuramoto_ring_parameters <: DEParameters
    N::Integer
    fak::Float64 # 2pi/N
    omega_0::Number
    phase_delay::Number
    coupling_function::Function

end
non_local_kuramoto_ring_parameters(N::Integer, omega_0::Number,phase_delay::Number, coupling_function::Function) = non_local_kuramoto_ring_parameters(N, (2pi)/N, omega_0, phase_delay, coupling_function)

"""
    non_local_kuramoto_ring(du, u, p::non_local_kuramoto_ring_parameters, t)

First order Kuramoto oscillators on a ring with non-local coupling.

``\\frac{\\theta_k}{dt} = \\omega_0 - \\sum_{j=1}^N G_1\\left(\\frac{2\\pi}{N}(k-j)\\right)\\sin\\left(\\theta_k(t) - \\theta_j(k) + \\alpha\\right)``
"""
function non_local_kuramoto_ring(du, u, p::non_local_kuramoto_ring_parameters, t)
    for istep=1:p.N
        du[istep] = 0
        for jstep=1:p.N
            du[istep] -= p.coupling_function(p.fak*(istep - jstep))*sin(u[istep] - u[jstep] + p.phase_delay)
        end
        du[istep] /= p.fak
        du[istep] += p.omega_0
    end
end

"""
    order_parameter(u::AbstractArray, N::Int)

Order Parameter of a Kuramoto System
"""
function order_parameter(u::AbstractArray, N::Int)
    1. /N*sqrt((sum(sin.(u)))^2+(sum(cos.(u)))^2)
end

"""
    roessler_parameters <: DEParameters

Parameters of a Roessler network

* `a::Array{Float64}`: `a` Parameters of all oscillators.
* `b::Array{Float64}`: `b` Parameters of all oscillators.
* `c::Array{Float64}`: `c` Parameters of all oscillators.
* `K::Float64`: Coupling Strength
* `L::Array{Float64}`: Laplacian matrix of the network
* `N::Int`: Number of nodes/oscilattors

    roessler_parameters(a, b, c, K, k::Int64, p::Float64, N)

Generates a set of roessler_parameters with a Watts Strogatz random Network with mean degreee `k` and rewiring probability `p`
"""
@with_kw struct roessler_parameters <: DEParameters
    a::Array{Float64}
    b::Array{Float64}
    c::Array{Float64}
    K::Float64 # coupling
    L::Array{Float64} # Laplacian
    N::Int # number of nodes
end

function roessler_parameters(a, b, c, K, k::Int64, p::Float64, N)
    G = watts_strogatz(N, k, p)
    L = laplacian_matrix(G)
    roessler_parameters(a,b,c,K,L,N)
end

"""
    roessler_network(du, u, p::roessler_parameters, t)

N Roessler Parameters coupled on their x-component
"""
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

"""
    lotka_volterra(du, u, p::lotka_volterra_parameters, t)

N-dimensional Lotka-Volterra system

NOT YET PROPERLY IMPLEMENTED
"""
function lotka_volterra(du, u, p::lotka_volterra_parameters, t)
    for i=1:p.N
        du[i] = 1.
        for j=1:p.N
            du[i] -= p.a[i,j]*p.N
        end
        du[i] *= p.b[i]*p.N
    end
end
