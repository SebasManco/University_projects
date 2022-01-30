using LinearAlgebra
using SparseArrays
using Plots; gr()
using Arpack
using BenchmarkTools

function schrodinger(gridSize::Int64, eig_evc)
    N = gridSize
    x = 1:N
    y = 1:N
    diff1(M) = [ [1.0 zeros(1,M-1)]; diagm(1 => ones(M-1)) - I(M) ]
    sdiff1(M) = sparse(diff1(M))
    
    #se crean los meshgrids para x y para y--------------------------------------------------------
    X = x' .* ones(N)
    Y = ones(N)' .* y
    
    ##construimos la matriz con la energia potencial-----------------------------------------------
    potencial(x,y) = 0*x
    V = potencial(X,Y)
    
    ## ahora construyamos la matriz del laplaciano ------------------------------------------------
    function Laplacian(Nx,Ny,Lx,Ly)
        dx = Lx / (Nx+1)
        dy = Ly / (Ny+1)
        Dx = sdiff1(Nx) / dx
        Dy = sdiff1(Ny) / dy
        Ax = Dx' * Dx
        Ay = Dy' * Dy
        return kron(sparse(I(Ny)),Ax) + kron(Ay,sparse(I(Nx)))
    end
    
    
    ## Construimos la matriz del hamiltoniano y se digonaliza -------------------------------------
    T = -1/2 .* Laplacian(N,N,1,1)
    U = sparse(Diagonal(reshape(V,N^2)))
    H = T + U
    
    λ, ϕ = eigs(H; nev=10, which=:SM)
    
    ## graficamos el resultado----------------------------------------------------------------------
    function get_e(n)
        return reshape(ϕ[:,n],(N,N))
    end
    
    return get_e(eig_evc).^2
end

heatmap(schrodinger(100,1),colorbar=false,axis=false,ticks=false)
print("hola")
