using Plots; plotly()
using LinearAlgebra
using SparseArrays
using Arpack
using Kronecker
using PlotlyJS

#----------------------------------------------------------------------------------------
#Se define el tamaño del grid (caja) asi como el numero de puntos a utilizar
N = 15
x = 1:N 
y = 1:N
z = 1:N

diff1(M) = [ [1.0 zeros(1,M-1)]; diagm(1 => ones(M-1)) - I(M) ]
sdiff1(M) = sparse(diff1(M))

# Se crean los meshgrids para x, y y z --------------------------------------------------
X = getindex.(Iterators.product(x, y, z), 1)
Y = getindex.(Iterators.product(x, y, z), 2)
Z = getindex.(Iterators.product(x, y, z), 3)

#Se define la energia potencial que se va a utilizar dentro de la caja ------------------
potencial(x,y,z) = 0*x
V = potencial(X,Y,Z)

# Se crea la matriz para las segundas derivadas y se forma el laplaciano ----------------
function Laplacian(Nx,Ny,Nz,Lx,Ly,Lz)
    dx = Lx / (Nx+1)
    dy = Ly / (Ny+1)
    dz = Lz / (Nz+1)
    Dx = sdiff1(Nx) / dx
    Dy = sdiff1(Ny) / dy
    Dz = sdiff1(Nz) / dz
    Ax = Dx' * Dx
    Ay = Dy' * Dy
    Az = Dz' * Dz
    MI = sparse(I(Nx))
    return (Ax ⊗ MI ⊗ MI) + (MI ⊗ Ay ⊗ MI) + (MI ⊗ MI ⊗ Az)
end

# Se forman las matrices del laplaciano y el potencial ----------------------------------
T = -1/2 .* sparse(Laplacian(N,N,N,1,1,1))
U = sparse(Diagonal(reshape(V,N^3)))
H = T + U #Matriz del Hmailtoniano

# Se hallan los vectores propios (autofunciones) y los valores propios (autoenergias) ---
λ, ϕ = eigs(H; nev=10, which=:SM)

# Se plotean los vectores propios (autofunciones) ---------------------------------------
function get_e(n)
    return reshape(ϕ[:,n],(N,N,N))
end

values = get_e(5).^2

PlotlyJS.plot(volume(
    x=X[:],
    y=Y[:],
    z=Z[:],
    value=values[:],
    opacity=0.1, 
    surface_count=17, 
))