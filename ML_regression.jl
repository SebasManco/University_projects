using StatsPlots
using RDatasets
using Query
using Statistics

iris = dataset("datasets","iris")

iris_reduced = @from i in iris begin
    @where i.Species != "setosa"
    @select{
        i.PetalLength,
        i.PetalWidth,
        intercept = 1,
        i.Species,
        label = i.Species == "virginica",
    }
    @collect DataFrame
end

X = Matrix(iris_reduced[:,1:3])
y = iris_reduced.label

@df iris_reduced scatter(
    :PetalLength,
    :PetalWidth;
    group = :Species,
    xlabel = "Petal Length",
    ylabel = "Petal Width",
    legend = :topleft,
    dpi=400
)

σ(z) = 1/(1 + exp(-z))

size(X,2)

function log_reg(X,y,w;max_iter=100,tol=1e-6)
    X_mult = [row*row' for row in eachrow(X)]
    for i in 1:max_iter
        y_hat = σ.(X*w)
        grad = X'*(y_hat.-y) / size(X,1)
        hess = y_hat.*(1 .-y_hat).*X_mult |> mean
        w -= hess \ grad
    end
    return w
end

w = log_reg(X,y,zeros(size(X,2)))

separ(x::Real,w) = (-w[3]-w[1]*x)/w[2]

xlims = extrema(iris_reduced.PetalLength) .+ [-0.1,0.1]
ylims = extrema(iris_reduced.PetalWidth) .+ [-0.1,0.1]

@df iris_reduced scatter(
    :PetalLength,
    :PetalWidth;
    group = :Species,
    xlabel = "PetalLength",
    ylabel = "Petal Width",
    legend = :topleft,
    xlims,
    ylims,
)

plot!(xlims, x->separ(x,w);label="separation",line=(:black,3),dpi=400)
savefig("/home/sebastian/Programación/proyectos/plots/plot1.png")


## Flux -------------------------------------------------------------------------------
using Flux

f(x) = 3x^2 + 2x + 1
df(x) = gradient(f,x)[1]

df(2)

d2f(x) = gradient(df,x)[1]
d2f(2)
#simple model
