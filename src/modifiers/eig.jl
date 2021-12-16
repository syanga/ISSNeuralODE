using LinearAlgebra
using Zygote: @adjoint

""" maximum eigenvalue of a symmetric square matrix """
function maxeig(symmat::Array{T,2})::T where T
    eig = eigen(Symmetric(symmat))
    max_idx = findmax(eig.values)[2]
    eigmax = eig.values[max_idx]
    return eigmax
end

@adjoint function maxeig(symmat::Array{T,2}) where T
    eig = eigen(Symmetric(symmat))
    max_idx = findmax(eig.values)[2]
    eigmax = eig.values[max_idx]
    vecmax = eig.vectors[:,max_idx]
    jac_adj = vecmax*(vecmax')
    return eigmax, c -> Tuple(jac_adj*c)
end

""" minimum eigenvalue of a symmetric square matrix """
function mineig(symmat::Array{T,2})::T where T
    eig = eigen(Symmetric(symmat))
    min_idx = findmin(eig.values)[2]
    eigmin = eig.values[min_idx]
    return eigmin
end

@adjoint function mineig(symmat::Array{T,2}) where T
    eig = eigen(Symmetric(symmat))
    min_idx = findmin(eig.values)[2]
    eigmin = eig.values[min_idx]
    vecmin = eig.vectors[:,min_idx]
    jac_adj = vecmin*(vecmin')
    return eigmax, c -> Tuple(jac_adj*c)
end

""" eigenvalue regularization """
function softplus_eigreg(symmat)
    eig = eigen(Symmetric(symmat))
    return mean(softplus.(eig.values))
end

@adjoint function softplus_eigreg(symmat)
    eig = eigen(Symmetric(symmat))
    return sum(softplus.(eig.values)), c -> Tuple(sum((sigmoid(eig.values[i])*eig.vectors[:,i]*(eig.vectors[:,i]') for i=1:size(symmat,1)))*c)
end

function relu_eigreg(symmat)
    eig = eigen(Symmetric(symmat))
    return sum(relu.(eig.values))
end

@adjoint function relu_eigreg(symmat)
    eig = eigen(Symmetric(symmat))
    return sum(relu.(eig.values)), c -> Tuple(sum((0.5*(sign(eig.values[i])+1)*eig.vectors[:,i]*(eig.vectors[:,i]') for i=1:size(symmat,1)))*c)
end

""" compute all eigenvalues """
function eigenvalues(symmat)
    eig = eigen(Symmetric(symmat))
    return eig.values
    # return eigvals(Symmetric(symmat))
end

@adjoint function eigenvalues(symmat)
    eig = eigen(Symmetric(symmat))
    return eig.values, c -> Tuple((eig.vectors[:,i]*(eig.vectors[:,i]'))*c for i=1:size(symmat,1))
    # vals = eigvals(Symmetric(symmat))
    # vecs = eigvecs(Symmetric(symmat))
    # return vals, c -> Tuple((vecs[:,i]*(vecs[:,i]'))*c for i=1:size(symmat,1))
end
