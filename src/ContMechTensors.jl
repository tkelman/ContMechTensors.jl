__precompile__()

module ContMechTensors

include("utilities.jl")

immutable InternalError <: Exception end

export SymmetricTensor, Tensor, Vec, SecondOrderTensor, FourthOrderTensor

export otimes, otimes_unsym, ⊗, dcontract, dev, dev!
export extract_components, load_components!, symmetrize, symmetrize!

#########
# Types #
#########
abstract AbstractTensor{order, dim, T <: Real} <: AbstractArray{T, order}

immutable SymmetricTensor{order, dim, T <: Real} <: AbstractTensor{order, dim, T}
   data::Vector{T}
end

immutable Tensor{order, dim, T} <: AbstractTensor{order, dim, T}
   data::Vector{T}
end


###############
# Typealiases #
###############
typealias Vec{dim, T} Tensor{1, dim, T}

typealias AllTensors{dim, T} Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T},
                                   SymmetricTensor{4, dim, T}, Tensor{4, dim, T},
                                   Vec{dim, T}}

typealias SecondOrderTensor{dim, T} Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T}}
typealias FourthOrderTensor{dim, T} Union{SymmetricTensor{4, dim, T}, Tensor{4, dim, T}}

typealias SymmetricTensors{dim, T} Union{SymmetricTensor{2, dim, T}, SymmetricTensor{4, dim, T}}
typealias Tensors{dim, T} Union{Tensor{2, dim, T}, Tensor{4, dim, T},
                                   Vec{dim, T}}

##############################
# Utility/Accessor Functions #
##############################

get_data(t::AbstractTensor) = t.data

@inline function n_independent_components(dim, issym)
    dim == 1 && return 1
    if issym
        dim == 2 && return 3
        dim == 3 && return 6
    else
        dim == 2 && return 4
        dim == 3 && return 9
    end
    return -1
end


get_main_type{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}) = SymmetricTensor
get_main_type{order, dim, T}(::Type{Tensor{order, dim, T}}) = Tensor
get_main_type{order, dim}(::Type{SymmetricTensor{order, dim}}) = SymmetricTensor
get_main_type{order, dim}(::Type{Tensor{order, dim}}) = Tensor

get_base{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}) = SymmetricTensor{order, dim}
get_base{order, dim, T}(::Type{Tensor{order, dim, T}}) = Tensor{order, dim}

get_lower_order_tensor{dim, T}(S::Type{SymmetricTensor{2, dim, T}}) = SymmetricTensor{2, dim}
get_lower_order_tensor{dim, T}(S::Type{Tensor{2, dim, T}}) = Tensor{2, dim}
get_lower_order_tensor{dim, T}(::Type{SymmetricTensor{4, dim, T}}) = SymmetricTensor{2, dim}
get_lower_order_tensor{dim, T}(::Type{Tensor{4, dim, T}}) = Tensor{2, dim}


n_components{dim}(::Type{SymmetricTensor{2, dim}}) = dim*dim - div((dim-1)*dim, 2)
function n_components{dim}(::Type{SymmetricTensor{4, dim}})
    n = n_components(SymmetricTensor{2, dim})
    return n*n
end

n_components{order, dim}(::Type{Tensor{order, dim}}) = dim^order


############################
# Abstract Array interface #
############################
Base.linearindexing(::SymmetricTensor) = Base.LinearSlow()
Base.linearindexing(::Tensors) = Base.LinearFast()

# Size #
########

Base.size(::Vec{1}) = (1,)
Base.size(::Vec{2}) = (2,)
Base.size(::Vec{3}) = (3,)

Base.size(::SecondOrderTensor{1}) = (1, 1)
Base.size(::SecondOrderTensor{2}) = (2, 2)
Base.size(::SecondOrderTensor{3}) = (3, 3)

Base.size(::FourthOrderTensor{1}) = (1, 1, 1, 1)
Base.size(::FourthOrderTensor{2}) = (2, 2, 2, 2)
Base.size(::FourthOrderTensor{3}) = (3, 3, 3, 3)

Base.similar(t::AbstractTensor) = typeof(t)(similar(get_data(t)))
Base.fill!(t::AbstractTensor, v) = (fill!(get_data(t), v); return t)

is_always_sym{dim, T}(::Type{Tensor{dim, T}}) = false
is_always_sym{dim, T}(::Type{SymmetricTensor{dim, T}}) = true


# Internal constructors #
#########################

# These are some kinda ugly stuff to create different type of constructors.
@gen_code function call{order, dim, T}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}},
                                       data::AbstractArray{T})
    # Check for valid orders
    if !(order in (1,2,4))
        @code (throw(ArgumentError("Only tensors of order 1, 2, 4 supported")))
    else
        @code :(n = n_independent_components(dim, Tt <: SymmetricTensor))
        # Storage format is of rank 1 for vectors and order / 2 for other tensors
        if order == 1
            @code(:(Tt <: SymmetricTensor && throw(ArgumentError("SymmetricTensor only supported for order 2, 4"))))
        end

        # Validate that the input array has the correct number of elements.
        if order == 1
            @code :(length(data) == dim || throw(ArgumentError("$(length(data)) != $dim")))
        elseif order == 2
            @code :(length(data) == n || throw(ArgumentError("$(length(data)) != $n")))
        elseif order == 4
            @code :(length(data) == n*n || throw(ArgumentError("$(length(data)) != $(n*n)")))
        end
        @code :(get_main_type(Tt){order, dim, T}(vec(data)))
    end
end

@gen_code function call{order, dim, T}(Tt::Union{Type{Tensor{order, dim, T}}, Type{SymmetricTensor{order, dim, T}}})
    @code :(n = n_independent_components(dim, Tt <: SymmetricTensor))
    # Validate that the input array has the correct number of elements.
    if order == 1
       @code :(data = zeros(T, dim))
    elseif order == 2
       @code :(data = zeros(T, n))
    elseif order == 4
        @code :(data = zeros(T, n*n))
    end
    @code :(get_main_type(Tt){order, dim}(vec(data)))
end

function call{order, dim}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}})
    get_main_type(Tt){order, dim, Float64}()
end

# Indexing #
############
@inline function get_index_from_symbol(sym::Symbol)
    if     sym == :x; return 1;
    elseif sym == :y; return 2;
    elseif sym == :z; return 3;
    else              return 0 # This will bound serror later
    end
end

@inline function compute_index{dim}(::Type{SymmetricTensor{2, dim}}, i::Int, j::Int)
    if i < j
        i, j  = j,i
    end
    # We are skipping triangle under diagonal = (j-1) * j / 2 indices
    skipped_indicies = div((j-1) * j, 2)
    return dim*(j-1) + i - skipped_indicies
end

@inline function compute_index{dim}(::Type{Tensor{2, dim}}, i::Int, j::Int)
    return dim*(j-1) + i
end


# getindex general tensor #
###########################
@inline function Base.getindex(S::Tensor, i::Int)
    checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

# getindex symmetric tensor #
#############################
@inline function Base.getindex{dim}(S::SymmetricTensor{2, dim}, i::Int, j::Int)
    checkbounds(S, i, j)
    @inbounds v = get_data(S)[compute_index(SymmetricTensor{2, dim}, i, j)]
    return v
end

@inline function Base.getindex{dim}(S::SymmetricTensor{4, dim}, i::Int, j::Int, k::Int, l::Int)
    checkbounds(S, i, j, k, l)
    lower_order = SymmetricTensor{2,dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    @inbounds v = get_data(S)[(J-1)*n + I]
    return v
end

# getindex symbol #
###################

@inline function Base.getindex(S::Vec, si::Symbol)
    i = get_index_from_symbol(si)
    checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

@inline function Base.getindex(S::SecondOrderTensor, si::Symbol, sj::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    checkbounds(S, i, j)
    @inbounds v = get_data(S)[compute_index(typeof(S), i, j)]
    return v
end

@inline function Base.getindex(S::FourthOrderTensor, si::Symbol, sj::Symbol, sk::Symbol, sl::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    k = get_index_from_symbol(sk)
    l = get_index_from_symbol(sl)
    checkbounds(S, i, j, k, l)
    lower_order = SymmetricTensor{2,dim}
    I = compute_index(get_lower_order_tensor(typeof(S)), i, j)
    J = compute_index(get_lower_order_tensor(typeof(S)), k, l)
    n = n_components(lower_order)
    @inbounds v = get_data(S)[(J-1)*n + I]
    return v
end

# setindex! general tensor #
############################
@inline function Base.setindex!(S::Tensor, v, i::Int)
    checkbounds(S, i)
    @inbounds get_data(S)[i] = v
    return v
end

# setindex! symmetric tensor #
##############################

@inline function Base.setindex!{dim}(S::SymmetricTensor{2, dim}, v, i::Int, j::Int)
    checkbounds(S, i, j)
    @inbounds get_data(S)[compute_index(SymmetricTensor{2, dim}, i, j)] = v
    return v
end

@inline function Base.setindex!{dim}(S::SymmetricTensor{4, dim}, v, i::Int, j::Int, k::Int, l::Int)
    checkbounds(S, i, j, k, l)
    lower_order = SymmetricTensor{2,dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    @inbounds get_data(S)[(J-1)*n + I] = v
    return v
end

# setindex! symbol #
####################

@inline function Base.setindex!(S::Vec, v, si::Symbol)
    i = get_index_from_symbol(si)
    checkbounds(S, i)
    @inbounds get_data(S)[i] = v
    return v
end

@inline function Base.setindex!(S::SecondOrderTensor, v, si::Symbol, sj::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    checkbounds(S, i, j)
    @inbounds get_data(S)[compute_index(typeof(S), i, j)] = v
    return v
end

@inline function Base.setindex!(S::FourthOrderTensor, v, si::Symbol, sj::Symbol, sk::Symbol, sl::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    k = get_index_from_symbol(sk)
    l = get_index_from_symbol(sl)
    checkbounds(S, i, j, k, l)
    lower_order = SymmetricTensor{2,dim}
    I = compute_index(get_lower_order_tensor(typeof(S)), i, j)
    J = compute_index(get_lower_order_tensor(typeof(S)), k, l)
    n = n_components(lower_order)
    @inbounds get_data(S)[(J-1)*n + I] = v
    return v
end


#############
# Promotion #
#############

function Base.promote_rule{dim , A <: Number, B <: Number, order}(::Type{SymmetricTensor{order, dim, A}},
                                                                     ::Type{SymmetricTensor{order, dim, B}})
    SymmetricTensor{order, dim, promote_type(A, B)}
end

function Base.promote_rule{dim , A <: Number, B <: Number, order}(::Type{Tensor{order, dim, A}},
                                                                  ::Type{Tensor{order, dim, B}})
    Tensor{order, dim, promote_type(A, B)}
end


# copy / copy! #
################
function Base.copy!(t1::AllTensors, t2::AllTensors)
    @assert get_base(typeof(t1)) == get_base(typeof(t2))
    copy!(get_data(t1), get_data(t2))
    return t1
end

Base.copy(t::AllTensors) = copy!(similar(t), t)


function Base.convert{dim}(::Type{SymmetricTensor{2, dim}}, t::Tensor{2, dim})
    t_sym = zero(SymmetricTensor{2, dim})
    @inbounds for i in 1:dim, j in 1:i
        t_sym[i,j] = 0.5 * (t[i,j] + t[j,i])
    end
    return t_sym
end


###############
# Simple Math #
###############

function Base.(:*)(n::Number, t::AllTensors)
    get_base(typeof(t))(n * get_data(t))
end

Base.(:*)(t::AllTensors, n::Number) = n * t

function Base.(:/)(t::AllTensors, n::Number)
    get_base(typeof(t))(get_data(t) / n)
end

function Base.(:-)(t1::AllTensors, t2::AllTensors)
    @assert get_base(typeof(t1)) == get_base(typeof(t2))
    get_base(typeof(t1))(get_data(t1) - get_data(t2))
end

function Base.(:-)(t::AllTensors)
    get_base(typeof(t))(- get_data(t))
end

function Base.(:+)(t1::AllTensors, t2::AllTensors)
    @assert get_base(typeof(t1)) == get_base(typeof(t2))
    get_base(typeof(t1))(get_data(t1) + get_data(t2))
end


###################
# Zero, one, rand #
###################

for (f, f!) in ((:zero, :zero!), (:rand, :rand!), (:one, :one!))
    @eval begin

        function Base.$(f){order, dim, T}(Tt::Union{Type{Tensor{order, dim, T}}, Type{SymmetricTensor{order, dim, T}}})
            $(f!)(Tt())
        end

        function Base.$(f){order, dim}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}})
            $(f!)(Tt())
        end

        Base.$(f)(t::AllTensors) = $(f!)(similar(t))
    end
end


zero!(t::AllTensors) = (fill!(get_data(t), 0.0); return t)

function rand!{dim, T}(t::AllTensors{dim, T})
    @inbounds for i in eachindex(t)
        t[i] = rand(T)
    end
    return t
end


# Helper function for `one`
set_diag!(S::Vec, v, i) = S[i] = v
set_diag!(S::SecondOrderTensor, v, i) = S[i,i] = v
set_diag!(S::FourthOrderTensor, v, i) = S[i,i,i,i] = v


function one!{dim}(t::Union{SecondOrderTensor{dim}, Vec{dim}})
    fill!(get_data(t), 0.0)
    for i in 1:dim
        set_diag!(t, 1, i)
    end
    return t
end

function one!{dim}(t::FourthOrderTensor{dim})
    @inbounds for i in 1:dim, j in 1:dim, k in 1:dim, l in 1:dim
        if i == k && j == l
            t[i,j,k,l] = 1
        else
            t[i,j,k,l] = 0
        end
    end
    return t
end


include("symmetric_ops.jl")
include("tensor_ops.jl")
include("data_functions.jl")

end # module
