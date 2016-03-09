#################################################
# Sepcialized Second Order Symmetric Operations #
#################################################

Base.transpose(S::SymmetricTensors) = S
Base.issym(S::SymmetricTensors) = true

######################
# Double contraction #
######################
@gen_code function dcontract{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, S2::SymmetricTensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    @code :(s = zero($Tv);
            data1 = get_data(S1);
            data2 = get_data(S2))
     for k in 1:n_independent_components(dim, true)
        if is_diagonal_index(dim, k)
            @code :(@inbounds s += data1[$k] * data2[$k])
        else
            @code :(@inbounds s += 2 * data1[$k] * data2[$k])
        end
    end
    @code :(return s)
end

@generated function dcontract{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, S2::SymmetricTensor{4, dim, T2})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(k,l) = compute_index(SymmetricTensor{2, dim}, k, l)
    exps = Expr(:tuple)
    for i in 1:dim, j in i:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
            for k in 1:dim, l in k:dim
            if k == l
                push!(exps_ele.args, :(data4[$(idx4(l, k, j, i))] * data2[$(idx2(l,k))]))
            else
                push!(exps_ele.args, :( 2 * data4[$(idx4(l, k, j, i))] * data2[$(idx2(l,k))]))
            end
        end
        push!(exps.args, exps_ele)
    end
    quote
         data2 = S1.data
         data4 = S2.data
         @inbounds r = $exps
         SymmetricTensor{2, dim}(r)
    end
end

@generated function dcontract{dim, T1, T2}(S1::SymmetricTensor{4, dim, T1}, S2::SymmetricTensor{2, dim, T2})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(k,l) = compute_index(SymmetricTensor{2, dim}, k, l)
    exps = Expr(:tuple)
    for i in 1:dim, j in i:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
            for k in 1:dim, l in k:dim
            if k == l
                push!(exps_ele.args, :(data4[$(idx4(j, i, l, k))] * data2[$(idx2(l,k))]))
            else
                push!(exps_ele.args, :( 2 * data4[$(idx4(j, i, l, k))] * data2[$(idx2(l,k))]))
            end
        end
        push!(exps.args, exps_ele)
    end
    quote
         data2 = S2.data
         data4 = S1.data
         @inbounds r = $exps
         SymmetricTensor{2, dim}(r)
    end
end

@generated function dcontract{dim, T1, T2}(S1::SymmetricTensor{4, dim, T1}, S2::SymmetricTensor{4, dim, T2})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    exps = Expr(:tuple)
    for k in 1:dim, l in k:dim, i in 1:dim, j in i:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
        for m in 1:dim, n in m:dim
            if m == n
                push!(exps_ele.args, :(data1[$(idx4(j,i,n,m))] * data2[$(idx4(m,n,l,k))]))
            else
                 push!(exps_ele.args, :(2*data1[$(idx4(j,i,n,m))] * data2[$(idx4(m,n,l,k))]))
            end
        end
        push!(exps.args, exps_ele)
    end
    quote
         data2 = S2.data
         data1 = S1.data
         @inbounds r = $exps
         SymmetricTensor{4, dim}(r)
    end
end



#######
# Dot #
#######

# TODO: Do not promote here but just write out the multiplication
@inline function Base.dot{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, v2::Vec{dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    S1_t = convert(Tensor{2, dim}, S1)
    return Vec{dim, Tv}(Am_mul_Bv(S1_t.data, v2.data))
end

@inline function Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, S2::SymmetricTensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    S2_t = convert(Tensor{2, dim}, S2)
    return Vec{dim, Tv}(Amt_mul_Bv(S2_t.data, v1.data))
end

########
# norm #
########
@gen_code function Base.norm{dim, T}(S::SymmetricTensor{4, dim, T})
    idx(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    @code :(data = get_data(S))
    @code :(s = zero(T))
    for k in 1:dim, l in 1:k, i in 1:dim, j in 1:i
        @code :(@inbounds v = data[$(idx(i,j,k,l))])
        if i == j && k == l
             @code :(s += v*v)
        elseif i == j || k == l
             @code :(s += 2*v*v)
        else
             @code :(s += 4*v*v)
        end
    end
    @code :(return sqrt(s))
end


################
# Open product #
################
function otimes{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, S2::SymmetricTensor{2, dim, T2})
    SymmetricTensor{4, dim}(A_otimes_B(S1.data, S2.data))
end
