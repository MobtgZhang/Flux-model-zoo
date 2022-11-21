using Flux,Statistics
using Flux:params
using Flux:onehotbatch,onecold,crossentropy,@epochs
using Base.Iterators:partition
using BSON:@load,@save
using Random
using Base:@kwdef
using MLDatasets

function define_model(;hidden)
    mlp = Chain(Dense(28^2,hidden,relu),
                Dense(hidden,hidden,relu),
                Dense(hidden,10),
                softmax)
    return mlp
end

@kwdef mutable struct Args
    epochs::Int64 = 10
    hid_dim::Int64 = 100
    batchsize::Int = 512
    use_cuda::Bool = true
    log_path::String = "./log"
end

function train(;kws...)
    args = Args(;kws...)
    println("Start to train")
    if !ispath(args.log_path)
        mkdir(args.log_path)
    end
    # Loading Dataset
    xtrain, ytrain = MLDatasets.MNIST(split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(split=:test)[:]

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    if args.use_cuda
        device = gpu
    else
        device = cpu
    end
    model = define_model(hidden=args.hid_dim) |> device 
    loss(x,y)= crossentropy(model(x),y)
    train_dataset = [(xtrain[:,batch] ,Float32.(ytrain[:,batch])) for batch in partition(1:size(ytrain)[2],args.batchsize)] |> device
    test_dataset = [(xtest[:,batch] ,Float32.(ytest[:,batch])) for batch in partition(1:size(ytest)[2],args.batchsize)] |> device

    callback_count = 0
    eval_callback = function callback()
        callback_count += 1
        if callback_count == length(train_dataset)
            println("action for each epoch")
            total_loss = 0
            total_acc = 0
            ntot = 0
            for (vx, vy) in test_dataset
                pred_y = model(vx)
                total_loss += loss(vx, vy)
                total_acc += sum(onecold(cpu(pred_y)) .== onecold(cpu(vy)))
                ntot += size(vx)[end]
            end
            total_loss /= ntot
            total_acc /= ntot
            println("total_acc $total_acc,total_loss $total_loss")
            callback_count = 0
        end
    end
    optimizer = ADAM()
    @epochs args.epochs Flux.train!(loss,params(model),train_dataset, optimizer, cb = eval_callback)
    pretrained = model |> cpu
    save_pretrained_file = joinpath(args.log_path,"pretrained.bson")
    @save save_pretrained_file pretrained
    weights =params(pretrained)
    save_weights_file = joinpath(args.log_path,"weights.bson")
    @save save_weights_file weights
    println("Finished to train")
end
function main()
    train()
end
main()

