# GPU benchmark 
using Flux, CUDA, InvariantPointAttention

# Test parameters
N_head = 12
N_query_points = 4
N_point_values = 8
N_frames_L = 500
N_frames_R = 500
batch_size = 30
dims = 1024
c = 16

# Create IPA settings
settings = IPA_settings(
    dims,
    c = c,
    N_head = N_head,
    N_query_points = N_query_points,
    N_point_values = N_point_values,
    c_z = 0,
    Typ = Float32,
    use_softmax1 = false,
    scaling_qk = :default,
)

# Create IPA layer and move to GPU
ipa = IPCrossA(settings) |> gpu

# Create test data on GPU
TiL = (
    randn(Float32, 3, 3, N_frames_L, batch_size) |> gpu,  # rotation matrices
    randn(Float32, 3, 1, N_frames_L, batch_size) |> gpu   # translation vectors
)
TiR = (
    randn(Float32, 3, 3, N_frames_R, batch_size) |> gpu,  # rotation matrices  
    randn(Float32, 3, 1, N_frames_R, batch_size) |> gpu   # translation vectors
)
siL = randn(Float32, dims, N_frames_L, batch_size) |> gpu
siR = randn(Float32, dims, N_frames_R, batch_size) |> gpu

# Warm up GPU
println("Warming up GPU...")
for i in 1:5
    ipa(TiL, siL, TiR, siR; old_eucdists = true, customgrad = false)
    ipa(TiL, siL, TiR, siR; old_eucdists = false, customgrad = false)
end
CUDA.synchronize()

println("\n=== GPU Benchmarking Results ===")
println("Configuration: N_head=$N_head, N_frames_L=$N_frames_L, N_frames_R=$N_frames_R, batch_size=$batch_size")

# Benchmark old method
println("\nBenchmarking old_eucdists = true:")
old_time = CUDA.@time begin
    for i in 1:10
        result_old = ipa(TiL, siL, TiR, siR; old_eucdists = true, customgrad = false)
    end
    CUDA.synchronize()
end

# Benchmark new method  
println("\nBenchmarking old_eucdists = false:")
new_time = CUDA.@time begin
    for i in 1:10
        result_new = ipa(TiL, siL, TiR, siR; old_eucdists = false, customgrad = false)
    end
    CUDA.synchronize()
end

# Memory usage comparison
println("\n=== Memory Usage ===")
println("Old method memory:")
CUDA.@time result_old = ipa(TiL, siL, TiR, siR; old_eucdists = true, customgrad = false);

println("\nNew method memory:")
CUDA.@time result_new = ipa(TiL, siL, TiR, siR; old_eucdists = false, customgrad = false);

# Verify results are still equivalent
println("\n=== Correctness Check ===")
max_diff = maximum(abs.(Array(result_old) - Array(result_new)))
println("Maximum absolute difference: $max_diff")
println("Results equivalent: ", isapprox(result_old, result_new, rtol=1e-5, atol=1e-6))

