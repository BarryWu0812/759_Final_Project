#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>

using std::vector;

__device__ double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// GPU kernel for dot product
__global__ void dot_product_kernel(const double* X, const double* w, double* y, int n_samples, int n_features, double b)
{
    // int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx % n_samples;

    // Each thread computes a partial dot product for one row
    if (idx < n_samples)
    {
        // temp[tid] = 0.0;
        // for (int col = tid; col < n_features; col += blockDim.x) {
        //     temp[tid] += X[row * n_features + col] * w[col];
        // }
        // __syncthreads();

        // // Reduction within the block
        // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        //     if (tid < stride) {
        //         temp[tid] += temp[tid + stride];
        //     }
        //     __syncthreads();
        // }
        double sum = 0;
        for (int k = 0; k < n_features; k++)
        {
            // printf("A: %f, B: %f\n", A[row * n + k], B[k * n + col]);
            sum += X[row * n_features + k] * w[k];
        }
        // printf("row: %d, col: %d\n", row, col);
        y[row] = sum;

        // Write the final result
        if (row < n_samples)
            y[row] += b; // Add the bias term
    }
}

// GPU kernel to update weights and biases
__global__ void update_weights_kernel(double* w, double* grad_w, double* b, double* grad_b, double lr, int n_features, double batch_size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_features) {
        w[idx] -= lr * (grad_w[idx] / batch_size);
    }
    if (idx == 0)
        atomicAddDouble(b, -lr * (*grad_b / batch_size));
}

__global__ void compute_gradients_kernel(const double* X, const double* y, const double* w, double* b, double* grad_w, double* grad_b, int n_samples, int n_features, double epsilon, double C)
{
    extern __shared__ double shared_mem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_grad_b = 0.0;

    if (idx < n_samples)
    {
        double y_pred = 0.0;

        // Compute y_pred for a single sample
        for (int j = 0; j < n_features; ++j)
        {
            y_pred += X[idx * n_features + j] * w[j];
        }
        y_pred += *b;

        // Compute loss and gradients
        double loss = y[idx] - y_pred;
        if (fabs(loss) > epsilon)
        {
            for (int j = 0; j < n_features; ++j)
            {
                atomicAddDouble(&grad_w[j], -C * X[idx * n_features + j] * ((loss > 0) - (loss < 0)));
            }
            local_grad_b = -C * ((loss > 0) - (loss < 0));
        }
    }

    // Use shared memory to reduce gradients for `grad_b`
    shared_mem[tid] = local_grad_b;
    __syncthreads();

    // Reduce shared memory to compute global gradient for bias
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 updates the global bias gradient
    if (tid == 0) {
        atomicAddDouble(grad_b, shared_mem[tid]);
    }
}


// StandardScaler class
class StandardScaler {
private:
    std::vector<double> means_;
    std::vector<double> std_devs_;
    bool is_fitted_;

    double compute_mean(const std::vector<double>& data) {
        return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    }

    double compute_std_dev(const std::vector<double>& data, double mean) {
        double variance = 0.0;
        for (const auto& value : data) {
            variance += (value - mean) * (value - mean);
        }
        return std::sqrt(variance / data.size());
    }

public:
    StandardScaler() : is_fitted_(false) {}

    // Fit for 1D or 2D data
    void fit(const std::vector<std::vector<double>>& X) {
        if (X.empty() || X[0].empty()) {
            throw std::invalid_argument("Input matrix is empty");
        }

        size_t n_features = X[0].size();
        means_.resize(n_features, 0);
        std_devs_.resize(n_features, 0);

        for (size_t j = 0; j < n_features; j++) {
            std::vector<double> column;
            for (const auto& row : X) {
                column.push_back(row[j]);
            }
            means_[j] = compute_mean(column);
            std_devs_[j] = compute_std_dev(column, means_[j]);
        }
        is_fitted_ = true;
    }

    void fit(const std::vector<double>& X) {
        means_.resize(1, compute_mean(X));
        std_devs_.resize(1, compute_std_dev(X, means_[0]));
        is_fitted_ = true;
    }

    // Transform for 1D or 2D data
    std::vector<double> transform(const std::vector<double>& X) const {
        if (!is_fitted_ || means_.size() != 1 || std_devs_.size() != 1) {
            throw std::logic_error("Scaler has not been fitted for 1D data");
        }

        std::vector<double> transformed(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            transformed[i] = (X[i] - means_[0]) / std_devs_[0];
        }
        return transformed;
    }

    vector<vector<double>> transform(const vector<vector<double>>& X) const {
        if (!is_fitted_) {
            throw std::logic_error("Scaler has not been fitted for 2D data");
        }

        std::vector<std::vector<double>> transformed(X.size(), std::vector<double>(X[0].size(), 0));
        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = 0; j < X[0].size(); j++) {
                transformed[i][j] = (X[i][j] - means_[j]) / std_devs_[j];
            }
        }
        return transformed;
    }

    // Inverse Transform for 1D or 2D data
    std::vector<double> inverse_transform(const std::vector<double>& X) const {
        if (!is_fitted_ || means_.size() != 1 || std_devs_.size() != 1) {
            throw std::logic_error("Scaler has not been fitted for 1D data");
        }

        std::vector<double> original(X.size());
        for (size_t i = 0; i < X.size(); i++) {
            original[i] = X[i] * std_devs_[0] + means_[0];
        }
        return original;
    }

    std::vector<std::vector<double>> inverse_transform(const std::vector<std::vector<double>>& X) const {
        if (!is_fitted_) {
            throw std::logic_error("Scaler has not been fitted for 2D data");
        }

        std::vector<std::vector<double>> original(X.size(), std::vector<double>(X[0].size(), 0));
        for (size_t i = 0; i < X.size(); i++) {
            for (size_t j = 0; j < X[0].size(); j++) {
                original[i][j] = X[i][j] * std_devs_[j] + means_[j];
            }
        }
        return original;
    }
};

// GPU-based StockSVR
class StockSVR {
private:
    double lr, C, epsilon;
    int n_iters;
    double* d_w, * d_grad_w, * d_grad_b;
    std::vector<double> w;
    double b;

public:
    StockSVR(double learning_rate, double penalty, int iterations, double eps)
        : lr(learning_rate), C(penalty), n_iters(iterations), epsilon(eps) {
        d_w = nullptr;
        d_grad_w = nullptr;
        d_grad_b = nullptr;
    }

    void fit(const vector<vector<double>>& X, const vector<double>& y, int batch_size)
    {
        size_t n_samples = X.size();
        size_t n_features = X[0].size();

        // Allocate device memory
        double *d_X, *d_y, *d_w, *d_b, *d_grad_w, *d_grad_b;
        cudaMalloc(&d_X, n_samples * n_features * sizeof(double));
        cudaMalloc(&d_y, n_samples * sizeof(double));
        cudaMalloc(&d_w, n_features * sizeof(double));
        cudaMalloc(&d_b, sizeof(double));
        cudaMalloc(&d_grad_w, n_features * sizeof(double));
        cudaMalloc(&d_grad_b, sizeof(double));

        // Initialize weights and bias on the host
        vector<double> w(n_features, 0.0); // Initialize weights to 0
        double b = 0.0;                   // Initialize bias to 0

        // Copy data to device
        cudaMemcpy(d_X, X.data()->data(), n_samples * n_features * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y.data(), n_samples * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), n_features * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, &b, sizeof(double), cudaMemcpyHostToDevice);

        // Gradient Descent Loop
        for (int iter = 0; iter < n_iters; iter++)
        {
            for (size_t batch_start = 0; batch_start < n_samples; batch_start += batch_size)
            {
                size_t batch_end = std::min(batch_start + batch_size, n_samples);
                int curr_batch_size = batch_end - batch_start;

                // Reset gradients for this batch
                cudaMemset(d_grad_w, 0, n_features * sizeof(double));
                cudaMemset(d_grad_b, 0, sizeof(double));

                // Compute gradients in parallel
                dim3 block_dim(256);
                dim3 grid_dim((curr_batch_size + block_dim.x - 1) / block_dim.x);
                size_t shared_mem_size = block_dim.x * sizeof(double);

                compute_gradients_kernel<<<grid_dim, block_dim, shared_mem_size>>>(
                    d_X + batch_start * n_features, d_y + batch_start, d_w, d_b,
                    d_grad_w, d_grad_b, curr_batch_size, n_features, epsilon, C);

                // Synchronize after gradient computation
                cudaDeviceSynchronize();

                // Update weights in parallel
                dim3 weight_block_dim(256);
                dim3 weight_grid_dim((n_features + weight_block_dim.x - 1) / weight_block_dim.x);

                update_weights_kernel<<<weight_grid_dim, weight_block_dim>>>(
                    d_w, d_grad_w, d_b, d_grad_b, lr, n_features, curr_batch_size);

                // Synchronize after weight update
                cudaDeviceSynchronize();
            }
        }

        // Copy back weights and bias from device to host
        cudaMemcpy(w.data(), d_w, n_features * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(&b, d_b, sizeof(double), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_X);
        cudaFree(d_y);
        cudaFree(d_w);
        cudaFree(d_b);
        cudaFree(d_grad_w);
        cudaFree(d_grad_b);

        // Store the final weights and bias in the class
        this->w = w;
        this->b = b;
    }


    vector<double> predict(const vector<vector<double>>& X)
    {
        size_t n_samples = X.size();
        size_t n_features = X[0].size();

        vector<double> predictions(n_samples);
        double* d_X, * d_w, * d_y;
        double* h_X = new double[n_samples * n_features];
        double* h_w = new double[n_features];

        // Flatten input matrices for GPU
        for (size_t i = 0; i < n_samples; i++)
        {
            for (size_t j = 0; j < n_features; j++)
            {
                h_X[i * n_features + j] = X[i][j];
            }
        }

        for (size_t j = 0; j < n_features; j++)
        {
            h_w[j] = w[j];
        }

        // Allocate device memory
        cudaMalloc(&d_X, n_samples * n_features * sizeof(double));
        cudaMalloc(&d_w, n_features * sizeof(double));
        cudaMalloc(&d_y, n_samples * sizeof(double));

        // Copy data to device
        cudaMemcpy(d_X, h_X, n_samples * n_features * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, h_w, n_features * sizeof(double), cudaMemcpyHostToDevice);

        // Launch kernel
        int threads_per_block = 256;
        int blocks_per_grid = (n_samples + threads_per_block - 1) / threads_per_block; // One block per row of `X`
        size_t shared_mem_size = threads_per_block * sizeof(double);

        dot_product_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_X, d_w, d_y, n_samples, n_features, b);

        // Copy predictions back to host
        cudaMemcpy(predictions.data(), d_y, n_samples * sizeof(double), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_X);
        cudaFree(d_w);
        cudaFree(d_y);
        delete[] h_X;
        delete[] h_w;

        return predictions;
    }

};

// Prepare data for training (rolling window)
void prepare_data(const vector<double>& data, int n_days, vector<vector<double>>& X, vector<double>& y)
{
    int n_samples = data.size() - n_days;

    // Resize X and y
    X.resize(n_samples, std::vector<double>(n_days));
    y.resize(n_samples);

    // Populate X and y
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_days; j++) {
            X[i][j] = data[i + j];  // Create rolling window features
        }
        y[i] = data[i + n_days];    // Target value
    }
}

vector<double> read_data(const std::string& filename)
{
    std::vector<double> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        // Convert line to double and store it
        data.push_back(std::stod(line));
    }

    return data;
}

void save_to_csv(const std::string& filename, const std::vector<double>& data)
{
    std::ofstream file(filename);
    for (const auto& value : data) {
        file << value << "\n";
    }
    file.close();
}

// Main Function
int main() {
    std::string data_filename = "AAPL_2024-11-28.csv";
    vector<double> data = read_data(data_filename);
    vector<vector<double>> X;
    vector<double> y;
    int n_days = 5, batch_size = 32;
    prepare_data(data, n_days, X, y);



    // Split into training and testing sets
    int split_idx = static_cast<int>(0.8 * X.size());
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + split_idx);
    std::vector<std::vector<double>> X_test(X.begin() + split_idx, X.end());
    std::vector<double> y_train(y.begin(), y.begin() + split_idx);
    std::vector<double> y_test(y.begin() + split_idx, y.end());


    // Fit scalers
    StandardScaler scaler_X, scaler_y;
    scaler_X.fit(X_train);
    scaler_y.fit(y_train);

    vector<vector<double>> X_scaled = scaler_X.transform(X_train);
    vector<double> y_scaled = scaler_y.transform(y_train);

    StockSVR svr(0.001, 1.0, 1000, 0.1);
    svr.fit(X_scaled, y, batch_size);

    // Make predictions
    X_test = scaler_X.transform(X_test);
    vector<double> predictions = svr.predict(X_test);
    predictions = scaler_y.inverse_transform(predictions);

    save_to_csv("y_test.csv", y_test);
    save_to_csv("predictions.csv", predictions);

    return 0;
}
