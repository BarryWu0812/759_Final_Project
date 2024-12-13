#include <iostream>
#include <stddef.h>
#include <stdexcept>
#include <numeric>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>
#include <memory>

using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::cout;
using std::endl;

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// CUDA kernel for dot product
__global__ void dot_product_kernel(const double* v1, const double* v2, double* result, int size) {
    extern __shared__ double shared_mem[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    double temp = 0.0;
    while (gid < size) {
        temp += v1[gid] * v2[gid];
        gid += gridDim.x * blockDim.x;
    }

    shared_mem[tid] = temp;
    __syncthreads();

    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = shared_mem[0];
    }
}

// GPU资源管理类
class CudaManager {
private:
    double *d_weights;     // GPU上的权重
    double *d_data;        // GPU上的输入数据
    double *d_result;      // GPU上的结果
    int max_size;          // 最大向量大小
    int block_size;        // CUDA block大小
    int num_blocks;        // CUDA block数量

public:
    CudaManager(int size) : max_size(size), block_size(256) {
        num_blocks = (size + block_size - 1) / block_size;
        num_blocks = std::min(num_blocks, 1024);  // 限制最大block数量

        // 分配GPU内存
        CHECK_CUDA_ERROR(cudaMalloc(&d_weights, size * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_data, size * sizeof(double)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_result, num_blocks * sizeof(double)));
    }

    ~CudaManager() {
        cudaFree(d_weights);
        cudaFree(d_data);
        cudaFree(d_result);
    }

    void update_weights(const std::vector<double>& w) {
        CHECK_CUDA_ERROR(cudaMemcpy(d_weights, w.data(), w.size() * sizeof(double),
                                    cudaMemcpyHostToDevice));
    }

    double compute_dot_product(const std::vector<double>& vec) {
        // 复制输入数据到GPU
        CHECK_CUDA_ERROR(cudaMemcpy(d_data, vec.data(), vec.size() * sizeof(double),
                                    cudaMemcpyHostToDevice));

        // 启动kernel
        dot_product_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(
                d_data, d_weights, d_result, vec.size());

        // 检查kernel执行错误
        CHECK_CUDA_ERROR(cudaGetLastError());

        // 获取结果
        std::vector<double> result(num_blocks);
        CHECK_CUDA_ERROR(cudaMemcpy(result.data(), d_result, num_blocks * sizeof(double),
                                    cudaMemcpyDeviceToHost));

        // 求和得到最终结果
        return std::accumulate(result.begin(), result.end(), 0.0);
    }
};

// StandardScaler类（保持不变）
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

    std::vector<double> transform(const std::vector<double>& X) const {
        if (!is_fitted_) {
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

    std::vector<double> inverse_transform(const std::vector<double>& X) const {
        if (!is_fitted_) {
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



// 优化后的StockSVR类
class StockSVR {
private:
    double lr;       // Learning rate
    double C;        // Regularization penalty
    int n_iters;     // Number of iterations
    double epsilon;  // Epsilon-insensitive margin
    std::vector<double> w;  // Weights
    double b;        // Bias
    std::unique_ptr<CudaManager> cuda_mgr;

public:
    StockSVR(double learning_rate, double penalty, int iterations, double eps)
            : lr(learning_rate), C(penalty), n_iters(iterations), epsilon(eps), b(0.0) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int batch_size) {
        int n_samples = X.size();
        int n_features = X[0].size();

        w.resize(n_features, 0.0);
        cuda_mgr = std::make_unique<CudaManager>(n_features);

        // 优化：预先更新GPU上的权重
        cuda_mgr->update_weights(w);

        for (int iter = 0; iter < n_iters; iter++) {
            for (int batch_start = 0; batch_start < n_samples; batch_start += batch_size) {
                int batch_end = std::min(batch_start + batch_size, n_samples);
                std::vector<double> grad_w(n_features, 0.0);
                double grad_b = 0.0;

                for (int i = batch_start; i < batch_end; i++) {
                    // 使用GPU计算点积
                    double y_pred = cuda_mgr->compute_dot_product(X[i]) + b;
                    double loss = y[i] - y_pred;

                    if (std::abs(loss) > epsilon) {
                        for (int j = 0; j < n_features; ++j) {
                            grad_w[j] += -C * X[i][j] * (loss > 0 ? 1 : -1);
                        }
                        grad_b += -C * (loss > 0 ? 1 : -1);
                    }
                    else {
                        for (int j = 0; j < n_features; ++j) {
                            grad_w[j] += w[j];
                        }
                    }
                }

                // 更新权重和偏置
                for (int j = 0; j < n_features; ++j) {
                    w[j] -= lr * (w[j] + grad_w[j] / batch_size);
                }
                b -= lr * (grad_b / batch_size);

                // 更新GPU上的权重
                cuda_mgr->update_weights(w);
            }
        }
    }

    vector<double> predict(const vector<vector<double>>& X) {
        std::vector<double> predictions;
        predictions.reserve(X.size());

        for (const auto& x : X) {
            predictions.push_back(cuda_mgr->compute_dot_product(x) + b);
        }
        return predictions;
    }
};

// 辅助函数
void prepare_data(const vector<double>& data, int n_days, vector<vector<double>>& X, vector<double>& y) {
    int n_samples = data.size() - n_days;
    X.resize(n_samples, std::vector<double>(n_days));
    y.resize(n_samples);

    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_days; j++) {
            X[i][j] = data[i + j];
        }
        y[i] = data[i + n_days];
    }
}

vector<double> read_data(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        data.push_back(std::stod(line));
    }

    return data;
}

void save_to_csv(const std::string& filename, const std::vector<double>& data) {
    std::ofstream file(filename);
    for (const auto& value : data) {
        file << value << "\n";
    }
    file.close();
}

// 主函数
int main() {
    // 检查并初始化CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using GPU device: " << deviceProp.name << std::endl;

    // 读取数据
    std::string data_filename = "AAPL_2024-11-28.csv";
    vector<double> data = read_data(data_filename);
    int n_days = 10;
    int batch_size = 200;  // 优化：增加batch size以提高GPU利用率

    vector<vector<double>> X;
    vector<double> y;
    prepare_data(data, n_days, X, y);

    // 分割训练集和测试集
    int split_idx = static_cast<int>(0.8 * X.size());
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + split_idx);
    std::vector<std::vector<double>> X_test(X.begin() + split_idx, X.end());
    std::vector<double> y_train(y.begin(), y.begin() + split_idx);
    std::vector<double> y_test(y.begin() + split_idx, y.end());

    // 数据标准化
    StandardScaler scaler_X, scaler_y;
    scaler_X.fit(X_train);
    scaler_y.fit(y_train);

    vector<vector<double>> X_scaled = scaler_X.transform(X_train);
    vector<double> y_scaled = scaler_y.transform(y_train);

    // 初始化并训练模型
    StockSVR svr(0.001, 1.0, 1000, 0.1);

    // 训练计时
    auto start_train = high_resolution_clock::now();
    svr.fit(X_scaled, y_scaled, batch_size);
    auto end_train = high_resolution_clock::now();
    auto duration_train = std::chrono::duration_cast<duration<double, std::milli>>(end_train-start_train);

    // 预测计时
    X_test = scaler_X.transform(X_test);
    auto start_predict = high_resolution_clock::now();
    vector<double> predictions = svr.predict(X_test);
    auto end_predict = high_resolution_clock::now();
    auto duration_predict = std::chrono::duration_cast<duration<double, std::milli>>(end_predict-start_predict);

    predictions = scaler_y.inverse_transform(predictions);

    // 计算并打印MSE
    double test_mse = 0.0;
    for (size_t i = 0; i < y_test.size(); i++) {
        test_mse += std::pow(y_test[i] - predictions[i], 2);
    }
    test_mse /= y_test.size();

    // 打印结果
    std::cout << "\nResults:" << std::endl;
    std::cout << "Test set MSE: " << test_mse << std::endl;
    std::cout << "Model training time: " << duration_train.count() << " ms" << std::endl;
    std::cout << "Prediction time: " << duration_predict.count() << " ms" << std::endl;

    std::cout << "\nTrue vs Predicted values (first 10 samples):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), y_test.size()); ++i) {
        std::cout << "True: " << std::fixed << std::setprecision(2) << y_test[i]
                  << ", Predicted: " << predictions[i] << std::endl;
    }

    // 保存结果
    save_to_csv("y_test.csv", y_test);
    save_to_csv("predictions.csv", predictions);

    std::cout << "\nPredictions saved to CSV files for visualization." << std::endl;

    return 0;
}