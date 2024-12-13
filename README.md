# Final-Project
This project is for ECE759 High Performance Computing final project in UW-Madison.
Project Name: High-Performance Financial Prediction for Stocks

main.py and svr_model.py is the python code for SVR
plot.py is to plot prediction and true data
save_stock is to generate stock data
stock_svr.cpp is the serial verison of SVR
stock_svr_openMP is the CPU parallel version of SVR
stock_svr_rx2060 is the GPU parallel version of SVR

Future Work:
Serial method (done)
CPU-openmp (done)  
GPU (done)  
GPU-shared memory (done)  
GPU-tiled matrix (done)  
fit parallel (done)  
batch update -> parallel (done)  
cross validation parameter selection parallel (done)  
parallel predict for different stock
