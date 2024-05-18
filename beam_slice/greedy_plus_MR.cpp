#include <iostream>
#include <complex>
#include "H5Cpp.h"
#include <cmath>
#include <iomanip>
// #include <armadillo>
#include <vector>
#include <algorithm> // For std::max_element and std::distance
#include <numeric>   // For std::accumulate
#include "max_rate.h"
#include "func.h"
#include <chrono>


bool any_positive(const std::vector<double>& data) {
    return std::any_of(data.begin(), data.end(), [](double x) { return x > 0; });
}



int main(){
    // Open the HDF5 file
    H5::H5File file("../DU_4_25_64_10_52.hdf5", H5F_ACC_RDONLY);

    // Read the dataset 'H_r' and 'H_i'
    H5::DataSet dataset_r = file.openDataSet("H_r");
    H5::DataSet dataset_i = file.openDataSet("H_i");

    // Get the dataspace of the datasets
    H5::DataSpace dataspace_r = dataset_r.getSpace();
    H5::DataSpace dataspace_i = dataset_i.getSpace();

    // Get the dimensions of the datasets
    hsize_t dims_r[5];
    hsize_t dims_i[5];
    dataspace_r.getSimpleExtentDims(dims_r);
    dataspace_i.getSimpleExtentDims(dims_i);


    arma::vec H_r(dims_r[0] * dims_r[1] * dims_r[2] * dims_r[3] * dims_r[4], arma::fill::zeros);
    arma::vec H_i(dims_r[0] * dims_r[1] * dims_r[2] * dims_r[3] * dims_r[4], arma::fill::zeros);

    // Read data from the datasets
    dataset_r.read(H_r.memptr(), H5::PredType::NATIVE_DOUBLE);
    dataset_i.read(H_i.memptr(), H5::PredType::NATIVE_DOUBLE);

    // Close the datasets and file
    dataset_r.close();
    dataset_i.close();
    file.close();

    // Print the shape
    std::cout << "(" << dims_r[0] << ", " << dims_r[1] << ", " << dims_r[2] << ", " << dims_r[3] << ", " << dims_r[4] << ")" << std::endl;

    const int Num_tti = 10;
    const int Num_RBG = 52;
    const int Num_BS = 64;
    const double corr_th = 0.5;
    const int total_tti = 10;
    // const int config = 4;

// ******************************************** CONFIGURE = 4 *******************************************************************
    const int Num_UE = 16;
    const int Num_slice = 4;
    const int Num_UE_ps = Num_UE / Num_slice;
    const int SEL_UE = 3;
    std::vector<int> SLAs = {135, 120, 130, 140};
    // std::vector<int> SLAs = {235, 220, 230, 240};

    arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    // std::cout << "Now here" << std::endl;
    // Evenly Distributed Case
    for (int ue = 0; ue < Num_UE; ++ue){
        int remain = ue % Num_slice;
        int div = ue / Num_slice;
        for (int j = 0; j < Num_BS; ++j){
            for (int k = 0; k < Num_tti; ++k){
                for (int l = 0; l < Num_RBG; ++l){
                    // std::cout << "Now here H_r:" <<H_r(remain*dims_i[0] + div*dims_i[1] + j*dims_r[2] + k*dims_r[3] + l) << std::endl;
                    H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(remain*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + div*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(remain*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + div*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
                }
            }
        }

    }

    // // Highly Correlated Case
    // for (int i = 0; i < Num_UE; ++i){
    //     int remain = i % Num_slice;
    //     int div = i / Num_slice;
    //     for (int j = 0; j < Num_BS; ++j){
    //         for (int k = 0; k < Num_tti; ++k){
    //             for (int l = 0; l < Num_RBG; ++l){
    //                 H(i*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r[div*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + remain*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l], H_i[div*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + remain*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l]);
    //             }
    //         }
    //     }

    // }
// ******************************************** CONFIGURE = 4 *******************************************************************

// ******************************************** CONFIGURE = 8 *******************************************************************
    // const int Num_UE = 80;
    // const int Num_slice = 8;
    // const int Num_UE_ps = Num_UE / Num_slice;
    // const int SEL_UE = 8;
    // std::vector<int> SLAs = {135, 120, 130, 140, 45, 110, 50, 125};
    // // std::vector<int> SLAs = {235, 220, 230, 240, 145, 210, 150, 225};

    // arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    // // std::cout << "Now here" << std::endl;
    // // Evenly Distributed Case
    // for (int ue = 0; ue < Num_UE; ++ue){
    //     int remain = ue % 10;
    //     int div = ue / 10;
    //     for (int j = 0; j < Num_BS; ++j){
    //         for (int k = 0; k < Num_tti; ++k){
    //             for (int l = 0; l < Num_RBG; ++l){
    //                 if (div<3){
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(0*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*3+div)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(0*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*3+div)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //                 }else if (div<6)
    //                 {
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(1*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*3+div-3)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(1*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*3+div-3)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //                 }else if (div<8)
    //                 {
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(2*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*2+div-6)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(2*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*2+div-6)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //                 }else{
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(3*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*2+div-8)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(3*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (remain*2+div-8)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //                 }
                    
                    
    //                 // std::cout << "Now here H_r:" <<H_r(remain*dims_i[0] + div*dims_i[1] + j*dims_r[2] + k*dims_r[3] + l) << std::endl;
    //                 // H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(remain*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + div*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(remain*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + div*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //             }
    //         }
    //     }

    // }

    // Highly Correlated Case
    // for (int i = 0; i < Num_UE; ++i){
    //     int remain = i % 20;
    //     int div = i / 20;
    //     for (int j = 0; j < Num_BS; ++j){
    //         for (int k = 0; k < Num_tti; ++k){
    //             for (int l = 0; l < Num_RBG; ++l){
    //                 H(i*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r[div*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + remain*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l], H_i[div*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + remain*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l]);
    //             }
    //         }
    //     }

    // }
// ******************************************** CONFIGURE = 8 *******************************************************************




    // Function to calculate combinations

    auto start = std::chrono::high_resolution_clock::now();
    double assigned_RBG = 0;
    // std::vector<std::array<std::array<double, Num_UE>, Num_slice>> cap_rbg_slice_ue (Num_RBG);
    std::vector<std::array<double, Num_slice>> cap_rbg_slice (Num_RBG);
    std::vector<std::array<int, Num_slice>> action_rbg_slice (Num_RBG);

    std::vector<double> total_slice_tp(Num_slice, 0); // Make room for Num_slice and initialize to 0
    std::vector<int> UE_list;
    for (int num = 0; num < Num_UE; ++num) {
        UE_list.push_back(num);
    }

    for (size_t tti = 0; tti < total_tti; tti++)
    {
        
        std::cout << "TTI:" << tti << std::endl;
        std::vector<int> rbgs(Num_slice, 0);
        std::vector<double> delta_data(Num_slice);
        for (size_t sl = 0; sl < Num_slice; sl++)
        {
            delta_data[sl] = SLAs[sl] * (tti+1) - total_slice_tp[sl];
        }
        // std::cout << "Start here:" << tti << std::endl;
        for (size_t rb = 0; rb < Num_RBG; rb++)
        {
            std::cout << "RBG:" << rb << std::endl;
            arma::cx_mat H_t(Num_UE, Num_BS, arma::fill::zeros);
            for (size_t i = 0; i < Num_UE; i++)
            {
                for (size_t j = 0; j < Num_BS; j++)
                {
                    H_t(i,j) = H(i*Num_BS*Num_tti*Num_RBG + j*Num_tti*Num_RBG + tti*Num_RBG + rb);
                }
            }
            
            for (int sl = 0; sl < Num_slice; ++sl) {
                std::cout << "Slice:" << sl << std::endl;
                auto slice_UEs = std::vector<double>(UE_list.begin() + sl * Num_UE_ps, UE_list.begin() + (sl + 1) * Num_UE_ps);
                
                auto [action, capacity] = choose_action(slice_UEs, H_t, SEL_UE);
                
                action_rbg_slice[rb][sl] = action;
                
                // for (size_t i = 0; i < Num_UE; i++)
                // {
                //     cap_rbg_slice_ue[rb][sl][i] = capacitiesUE[i];
                // }
                cap_rbg_slice[rb][sl] = capacity;
            }
        }

        std::vector<std::array<double, Num_slice>> new_matrix(cap_rbg_slice);
        
        while (any_positive(delta_data)) {
            auto order = std::distance(delta_data.begin(), std::max_element(delta_data.begin(), delta_data.end()));
            std::vector<double> cap_rb_sl(new_matrix.size());
            for (size_t i = 0; i < new_matrix.size(); i++)
            {
                cap_rb_sl[i] = new_matrix[i][order];
            }
            auto rbg_num = std::distance(cap_rb_sl.begin(), std::max_element(cap_rb_sl.begin(), cap_rb_sl.end()));
            rbgs[order] += 1;


            delta_data[order] -= new_matrix[rbg_num][order];
            total_slice_tp[order] += new_matrix[rbg_num][order];
            new_matrix.erase(new_matrix.begin() + rbg_num);
        }

        int sum = std::accumulate(rbgs.begin(), rbgs.end(), 0);
        assigned_RBG += sum;
        // std::cout << "Bugs here:" << tti << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << duration.count() / total_tti << " ms" << std::endl;

    std::vector<double> avg_slice_tp(Num_slice);
    for (size_t i = 0; i < Num_slice; i++)
    {
        avg_slice_tp[i] = total_slice_tp[i] / total_tti;
        std::cout << "Average TP is: " << avg_slice_tp[i] << std::endl;
    }

    double avg_num_rb = assigned_RBG / total_tti;
    std::cout << "Average Allocated RBG is: "<< avg_num_rb << std::endl;
    
    return 0;
}
