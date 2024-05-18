#include <iostream>
#include <complex>
#include "H5Cpp.h"
#include <cmath>
#include <iomanip>
// #include <armadillo>
#include <vector>
#include <algorithm> // For std::max_element and std::distance
#include <numeric>   // For std::accumulate
// #include "max_rate.h"
// #include "func.h"
#include <chrono>
#include "alloc_rb.h"

using namespace H5;

bool any_positive(const std::vector<double>& data) {
    return std::any_of(data.begin(), data.end(), [](double x) { return x > 1; });
}



int main(){
    //************************************************** Load Channel Dataset *******************************************************************
    // H5File file("../DU_8_26_64_100_52.hdf5", H5F_ACC_RDONLY);

    // // Read the dataset 'H_r' and 'H_i'
    // DataSet dataset_r = file.openDataSet("H_r");
    // DataSet dataset_i = file.openDataSet("H_i");

    // // Get the dataspace of the datasets
    // DataSpace dataspace_r = dataset_r.getSpace();
    // DataSpace dataspace_i = dataset_i.getSpace();

    // // Get the dimensions of the datasets
    // hsize_t dims_r[5];
    // hsize_t dims_i[5];
    // dataspace_r.getSimpleExtentDims(dims_r);
    // dataspace_i.getSimpleExtentDims(dims_i);


    // arma::vec H_r(dims_r[0] * dims_r[1] * dims_r[2] * dims_r[3] * dims_r[4], arma::fill::zeros);
    // arma::vec H_i(dims_r[0] * dims_r[1] * dims_r[2] * dims_r[3] * dims_r[4], arma::fill::zeros);

    // // Read data from the datasets
    // dataset_r.read(H_r.memptr(), PredType::NATIVE_DOUBLE);
    // dataset_i.read(H_i.memptr(), PredType::NATIVE_DOUBLE);

    // // arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    // // Close the datasets and file
    // dataset_r.close();
    // dataset_i.close();
    // file.close();

    // // Print the shape
    // std::cout << "(" << dims_r[0] << ", " << dims_r[1] << ", " << dims_r[2] << ", " << dims_r[3] << ", " << dims_r[4] << ")" << std::endl;
    // //************************************************** Load Channel Dataset *******************************************************************

    //************************************************** Load Gain and Grouping *******************************************************************
    H5File file_2("../DU_config_400_cg_ug_even.hdf5", H5F_ACC_RDONLY);

    // Try to open the dataset
    DataSet dataset = file_2.openDataSet("var_length_lists");
    DataType datatype = dataset.getDataType();

    // Get dataspace of the dataset.
    DataSpace dataspace = dataset.getSpace();

    // Get the dimension size of each dimension in the dataspace and display them.
    hsize_t num_elements;
    dataspace.getSimpleExtentDims(&num_elements);


    std::vector<hvl_t> read_data(num_elements); // For variable-length data
    dataset.read(read_data.data(), datatype); // Read directly into hvl_t array

    // Store group in a new vector
    std::vector<std::vector<int>> group_vector;

    for (const auto& elem : read_data) {
        // Cast the void pointer to an int pointer
        int* int_array = static_cast<int*>(elem.p);

        // Create a vector from the int array
        std::vector<int> temp(int_array, int_array + elem.len);  // Using the vector range constructor

        // Add the newly created vector to the destination vector
        group_vector.push_back(temp);
    }

    for (size_t i = 0; i < num_elements; ++i) {
        free(read_data[i].p);
    }

    // for (size_t i = 0; i < num_elements; ++i) {
    //     int* data_array = static_cast<int*>(read_data[i].p);
    //     size_t len = read_data[i].len;
    //     std::cout << "Data " << i << ": ";
    //     for (size_t j = 0; j < len; ++j) {
    //         std::cout << data_array[j] << " ";
    //     }
    //     std::cout << std::endl;
    //     // free(read_data[i].p);
    // }

    // int* list = static_cast<int*>(read_data[30*52*3].p);
    // size_t len = read_data[30*52*3].len;
    // std::vector<int> list_l(len);
    // for (size_t j = 0; j < len; ++j) {
    //     list_l[j] = list[j];
    // }
    // for (size_t i = 0; i < list_l.size(); i++)
    // {
    //     std::cout << list_l[i] << std::endl;
    // }
    
    DataSet dataset_len = file_2.openDataSet("len");
    DataSet dataset_cg = file_2.openDataSet("cg");

    // Get the dataspace of the datasets
    DataSpace dataspace_len = dataset_len.getSpace();
    DataSpace dataspace_cg = dataset_cg.getSpace();

    // Get the dimensions of the datasets
    hsize_t dims_len[1];
    hsize_t dims_cg[3];
    dataspace_len.getSimpleExtentDims(dims_len);
    dataspace_cg.getSimpleExtentDims(dims_cg);

    // std::cout << "(" << num_elements << ", " << dims_len[0] << ", " << dims_cg[0] << ", "<< dims_cg[1] << ", " << dims_cg[2] << ")" << std::endl;

    arma::vec group_len(dims_len[0], arma::fill::zeros);
    arma::vec cg_tti_rb_ue(dims_cg[0] * dims_cg[1] * dims_cg[2], arma::fill::zeros);;

    
    // Read data from the datasets
    dataset_len.read(group_len.memptr(), PredType::NATIVE_DOUBLE);
    dataset_cg.read(cg_tti_rb_ue.memptr(), PredType::NATIVE_DOUBLE);

    

    // Close the datasets and file
    // dataset.close();
    dataset_len.close();
    dataset_cg.close();
    file_2.close();

    std::cout << "READING DATA GOOD FOR NOW" << std::endl;

    //************************************************** Load Gain and Grouping *******************************************************************

    const int Num_tti = 10;
    const int Num_RBG = 52;
    const int Num_BS = 64;
    const double corr_th = 0.5;
    const int total_tti = 10;
    // const int config = 4;

// ******************************************** CONFIGURE = 4 *******************************************************************
    // const int Num_UE = 16;
    // const int Num_slice = 4;
    // const int num_ps = Num_UE / Num_slice;
    // std::vector<int> Num_UE_ps(Num_RBG, num_ps);
    // const int SEL_UE = 3;
    // std::vector<int> SLAs = {135, 120, 130, 140};
    // // std::vector<int> SLAs = {235, 220, 230, 240};

    // arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    // std::cout << "Now here" << std::endl;
    // // Evenly Distributed Case
    // for (int ue = 0; ue < Num_UE; ++ue){
    //     int remain = ue % Num_slice;
    //     int div = ue / Num_slice;
    //     for (int j = 0; j < Num_BS; ++j){
    //         for (int k = 0; k < Num_tti; ++k){
    //             for (int l = 0; l < Num_RBG; ++l){
    //                 // std::cout << "Now here H_r:" <<H_r(remain*dims_i[0] + div*dims_i[1] + j*dims_r[2] + k*dims_r[3] + l) << std::endl;
    //                 H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(remain*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + div*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(remain*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + div*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //             }
    //         }
    //     }

    // }

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
    // const int num_ps = Num_UE / Num_slice;
    // std::vector<int> Num_UE_ps(Num_RBG, num_ps);
    // const int SEL_UE = 16;

    // std::vector<double> SLAs = {135, 120, 130, 140, 45, 110, 50, 125};
    // // std::vector<double> SLAs = {235, 220, 230, 240, 145, 210, 150, 225};

    // arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    // // // std::cout << "Now here" << std::endl;
    // // // Evenly Distributed Case
    // for (int ue = 0; ue < Num_UE; ++ue){
    //     int remain = ue % 10;
    //     int div = ue / 10;
    //     for (int j = 0; j < Num_BS; ++j){
    //         for (int k = 0; k < Num_tti; ++k){

    //             for (int l = 0; l < Num_RBG; ++l){
    //                 if (remain<3){
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(0*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*3+remain)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(0*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*3+remain)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //                 }else if (remain<6)
    //                 {
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(1*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*3+remain-3)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(1*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*3+remain-3)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //                 }else if (remain<8)
    //                 {
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(2*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*2+remain-6)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(2*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*2+remain-6)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
    //                 }else{
    //                     H(ue*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r(3*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*2+remain-8)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l), H_i(3*dims_i[1]*dims_i[2]*dims_i[3]*dims_i[4] + (div*2+remain-8)*dims_i[2]*dims_i[3]*dims_i[4] + j*dims_i[3]*dims_i[4] + k*dims_i[4] + l));
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

// ******************************************** CONFIGURE = 400 *******************************************************************
    const int Num_UE = 200;
    const int Num_slice = 8;
    // const int num_ps = Num_UE / Num_slice;
    std::vector<int> Num_UE_ps = {10, 12, 18, 20, 25, 33, 45, 37};
    const int SEL_UE = 16;
    // std::vector<double> SLAs = {135, 120, 130, 140, 45, 110, 50, 125};
    std::vector<double> SLAs = {235, 220, 230, 240, 145, 210, 150, 225};

    // arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    H5File file_200("../DU_200_H.hdf5", H5F_ACC_RDONLY);

    // Read the dataset 'H_r' and 'H_i'
    DataSet dataset_r_200 = file_200.openDataSet("H_r_200");
    DataSet dataset_i_200 = file_200.openDataSet("H_i_200");

    // Get the dataspace of the datasets
    DataSpace dataspace_r_200 = dataset_r_200.getSpace();
    DataSpace dataspace_i_200 = dataset_i_200.getSpace();

    // Get the dimensions of the datasets
    hsize_t dims_r_200[4];
    hsize_t dims_i_200[4];
    dataspace_r_200.getSimpleExtentDims(dims_r_200);
    dataspace_i_200.getSimpleExtentDims(dims_i_200);


    arma::vec H_r_200(dims_r_200[0] * dims_r_200[1] * dims_r_200[2] * dims_r_200[3], arma::fill::zeros);
    arma::vec H_i_200(dims_i_200[0] * dims_i_200[1] * dims_i_200[2] * dims_i_200[3], arma::fill::zeros);

    // Read data from the datasets
    dataset_r_200.read(H_r_200.memptr(), PredType::NATIVE_DOUBLE);
    dataset_i_200.read(H_i_200.memptr(), PredType::NATIVE_DOUBLE);

    // std::cout << "(" << num_elements << ", " << dims_r_200[0] << ", " << dims_r_200[1] << ", "<< dims_r_200[2] << ", " << dims_r_200[3] << ")" << std::endl;

    // arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    // Close the datasets and file
    dataset_r_200.close();
    dataset_i_200.close();
    file_200.close();

    arma::cx_vec H(Num_UE * Num_BS * Num_tti * Num_RBG, arma::fill::zeros);

    for (int i = 0; i < Num_UE; ++i){
        for (int j = 0; j < Num_BS; ++j){
            for (int k = 0; k < Num_tti; ++k){
                for (int l = 0; l < Num_RBG; ++l){
                    // std::cout << "Now here H_r:" <<H_r(remain*dims_i[0] + div*dims_i[1] + j*dims_r[2] + k*dims_r[3] + l) << std::endl;
                    H(i*Num_BS*Num_tti*Num_RBG+j*Num_tti*Num_RBG+k*Num_RBG+l) = std::complex<double>(H_r_200(i*dims_r_200[1]*dims_r_200[2]*dims_r_200[3] + j*dims_r_200[2]*dims_r_200[3] + k*dims_r_200[3] + l), H_i_200(i*dims_i_200[1]*dims_i_200[2]*dims_i_200[3] + j*dims_i_200[2]*dims_i_200[3] + k*dims_i_200[3] + l));
                }
            }
        }

    }





// ******************************************** CONFIGURE = 400 *******************************************************************

    std::vector<double> total_slice_tp(Num_slice);
    for (size_t i = 0; i < Num_slice; i++)
    {
        total_slice_tp[i] = SLAs[i] * 100;
    }
    double total_rb_allocated = 0;
    int period = 1;
    int new_rb_para = 1;
    double total_time = 0;
    double avg_rb = 0;
    for (size_t tti = 0; tti < total_tti; tti++)
    {   
        std::cout << "TTI:" << tti << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> delta_slice(Num_slice);
        for (size_t sl = 0; sl < Num_slice; sl++)
        {
            delta_slice[sl] = SLAs[sl] * (100+tti+1) - total_slice_tp[sl];
        }

        std::vector<double> last_slice_tp = total_slice_tp;
        std::vector<int> RBG_remain;
        for (int num = 0; num < Num_RBG; ++num) {
            RBG_remain.push_back(num);
        }
        arma::mat cg_rb_ue(Num_RBG, Num_UE, arma::fill::zeros);
        for (size_t i = 0; i < Num_RBG; i++)
        {
            for (size_t j = 0; j < Num_UE; j++)
            {
                cg_rb_ue(i,j) = cg_tti_rb_ue[tti*Num_RBG*Num_UE + i*Num_UE + j];
            }
            
        }

        // if (!new_rb_para)
        // {
        //     if (tti>1)
        //     {
        //         int parallel = std::sqrt(current_alloc_rb * 2.0) +1;
        //     }else{
        //         int parallel = 1;
        //     }
            
        // }
        auto cg_time = std::chrono::high_resolution_clock::now();

        
        double rb_alloc_time = 0;
        while (any_positive(delta_slice) && !RBG_remain.empty()) {
            
            auto loop_time = std::chrono::high_resolution_clock::now();
            std::vector<int> UE_list;
            for (int num = 0; num < Num_UE; ++num) {
                UE_list.push_back(num);
            }

            std::vector<double> non_zero;
            double mean_delta = 0;
            
            int parallel = 0;
            // Find Non-zero Elements in Delta_slice
            for (double x : delta_slice) {
                if (x > 0) {
                    non_zero.push_back(x);
                }
            }
            // Define Mean_delta to classify Large Group and Small Group
            if (!non_zero.empty()) {
                mean_delta = std::accumulate(non_zero.begin(), non_zero.end(), 0.0) / non_zero.size();
            } else {
                // Calculate mean of delta_slice if non_zero is empty
                if (!delta_slice.empty()) {
                    mean_delta = std::accumulate(delta_slice.begin(), delta_slice.end(), 0.0) / delta_slice.size();
                }
            }
            // Define Parallel
            if (new_rb_para)
            {
                if (tti>1)
                {
                    double max_non_zero = *std::max_element(non_zero.begin(), non_zero.end());
                    parallel = static_cast<int>((max_non_zero * non_zero.size() / avg_rb)) + 1;
                    // std::cout << "max_non_zero, parallel and avg_rb:" << max_non_zero <<"," <<parallel << "," << avg_rb << std::endl; 
                    // parallel = 1;
                }else{
                    parallel = 1;
                }
                
            }
            std::cout << "Current Parallel:" << parallel << std::endl;
            std::vector<int> large_ue_list;
            std::vector<int> small_ue_list;
            std::vector<int> prefix_sum(Num_UE_ps.size() + 1, 0);
            std::partial_sum(Num_UE_ps.begin(), Num_UE_ps.end(), prefix_sum.begin() + 1);
            // std::cout << "mean_delta:" << mean_delta << std::endl;
            for (int sl = 0; sl < Num_slice; ++sl) {
                if (delta_slice[sl] >= mean_delta) {
                    large_ue_list.insert(large_ue_list.end(), 
                                        UE_list.begin() + prefix_sum[sl], 
                                        UE_list.begin() + prefix_sum[sl + 1]);
                } else if (delta_slice[sl] < mean_delta && delta_slice[sl] > 0) {
                    small_ue_list.insert(small_ue_list.end(), 
                                        UE_list.begin() + prefix_sum[sl], 
                                        UE_list.begin() + prefix_sum[sl + 1]);
                }
            }

            // std::cout << "Large UE lists: ";
            // std::for_each(large_ue_list.begin(), large_ue_list.end(), [](int n) { std::cout << n << " "; });
            // std::cout << std::endl;

            // Get cg_large
            arma::mat cg_large(RBG_remain.size(), large_ue_list.size(), arma::fill::zeros);

            for (size_t i = 0; i < RBG_remain.size(); i++)
            {
                for (size_t j = 0; j < large_ue_list.size(); j++)
                {
                    cg_large(i,j) = cg_rb_ue(i,large_ue_list[j]);
                }
            }
            
            if (parallel == 1)
            {
                std::cout << "One RB" << std::endl;
                double max_value = 0;
                std::pair<int, int> max_index(-1, -1);
                // Search for the maximum value
                for (size_t i = 0; i < cg_large.n_rows; i++) {
                    for (size_t j = 0; j < cg_large.n_cols; j++) {
                        if (cg_large(i,j) > max_value) {
                            max_value = cg_large(i,j);
                            max_index = {i, j};  // Store the new indices
                        }
                    }
                }
                int rbg_index = max_index.first;
                int sel_rbg = RBG_remain[rbg_index];
                int ue_index = large_ue_list[max_index.second];

                // std::cout << "RBG(rbg_index):"<< sel_rbg << "," << rbg_index << std::endl;

                arma::cx_mat H_t(Num_UE, Num_BS, arma::fill::zeros);
                for (size_t i = 0; i < Num_UE; i++)
                {
                    for (size_t j = 0; j < Num_BS; j++)
                    {
                        H_t(i,j) = H(i*Num_BS*Num_tti*Num_RBG + j*Num_tti*Num_RBG + tti*Num_RBG + sel_rbg);
                    }
                }

                arma::vec cg_ue_vec(Num_UE, arma::fill::zeros);
                for (size_t i = 0; i < Num_UE; i++)
                {
                    cg_ue_vec[i] = cg_rb_ue(rbg_index,i);
                }
                
                int sum_lower; 
                if (tti==0 && sel_rbg==0)
                {
                    sum_lower = 0;
                }else{
                    sum_lower = arma::sum(group_len.subvec(0, tti*Num_RBG+sel_rbg-1));
                }
                
                int sum_upper = arma::sum(group_len.subvec(0, tti*Num_RBG+sel_rbg));

                std::vector<std::vector<int>> group_target(group_vector.begin()+ sum_lower, group_vector.begin()+ sum_upper);
                
                auto[sel_UE_list, cap_per_ue] = alloc_rb(group_target, SEL_UE, cg_ue_vec, sel_rbg, tti, H_t, ue_index, large_ue_list, small_ue_list);

                // Equal Num of UEs in Each Slice
                // for (size_t i = 0; i < sel_UE_list.size(); ++i) {
                //     int ue = sel_UE_list[i];
                //     int index = ue / num_ps; // Integer division
                //     total_slice_tp[index] += cap_per_ue[i]; // Update total_slice_tp at index
                // }

        
                // std::cout << "cap_per_ue: ";
                // std::for_each(cap_per_ue.begin(), cap_per_ue.end(), [](int n) { std::cout << n << " "; });
                // std::cout << std::endl;


                // Different Num of UEs in Each Slice
                for (size_t i = 0; i < sel_UE_list.size(); i++) {
                    int ue = sel_UE_list[i];
                    if (ue < 10)
                    {
                        total_slice_tp[0] += cap_per_ue[i];
                    }else if (ue < 22)
                    {
                        total_slice_tp[1] += cap_per_ue[i];
                    }else if (ue < 40)
                    {
                        total_slice_tp[2] += cap_per_ue[i];
                    }else if (ue < 60)
                    {
                        total_slice_tp[3] += cap_per_ue[i];
                    }else if (ue < 85)
                    {
                        total_slice_tp[4] += cap_per_ue[i];
                    }else if (ue < 118)
                    {
                        total_slice_tp[5] += cap_per_ue[i];
                    }else if (ue < 163)
                    {
                        total_slice_tp[6] += cap_per_ue[i];
                    }else if (ue < 200)
                    {
                        total_slice_tp[7] += cap_per_ue[i];
                    }   
                }

                cg_rb_ue.shed_row(rbg_index);;
                RBG_remain.erase(RBG_remain.begin() + rbg_index);
                avg_rb = std::accumulate(cap_per_ue.begin(), cap_per_ue.end(), 0.0) / parallel;
                auto end_in = std::chrono::high_resolution_clock::now();
                auto loop_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_in - loop_time);
                rb_alloc_time += loop_duration.count();

            }else{
                std::cout << "Multiple RBs and parallel:"<< parallel << std::endl;
                // std::vector<int> large_ue_list;
                // std::vector<int> small_ue_list;
                double total_rate = 0;
                auto iter_time = std::chrono::high_resolution_clock::now();
                for (size_t count= 0; count < parallel; count++)
                {
                    
                    double max_value = 0;
                    std::pair<int, int> max_index(-1, -1);
                    // std::cout << "Now Parallel:" << count << std::endl;
                    // Search for the maximum value
                    for (size_t i = 0; i < cg_large.n_rows; i++) {
                        for (size_t j = 0; j < cg_large.n_cols; j++) {
                            if (cg_large(i,j) > max_value) {
                                max_value = cg_large(i,j);
                                max_index = {i, j};  // Store the new indices
                            }
                        }
                    }
                    
                    int rbg_index = max_index.first;
                    int sel_rbg = RBG_remain[rbg_index];
                    int ue_index = large_ue_list[max_index.second];
                    
                    // std::cout << "RBG:"<< sel_rbg << std::endl;

                    std::vector<int> large_ue = large_ue_list;
                    std::vector<int> small_ue = small_ue_list;
                    

                    arma::cx_mat H_t(Num_UE, Num_BS, arma::fill::zeros);
                    for (size_t i = 0; i < Num_UE; i++)
                    {
                        for (size_t j = 0; j < Num_BS; j++)
                        {
                            H_t(i,j) = H(i*Num_BS*Num_tti*Num_RBG + j*Num_tti*Num_RBG + tti*Num_RBG + sel_rbg);
                        }
                    }
                    // std::cout << "Now Good:" << cg_rb_ue.n_rows << "," << cg_large.n_rows << std::endl;
                    arma::vec cg_ue_vec(Num_UE, arma::fill::zeros);

                    for (size_t i = 0; i < Num_UE; i++)
                    {
                        cg_ue_vec[i] = cg_rb_ue(rbg_index,i);
                        // cg_ue_vec[i] = 1;
                    }
                    // std::cout << "Now Also Good:" << count << std::endl;
                    int sum_lower; 
                    if (tti==0 && sel_rbg==0)
                    {
                        sum_lower = 0;
                    }else{
                        sum_lower = arma::sum(group_len.subvec(0, tti*Num_RBG+sel_rbg-1));
                    }
                    int sum_upper = arma::sum(group_len.subvec(0, tti*Num_RBG+sel_rbg));
                    

                    std::vector<std::vector<int>> group_target(group_vector.begin()+ sum_lower, group_vector.begin()+ sum_upper);
                    // std::cout << "Before alloc_rb Good:" << count << std::endl;
                    auto[sel_UE_list, cap_per_ue] = alloc_rb(group_target, SEL_UE, cg_ue_vec, sel_rbg, tti, H_t, ue_index, large_ue, small_ue);
                    // std::cout << "After alloc_rb Good:" << count << std::endl;
                    // Equal Num of UEs in Each Slice
                    // for (size_t i = 0; i < sel_UE_list.size(); ++i) {
                    //     int ue = sel_UE_list[i];
                    //     int index = ue / num_ps; // Integer division
                    //     total_slice_tp[index] += cap_per_ue[i]; // Update total_slice_tp at index
                    // }
                    
                    // Different Num of UEs in Each Slice
                    for (size_t i = 0; i < sel_UE_list.size(); i++) {
                        int ue = sel_UE_list[i];
                        if (ue < 10)
                        {
                            total_slice_tp[0] += cap_per_ue[i];
                        }else if (ue < 22)
                        {
                            total_slice_tp[1] += cap_per_ue[i];
                        }else if (ue < 40)
                        {
                            total_slice_tp[2] += cap_per_ue[i];
                        }else if (ue < 60)
                        {
                            total_slice_tp[3] += cap_per_ue[i];
                        }else if (ue < 85)
                        {
                            total_slice_tp[4] += cap_per_ue[i];
                        }else if (ue < 118)
                        {
                            total_slice_tp[5] += cap_per_ue[i];
                        }else if (ue < 163)
                        {
                            total_slice_tp[6] += cap_per_ue[i];
                        }else if (ue < 200)
                        {
                            total_slice_tp[7] += cap_per_ue[i];
                        }   
                    }

                    cg_rb_ue.shed_row(rbg_index);
                    // std::cout << "After cg_rb Good:" << count << std::endl;
                    cg_large.shed_row(rbg_index);
                    // std::cout << "After cg_large Good:" << count << std::endl;
                    RBG_remain.erase(RBG_remain.begin() + rbg_index);
                    total_rate += std::accumulate(cap_per_ue.begin(), cap_per_ue.end(), 0.0);
                    // std::cout << "After RB_remain Good:" << count << std::endl;
                
                }

                avg_rb = total_rate / parallel;
                auto iter_over = std::chrono::high_resolution_clock::now();
                auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>(iter_time - loop_time);
                auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>(iter_over - iter_time);
                // std::cout << "rb_alloc_time:" << rb_alloc_time << std::endl;
                rb_alloc_time += duration_1.count() + duration_2.count()/parallel;
                // std::cout << "Iter_pre and iter_over:" << rb_alloc_time << "=" << duration_1.count() << "," << duration_2.count()/parallel << std::endl;

            }

            for (size_t sl = 0; sl < Num_slice; sl++)
            {
                delta_slice[sl] = SLAs[sl] * (100+tti+1) - total_slice_tp[sl];
                // std::cout << "Delta of slice" << sl << ":" << delta_slice[sl] << std::endl;
            }
            // std::cout << "Delta of slice: ";
            // std::for_each(delta_slice.begin(), delta_slice.end(), [](double n) { std::cout << n << " "; });
            // std::cout << std::endl;
            
            // if (parallel > 1)
            // {
            //     rb_time = end_in - iter_over + (iter_over-iter_time)/parallel + iter_time - loop_time;
            // }else{
            //     rb_time = end_in - loop_time;
            // }

            // rb_alloc_time += rb_time;
        }
        auto rb_finish = std::chrono::high_resolution_clock::now();

        std::vector<double> slice_tp (Num_slice);

        for (size_t i = 0; i < Num_slice; i++)
        {
            slice_tp[i] = total_slice_tp[i] - last_slice_tp[i];
        }


        // std::cout << "last_slice_tp: ";
        // std::for_each(last_slice_tp.begin(), last_slice_tp.end(), [](int n) { std::cout << n << " "; });
        // std::cout << std::endl;

        double tp_tti = std::accumulate(slice_tp.begin(), slice_tp.end(), 0.0);

        double current_alloc_rb = Num_RBG - RBG_remain.size();
        // std::cout << "# of RBs:" << current_alloc_rb << std::endl;
        total_rb_allocated += current_alloc_rb;
        avg_rb = tp_tti / current_alloc_rb;
        // std::cout << "avg_rb = TP_TTI/ current_alloc_rb: " << avg_rb << "," << tp_tti << "," << current_alloc_rb << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        if (tti>1)
        {
            // std::chrono::microseconds tti_time = end - rb_finish  + cg_time - start;
            auto mid_time = std::chrono::duration_cast<std::chrono::microseconds>(end - rb_finish  + cg_time - start);
            double tti_time = mid_time.count() + rb_alloc_time;
            total_time += tti_time;
            std::cout << "One TTI time:: " << tti_time << "," << mid_time.count() << "," << rb_alloc_time << " us" << std::endl;
        }
    
    }
    
    std::vector<double> avg_slice_tp(Num_slice);
    for (size_t i = 0; i < Num_slice; i++)
    {
        avg_slice_tp[i] = total_slice_tp[i] / total_tti;
        std::cout << "Average TP is: " << avg_slice_tp[i] << std::endl;
    }

    double avg_num_rb = total_rb_allocated / total_tti;
    std::cout << "Average Allocated RBG is: "<< avg_num_rb << std::endl;

    std::cout << "Time taken: " << total_time / (total_tti-2) << " us" << std::endl;

    return 0;
}