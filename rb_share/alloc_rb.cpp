#include <iostream>
#include <complex>
//#include "H5Cpp.h"
#include <cmath>
#include <vector>
#include <algorithm> // For std::max_element and std::distance
#include <numeric>   // For std::accumulate
//#include <armadillo>
#include "csi_mod.hpp"

// using namespace H5;

inline bool any_positive(const std::vector<double>& data) {
    return std::any_of(data.begin(), data.end(), [](double x) { return x > 1; });
}


std::pair<std::vector<size_t>, arma::vec> ref_design_csi_mod::alloc_rb(std::vector<std::vector<size_t>>& group_list, size_t SEL_UE, const arma::vec& cg_ue_vec, const arma::cx_mat& H, size_t ue_index, std::vector<size_t>& large_ue_list, std::vector<size_t>& small_ue_list) {
    size_t len_remain = SEL_UE;
    std::vector<size_t> sel_UE_list;

    // for (const auto& group : group_list) {
    //     // Iterate over the inner vector
    //     for (int num : group) {
    //         std::cout << num << " ";
    //     }
    //     std::cout << std::endl;  // Newline for each inner vector
    // }

    // std::cout << "cg_ue_vec: ";
    // std::for_each(cg_ue_vec.begin(), cg_ue_vec.end(), [](double n) { std::cout << n << " "; });
    // std::cout << std::endl;

    // arma::vec cap_per_ue(80, arma::fill::zeros);
    while (len_remain > 0) {
        // std::cout << "ue_index:" << ue_index << std::endl;
        for (auto it = group_list.begin(); it != group_list.end(); ++it) {
            auto& group = *it;
            // size_t index = std::distance(group_list.begin(), it);
            if (std::find(group.begin(), group.end(), ue_index) != group.end()) {
                if (group.size() <= len_remain) {
                    for (size_t ele : group) {
                        if (std::find(sel_UE_list.begin(), sel_UE_list.end(), ele) == sel_UE_list.end()) {
                            auto idx = std::find(large_ue_list.begin(), large_ue_list.end(), ele);
                            auto idx_1 = std::find(small_ue_list.begin(), small_ue_list.end(), ele);
                            if (idx != large_ue_list.end()) {
                                sel_UE_list.push_back(ele);
                                large_ue_list.erase(idx);
                                --len_remain;
                            } else if (idx_1 != small_ue_list.end()) {
                                sel_UE_list.push_back(ele);
                                small_ue_list.erase(idx_1);
                                --len_remain;
                            }
                        }
                    }
                    group_list.erase(it);
                    break;
                } else {
                    for (size_t ele : group) {
                        if (std::find(sel_UE_list.begin(), sel_UE_list.end(), ele) == sel_UE_list.end()) {
                            auto idx_3 = std::find(large_ue_list.begin(), large_ue_list.end(), ele);
                            if (idx_3 != large_ue_list.end()) {
                                sel_UE_list.push_back(ele);
                                large_ue_list.erase(idx_3);
                                --len_remain;
                            }
                        }
                        if (len_remain<=0) break;
                        
                    }

                    if (len_remain > 0) {
                        for (size_t ele : group) {
                            if (std::find(sel_UE_list.begin(), sel_UE_list.end(), ele) == sel_UE_list.end()) {
                                auto idx_4 = std::find(small_ue_list.begin(), small_ue_list.end(), ele);
                                if (idx_4 != small_ue_list.end()) {
                                    sel_UE_list.push_back(ele);
                                    small_ue_list.erase(idx_4);
                                    --len_remain;
                                }
                            }
                            if (len_remain<=0) break;
                        }
                    }
                    break;
                }
            }
        }

        if (len_remain>0) {
            if (!large_ue_list.empty())
            {
                double max_value = 0;
                size_t max_index = SIZE_MAX;
                for (int index : large_ue_list) {
                    if (cg_ue_vec(index) > max_value) {
                        max_value = cg_ue_vec(index);
                        max_index = index;  // Store the actual index within the original vector
                    
                    }
                    // std::cout << "Max Index:" << max_index << std::endl;
                }
                ue_index = max_index;

            }else if (!small_ue_list.empty())
            {
                double max_value = 0;
                size_t max_index = SIZE_MAX;
                for (size_t index : small_ue_list) {
                    if (cg_ue_vec(index) > max_value) {
                        max_value = cg_ue_vec(index);
                        max_index = index;  // Store the actual index within the original vector
                    }
                }
                ue_index = max_index;
            }else{
                break;
            }
        }
    }

    arma::cx_mat H_s(sel_UE_list.size(), H.n_cols);

    for (size_t i = 0; i < sel_UE_list.size(); ++i) {
        H_s.row(i) = H.row(sel_UE_list[i]);
    }

    // std::cout << "Sel UE lists: ";
    // std::for_each(sel_UE_list.begin(), sel_UE_list.end(), [](int n) { std::cout << n << " "; });
    // std::cout << std::endl;
    
    // std::cout << "H_s:" << H_s << std::endl;
    // std::cout << "Hermitian H_s:" << H_s.t() << std::endl;


    arma::vec sig_user = arma::real(arma::diagvec(arma::inv(H_s * H_s.t())));

    
    
    arma::vec data_rate = arma::log2(1 + 10 / sig_user);

    // std::cout << "CAP: ";
    // std::for_each(data_rate.begin(), data_rate.end(), [](double n) { std::cout << n << " "; });
    // std::cout << std::endl;
    // std::cout << "Finish inv:" << std::endl;
    
    // cap_per_ue.row(action).head(idx) = data_rate.t();
    
    // cap_total(action) = arma::accu(data_rate);

    return std::make_pair(sel_UE_list, data_rate);
     
}

std::vector<std::vector<size_t>> ref_design_csi_mod::rb_share(const arma::cx_vec& csi, std::vector<double>& total_slice_tp, size_t tti, double& avg_rb, double& total_rb_allocated){
// csi shape: 1-D arma::cx_vec (Num_UE * Num_BS * Num_RBG); 
// total_slice_tp: accumulated data rate of slices
// avg_rb only for Multi-RB to compute parallel
// total_rb_allocated: count total # of allocated RBs over TTIs
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> delta_slice(Num_slice);
    for (size_t sl = 0; sl < Num_slice; sl++)
    {
        delta_slice[sl] = SLAs[sl] * (tti+1) - total_slice_tp[sl];
    }

    std::vector<double> last_slice_tp = total_slice_tp;
    std::vector<size_t> RBG_remain;
    for (size_t num = 0; num < Num_RBG; ++num) {
        RBG_remain.push_back(num);
    }
    arma::mat cg_rb_ue(Num_RBG, Num_UE, arma::fill::zeros);
    for (size_t i = 0; i < Num_RBG; i++)
    {
        for (size_t j = 0; j < Num_UE; j++)
        {
            cg_rb_ue(i,j) = cg_tti_rb_ue[i*Num_UE + j];
        }
        
    }

    auto cg_time = std::chrono::high_resolution_clock::now();
    
    //***************** Output to AraMIMO ********************
    std::vector<std::vector<size_t>> sel_UE_list_rb;
    std::vector<size_t> sel_rb;
    //***************** Output to AraMIMO ********************

    double rb_alloc_time = 0;
    while (any_positive(delta_slice) && !RBG_remain.empty()) {
        
        auto loop_time = std::chrono::high_resolution_clock::now();
        std::vector<size_t> UE_list;
        for (size_t num = 0; num < Num_UE; ++num) {
            UE_list.push_back(num);
        }

        std::vector<double> non_zero;
        double mean_delta = 0;
        
        size_t parallel = 1;
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
                parallel = static_cast<size_t>((max_non_zero * non_zero.size() / avg_rb)) + 1;

            }else{
                parallel = 1;
            }
            
        }
        // std::cout << "Current Parallel:" << parallel << std::endl;
        std::vector<size_t> large_ue_list;
        std::vector<size_t> small_ue_list;
        std::vector<size_t> prefix_sum(Num_UE_ps.size() + 1, 0);
        std::partial_sum(Num_UE_ps.begin(), Num_UE_ps.end(), prefix_sum.begin() + 1);

        for (size_t sl = 0; sl < Num_slice; ++sl) {
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
            std::pair<size_t, size_t> max_index(SIZE_MAX, SIZE_MAX);
            // Search for the maximum value
            for (size_t i = 0; i < cg_large.n_rows; i++) {
                for (size_t j = 0; j < cg_large.n_cols; j++) {
                    if (cg_large(i,j) > max_value) {
                        max_value = cg_large(i,j);
                        max_index = {i, j};  // Store the new indices
                    }
                }
            }
            size_t rbg_index = max_index.first;
            size_t sel_rbg = RBG_remain[rbg_index];
            size_t ue_index = large_ue_list[max_index.second];

            arma::cx_mat H_t(Num_UE, Num_BS, arma::fill::zeros);
            for (size_t i = 0; i < Num_UE; i++)
            {
                for (size_t j = 0; j < Num_BS; j++)
                {
                    H_t(i,j) = csi(i*Num_BS*Num_RBG + j*Num_RBG + sel_rbg);
                }
            }

            arma::vec cg_ue_vec(Num_UE, arma::fill::zeros);
            for (size_t i = 0; i < Num_UE; i++)
            {
                cg_ue_vec[i] = cg_rb_ue(rbg_index,i);
            }
            
            std::vector<std::vector<size_t>> group_target = group_vector;
            
            auto[sel_UE_list, cap_per_ue] = alloc_rb(group_target, SEL_UE, cg_ue_vec, H_t, ue_index, large_ue_list, small_ue_list);

            sel_rb.push_back(sel_rbg);
            sel_UE_list_rb.push_back(sel_UE_list);


            for (size_t i = 0; i < sel_UE_list.size(); i++) {
                size_t ue = sel_UE_list[i];
                if (ue % 2) // {0, 2, 4, 6, 8, 10} is S1
                {
                    total_slice_tp[0] += cap_per_ue[i];
                }else // {1, 3, 5, 7, 9, 11} is S2
                {
                    total_slice_tp[1] += cap_per_ue[i];
                }
            }

            cg_rb_ue.shed_row(rbg_index);
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
                std::pair<size_t, size_t> max_index(SIZE_MAX, SIZE_MAX);
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
                
                size_t rbg_index = max_index.first;
                size_t sel_rbg = RBG_remain[rbg_index];
                size_t ue_index = large_ue_list[max_index.second];
                
                // std::cout << "RBG:"<< sel_rbg << std::endl;

                std::vector<size_t> large_ue = large_ue_list;
                std::vector<size_t> small_ue = small_ue_list;
                

                arma::cx_mat H_t(Num_UE, Num_BS, arma::fill::zeros);
                for (size_t i = 0; i < Num_UE; i++)
                {
                    for (size_t j = 0; j < Num_BS; j++)
                    {
                        H_t(i,j) = csi(i*Num_BS*Num_RBG + j*Num_RBG + sel_rbg);
                    }
                }
                // std::cout << "Now Good:" << cg_rb_ue.n_rows << "," << cg_large.n_rows << std::endl;
                arma::vec cg_ue_vec(Num_UE, arma::fill::zeros);

                for (size_t i = 0; i < Num_UE; i++)
                {
                    cg_ue_vec[i] = cg_rb_ue(rbg_index,i);
                    // cg_ue_vec[i] = 1;
                }

                std::vector<std::vector<size_t>> group_target = group_vector;

                // std::cout << "Before alloc_rb Good:" << count << std::endl;
                auto[sel_UE_list, cap_per_ue] = alloc_rb(group_target, SEL_UE, cg_ue_vec, H_t, ue_index, large_ue, small_ue);
                
                for (size_t i = 0; i < sel_UE_list.size(); i++) {
                    size_t ue = sel_UE_list[i];
                    if (ue % 2) // {0, 2, 4, 6, 8, 10} is S1
                    {
                        total_slice_tp[0] += cap_per_ue[i];
                    }else // {1, 3, 5, 7, 9, 11} is S2
                    {
                        total_slice_tp[1] += cap_per_ue[i];
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
            delta_slice[sl] = SLAs[sl] * (tti+1) - total_slice_tp[sl];
            // std::cout << "Delta of slice" << sl << ":" << delta_slice[sl] << std::endl;
        }
    }
    auto rb_finish = std::chrono::high_resolution_clock::now();

// ************************* For Multi-RB *************************************
    std::vector<double> slice_tp (Num_slice);

    for (size_t i = 0; i < Num_slice; i++)
    {
        slice_tp[i] = total_slice_tp[i] - last_slice_tp[i];
    }
    double tp_tti = std::accumulate(slice_tp.begin(), slice_tp.end(), 0.0);
    double current_alloc_rb = Num_RBG - RBG_remain.size();
    avg_rb = tp_tti / current_alloc_rb; // Update avg_rb to compute parallel next TTI
// ************************* For Multi-RB *************************************
    total_rb_allocated += current_alloc_rb; // Update # of allocated RBs

    auto end = std::chrono::high_resolution_clock::now();
    if (tti>1)
    {
        auto mid_time = std::chrono::duration_cast<std::chrono::microseconds>(end - rb_finish  + cg_time - start);
        double tti_time = mid_time.count() + rb_alloc_time;
        std::cout << "One TTI time: " << tti_time << "," << mid_time.count() << "," << rb_alloc_time << " us" << std::endl;
    }

    std::vector<std::vector<size_t>> alloc_result(Num_RBG);
    for (size_t i = 0; i < sel_rb.size(); ++i) {
        if (sel_rb[i] < alloc_result.size()) {
            alloc_result[sel_rb[i]] = sel_UE_list_rb[i];
        }
    }

    return alloc_result;
}

