#include <iostream>
#include <complex>
#include "H5Cpp.h"
#include <cmath>
#include <vector>
#include <algorithm> // For std::max_element and std::distance
#include <numeric>   // For std::accumulate
#include <armadillo>

std::pair<std::vector<int>, arma::vec> alloc_rb(std::vector<std::vector<int>>& group_list, const int SEL_UE, const arma::vec& cg_ue_vec, int num_rb, int tti, const arma::cx_mat& H, int ue_index, std::vector<int>& large_ue_list, std::vector<int>& small_ue_list) {
    int len_remain = SEL_UE;
    std::vector<int> sel_UE_list;

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
                    for (int ele : group) {
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
                    for (int ele : group) {
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
                        for (int ele : group) {
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
                int max_index = -1;
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
                int max_index = -1;
                for (int index : small_ue_list) {
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