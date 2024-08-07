#include <iostream>
#include <complex>
#include "H5Cpp.h"
#include <cmath>
#include <vector>
#include <algorithm> // For std::max_element and std::distance
#include <numeric>   // For std::accumulate
#include <armadillo>

std::pair<std::vector<int>, double> alloc_rb(std::vector<std::vector<int>>& group_list, const int SEL_UE, const std::vector<int>& Num_UE_ps, int sel_sl, int num_rb, int tti, const arma::cx_mat& H, int ue_index, std::vector<int>& sl_ue_list, arma::vec& cg_ue_vec) {
    int len_remain = SEL_UE;
    std::vector<int> sel_UE_list;

    std::cout << "ue_idx:" << ue_index << ", Slice:" << sel_sl << std::endl;

    if(Num_UE_ps[sel_sl] <= SEL_UE){
        sel_UE_list = sl_ue_list;
        len_remain = 0;
    } else{
        while(len_remain > 0){
            std::cout << "len_remain" << len_remain << std::endl;
            // Need to find another max index
            if(len_remain != SEL_UE){
                std::cout << "Finding New index" << std::endl;
                double max_value = 0;
                int max_index = -1;
                for (size_t j = 0; j < sl_ue_list.size(); j++) {
                    if (cg_ue_vec[sl_ue_list[j]] > max_value) {
                        max_value = cg_ue_vec[sl_ue_list[j]];
                        max_index = sl_ue_list[j];  // Store the new indices
                    }
                }
                ue_index = max_index;
            }

            for (auto it = group_list.begin(); it != group_list.end(); ++it) {
                auto& group = *it;

                if (std::find(group.begin(), group.end(), ue_index) != group.end()) {
                    std::cout << "Find Group" << std::endl;
                    std::cout << "Group lists: ";
                    std::for_each(group.begin(), group.end(), [](int n) { std::cout << n << " "; });
                    std::cout << std::endl;
                    for (int ele : group) {
                        if (std::find(sel_UE_list.begin(), sel_UE_list.end(), ele) == sel_UE_list.end() && std::find(sl_ue_list.begin(), sl_ue_list.end(), ele) != sl_ue_list.end()) {
                            sel_UE_list.push_back(ele);
                            --len_remain;
                            auto idx = std::find(sl_ue_list.begin(), sl_ue_list.end(), ele);
                            sl_ue_list.erase(idx);
                        }
                        if (len_remain<=0) break;
                    }
                    group_list.erase(it);
                    break;
                }
            }
        }
    }


    arma::cx_mat H_s(sel_UE_list.size(), H.n_cols);

    for (size_t i = 0; i < sel_UE_list.size(); ++i) {
        H_s.row(i) = H.row(sel_UE_list[i]);
    }

    std::cout << "Sel UE lists: ";
    std::for_each(sel_UE_list.begin(), sel_UE_list.end(), [](int n) { std::cout << n << " "; });
    std::cout << std::endl;
    
    // std::cout << "H_s:" << H_s << std::endl;
    // std::cout << "Hermitian H_s:" << H_s.t() << std::endl;


    arma::vec sig_user = arma::real(arma::diagvec(arma::inv(H_s * H_s.t())));

    
    
    arma::vec data_rate = arma::log2(1 + 10 / sig_user);
    double sum_of_rate = arma::sum(data_rate);
    std::cout << "Total rate:" << sum_of_rate << std::endl;

    // std::cout << "CAP: ";
    // std::for_each(data_rate.begin(), data_rate.end(), [](double n) { std::cout << n << " "; });
    // std::cout << std::endl;
    // std::cout << "Finish inv:" << std::endl;
    
    // cap_per_ue.row(action).head(idx) = data_rate.t();
    
    // cap_total(action) = arma::accu(data_rate);

    return std::make_pair(sel_UE_list, sum_of_rate);
     
}