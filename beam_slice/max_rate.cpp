#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <algorithm> // For std::max_element and std::distance
#include <numeric>   // For std::accumulate
#include <armadillo>
#include "func.h"

// int P_N_UE = 3; // How many UEs are scheduled each TTI
int BS_ANT = 64;

// Function to select a set of users based on the given action
std::pair<std::vector<int>, int> sel_ue(const std::vector<double>& user_set, int action, const int P_N_UE) {
    int sum_before = 0;
    int idx = 0;
    std::vector<int> ue_select;

    // Calculate sum of combinations before each iteration
    for (int i = 1; i <= P_N_UE; ++i) {
        sum_before += nCr(user_set.size(), i);
        if ((action + 1) > sum_before) {
            continue;
        } else {
            idx = i;
            sum_before -= nCr(user_set.size(), i);
            // Calculate the index of selected combination
            int comb_idx = action - sum_before;
            // Generate the selected combination
            std::vector<bool> bitmask(user_set.size());
            std::fill(bitmask.begin(), bitmask.begin() + i, true);
            do {
                std::vector<int> current_comb;
                for (int j = 0; j < user_set.size(); ++j) {
                    if (bitmask[j]) {
                        current_comb.push_back(user_set[j]);
                    }
                }
                ue_select = current_comb;
                --comb_idx;
            } while (comb_idx > 0 && std::prev_permutation(bitmask.begin(), bitmask.end()));
            break;
        }
    }

    return std::make_pair(ue_select, idx);
}

std::tuple<int, double> choose_action(const std::vector<double>& user_set, const arma::cx_mat& H, const int P_N_UE) {
    int actions_num = 0;
    for (int i = 1; i <= P_N_UE; ++i) {
        actions_num += nCr(user_set.size(), i);
        // std::cout << "nCr:" << nCr(user_set.size(), i) << std::endl;
    }

    arma::vec cap_total(actions_num, arma::fill::zeros);
    arma::mat cap_per_ue(actions_num, 80, arma::fill::zeros);
    
    for (int action = 0; action < actions_num; ++action) {
        auto [ue_select, idx] = sel_ue(user_set, action, P_N_UE);


        arma::cx_mat H_s(ue_select.size(), H.n_cols);
    
        for (size_t i = 0; i < ue_select.size(); ++i) {
            H_s.row(i) = H.row(ue_select[i]);
        }
        
        // std::cout << "Now here:" << idx << H_s * H_s.t() << std::endl;

        arma::vec sig_user = arma::real(arma::diagvec(arma::inv(H_s * H_s.t())));
        
        arma::vec data_rate = arma::log2(1 + 10 / sig_user);
        
        cap_per_ue.row(action).head(idx) = data_rate.t();
        
        cap_total(action) = arma::accu(data_rate);
    }
    // std::cout << "Now here:" << std::endl;
    int max_cap_index = cap_total.index_max();
    // arma::vec cap_per_ue_vec = cap_per_ue.row(max_cap_index);
    // std::vector<double> cap_per_ue_std(cap_per_ue_vec.begin(), cap_per_ue_vec.end());

    return std::make_tuple(max_cap_index, cap_total(max_cap_index));
}
