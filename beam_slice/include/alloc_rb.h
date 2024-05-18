#include <iostream>
#include <armadillo>
// Function declaration
std::pair<std::vector<int>, arma::vec> alloc_rb(std::vector<std::vector<int>>& group_list, const int SEL_UE, const arma::vec& cg_ue_vec, int num_rb, int tti, const arma::cx_mat& H, int ue_index, std::vector<int>& large_ue_list, std::vector<int>& small_ue_list);
