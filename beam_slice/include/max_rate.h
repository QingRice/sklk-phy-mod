#include <iostream>
#include <armadillo>
// Function declaration
int nCr(int n, int r);
std::pair<std::vector<int>, int> sel_ue(const std::vector<int>& user_set, int action, const int P_N_UE);
std::tuple<int, double> choose_action(const std::vector<double>& user_set, const arma::cx_mat& H, const int P_N_UE);
