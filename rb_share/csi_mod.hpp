#pragma once

#include "api.hpp"

#include <sklkphy/common.hpp>
#include <sklkphy/mimo_rrh_scheduler.hpp>
#include <sklkphy/modding.hpp>

#define ARMA_MAT_PREALLOC (SKLK_PHY_MAX_RADIOS*SKLK_PHY_MAX_MIMO_USERS)
#include <armadillo>
#include <array>
#include <random>

extern const std::string ref_design_csi_mod_name;
class ref_design_mod_loader;

class SKLK_PHY_MOD_REFDESIGN_API ref_design_csi_estimation
{
    sklk_phy_csi_vec _data{};
    bool _valid{false};
    size_t _frame_time;
public:
    void set_csi(size_t frame_time, const sklk_phy_csi_vec &csi) {
        _frame_time = frame_time;
        _data = csi;
        _valid = true;
    }

    const sklk_phy_csi_vec & data() const { return _data; }
    [[nodiscard]] bool is_valid() const {return _valid;}
};

class ref_design_csi_estimations : public std::array<ref_design_csi_estimation, SKLK_PHY_MAX_ESTIMATIONS>
{
public:
    [[nodiscard]] bool ready() const {
        return std::all_of(this->begin(), this->end(),[](const auto & est) {return est.is_valid(); });
    }
};

class SKLK_PHY_MOD_REFDESIGN_API ref_design_csi_radio_container : public sklk_phy_mod_container
{
public:
    std::array<ref_design_csi_estimations, SKLK_PHY_MAX_BANDS> csi{};
};

class SKLK_PHY_MOD_REFDESIGN_API ref_design_csi_mod : public sklk_phy_modding
{
    bool _initialized{false};
    ref_design_mod_loader *_loader;
    const size_t _num_resouce_blks;
    const size_t _max_spatial_streams;
    const size_t _num_estimations;
    std::mt19937 _randomizer;
    std::array<bool, SKLK_PHY_MAX_RADIOS> _radio_enabled{};
    std::array<
        std::array<sklk_phy_csi_vec, SKLK_PHY_MAX_ESTIMATIONS>,
        SKLK_PHY_MAX_BANDS> _cc_values{};

    size_t _last_frame_time{0};

public:
    ref_design_csi_mod(ref_design_mod_loader *loader, const mimo_rrh_scheduler_config &config);
    ~ref_design_csi_mod() override = default;

    void ue_changed(size_t key, const sklk_phy_ue &ue, bool is_new) override;
    void ue_radio_changed(size_t key, const sklk_phy_ue_radio &ue_radio, bool is_new) override;
    void ue_stream_changed(size_t key, const sklk_phy_ue_stream &ue_stream, bool is_new)  override;

    bool run_once() override;

    sklk_phy_mod_container_ptr_t allocate_ue_radio() override;

    void csi_update(size_t frame_time, size_t key, const sklk_phy_ue_radio &ue_radio, size_t resource_blk_no, size_t est_idx, const sklk_phy_csi_vec &vec);

private:
    void _calculate_weights();
    void _calculate_weights(size_t resource_blk_no, bool is_downlink);
    void _calculate_weight_page(const std::vector<sklk_phy_ue_stream> &ue_streams, size_t resource_blk_no, bool is_downlink);
    bool _calculate_weight_page_estimate(const sklk_phy_weight_page_id_t &page_hdl, size_t resource_blk_no, size_t est_idx, bool is_downlink);

    void _scale_pages_for_uplink(sklk_phy_weight_page &page, size_t num_users, size_t num_radios, float amplitude_ceiling);
    void _scale_pages_for_downlink(sklk_phy_weight_page &page, size_t num_users, size_t num_radios, float amplitude_ceiling);
    std::pair<std::vector<size_t>, arma::vec> alloc_rb(std::vector<std::vector<size_t>>& group_list, size_t SEL_UE, const arma::vec& cg_ue_vec, const arma::cx_mat& H, size_t ue_index, std::vector<size_t>& large_ue_list, std::vector<size_t>& small_ue_list);
    std::vector<std::vector<size_t>> rb_share(const arma::cx_vec& csi, std::vector<double>& total_slice_tp, size_t tti, double& avg_rb, double& total_rb_allocated);

//************************* Configurations ***********************************
const size_t Num_RBG = 52;
const size_t Num_BS = 64;
const double corr_th = 0.5;
// const int total_tti = 10;
const size_t Num_UE = 12;
const size_t Num_slice = 2;

std::vector<size_t> Num_UE_ps = {6,6};
const size_t SEL_UE = 8;
std::vector<double> SLAs = {235, 220};

int new_rb_para = 0; // If RB Parallel

//************************* Configurations ***********************************

arma::vec cg_tti_rb_ue = 10 * arma::randu<arma::vec>(Num_RBG * Num_UE);
std::vector<std::vector<size_t>> group_vector = {
    {0, 2, 4, 6, 8, 10},
    {1, 3, 5, 7, 9, 11}
};
// double total_rb_allocated = 0;
// double total_time = 0;
// double avg_rb = 0;

};
