#pragma once

#include "api.hpp"

#include <sklk-mii/message_queue.hpp>

#include <sklkphy/common.hpp>
#include <sklkphy/modding.hpp>

#include <cstdint>
#include <map>
#include <memory>

class ref_design_rpc_handler;
class ref_design_csi_mod;
class ref_design_schedule_mod;

class SKLK_PHY_MOD_REFDESIGN_API ref_design_mod_loader : public sklk_phy_mod_loader {
    static constexpr size_t _max_frame_delay{10};

    //! [weight page queue]
    sklk_mii_message_queue<std::tuple<size_t, sklk_phy_weight_page_id_t>, _max_frame_delay*SKLK_PHY_MAX_BANDS> _dl_schedule_weight_pages;
    sklk_mii_message_queue<std::tuple<size_t, sklk_phy_weight_page_id_t>, _max_frame_delay*SKLK_PHY_MAX_BANDS> _ul_schedule_weight_pages;
    //! [weight page queue]

public:
    explicit ref_design_mod_loader(const sklk_phy_scheduler_config & config);
    ~ref_design_mod_loader() override = default;

    std::shared_ptr<ref_design_rpc_handler> rpc_hdl;
    std::weak_ptr<ref_design_csi_mod> csi_mod;
    std::weak_ptr<ref_design_schedule_mod> scedule_mod;

    //! [send the weight page]
    void send_weight_page(size_t resource_blk_no, bool is_downlink, const sklk_phy_weight_page_id_t &page_hdl);
    //! [send the weight page]

    //! [receiving the weight]
    template<typename Callback>
    void get_weight_pages(bool is_downlink, Callback callback) {
        auto &queue = is_downlink ? _dl_schedule_weight_pages : _ul_schedule_weight_pages;
        std::tuple<size_t, sklk_phy_weight_page_id_t> msg;
        while (queue.pop(msg)) {
            const auto &[resource_blk_no, page_hdl] = msg;
            callback(resource_blk_no, page_hdl);
        }
    }
    //! [receiving the weight]

    void add_rpc_commands(jsonrpccxx::JsonRpc2Server &rpc_server [[maybe_unused]]) override;

    /**
     * Update the RPC thread.
     */
    void rpc_get_updates() override;
};

/**
 * Overloads the base modding factory to create a new factory for a custom loader.
 */
class SKLK_PHY_MOD_REFDESIGN_API ref_design_mod_loader_factory : public sklk_phy_mod_loader_factory {
public:
    ref_design_mod_loader_factory() = default;
    virtual ~ref_design_mod_loader_factory() = default;

    std::shared_ptr<sklk_phy_mod_loader> create(const sklk_phy_scheduler_config & config) override;
};
