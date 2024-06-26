#include "refplat_test.hpp"

#include <sklk-mii/simple_log.hpp>

void RefplatTest::test(sklk_phy_refplat &refplat)
{
    refplat.run_one();

    cpe cpe(1);

    std::array<const char *, 2> pilots_files = {"uaa_900795_upl_1_ch2.ndjson", "uaa_900795_upl_1_ch3.ndjson"};

    for (size_t i = 0; i < pilots_files.size(); ++i)
    {
        pilot_test_vec_t pilots;
        read_pilots_from_file(_pilots_dir / pilots_files[i], pilots);
        cpe.load_pilots(refplat, pilots[0], i);
    }

    cpe.connect(refplat);

    for (int i = 0; i < 500; ++i)
    {
        refplat.run_one();
    }

    check_total_stats(
        refplat,
        {
            {"num_bands", 16},
            {"num_conns", 2},
        }
    );

    cpe.disconnect(refplat);

    for (int i = 0; i < 500; ++i)
    {
        refplat.run_one();
    }

    check_total_stats(
        refplat,
        {
            {"num_bands", 0},
            {"num_conns", 0},
        }
    );

    sklk_mii_log::notice("Reconnecting...");
    cpe.connect(refplat);

    for (int i = 0; i < 500; ++i)
    {
        refplat.run_one();
    }

    check_total_stats(
        refplat,
        {
            {"num_bands", 16},
            {"num_conns", 2},
        }
    );
}
