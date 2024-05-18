cmake_minimum_required(VERSION 3.10)
project(beam_slice)

# Find Armadillo package
include_directories(./include)
find_package(Armadillo "11.0.0" REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIR})
find_package(HDF5 1.10 REQUIRED COMPONENTS CXX)
message(VERBOSE "  HDF5 Includes: ${HDF5_INCLUDE_DIRS} Libraries: ${HDF5_LIBRARIES}")
include_directories(${HDF5_INCLUDE_DIRS})

# if(TARGET hdf5::hdf5)
#     # cmake >= 3.19
#     target_link_libraries(hdf PRIVATE hdf5::hdf5 hdf5::hdf5_cpp)
# else()
#     # cmake < 3.19
#     target_link_libraries(hdf PRIVATE ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
#     target_include_directories(hdf PRIVATE ${HDF5_C_INCLUDE_DIRS} ${HDF5_CXX_INCLUDE_DIRS}
# endif()

# Add executable
# add_executable(greedy_plus greedy_plus_MR.cpp max_rate.cpp func.cpp)
# target_link_libraries(greedy_plus ${ARMADILLO_LIBRARIES} ${HDF5_LIBRARIES})

# add_executable(test test.cpp)
# target_link_libraries(test ${ARMADILLO_LIBRARIES} ${HDF5_LIBRARIES})


add_executable(RB_share RB_share_para.cpp alloc_rb.cpp)
target_link_libraries(RB_share ${ARMADILLO_LIBRARIES} ${HDF5_LIBRARIES})