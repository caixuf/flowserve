cmake_minimum_required(VERSION 3.16)
project(flowserve VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_library(flowserve
    src/block_manager.cpp
    src/scheduler.cpp
    src/engine.cpp
)
target_include_directories(flowserve PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

enable_testing()

add_executable(test_block_manager tests/test_block_manager.cpp)
target_link_libraries(test_block_manager PRIVATE flowserve)
target_compile_options(test_block_manager PRIVATE -UNDEBUG)
add_test(NAME test_block_manager COMMAND test_block_manager)

add_executable(test_scheduler tests/test_scheduler.cpp)
target_link_libraries(test_scheduler PRIVATE flowserve)
target_compile_options(test_scheduler PRIVATE -UNDEBUG)
add_test(NAME test_scheduler COMMAND test_scheduler)

add_executable(test_engine_ablation tests/test_engine_ablation.cpp)
target_link_libraries(test_engine_ablation PRIVATE flowserve)
target_compile_options(test_engine_ablation PRIVATE -UNDEBUG)
add_test(NAME test_engine_ablation COMMAND test_engine_ablation)

add_executable(bench_serving tools/bench_serving.cpp)
target_link_libraries(bench_serving PRIVATE flowserve)
