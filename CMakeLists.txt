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

add_executable(test_nn_primitives tests/test_nn_primitives.cpp)
target_link_libraries(test_nn_primitives PRIVATE flowserve)
target_compile_options(test_nn_primitives PRIVATE -UNDEBUG)
add_test(NAME test_nn_primitives COMMAND test_nn_primitives)

add_executable(test_speculative_and_mla tests/test_speculative_and_mla.cpp)
target_link_libraries(test_speculative_and_mla PRIVATE flowserve)
target_compile_options(test_speculative_and_mla PRIVATE -UNDEBUG)
add_test(NAME test_speculative_and_mla COMMAND test_speculative_and_mla)

add_executable(test_weight_loader tests/test_weight_loader.cpp)
target_link_libraries(test_weight_loader PRIVATE flowserve)
target_compile_options(test_weight_loader PRIVATE -UNDEBUG)
add_test(NAME test_weight_loader COMMAND test_weight_loader)

add_executable(test_async_engine tests/test_async_engine.cpp)
target_link_libraries(test_async_engine PRIVATE flowserve)
target_compile_options(test_async_engine PRIVATE -UNDEBUG)
add_test(NAME test_async_engine COMMAND test_async_engine)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE flowserve)
target_compile_options(test_tokenizer PRIVATE -UNDEBUG)
add_test(NAME test_tokenizer COMMAND test_tokenizer)

add_executable(test_transformer_forward tests/test_transformer_forward.cpp)
target_link_libraries(test_transformer_forward PRIVATE flowserve)
target_compile_options(test_transformer_forward PRIVATE -UNDEBUG)
add_test(NAME test_transformer_forward COMMAND test_transformer_forward)

add_executable(bench_serving tools/bench_serving.cpp)
target_link_libraries(bench_serving PRIVATE flowserve)

add_executable(chat_cli tools/chat_cli.cpp)
target_link_libraries(chat_cli PRIVATE flowserve)

