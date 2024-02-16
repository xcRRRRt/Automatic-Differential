cmake_minimum_required(VERSION 3.27)
project(new_dl)

set(CMAKE_CXX_STANDARD 17)

add_executable(new_dl main.cpp
        src/node/node.h
        src/operator/operator.h
        src/operator/basic_operator.h
        src/calculate/cal_map.h
        src/operator/other_operator.h
        src/active_func/sigmoid.h
        src/datatype.h
        src/layer/linear.h
        src/nn.h
        src/loss/cross_entropy.h
        src/utils/init_weights.h
        src/utils/read_data.h
)

include_directories(${PROJECT_SOURCE_DIR}/lib/eigen)