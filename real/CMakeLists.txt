# Programs for the real Panda, built against libfranka. The robot's system version decides the
# libfranka version (https://frankaemika.github.io/docs/compatibility.html); a Panda with server
# version 4 needs libfranka 0.8.0, built from source and installed, for example, to
# ~/opt/libfranka-0.8.0:
#
#   cmake -S real -B real/build -DCMAKE_PREFIX_PATH=$HOME/opt/libfranka-0.8.0 -DCMAKE_BUILD_TYPE=Release
#   cmake --build real/build
cmake_minimum_required(VERSION 3.10)
project(panda_sysid_real CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Franka REQUIRED)
message(STATUS "libfranka ${Franka_VERSION} in ${Franka_DIR}")

foreach(program joint_impedance joint_position stream_state recover)
  add_executable(${program} ${program}.cpp)
  target_link_libraries(${program} Franka::Franka)
  # Find libfranka at run time without LD_LIBRARY_PATH.
  set_target_properties(${program} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE
                        BUILD_RPATH "${Franka_DIR}/../..")
endforeach()
