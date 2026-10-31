cmake_minimum_required(VERSION 3.20)
project(netprecond CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PETSc through pkg-config: set PKG_CONFIG_PATH to <petsc prefix>/lib/pkgconfig, or PETSC_DIR/PETSC_ARCH
if(DEFINED ENV{PETSC_DIR})
  set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
endif()
find_package(PkgConfig REQUIRED)
pkg_search_module(PETSC REQUIRED IMPORTED_TARGET PETSc)
find_package(HDF5)

option(NETPRECOND_PARHIP "build the ParHIP partitioner backend of net2as" OFF)
option(NETPRECOND_CHOLMOD "build tools/frontal_distribution (needs cholmod from PETSc's SuiteSparse)" ON)
option(NETPRECOND_HYPERHDG "build examples/hyperhdg/network (needs HYPERHDG_DIR)" OFF)

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-format-security -Wno-format-truncation)

add_library(petsc_util src/petsc_util.cxx)
target_link_libraries(petsc_util PUBLIC PkgConfig::PETSC)
target_include_directories(petsc_util PUBLIC src)

add_library(net2as src/net2as.cxx)
target_link_libraries(net2as PRIVATE petsc_util $<$<BOOL:${HDF5_FOUND}>:HDF5::HDF5> PUBLIC PkgConfig::PETSC)
target_include_directories(net2as PUBLIC src)

if(NETPRECOND_PARHIP)
  find_library(PARHIP_LIBRARY parhip_interface HINTS ${PARHIP_ROOT}/lib REQUIRED)
  find_path(PARHIP_INCLUDE_DIR parhip_interface.h HINTS ${PARHIP_ROOT}/include REQUIRED)
  add_library(petsc_parhip src/petsc_parhip.cxx)
  target_link_libraries(petsc_parhip PRIVATE ${PARHIP_LIBRARY} PUBLIC PkgConfig::PETSC)
  target_include_directories(petsc_parhip PRIVATE ${PARHIP_INCLUDE_DIR})
  target_link_libraries(net2as PRIVATE petsc_parhip)
  target_compile_definitions(net2as PUBLIC HYPERHDG_PARHIP)
endif()

if(NETPRECOND_CHOLMOD)
  # cholmod comes with PETSc's suite-sparse, but petsc.pc only carries it in Libs.private
  find_library(CHOLMOD_LIBRARY cholmod HINTS ${PETSC_STATIC_LIBRARY_DIRS} ${PETSC_LIBRARY_DIRS})
  if(CHOLMOD_LIBRARY)
    cmake_path(GET CHOLMOD_LIBRARY PARENT_PATH CHOLMOD_LIBDIR)
    find_path(CHOLMOD_INCLUDE_DIR cholmod.h HINTS ${CHOLMOD_LIBDIR}/../include ${CHOLMOD_LIBDIR}/../include/suitesparse)
    add_executable(frontal_distribution tools/frontal_distribution.cxx)
    target_link_libraries(frontal_distribution PkgConfig::PETSC ${CHOLMOD_LIBRARY})
    target_include_directories(frontal_distribution PRIVATE ${CHOLMOD_INCLUDE_DIR})
  else()
    message(STATUS "cholmod not found: skipping tools/frontal_distribution")
  endif()
endif()

if(NETPRECOND_HYPERHDG)
  # the HyperHDG driver: a checkout of HyperHDG (with its submodules) in HYPERHDG_DIR
  add_subdirectory(${HYPERHDG_DIR} hyperhdg EXCLUDE_FROM_ALL)
  add_executable(network examples/hyperhdg/network.cxx)
  target_include_directories(network PRIVATE examples/hyperhdg)
  target_link_libraries(network HyperHDG net2as petsc_util)
endif()
