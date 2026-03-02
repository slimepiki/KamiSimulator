cmake_minimum_required(VERSION 3.10)
project(HairAnimWithMotionGen CXX CUDA)

set(EXEC_NAME kami)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG -Wall -Wno-comment -O3")
set(CMAKE_C_FLAGS_RELEASE "-DNDEBUG -Wall -Wno-comment -O3")
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
    ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()
set(CMAKE_BUILD_RPATH_USE_ORIGIN on)
set(CMAKE_OPTIMIZE_DEPENDENCIES ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/modules/" ${CMAKE_MODULE_PATH})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -diag-suppress 20012 --expt-relaxed-constexpr")

if(WIN32)

else()
  find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif(WIN32)

find_package(GLUT REQUIRED)
find_package(assimp REQUIRED)
find_package(Boost REQUIRED NO_MODULE)
find_package(Python3 COMPONENTS Interpreter Development)
find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#for MSVC
function(ExternalTarget folder target)
  set_property(TARGET ${target} PROPERTY FOLDER ${folder})
endfunction(ExternalTarget)

function(InternalTarget folder target)
  ExternalTarget("${folder}" ${target})
  if(MSVC)
    get_target_property(targetSources ${target} SOURCES)
    foreach(sourceFile IN ITEMS ${targetSources})
      if(IS_ABSOLUTE "${sourceFile}")
        file(RELATIVE_PATH sourceFile "${CMAKE_CURRENT_SOURCE_DIR}" "${sourceFile}")
      endif()
      get_filename_component(sourceDir "${sourceFile}" PATH)
      string(REPLACE "/" "\\" sourceDir "${sourceDir}")
      source_group("${sourceDir}" FILES "${sourceFile}")
    endforeach()
  endif()
endfunction(InternalTarget)


add_subdirectory("${PROJECT_SOURCE_DIR}/src")
add_subdirectory("${PROJECT_SOURCE_DIR}/extern")

include_directories(${CUDA_INCLUDE_DIRS} "/usr/local/cuda/include")

set_target_properties(${EXEC_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_ARCHITECTURES "86;89;120"
)

add_subdirectory(extern/Discregrid)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-diag-suppress 20012")
set(EXTERN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/extern")
