file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.h ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE CU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
list(FILTER SOURCES EXCLUDE REGEX "/tests/*")
list(FILTER SOURCES EXCLUDE REGEX "/rubbish_bin/*")
if(WIN32)
  add_executable(${EXEC_NAME} WIN32 ${SOURCES} ${CU_SOURCES})
else()
  add_executable(${EXEC_NAME} ${SOURCES} ${CU_SOURCES})
endif(WIN32)

if(WIN32)
  target_include_directories(${EXEC_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLM_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
    ${EXTERN_INCLUDE_DIR}
  )
else()
  target_include_directories(${EXEC_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLM_INCLUDE_DIR}
    "/usr/local/cuda/include"
  )
  target_link_libraries(${EXEC_NAME} Eigen3::Eigen GLEW libGLEW.so glfw libglut.so libGLU.so libGL.so assimp boost_system boost_filesystem Discregrid)
endif(WIN32)
InternalTarget("" ${EXEC_NAME})
