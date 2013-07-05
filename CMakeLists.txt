CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CLASSCODE)
 
# Point CMake to the directory containing various build scripts
set(CMakeScripts_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMakeScripts_MODULE_DIR})

include(CacheInternalHelpers)

find_package(GSL)
find_library(GSL_LIBRARY gsl)
find_library(GSLCBLAS_LIBRARY gslcblas)
include_directories(${GSL_INCLUDE_DIR})

find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
		      ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ALL ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			       COMMENT "Generating API documentation with Doxygen" VERBATIM)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc)
endif(DOXYGEN_FOUND)

set(SOURCES
	src/GP_Exception.cc
	src/GP_InputParams.cc 
	src/GP_Vector.cc
	src/GP_Matrix.cc
	src/Thresholding.cc
	src/GP_Evaluation.cc
	src/GP_Optimizer.cc
	src/GP_SigmoidFunction.cc
	)

set(HEADER_SUBDIR include/Classcode)

set(PUBLIC_HEADERS
	${HEADER_SUBDIR}/GP_Constants.hh
	${HEADER_SUBDIR}/GP_InputParams.hh
	${HEADER_SUBDIR}/GP_Vector.hh
	${HEADER_SUBDIR}/GP_Matrix.hh
	${HEADER_SUBDIR}/GP_DataSet.hh
	${HEADER_SUBDIR}/GP_DataReader.hh
	${HEADER_SUBDIR}/GP_UniversalDataReader.hh
	${HEADER_SUBDIR}/GP_Exception.hh
	${HEADER_SUBDIR}/BinaryClassifier.hh
	${HEADER_SUBDIR}/BinaryClassifierWithWeights.hh
	${HEADER_SUBDIR}/Thresholding.hh
	${HEADER_SUBDIR}/AdaBoost.hh
)

cache_internal_init_unique(CLASSCODE_INCLUDE 
	${GSL_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${CLASSCODE_INCLUDE})

add_library(classcode ${SOURCES} ${PUBLIC_HEADERS} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
target_link_libraries(classcode ${GSL_LIBRARY})

#add_executable(regression regression/regression.cc ${CONF_FILE})
#target_link_libraries(regression classcode ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

#add_executable(adaboost classification/adaboost.cc ${CONF_FILE})
#target_link_libraries(adaboost classcode ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

add_executable(clustering mixturemodels/clustering.cc ${CONF_FILE})
target_link_libraries(clustering classcode ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

add_executable(gp-classification mixturemodels/gp-classification.cc ${CONF_FILE})
target_link_libraries(gp-classification classcode ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

install(FILES ${PUBLIC_HEADERS} DESTINATION ${HEADER_SUBDIR})
#install(TARGETS regression GP_predict GP_active_learning RUNTIME DESTINATION bin)

export(TARGETS classcode FILE UseClasscode.cmake)

configure_file( "cmake/ClasscodeConfig.cmake.in" "${PROJECT_BINARY_DIR}/ClasscodeConfig.cmake" @ONLY )

# Stick build location in the central CMake user package registry, so that it may be easily found
if ("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.7)
  export(PACKAGE Classcode)
endif()

