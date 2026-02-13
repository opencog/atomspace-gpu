#
# Master AtomSpace GPU CMake file.
#
# General organization:
# -- check for different compilers, OS'es
# -- search for various required & optional libraries/tools
# -- decide what to build based on above results.
# -- configure various config files.
# -- print pretty summary
#
# Supports CUDA (preferred for NVIDIA) and OpenCL (fallback for AMD/Intel).
# At least one GPU backend must be available.
#

# cogutils already requires 3.12, so may as well ask for that.
CMAKE_MINIMUM_REQUIRED(VERSION 3.12)
IF (COMMAND CMAKE_POLICY)
	# These must be explicitly set, as otherwise warnings are printed.
	CMAKE_POLICY(SET CMP0003 NEW)
	CMAKE_POLICY(SET CMP0037 NEW)
	CMAKE_POLICY(SET CMP0045 NEW)
ENDIF (COMMAND CMAKE_POLICY)

IF(CMAKE_VERSION VERSION_GREATER 3.31)
	CMAKE_POLICY(SET CMP0177 NEW)
ENDIF(CMAKE_VERSION VERSION_GREATER 3.31)

PROJECT(atomspace-gpu VERSION 1.0.0)

# ----------------------------------------------------------
# User-modifiable options. Feel free to change these!
#
# Uncomment to be in Release mode [default].
# SET(CMAKE_BUILD_TYPE Release)

# Uncomment to build in debug mode.
# SET(CMAKE_BUILD_TYPE Debug)

# Uncomment to be in coverage testing mode.
# SET(CMAKE_BUILD_TYPE Coverage)

# Uncomment to build in profile mode.
# SET(CMAKE_BUILD_TYPE Profile)

# Uncomment to build in release mode with debug information.
# SET(CMAKE_BUILD_TYPE RelWithDebInfo)

# default build type
IF (CMAKE_BUILD_TYPE STREQUAL "")
	SET(CMAKE_BUILD_TYPE Release)
ENDIF (CMAKE_BUILD_TYPE STREQUAL "")

MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

ADD_DEFINITIONS(-DPROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
                -DPROJECT_BINARY_DIR="${CMAKE_BINARY_DIR}")

# ===============================================================
# Check for existence of various required, optional packages.
# CogUtil must come first, because it supplies various FindXXX macros.

# Cogutil
FIND_PACKAGE(CogUtil 2.0.1 CONFIG REQUIRED)
IF (NOT COGUTIL_FOUND)
	MESSAGE(FATAL_ERROR "CogUtil missing: it is needed!")
ENDIF ()

include(OpenCogGccOptions)
include(OpenCogLibOptions)
include(OpenCogInstallOptions)
include(Summary)

# ===================================================================
# Check for existence of various required, optional packages.

# AtomSpace
FIND_PACKAGE(AtomSpace 5.0.3 CONFIG REQUIRED)
IF (NOT ATOMSPACE_FOUND)
	MESSAGE(FATAL_ERROR "AtomSpace missing: it is needed!")
ENDIF ()

FIND_PACKAGE(AtomSpaceStorage CONFIG REQUIRED)
IF (NOT ATOMSPACE_STORAGE_FOUND)
	MESSAGE(FATAL_ERROR "AtomSpace Storage missing: it is needed!")
ENDIF ()

# ----------------------------------------------------------
# Find CUDA (optional, preferred for NVIDIA GPUs).
INCLUDE(CheckLanguage)
CHECK_LANGUAGE(CUDA)
IF (CMAKE_CUDA_COMPILER)
	MESSAGE(STATUS "CUDA found.")
	ADD_DEFINITIONS(-DHAVE_CUDA)
	SET(HAVE_CUDA 1)

	ENABLE_LANGUAGE(CUDA)
	CMAKE_POLICY(SET CMP0104 NEW)

ENDIF (CMAKE_CUDA_COMPILER)

# ----------------------------------------------------------
# Find OpenCL (optional, fallback for AMD/Intel GPUs).
FIND_PACKAGE(OpenCL)
IF (OPENCL_FOUND)
	MESSAGE(STATUS "OpenCL found.")
	MESSAGE(STATUS "OpenCL includes: ${OpenCL_INCLUDE_DIRS}")
	MESSAGE(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")

	INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIRS})
	ADD_DEFINITIONS(-DHAVE_OPENCL)
	SET(HAVE_OPENCL 1)
ENDIF (OPENCL_FOUND)

# Require at least one GPU backend.
IF (NOT HAVE_OPENCL AND NOT HAVE_CUDA)
	MESSAGE(FATAL_ERROR
		"No GPU backend found! Need at least one of CUDA or OpenCL.")
ENDIF (NOT HAVE_OPENCL AND NOT HAVE_CUDA)

# ----------------------------------------------------------
# Guile Python and Cython

include(OpenCogFindGuile)
include(OpenCogFindPython)

# ===================================================================
# Include configuration.

# Set default include paths.
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}
	${COGUTIL_INCLUDE_DIR} ${ATOMSPACE_INCLUDE_DIR})

INCLUDE(OpenCogMacros)
INCLUDE(OpenCogGuile)
INCLUDE(OpenCogCython)

# ==========================================================
# Decide what to build, based on the packages found.

ADD_SUBDIRECTORY(cmake)
ADD_SUBDIRECTORY(opencog)

INCLUDE(OpenCogTestOptions)
OPENCOG_SETUP_TESTING()
OPENCOG_ADD_TEST_TARGET(test_gpu
	tests/persist/gpu "Running GPU storage tests...")

ADD_CUSTOM_TARGET(cscope
	COMMAND find opencog tests -name '*.cc' -o -name '*.h' -o -name '*.cxxtest' -o -name '*.scm' -o -name '*.cl' -o -name '*.cu' > ${CMAKE_SOURCE_DIR}/cscope.files
	COMMAND cscope -b
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
	COMMENT "Generating CScope database"
)

# ===================================================================
# Packaging
## Architecture the package is for.
EXECUTE_PROCESS(COMMAND  dpkg --print-architecture
	OUTPUT_VARIABLE PACKAGE_ARCHITECTURE
	OUTPUT_STRIP_TRAILING_WHITESPACE)
STRING(TIMESTAMP UTC_DATE %Y%m%d UTC)
FILE(WRITE "${PROJECT_BINARY_DIR}/install_manifest.txt")

## Cpack configuration
SET(CPACK_GENERATOR "DEB")
SET(CPACK_PACKAGE_CONTACT "opencog@googlegroups.com")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/packages")
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The AtomSpace GPU (CUDA + OpenCL)")
SET(CPACK_PACKAGE_NAME "atomspace-gpu-dev")
SET(CPACK_PACKAGE_VENDOR "opencog.org")
SET(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}-${UTC_DATE}")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
SET(CPACK_PACKAGE_FILE_NAME
	"${CPACK_PACKAGE_NAME}_${CPACK_PACKAGE_VERSION}_${PACKAGE_ARCHITECTURE}")
SET(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")

## Debian specific configurations
SET(DEPENDENCY_LIST
	"guile-3.0-dev (>= 3.0.0)"
	"libstdc++6 (>= 7.0)"
	"libcogutil-dev (>= 2.0.1)"
	"atomspace-dev (>= 5.0.3)"
)

STRING(REPLACE ";" ", " MAIN_DEPENDENCIES "${DEPENDENCY_LIST}")
SET(CPACK_DEBIAN_PACKAGE_DEPENDS "${MAIN_DEPENDENCIES}")
SET(CPACK_DEBIAN_PACKAGE_SECTION "libdevel")
SET(CPACK_DEBIAN_PACKAGE_HOMEPAGE "http://opencog.org")
INCLUDE(CPack)

# ===================================================================
# Documentation.
FIND_PACKAGE(Doxygen)

# ===================================================================
# Show a summary of what we found, what we will do.

SUMMARY_ADD("CUDA Backend"   "CUDA GPU AtomSpace"   HAVE_CUDA)
SUMMARY_ADD("OpenCL Backend" "OpenCL GPU AtomSpace"  HAVE_OPENCL)
SUMMARY_ADD("Doxygen"        "Code documentation" DOXYGEN_FOUND)
SUMMARY_ADD("Unit tests"     "Unit tests" CXXTEST_FOUND)

SUMMARY_SHOW()
