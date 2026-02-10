
# CMake boilerplate that allows users to do
#    find_package(AtomSpaceGPU REQUIRED 1.0.0)
# and have it work.

include(CMakePackageConfigHelpers)

set(ConfigPackageLocation lib/cmake/AtomSpaceGPU)

install(EXPORT AtomSpaceGPUTargets
	FILE AtomSpaceGPUTargets.cmake
	DESTINATION ${ConfigPackageLocation}
)

configure_package_config_file(AtomSpaceGPUConfig.cmake.in
	${CMAKE_CURRENT_BINARY_DIR}/AtomSpaceGPUConfig.cmake
	INSTALL_DESTINATION ${ConfigPackageLocation}
	PATH_VARS CMAKE_INSTALL_PREFIX
)

write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/AtomSpaceGPUConfigVersion.cmake"
	VERSION ${PROJECT_VERSION}
	COMPATIBILITY SameMajorVersion
)

INSTALL(FILES
	${CMAKE_CURRENT_BINARY_DIR}/AtomSpaceGPUConfigVersion.cmake
	${CMAKE_CURRENT_BINARY_DIR}/AtomSpaceGPUConfig.cmake
	DESTINATION ${ConfigPackageLocation}
)

# -----------------------------------------------------------
