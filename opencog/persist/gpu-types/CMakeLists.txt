# Generate the atom types
OPENCOG_GEN_CXX_ATOMTYPES(atom_types.script
	atom_types.h
	atom_types.definitions
	atom_types.inheritance)

OPENCOG_GEN_SCM_ATOMTYPES(atom_types.script persist_gpu_types.scm)
OPENCOG_GEN_PYTHON_ATOMTYPES(atom_types.script persist_gpu_types.pyx)

ADD_CUSTOM_TARGET(persist_gpu_atom_types
	DEPENDS atom_types.h persist_gpu_types.scm persist_gpu_types.pyx)

# Build the shared library
ADD_LIBRARY (persist-gpu-types SHARED
	atom_types_init.cc
)

# The atom_types.h file is written to the build directory
TARGET_INCLUDE_DIRECTORIES(persist-gpu-types PRIVATE ${CMAKE_BINARY_DIR})

TARGET_LINK_LIBRARIES(persist-gpu-types
	${ATOMSPACE_STORAGE_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
)

ADD_DEPENDENCIES(persist-gpu-types persist_gpu_atom_types)

# Install targets
INSTALL (TARGETS persist-gpu-types EXPORT AtomSpaceGPUTargets
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

INSTALL (FILES
	${CMAKE_CURRENT_BINARY_DIR}/atom_types.h
	DESTINATION "include/opencog/persist/gpu-types"
)

ADD_GUILE_MODULE (FILES
	${CMAKE_CURRENT_BINARY_DIR}/persist_gpu_types.scm
	MODULE_DESTINATION "${GUILE_SITE_DIR}/opencog/persist/gpu-types"
	DEPENDS persist_gpu_atom_types
)

ADD_GUILE_EXTENSION(SCM_CONFIG persist-gpu-types "opencog-ext-path-persist-gpu-types")
