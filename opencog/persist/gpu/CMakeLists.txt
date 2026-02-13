#
# Build the GPU AtomTable library.
# Supports CUDA (preferred) and OpenCL (fallback).
#
# Step 2: AoS atom table (GpuAtomTable)
# Step 3: BackingStore wrapper (GpuStorageNode)
#

# -- Sources: dispatch layer always compiled, backends conditional --
SET(GPU_SOURCES
	GpuAtomTable_dispatch.cc
	GpuStorageNode.cc
)
SET(GPU_LINK_LIBS
	persist-gpu-types
	${ATOMSPACE_STORAGE_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
)
SET(GPU_INSTALL_HEADERS
	GpuAtomTable.h
	GpuStorageNode.h
)

IF (HAVE_CUDA)
	LIST(APPEND GPU_SOURCES GpuAtomTable.cu)
ENDIF (HAVE_CUDA)

IF (HAVE_OPENCL)
	LIST(APPEND GPU_SOURCES GpuAtomTable_ocl.cc)
	LIST(APPEND GPU_LINK_LIBS OpenCL::OpenCL)
	LIST(APPEND GPU_INSTALL_HEADERS opencl-headers.h)
ENDIF (HAVE_OPENCL)

# -- Build the library --
ADD_LIBRARY(persist-gpu SHARED ${GPU_SOURCES})

TARGET_INCLUDE_DIRECTORIES(persist-gpu PRIVATE
	${PROJECT_SOURCE_DIR}
	${CMAKE_BINARY_DIR}
)

TARGET_LINK_LIBRARIES(persist-gpu ${GPU_LINK_LIBS})

# GpuStorageNode depends on generated atom types header.
ADD_DEPENDENCIES(persist-gpu persist_gpu_atom_types)

IF (HAVE_CUDA)
	SET_TARGET_PROPERTIES(persist-gpu PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_ARCHITECTURES "50;72;75;86;89"
	)
ENDIF (HAVE_CUDA)

INSTALL (TARGETS persist-gpu
	EXPORT AtomSpaceGPUTargets
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

INSTALL (FILES ${GPU_INSTALL_HEADERS}
	DESTINATION "include/opencog/persist/gpu"
)
