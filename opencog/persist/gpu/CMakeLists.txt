#
# Build the GPU driver for the AtomSpace.
# Supports CUDA (preferred) and OpenCL (fallback).
#

# -- Core StorageNode (backend-agnostic) --
SET(GPU_SOURCES GpuStorageNode.cc)
SET(GPU_LINK_LIBS
	persist-gpu-types
	persist
	${ATOMSPACE_STORAGE_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
)
SET(GPU_INSTALL_HEADERS
	GpuStorageNode.h
	GpuBackend.h
	gpu-pool-defs.h
)

# -- OpenCL backend (optional) --
IF (HAVE_OPENCL)
	LIST(APPEND GPU_SOURCES OpenCLBackend.cc)
	LIST(APPEND GPU_LINK_LIBS OpenCL::OpenCL)
	LIST(APPEND GPU_INSTALL_HEADERS opencl-headers.h OpenCLBackend.h)
ENDIF (HAVE_OPENCL)

# -- CUDA backend (optional, preferred) --
IF (HAVE_CUDA)
	LIST(APPEND GPU_SOURCES CudaBackend.cu)
	LIST(APPEND GPU_INSTALL_HEADERS CudaBackend.h)
ENDIF (HAVE_CUDA)

# -- Build the library --
ADD_LIBRARY(persist-gpu SHARED ${GPU_SOURCES})

# The atom_types.h file is written to the build directory
TARGET_INCLUDE_DIRECTORIES(persist-gpu PRIVATE ${CMAKE_BINARY_DIR})

TARGET_LINK_LIBRARIES(persist-gpu ${GPU_LINK_LIBS})

IF (HAVE_CUDA)
	SET_TARGET_PROPERTIES(persist-gpu PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_ARCHITECTURES "50;72;75;86;89"
	)
ENDIF (HAVE_CUDA)

ADD_DEPENDENCIES(persist-gpu persist_gpu_atom_types)

ADD_GUILE_EXTENSION(SCM_CONFIG persist-gpu
	"opencog-ext-path-persist-gpu")

# The EXPORT is needed to autogenerate CMake boilerplate in the lib
# directory that lets other packages FIND_PACKAGE(AtomSpaceGPU)
INSTALL (TARGETS persist-gpu
	EXPORT AtomSpaceGPUTargets
	DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
)

INSTALL (FILES ${GPU_INSTALL_HEADERS}
	DESTINATION "include/opencog/persist/gpu"
)

# Install OpenCL kernel files (needed at runtime for JIT compilation)
IF (HAVE_OPENCL)
	INSTALL (FILES
		gpu-hashtable.cl
		gpu-atomspace.cl
		DESTINATION "share/opencog/opencl/gpu"
	)
ENDIF (HAVE_OPENCL)
