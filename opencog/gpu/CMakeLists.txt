#
# GPU AtomSpace -- kernels and computation.
# This is the actual AtomSpace implementation on the GPU.
#
# OpenCL kernels (.cl) are JIT-compiled at runtime.
# CUDA kernels (.cu) are compiled at build time.
#

# -- OpenCL kernels (installed for runtime JIT compilation) --
SET(OPENCL_KERNEL_FILES
	gpu-hashtable.cl
	gpu-atomspace.cl
	gpu-mi.cl
	gpu-cosine.cl
	gpu-counting.cl
	gpu-sections.cl
	gpu-substitute.cl
)

INSTALL(FILES ${OPENCL_KERNEL_FILES}
	DESTINATION "share/opencog/opencl/gpu"
)

# -- CUDA kernels (compiled into shared library) --
IF (HAVE_CUDA)
	SET(CUDA_KERNEL_SOURCES
		gpu-learning-loop.cu
		gpu-learning-bridge.cu
		gpu-compartment-kernel.cu
		gpu-connected-components.cu
		gpu-connector-rewrite.cu
		gpu-spanning-forest.cu
		gpu-divergence.cu
	)

	ADD_LIBRARY(gpu-kernels SHARED ${CUDA_KERNEL_SOURCES})

	SET_TARGET_PROPERTIES(gpu-kernels PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_ARCHITECTURES "50;72;75;86;89"
	)

	INSTALL(TARGETS gpu-kernels
		EXPORT AtomSpaceGPUTargets
		DESTINATION "${CMAKE_INSTALL_LIBDIR}/opencog"
	)
ENDIF (HAVE_CUDA)

# -- Header files --
INSTALL(FILES gpu-learning-types.h
	DESTINATION "include/opencog/gpu"
)
