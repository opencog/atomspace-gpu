#
# GPU AtomSpace kernel tests.
# OpenCL tests (.c) test the OpenCL kernels.
# CUDA tests (.cu) test the CUDA kernels.
#

# -- OpenCL kernel tests --
IF (HAVE_OPENCL)
	SET(OPENCL_TESTS
		test-hashtable
		test-atomspace
		test-mi
		test-cosine
		test-counting
		test-sections
		test-substitute
	)

	FOREACH(TEST_NAME ${OPENCL_TESTS})
		ADD_EXECUTABLE(${TEST_NAME} ${TEST_NAME}.c)
		TARGET_LINK_LIBRARIES(${TEST_NAME} OpenCL::OpenCL m)
		TARGET_INCLUDE_DIRECTORIES(${TEST_NAME} PRIVATE
			${PROJECT_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}/opencog/gpu)
		ADD_TEST(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
			WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	ENDFOREACH()
ENDIF (HAVE_OPENCL)

# -- CUDA kernel tests --
IF (HAVE_CUDA)
	SET(CUDA_TESTS
		test-connected-components
		test-connector-rewrite
		test-spanning-forest
		test-divergence
		test-learning-loop
		test-compartment-kernel
	)

	FOREACH(TEST_NAME ${CUDA_TESTS})
		ADD_EXECUTABLE(${TEST_NAME} ${TEST_NAME}.cu)
		SET_TARGET_PROPERTIES(${TEST_NAME} PROPERTIES
			CUDA_SEPARABLE_COMPILATION ON
			CUDA_ARCHITECTURES "50;72;75;86;89"
		)
		TARGET_INCLUDE_DIRECTORIES(${TEST_NAME} PRIVATE
			${PROJECT_SOURCE_DIR}
			${PROJECT_SOURCE_DIR}/opencog/gpu)
		TARGET_LINK_LIBRARIES(${TEST_NAME} gpu-kernels)
		ADD_TEST(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
			WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
	ENDFOREACH()
ENDIF (HAVE_CUDA)
