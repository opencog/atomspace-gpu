
# GpuAtomTable standalone test (not CxxTest -- uses main()).

IF (HAVE_CUDA)
	ADD_EXECUTABLE(test-atom-table test_atom_table.cu)
	TARGET_INCLUDE_DIRECTORIES(test-atom-table PRIVATE ${PROJECT_SOURCE_DIR})
	TARGET_LINK_LIBRARIES(test-atom-table persist-gpu)
	SET_TARGET_PROPERTIES(test-atom-table PROPERTIES
		CUDA_SEPARABLE_COMPILATION ON
		CUDA_ARCHITECTURES "50;72;75;86;89"
	)
	ADD_TEST(NAME test-atom-table COMMAND test-atom-table)
ENDIF (HAVE_CUDA)

# Step 3: BackingStore integration test (CxxTest).
LINK_LIBRARIES(
	persist-gpu
	persist-gpu-types
	atomspace
)

ADD_CXXTEST(GpuBasicSaveUTest)
