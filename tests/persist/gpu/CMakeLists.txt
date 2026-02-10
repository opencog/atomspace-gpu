
# Unit tests

LINK_LIBRARIES(
	persist-gpu
	persist
	${ATOMSPACE_STORAGE_LIBRARIES}
	${ATOMSPACE_LIBRARIES}
	OpenCL::OpenCL
)

ADD_CXXTEST(GpuStorageNodeUTest)
ADD_CXXTEST(GpuIncomingUTest)
ADD_CXXTEST(BasicSaveUTest)
ADD_CXXTEST(GpuBulkUTest)
