
INCLUDE(AddCxxtest)

# All tests should load the atomspace scm from the build dir
SET(GUILE_LOAD_PATH "${PROJECT_BINARY_DIR}/opencog/scm")

# GPU AtomSpace kernel tests (standalone, no CxxTest needed)
ADD_SUBDIRECTORY (gpu)

# StorageNode tests (require CxxTest)
IF (CXXTEST_FOUND)
   ADD_SUBDIRECTORY (persist)
ENDIF (CXXTEST_FOUND)
