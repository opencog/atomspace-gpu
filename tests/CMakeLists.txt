
INCLUDE(AddCxxtest)

# All tests should load the atomspace scm from the build dir
SET(GUILE_LOAD_PATH "${PROJECT_BINARY_DIR}/opencog/scm")

# Perform tests in component-dependency order.
IF (CXXTEST_FOUND)
   ADD_SUBDIRECTORY (persist)
ENDIF (CXXTEST_FOUND)
