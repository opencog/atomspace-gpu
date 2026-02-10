
# This declaration must happen before subdirectory traversal.
DECLARE_GUILE_CONFIG_TARGET(SCM_CONFIG "opencog gpu-config" "OPENCOG_TEST")

ADD_SUBDIRECTORY (gpu)
ADD_SUBDIRECTORY (persist)

# ------------------------------------------------------------------
ADD_GUILE_MODULE (FILES
   scm/persist-gpu.scm
   MODULE_DESTINATION "${GUILE_SITE_DIR}/opencog"
)

WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/gpu-config.scm SCM_CONFIG TRUE)
WRITE_GUILE_CONFIG(${GUILE_BIN_DIR}/opencog/gpu-config-installable.scm SCM_CONFIG FALSE)

INSTALL(FILES ${GUILE_BIN_DIR}/opencog/gpu-config-installable.scm DESTINATION ${GUILE_SITE_DIR}/opencog RENAME gpu-config.scm)
