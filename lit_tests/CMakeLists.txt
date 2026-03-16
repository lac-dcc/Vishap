set(LIT_CFG_PY "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${LIT_CFG_PY}
)

set(VISHAP_TEST_DEPENDS
  FileCheck
  vishap-opt
)

add_lit_testsuite(check-vishap "Running Vishap regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${VISHAP_TEST_DEPENDS}
)

set_target_properties(check-vishap PROPERTIES FOLDER "Tests")
