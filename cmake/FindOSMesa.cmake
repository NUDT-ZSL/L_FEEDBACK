find_path(OSMESA_INCLUDE_DIR NAMES GL/osmesa.h)
find_library(OSMESA_LIBRARIES NAMES OSMesa libOSMesa)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OSMesa DEFAULT_MSG
    OSMESA_INCLUDE_DIR OSMESA_LIBRARIES)

mark_as_advanced(OSMESA_INCLUDE_DIR OSMESA_LIBRARIES)

