#pragma once

#include <GL/glew.h>

#if defined(GLEW_OSMESA)
#  ifndef APIENTRY
#    define APIENTRY
#  endif
#  ifndef GLAPI
#    define GLAPI extern
#  endif
#endif

