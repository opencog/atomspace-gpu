/*
 * opencog/persist/gpu/opencl-headers.h
 *
 * Portable OpenCL C++ header include.
 *
 * Copyright (C) 2025 Linas Vepstas
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_GPU_OPENCL_HEADERS_H
#define _OPENCOG_GPU_OPENCL_HEADERS_H

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300

#if defined __has_include
	#if __has_include(<CL/opencl.hpp>)
		#include <CL/opencl.hpp>
	#else
		#include <CL/cl.hpp>
	#endif
#else
	#include <CL/opencl.hpp>
#endif

#endif // _OPENCOG_GPU_OPENCL_HEADERS_H
