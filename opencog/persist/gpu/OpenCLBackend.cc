/*
 * opencog/persist/gpu/OpenCLBackend.cc
 *
 * OpenCL implementation of GpuBackend.
 * Manages OpenCL context, command queue, kernel compilation,
 * and GPU buffer operations for SoA pools.
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <fstream>
#include <sstream>
#include <vector>

#include <opencog/util/exceptions.h>
#include <opencog/util/Logger.h>

#include "OpenCLBackend.h"

using namespace opencog;

// ==============================================================

OpenCLBackend::~OpenCLBackend()
{
	// OpenCL resources are RAII via cl::* wrappers
}

// ==============================================================
// Device discovery

void OpenCLBackend::find_device(const std::string& splat,
                                const std::string& sdev)
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (const auto& plat : platforms)
	{
		std::string pname = plat.getInfo<CL_PLATFORM_NAME>();
		if (0 < splat.size() and pname.find(splat) == std::string::npos)
			continue;

		std::vector<cl::Device> devices;
		plat.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		for (const cl::Device& dev : devices)
		{
			std::string dname = dev.getInfo<CL_DEVICE_NAME>();
			if (0 < sdev.size() and dname.find(sdev) == std::string::npos)
				continue;

			_platform = plat;
			_device = dev;
			_platform_name = pname;
			_device_name = dname;

			logger().info("OpenCLBackend: Using platform '%s' device '%s'\n",
				pname.c_str(), dname.c_str());
			return;
		}
	}

	throw RuntimeException(TRACE_INFO,
		"OpenCLBackend: unable to find platform '%s' device '%s'\n",
		splat.c_str(), sdev.c_str());
}

// ==============================================================
// Kernel compilation

void OpenCLBackend::compile_kernels()
{
	std::ostringstream opts;
	opts << "-D WORD_CAPACITY=" << GPU_WORD_CAPACITY
	     << " -D PAIR_CAPACITY=" << GPU_PAIR_CAPACITY
	     << " -D SECTION_CAPACITY=" << GPU_SECTION_CAPACITY
	     << " -D WORD_HT_CAPACITY=" << GPU_WORD_HT_CAPACITY
	     << " -D PAIR_HT_CAPACITY=" << GPU_PAIR_HT_CAPACITY
	     << " -D SECTION_HT_CAPACITY=" << GPU_SECTION_HT_CAPACITY;

	std::string ht_src, as_src, inc_src;
	std::vector<std::string> search_paths = {
		"/usr/local/share/opencog/opencl/gpu/",
		"/usr/share/opencog/opencl/gpu/",
	};

	search_paths.push_back("opencog/gpu/");
	search_paths.push_back("../opencog/gpu/");
	search_paths.push_back("../../opencog/gpu/");

	for (const auto& prefix : search_paths)
	{
		if (not ht_src.empty() and not as_src.empty()
		    and not inc_src.empty()) break;

		if (ht_src.empty())
		{
			std::ifstream f(prefix + "gpu-hashtable.cl");
			if (f.is_open())
				ht_src.assign(std::istreambuf_iterator<char>(f),
				              std::istreambuf_iterator<char>());
		}
		if (as_src.empty())
		{
			std::ifstream f(prefix + "gpu-atomspace.cl");
			if (f.is_open())
				as_src.assign(std::istreambuf_iterator<char>(f),
				              std::istreambuf_iterator<char>());
		}
		if (inc_src.empty())
		{
			std::ifstream f(prefix + "gpu-incoming.cl");
			if (f.is_open())
				inc_src.assign(std::istreambuf_iterator<char>(f),
				              std::istreambuf_iterator<char>());
		}
	}

	if (ht_src.empty() or as_src.empty())
		throw RuntimeException(TRACE_INFO,
			"OpenCLBackend: cannot find gpu-hashtable.cl "
			"or gpu-atomspace.cl kernel source files\n");

	std::string combined = ht_src + "\n" + as_src;
	if (not inc_src.empty())
		combined += "\n" + inc_src;

	cl::Program::Sources sources;
	sources.push_back(combined);
	_program = cl::Program(_context, sources);

	try
	{
		_program.build(opts.str().c_str());
	}
	catch (const cl::Error& e)
	{
		std::string log = _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_device);
		logger().error("OpenCLBackend kernel build failed:\n%s\n", log.c_str());
		throw RuntimeException(TRACE_INFO,
			"OpenCLBackend: kernel compilation failed\n%s\n",
			log.c_str());
	}

	logger().info("OpenCLBackend: kernels compiled successfully\n");
}

// ==============================================================
// Lifecycle

void OpenCLBackend::init(const std::string& platform_hint,
                         const std::string& device_hint)
{
	find_device(platform_hint, device_hint);
	_context = cl::Context(_device);
	_queue = cl::CommandQueue(_context, _device);
	compile_kernels();
}

void OpenCLBackend::shutdown()
{
	_queue.finish();
}

std::string OpenCLBackend::device_info()
{
	return _platform_name + " / " + _device_name;
}

// ==============================================================
// Pool allocation

void OpenCLBackend::alloc_pools()
{
	// -- WordPool --
	_word_name_hash = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_ulong) * GPU_WORD_CAPACITY);
	_word_count = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_double) * GPU_WORD_CAPACITY);
	_word_mi_marginal = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_double) * GPU_WORD_CAPACITY);
	_word_class_id = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_WORD_CAPACITY);
	_word_type = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_ushort) * GPU_WORD_CAPACITY);
	_word_next_free = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint));

	// -- PairPool --
	_pair_word_a = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_PAIR_CAPACITY);
	_pair_word_b = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_PAIR_CAPACITY);
	_pair_count = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_double) * GPU_PAIR_CAPACITY);
	_pair_mi = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_double) * GPU_PAIR_CAPACITY);
	_pair_flags = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_PAIR_CAPACITY);
	_pair_next_free = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint));

	// -- SectionPool --
	_sec_word = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_SECTION_CAPACITY);
	_sec_djh = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_ulong) * GPU_SECTION_CAPACITY);
	_sec_count = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_double) * GPU_SECTION_CAPACITY);
	_sec_next_free = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint));

	// -- Hash tables --
	_word_ht_keys = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_ulong) * GPU_WORD_HT_CAPACITY);
	_word_ht_values = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_WORD_HT_CAPACITY);
	_pair_ht_keys = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_ulong) * GPU_PAIR_HT_CAPACITY);
	_pair_ht_values = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_PAIR_HT_CAPACITY);
	_sec_ht_keys = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_ulong) * GPU_SECTION_HT_CAPACITY);
	_sec_ht_values = cl::Buffer(_context, CL_MEM_READ_WRITE,
		sizeof(cl_uint) * GPU_SECTION_HT_CAPACITY);
}

void OpenCLBackend::init_pools()
{
	cl_uint zero = 0;
	_queue.enqueueWriteBuffer(_word_next_free, CL_TRUE, 0, sizeof(cl_uint), &zero);
	_queue.enqueueWriteBuffer(_pair_next_free, CL_TRUE, 0, sizeof(cl_uint), &zero);
	_queue.enqueueWriteBuffer(_sec_next_free, CL_TRUE, 0, sizeof(cl_uint), &zero);

	// Zero out word type array
	{
		std::vector<cl_ushort> zt(GPU_WORD_CAPACITY, 0);
		_queue.enqueueWriteBuffer(_word_type, CL_TRUE, 0,
			sizeof(cl_ushort) * GPU_WORD_CAPACITY, zt.data());
	}

	// Initialize hash table keys to EMPTY, values to EMPTY.
	// Use CL_TRUE (blocking) because host vectors go out of scope.
	{
		std::vector<cl_ulong> ek(GPU_WORD_HT_CAPACITY, GPU_HT_EMPTY_KEY);
		_queue.enqueueWriteBuffer(_word_ht_keys, CL_TRUE, 0,
			sizeof(cl_ulong) * GPU_WORD_HT_CAPACITY, ek.data());
	}
	{
		std::vector<cl_uint> ev(GPU_WORD_HT_CAPACITY, GPU_HT_EMPTY_VALUE);
		_queue.enqueueWriteBuffer(_word_ht_values, CL_TRUE, 0,
			sizeof(cl_uint) * GPU_WORD_HT_CAPACITY, ev.data());
	}
	{
		std::vector<cl_ulong> ek(GPU_PAIR_HT_CAPACITY, GPU_HT_EMPTY_KEY);
		_queue.enqueueWriteBuffer(_pair_ht_keys, CL_TRUE, 0,
			sizeof(cl_ulong) * GPU_PAIR_HT_CAPACITY, ek.data());
	}
	{
		std::vector<cl_uint> ev(GPU_PAIR_HT_CAPACITY, GPU_HT_EMPTY_VALUE);
		_queue.enqueueWriteBuffer(_pair_ht_values, CL_TRUE, 0,
			sizeof(cl_uint) * GPU_PAIR_HT_CAPACITY, ev.data());
	}
	{
		std::vector<cl_ulong> ek(GPU_SECTION_HT_CAPACITY, GPU_HT_EMPTY_KEY);
		_queue.enqueueWriteBuffer(_sec_ht_keys, CL_TRUE, 0,
			sizeof(cl_ulong) * GPU_SECTION_HT_CAPACITY, ek.data());
	}
	{
		std::vector<cl_uint> ev(GPU_SECTION_HT_CAPACITY, GPU_HT_EMPTY_VALUE);
		_queue.enqueueWriteBuffer(_sec_ht_values, CL_TRUE, 0,
			sizeof(cl_uint) * GPU_SECTION_HT_CAPACITY, ev.data());
	}

	logger().info("OpenCLBackend: GPU pools initialized\n");
}

// ==============================================================
// Word pool operations

uint32_t OpenCLBackend::word_find_or_create(uint64_t nhash)
{
	cl::Kernel kern(_program, "word_find_or_create");

	cl::Buffer in_hash(_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_ulong), &nhash);
	cl::Buffer out_buf(_context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
	cl_uint num_items = 1;

	kern.setArg(0, _word_ht_keys);
	kern.setArg(1, _word_ht_values);
	kern.setArg(2, _word_name_hash);
	kern.setArg(3, _word_count);
	kern.setArg(4, _word_class_id);
	kern.setArg(5, _word_next_free);
	kern.setArg(6, in_hash);
	kern.setArg(7, out_buf);
	kern.setArg(8, num_items);

	_queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(1));
	cl_uint out_idx = GPU_NOT_FOUND;
	_queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(cl_uint), &out_idx);

	return out_idx;
}

uint32_t OpenCLBackend::word_lookup(uint64_t nhash)
{
	cl::Kernel kern(_program, "ht_lookup");

	cl::Buffer query_key(_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_ulong), &nhash);
	cl::Buffer out_buf(_context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
	cl_ulong capacity = GPU_WORD_HT_CAPACITY;
	cl_uint num_queries = 1;

	kern.setArg(0, _word_ht_keys);
	kern.setArg(1, _word_ht_values);
	kern.setArg(2, capacity);
	kern.setArg(3, query_key);
	kern.setArg(4, out_buf);
	kern.setArg(5, num_queries);

	_queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(1));
	cl_uint result = GPU_NOT_FOUND;
	_queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(cl_uint), &result);

	return result;
}

void OpenCLBackend::word_write_count(uint32_t idx, double count)
{
	cl_double v = count;
	_queue.enqueueWriteBuffer(_word_count, CL_FALSE,
		sizeof(cl_double) * idx, sizeof(cl_double), &v);
}

void OpenCLBackend::word_write_marginal(uint32_t idx, double marginal)
{
	cl_double v = marginal;
	_queue.enqueueWriteBuffer(_word_mi_marginal, CL_FALSE,
		sizeof(cl_double) * idx, sizeof(cl_double), &v);
}

void OpenCLBackend::word_read_values(uint32_t idx,
                                     double& count, double& marginal)
{
	cl_double c = 0.0, m = 0.0;
	_queue.enqueueReadBuffer(_word_count, CL_TRUE,
		sizeof(cl_double) * idx, sizeof(cl_double), &c);
	_queue.enqueueReadBuffer(_word_mi_marginal, CL_TRUE,
		sizeof(cl_double) * idx, sizeof(cl_double), &m);
	count = c;
	marginal = m;
}

void OpenCLBackend::word_write_type(uint32_t idx, uint16_t type)
{
	cl_ushort v = type;
	_queue.enqueueWriteBuffer(_word_type, CL_TRUE,
		sizeof(cl_ushort) * idx, sizeof(cl_ushort), &v);
}

uint16_t OpenCLBackend::word_read_type(uint32_t idx)
{
	cl_ushort v = 0;
	_queue.enqueueReadBuffer(_word_type, CL_TRUE,
		sizeof(cl_ushort) * idx, sizeof(cl_ushort), &v);
	return (uint16_t)v;
}

void OpenCLBackend::word_delete(uint64_t nhash)
{
	cl::Kernel kern(_program, "ht_delete");
	cl::Buffer del_key(_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_ulong), &nhash);
	cl_ulong capacity = GPU_WORD_HT_CAPACITY;
	cl_uint num = 1;

	kern.setArg(0, _word_ht_keys);
	kern.setArg(1, _word_ht_values);
	kern.setArg(2, capacity);
	kern.setArg(3, del_key);
	kern.setArg(4, num);

	_queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(1));
	_queue.finish();
}

uint32_t OpenCLBackend::word_pool_count()
{
	cl_uint cnt = 0;
	_queue.enqueueReadBuffer(_word_next_free, CL_TRUE, 0, sizeof(cl_uint), &cnt);
	return cnt;
}

void OpenCLBackend::word_read_bulk(uint32_t n, uint64_t* hashes,
                                   double* counts, double* marginals,
                                   uint16_t* types)
{
	_queue.enqueueReadBuffer(_word_name_hash, CL_TRUE, 0,
		sizeof(cl_ulong) * n, hashes);
	_queue.enqueueReadBuffer(_word_count, CL_TRUE, 0,
		sizeof(cl_double) * n, counts);
	_queue.enqueueReadBuffer(_word_mi_marginal, CL_TRUE, 0,
		sizeof(cl_double) * n, marginals);
	_queue.enqueueReadBuffer(_word_type, CL_TRUE, 0,
		sizeof(cl_ushort) * n, types);
}

// ==============================================================
// Pair pool operations

uint32_t OpenCLBackend::pair_find_or_create(uint32_t word_a, uint32_t word_b)
{
	cl::Kernel kern(_program, "pair_find_or_create");

	cl::Buffer in_wa(_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint), &word_a);
	cl::Buffer in_wb(_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_uint), &word_b);
	cl::Buffer out_buf(_context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
	cl_uint num_items = 1;

	kern.setArg(0, _pair_ht_keys);
	kern.setArg(1, _pair_ht_values);
	kern.setArg(2, _pair_word_a);
	kern.setArg(3, _pair_word_b);
	kern.setArg(4, _pair_count);
	kern.setArg(5, _pair_mi);
	kern.setArg(6, _pair_flags);
	kern.setArg(7, _pair_next_free);
	kern.setArg(8, in_wa);
	kern.setArg(9, in_wb);
	kern.setArg(10, out_buf);
	kern.setArg(11, num_items);

	_queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(1));
	cl_uint out_idx = GPU_NOT_FOUND;
	_queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(cl_uint), &out_idx);

	return out_idx;
}

uint32_t OpenCLBackend::pair_lookup(uint32_t word_a, uint32_t word_b)
{
	uint32_t lo = std::min(word_a, word_b);
	uint32_t hi = std::max(word_a, word_b);
	uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;

	cl::Kernel kern(_program, "ht_lookup");

	cl::Buffer query_key(_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_ulong), &key);
	cl::Buffer out_buf(_context, CL_MEM_WRITE_ONLY, sizeof(cl_uint));
	cl_ulong capacity = GPU_PAIR_HT_CAPACITY;
	cl_uint num_queries = 1;

	kern.setArg(0, _pair_ht_keys);
	kern.setArg(1, _pair_ht_values);
	kern.setArg(2, capacity);
	kern.setArg(3, query_key);
	kern.setArg(4, out_buf);
	kern.setArg(5, num_queries);

	_queue.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(1));
	cl_uint result = GPU_NOT_FOUND;
	_queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(cl_uint), &result);

	return result;
}

void OpenCLBackend::pair_write_count(uint32_t idx, double count)
{
	cl_double v = count;
	_queue.enqueueWriteBuffer(_pair_count, CL_FALSE,
		sizeof(cl_double) * idx, sizeof(cl_double), &v);
}

void OpenCLBackend::pair_write_mi(uint32_t idx, double mi)
{
	cl_double v = mi;
	_queue.enqueueWriteBuffer(_pair_mi, CL_FALSE,
		sizeof(cl_double) * idx, sizeof(cl_double), &v);
}

void OpenCLBackend::pair_read_values(uint32_t idx,
                                     double& count, double& mi)
{
	cl_double c = 0.0, m = 0.0;
	_queue.enqueueReadBuffer(_pair_count, CL_TRUE,
		sizeof(cl_double) * idx, sizeof(cl_double), &c);
	_queue.enqueueReadBuffer(_pair_mi, CL_TRUE,
		sizeof(cl_double) * idx, sizeof(cl_double), &m);
	count = c;
	mi = m;
}

void OpenCLBackend::pair_write_type(uint32_t idx, uint16_t type)
{
	cl_uint val = (cl_uint)type;
	_queue.enqueueWriteBuffer(_pair_flags, CL_TRUE,
		sizeof(cl_uint) * idx, sizeof(cl_uint), &val);
}

uint16_t OpenCLBackend::pair_read_type(uint32_t idx)
{
	cl_uint val = 0;
	_queue.enqueueReadBuffer(_pair_flags, CL_TRUE,
		sizeof(cl_uint) * idx, sizeof(cl_uint), &val);
	return (uint16_t)val;
}

uint32_t OpenCLBackend::pair_pool_count()
{
	cl_uint cnt = 0;
	_queue.enqueueReadBuffer(_pair_next_free, CL_TRUE, 0, sizeof(cl_uint), &cnt);
	return cnt;
}

// ==============================================================
// Incoming-set scan (Phase 2)

uint32_t OpenCLBackend::incoming_scan(uint32_t target_word_idx,
                                       uint32_t* out_pair_indices,
                                       uint32_t max_results)
{
	uint32_t pool_count = pair_pool_count();
	if (pool_count == 0) return 0;

	cl::Kernel kern(_program, "incoming_scan");

	// Output buffers
	cl::Buffer match_buf(_context, CL_MEM_WRITE_ONLY,
		sizeof(cl_uint) * max_results);
	cl::Buffer count_buf(_context, CL_MEM_READ_WRITE, sizeof(cl_uint));

	// Initialize match count to zero
	cl_uint zero = 0;
	_queue.enqueueWriteBuffer(count_buf, CL_TRUE, 0, sizeof(cl_uint), &zero);

	kern.setArg(0, _pair_word_a);
	kern.setArg(1, _pair_word_b);
	kern.setArg(2, (cl_uint)target_word_idx);
	kern.setArg(3, (cl_uint)pool_count);
	kern.setArg(4, match_buf);
	kern.setArg(5, count_buf);
	kern.setArg(6, (cl_uint)max_results);

	// Launch one thread per pair slot
	size_t global_size = ((pool_count + 255) / 256) * 256;
	_queue.enqueueNDRangeKernel(kern, cl::NullRange,
		cl::NDRange(global_size), cl::NDRange(256));

	// Read back match count
	cl_uint match_count = 0;
	_queue.enqueueReadBuffer(count_buf, CL_TRUE, 0,
		sizeof(cl_uint), &match_count);

	if (match_count > max_results)
		match_count = max_results;

	if (match_count > 0)
	{
		_queue.enqueueReadBuffer(match_buf, CL_TRUE, 0,
			sizeof(cl_uint) * match_count, out_pair_indices);
	}

	return match_count;
}

uint32_t OpenCLBackend::pair_read_bulk(uint32_t n,
                                       uint32_t* word_a, uint32_t* word_b,
                                       double* counts, double* mis,
                                       uint16_t* types)
{
	uint32_t pool_count = pair_pool_count();
	if (pool_count == 0) return 0;
	if (n > pool_count) n = pool_count;

	_queue.enqueueReadBuffer(_pair_word_a, CL_TRUE, 0,
		sizeof(cl_uint) * n, word_a);
	_queue.enqueueReadBuffer(_pair_word_b, CL_TRUE, 0,
		sizeof(cl_uint) * n, word_b);
	_queue.enqueueReadBuffer(_pair_count, CL_TRUE, 0,
		sizeof(cl_double) * n, counts);
	_queue.enqueueReadBuffer(_pair_mi, CL_TRUE, 0,
		sizeof(cl_double) * n, mis);

	// Read pair flags and convert uint32_t -> uint16_t types
	std::vector<cl_uint> flags(n);
	_queue.enqueueReadBuffer(_pair_flags, CL_TRUE, 0,
		sizeof(cl_uint) * n, flags.data());
	for (uint32_t i = 0; i < n; i++)
		types[i] = (uint16_t)flags[i];

	return n;
}

// ==============================================================
// Section pool operations

uint32_t OpenCLBackend::section_pool_count()
{
	cl_uint cnt = 0;
	_queue.enqueueReadBuffer(_sec_next_free, CL_TRUE, 0, sizeof(cl_uint), &cnt);
	return cnt;
}

// ==============================================================

void OpenCLBackend::barrier()
{
	_queue.finish();
}

// ==============================================================
// Factory function

namespace opencog {

GpuBackend* create_opencl_backend()
{
	return new OpenCLBackend();
}

} // namespace opencog

// ==============================================================
