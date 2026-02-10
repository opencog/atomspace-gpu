
AtomSpace GPU Storage Backend
=====================================

A GPU-backed [StorageNode](https://wiki.opencog.org/w/StorageNode)
for the [OpenCog AtomSpace](https://github.com/opencog/atomspace).
Allows atoms and values to be stored in GPU VRAM for high-throughput
GPU-resident computation.

Same API as [RocksStorageNode](https://github.com/opencog/atomspace-rocks)
(disk) and [CogStorageNode](https://github.com/opencog/atomspace-cog)
(network), but backed by GPU SoA (Structure-of-Arrays) pools.

Supports **CUDA** (preferred for NVIDIA GPUs) and **OpenCL** (fallback
for AMD/Intel). At least one must be available at build time. Backend
is selected at runtime based on URI and hardware availability.

Architecture
------------
```
CPU AtomSpace
    |
    +-- RocksStorageNode  -> disk (RocksDB)
    +-- CogStorageNode    -> network (CogServer)
    +-- GpuStorageNode    -> GPU VRAM (CUDA or OpenCL)
            |
            +-- GpuBackend (abstract interface)
            |       |
            |       +-- CudaBackend    (preferred, NVIDIA)
            |       +-- OpenCLBackend  (fallback, AMD/Intel/NVIDIA)
            |
            +-- WordPool      128K nodes (name_hash, count, marginal, class_id)
            +-- PairPool      4M binary links (word_a, word_b, count, mi, flags)
            +-- SectionPool   1M sections (word, disjunct_hash, count)
```

Each pool uses a lock-free GPU hash table for O(1) lookup and a bump
allocator for slot allocation. CUDA backend uses device memory with
managed staging buffers; OpenCL backend uses cl::Buffer with JIT
kernel compilation.

URI Format
----------
```
gpu://:              first available device (CUDA preferred)
gpu://NVIDIA:RTX     NVIDIA device containing "RTX" (auto backend)
gpu://cuda::         force CUDA backend, any device
gpu://opencl::       force OpenCL backend, any device
gpu://opencl:Intel:  OpenCL on first Intel device
```

Building
--------
Prerequisites: [CogUtil](https://github.com/opencog/cogutil),
[AtomSpace](https://github.com/opencog/atomspace),
[AtomSpace-Storage](https://github.com/opencog/atomspace-storage),
and at least one of:
- **CUDA Toolkit** (11.0+) for NVIDIA GPUs
- **OpenCL** (1.2+ runtime and headers) for AMD/Intel/NVIDIA GPUs

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
make test
sudo make install
```

CMake will auto-detect available backends. To force a specific backend:
```bash
cmake .. -DHAVE_CUDA=OFF        # OpenCL only
cmake .. -DHAVE_OPENCL=OFF      # CUDA only (not recommended)
```

Usage (C++)
-----------
```cpp
#include <opencog/persist/gpu/GpuStorageNode.h>

AtomSpacePtr as = createAtomSpace();
Handle hsn = as->add_node(GPU_STORAGE_NODE, std::string("gpu://:"));
GpuStorageNodePtr store = GpuStorageNodeCast(hsn);

store->open();   // Selects CUDA if available, else OpenCL

// Store a node with values
Handle a(createNode(SCHEMA_NODE, "someWord"));
a->setValue(truth_key(), createFloatValue({42.0, 3.7}));
store->storeAtom(a, true);
store->barrier();

// Fetch it back
Handle b(createNode(SCHEMA_NODE, "someWord"));
store->getAtom(b);
// b now has the same FloatValue

store->close();
```

Usage (Scheme)
--------------
```scheme
(use-modules (opencog) (opencog persist) (opencog persist-gpu))

(define gpu (GpuStorageNode "gpu://:"))
(cog-open gpu)

(define a (Concept "someWord"))
(cog-set-value! a (Predicate "*-TruthValueKey-*")
    (FloatValue 42.0 3.7))
(store-atom a)
(barrier gpu)

; Fetch back
(define b (Concept "someWord"))
(fetch-atom b)

(cog-close gpu)
```

GPU Memory Usage
----------------
| Pool     | Capacity | Hash Table | GPU Memory |
|----------|----------|------------|------------|
| Words    | 128K     | 256K       | ~4 MB      |
| Pairs    | 4M       | 8M         | ~160 MB    |
| Sections | 1M       | 2M         | ~40 MB     |
| **Total** |         |            | **~204 MB** |

Phases
------
- **Phase 1** (complete): Store/fetch round-trip. 8/8 tests pass.
- **Phase 2** (planned): `runQuery()` -- execute Atomese expressions on GPU.
- **Phase 3** (planned): JIT compilation -- fused kernels from AST.

Known Limitations
-----------------
- GPU storage is **volatile** -- data is lost when the connection closes.
  Use RocksStorageNode for persistence.
- Only `truth_key()` values are stored. Other value keys are silently
  ignored (GPU has fixed SoA layout).
- Only binary links map to the pair pool. Higher-arity links store
  their outgoing nodes but not link-level values.
- CUDA backend requires Compute Capability 5.0+ (Maxwell or newer).
- OpenCL backend requires `enqueueWriteBuffer` with `CL_TRUE` (blocking)
  when the host buffer goes out of scope immediately.
