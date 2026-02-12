**warning this is curently an ai-generated experiment so as a result should be considered slop until otherwise proven**

AtomSpace GPU
=====================================

An [AtomSpace](https://github.com/opencog/atomspace) on the GPU.

Stores atoms as flattened Structure-of-Arrays (SoA) in VRAM so thousands
of GPU threads can operate on them simultaneously. Kernels compute MI,
cosine similarity, clustering, connected components, and the full
learning loop (count → MI → cluster → grammar) directly on the GPU.

[GpuStorageNode](https://wiki.opencog.org/w/StorageNode) is the bridge
that sends atoms from the CPU to the AtomSpace on the GPU and fetches
results back. Same API as
[RocksStorageNode](https://github.com/opencog/atomspace-rocks) (disk)
and [CogStorageNode](https://github.com/opencog/atomspace-cog) (network).

Supports **CUDA** (preferred for NVIDIA GPUs) and **OpenCL** (fallback
for AMD/Intel). At least one must be available at build time.

Architecture
------------
```
CPU (control plane)          GPU (the AtomSpace)
  Scheme / C++ code            SoA pools + kernels
        │                           │
        ├── storeAtom() ──────────→ │  receives atom
        ├── getAtom()   ←────────── │  returns results
        └── runQuery()  ──────────→ │  runs computation
```

The repo has two layers with different dependencies:

- **`opencog/gpu/`** -- The GPU AtomSpace itself. Fully standalone: pure
  CUDA and OpenCL with zero OpenCog dependencies. Can be built and used
  independently for any GPU computing project.
- **`opencog/persist/gpu/`** -- The GpuStorageNode bridge (~350 lines).
  Implements the StorageNode API so the CPU AtomSpace can send atoms to
  the GPU. Requires [CogUtil](https://github.com/opencog/cogutil),
  [AtomSpace](https://github.com/opencog/atomspace), and
  [AtomSpace-Storage](https://github.com/opencog/atomspace-storage).

Building
--------
Requires at least one of:
- **CUDA Toolkit** (11.0+) for NVIDIA GPUs
- **OpenCL** (1.2+ runtime and headers) for AMD/Intel/NVIDIA GPUs

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
make test
sudo make install
```

Usage
-----
```cpp
#include <opencog/persist/gpu/GpuStorageNode.h>

Handle hsn = as->add_node(GPU_STORAGE_NODE, std::string("gpu://:"));
GpuStorageNodePtr store = GpuStorageNodeCast(hsn);
store->open();

Handle a(createNode(CONCEPT_NODE, "someWord"));
a->setValue(truth_key(), createFloatValue({42.0, 3.7}));
store->storeAtom(a, true);
store->barrier();

Handle b(createNode(CONCEPT_NODE, "someWord"));
store->getAtom(b);  // b now has the same FloatValue

store->close();
```

Or from Scheme:
```scheme
(use-modules (opencog) (opencog persist) (opencog persist-gpu))
(define gpu (GpuStorageNode "gpu://:"))
(cog-open gpu)
(store-atom (Concept "someWord"))
(fetch-atom (Concept "someWord"))
(cog-close gpu)
```

Phases
------
- **Phase 1** (complete): Store/fetch round-trip. 13/13 tests pass.
- **Phase 2** (planned): `runQuery()` -- execute Atomese expressions on GPU.
- **Phase 3** (planned): JIT compilation -- fused kernels from AST.

Known Limitations
-----------------
- GPU storage is **volatile** -- data is lost when the connection closes.
- Only `truth_key()` values are stored (GPU has fixed SoA layout).
- CUDA backend requires Compute Capability 5.0+ (Maxwell or newer).
