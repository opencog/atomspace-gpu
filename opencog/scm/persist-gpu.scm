;
; persist-gpu.scm -- OpenCL GPU storage backend for the AtomSpace
;
; Load the GPU persistence module.
;

(define-module (opencog persist-gpu))

(use-modules (opencog) (opencog exec) (opencog persist))
(use-modules (opencog gpu-config))

; Load the C++ library that implements the GPU storage node.
(load-extension (string-append opencog-ext-path-persist-gpu "libpersist-gpu") "opencog_persist_gpu_init")

(export GpuStorageNode)
