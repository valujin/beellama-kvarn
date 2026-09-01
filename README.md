# beellama-kvarn

*[Русская версия](README.ru.md)*

A fork of [`Anbeeld/beellama.cpp`](https://github.com/Anbeeld/beellama.cpp) (itself a
fork of [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)) carrying changes
to the **KVarN** KV cache: five correctness fixes, seven decode-kernel optimizations,
two memory-footprint options, and test coverage for the lot.

The base is commit `a749684` of the upstream `v0.4.4` branch, tagged `preview-v0.4.4`
when it was taken. Our changes sit on top as a series of commits, one per change, and
upstream history is preserved in full.

The upstream README is kept as [README.upstream.md](README.upstream.md). The licence and
the author list are unchanged: [LICENSE](LICENSE) (MIT), [AUTHORS](AUTHORS).

## Read the numbers with their conditions

Everything measured below comes from **one machine**: a single **RTX 3090** (sm_86,
24 GiB) running **Qwen3.8-27B `UD-Q4_K_XL`** through `llama-server`. Each table states
its own context size, depth and slot count, because the effects here depend on all
three. None of it is a claim about other GPUs, other models, or configurations that
were not run.

## What differs from `Anbeeld/beellama.cpp`

### Five defects fixed

| defect | how it showed up |
| --- | --- |
| `structured KV live groups alias one F16 stage slot` | crash with `--kv-unified` and more than one slot: two live groups shared one F16 stage slot and overwrote each other |
| `failed to find a memory slot` on a nearly empty cache | a group counted as occupied only once cells were filled, so concurrent requests claimed the same one |
| stale read plan between ubatches | the plan was cached for a whole batch although cell membership changes inside a batch — silently wrong output, dependent on the slot count |
| draft rollback taken on the target's position axis | any image request aborted the server when the MTP draft cache ran on KVarN (see below) |
| out-of-bounds read of `p_sh` in the CUDA decode kernel | no observable effect, but a genuine read past the end of a shared-memory array |

Two of these are crashes, one is silent output corruption, one is a crash confined to
vision, one is invisible in behaviour. A needle-in-a-haystack test did not catch the
third; what catches it is the result depending on the slot count.

### Decoding made faster

Measured at depth 65000 tokens per slot, context 163840, `--parallel 2 --kv-unified`,
`kvarn5` for K and V plus `--kv-tail-tokens 1024`. The two-slot metric is the window in
which both slots are active.

| | one slot | two slots, total | VRAM |
| --- | ---: | ---: | ---: |
| `preview-v0.4.4` as it stands | 23.87 tok/s | 7.10 tok/s | 20066 MiB |
| **this fork** | **32.50 tok/s** | **41.92 tok/s** | 20066 MiB |
| `q8_0`, no tail, for reference | 32.79 tok/s | 40.50 tok/s | 22648 MiB |

That is **×1.36 on one slot and ×5.9 on two** against the base, with no loss of
accuracy: the generation checksum never moved across the whole of this work.

At `--parallel 1` KVarN beats `q8_0` on both speed and memory:

| `--parallel 1`, depth 65000 | tok/s | VRAM |
| --- | ---: | ---: |
| **`kvarn5` + 1024-token tail** | **33.48** | 19996 MiB |
| `q8_0` | 32.77 | 22498 MiB |

Contribution of the individual changes:

| change | one slot | two slots |
| --- | ---: | ---: |
| compact read plan without a red-black tree | +9.8% | +13.7% |
| read plan ordered by cell | +0.5% | ×2.19 |
| `n_q` threshold for split decode | +0.3% | ×1.30 |
| skip fully masked splits | −0.5% | +27% |
| descriptor kernel: 128 → 1024 threads per block | — | −13% step time |
| 128-token splits + fixed geometry selection | +4.7% | 0% |
| matrix-fragment row permutation | +2.5% | +5.0% |
| register ceiling for four blocks per SM | +1.5% | +15% |

### Speculation: a compact MTP draft cache

The MTP draft context used to hold its single `nextn` layer in plain f16 and pay
**832.00 MiB** for it at context 212992 with four slots — the second largest line in the
cost of MTP after the target's recurrent state. It can now run on KVarN, where the same
layer costs **131 MiB** at `kvarn2`.

This is the one cache where lossy compression cannot change the answer: the MTP head
only *proposes* tokens, and the target verifies every proposal against its own cache. A
weaker draft cache can lower the acceptance rate; it cannot corrupt the output. On the
runs measured, acceptance did not fall — it came out marginally *higher* than f16
(0.778 against 0.769 at `--spec-draft-n-max 2`, 0.875 against 0.871 at `n_max 1`,
depth 45283). That difference is small, it repeated across runs, and we cannot explain
it. Do not read it as a claim that a coarser draft predicts better.

Turn it on with:

```bash
--spec-draft-type-k kvarn2 --spec-draft-type-v kvarn2
```

`kvarn2` through `kvarn8` are accepted, half a request is completed rather than
rejected, and the default is unchanged: without these flags the draft cache stays f16.

Running a KVarN draft cache also exposed a throughput cliff that turned out to be a
false capability claim rather than a cache cost. With `--spec-draft-n-max 2` the cache
advertised a suffix-rollback bound of one token that `can_seq_rm()` does not actually
enforce, so the server created a checkpoint on every speculation round and serialized
the entire draft cache through a quadratic path. At context 212992, four slots, depth
20115, `kvarn2` draft: **42.70 → 59.40 tok/s**, GPU utilization **63% → 99%**.

Cost of the whole embedded-MTP setup on this machine, at context 212992 with four
slots: **2522 MiB before, 1814 MiB now.**

### Vision: projector weights in pinned host memory

The multimodal projector's weights can be placed in pinned host memory while the
arithmetic stays on the GPU. On this machine, with Qwen3.8-27B and its vision projector:
**886 MiB of VRAM freed**, image encoding **5-16% slower**, language-model throughput
unaffected, and the generated text byte-identical to the device-memory baseline.

That byte-identity was checked at **`--parallel 1` only**. It says nothing about higher
slot counts, and should not be quoted as if it did.

**Off by default.** Turn it on explicitly:

```bash
--mmproj-host-weights            # or LLAMA_ARG_MMPROJ_HOST_WEIGHTS=1
--no-mmproj-host-weights         # the paired off switch
```

Without the flag the fork behaves exactly as before and no code path reads the
environment for this.

#### Why upstream has this path closed, and what we found

Upstream pins `devices[].integrated` to `false` with a comment about corrupted output.
The corruption is real. It is not in the hardware and not in the weights.

llama.cpp deliberately places the model's *input* tensors in the same pinned host buffer
("use the host buffer of the first device CPU for faster transfer of the intermediate
state"). While CUDA declares that buffer type unsupported, the scheduler is obliged to
copy those inputs into device memory, and it calls `ggml_backend_synchronize()` before
the copy. **That copy is the only barrier** between the CPU writing the inputs and the
device reading them — upstream synchronizes explicitly only under
`cparams.pipeline_parallel`. Declare the buffer type supported and the copy disappears,
the barrier with it, and `set_input()` for the next ubatch overwrites memory the
previous ubatch's kernels are still reading.

Three measurements pin it down: opening the shared buffer type *without* moving a single
projector weight corrupts the output with a bit-identical hash, so the weights are not
involved; `CUDA_LAUNCH_BLOCKING=1` restores byte-identical output while changing nothing
else; and the scheduler dump turns `## SPLIT #1: CUDA0 # 8 inputs: [model.input_embed]
... [attn_inp_kq_mask]` into `# 0 inputs`.

The fix is a separate buffer type, `CUDA_Host_W`, for weights only. It is reachable only
by asking the backend registry for its address by name, so no language-model tensor can
land in it and the scheduler's input barrier stays where it is. Weights are written once
at load time, before the first compute, so the race cannot apply to them by
construction.

We think this analysis also explains upstream issue
[#25992](https://github.com/ggml-org/llama.cpp/issues/25992), where a server with four
slots returns another request's answer verbatim. We have not reproduced that issue or
submitted a fix for it — the mechanism is the same shape, and that is as far as the
claim goes.

### Test coverage

The suite is in an honest state — **97 of 97** green, about 216 s. Three failures
present in the base were dealt with (a stale expectation in `test-kvarn`, a silent
failure in `test-arg-parser`, four model tests under `GGML_BACKEND_DL`), and two coverage
holes were closed: a dedicated `tests/test-kv-group-claim.cpp` with a negative control on
group claiming, and a matrix over the natural choice of query-tile width. The KVarN draft
request added four more blocks to `test-arg-parser`.

Run it with:

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON <your other options>
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

## What is not settled

* **First request after startup can differ, with a KVarN draft cache.** At
  `--parallel 1`, temperature 0, top-k 1, `--spec-draft-n-max 2`, depth ~2123 and one
  specific prompt, the first request after a server start produces a different (not
  corrupted — equally stable) continuation from every request after it. It needs
  `--parallel 1` *and* that depth *and* that prompt; a warm-up request of nine tokens
  removes it for the life of the process. `GGML_KVARN_SPLIT_MAX_Q=1` removes it
  entirely, which places it in the split-decode route at `n_q > 1`. Draft checkpoints,
  uninitialized memory, lazy geometry selection, the `p_sh` fix and CUDA graphs have all
  been ruled out by measurement. **The cause is not found.**
* **Why the coarser draft cache accepts slightly more is unexplained**, as noted above.
* The `--mmproj-host-weights` byte-identity check was run at one slot only.

## Building

### Docker (recommended)

The image builds straight from the repository tree; nothing else to clone.

```bash
docker build -f docker/Dockerfile.ci -t beellama-kvarn:local .
```

The default CUDA architecture set is
`75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real` — the same set
ggml picks for CUDA 13. With one card, naming only its architecture makes the build far
quicker:

```bash
docker build -f docker/Dockerfile.ci --build-arg CUDA_ARCHS=86 -t beellama-kvarn:local .
```

### Prebuilt image from GHCR

```bash
docker pull ghcr.io/valujin/beellama-kvarn:latest
```

### How CI builds it

The all-architectures image is built in GitHub Actions
([.github/workflows/docker-image.yml](.github/workflows/docker-image.yml)) and published
to GHCR. It does not fit in one job: measured on the free runner, **137 minutes for a
single architecture** against a 6-hour job limit, and the set has seven
`--generate-code` entries. Disk is not the constraint — the runner has 145 GB and the
build takes 11.

So the object files are split across six shards. Every shard configures the build
identically through [.github/ci-configure.sh](.github/ci-configure.sh), so their ccache
keys match the final build's: each shard compiles its share and saves its slice of the
cache, and the final job merges the slices and walks the build on hits, left with the
link. The binaries are then packaged by
[docker/Dockerfile.package](docker/Dockerfile.package).

Sharding by *architecture* is not possible: ccache keys on the whole command line,
`--generate-code` list included, so a single-architecture compile yields no hit for a
multi-architecture one.

### Natively

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_BACKEND_DL=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build build -j"$(nproc)" --target llama-server
```

**`GGML_NATIVE=OFF` is deliberate.** `-march=native` makes llama.cpp's CPU kernels
roughly 40% slower: the kernels are written in intrinsics, the variants are built with
explicit feature flags and an ordinary `-mtune` and selected at runtime, and a native
build breaks that mechanism and spoils the auto-vectorizer's decisions. So this fork
takes upstream's approach: `GGML_CPU_ALL_VARIANTS` + `GGML_BACKEND_DL`.

## Running

```bash
docker run --rm --gpus all -p 8080:8080 \
  -v /path/to/models:/models \
  ghcr.io/valujin/beellama-kvarn:latest \
  --model /models/your-model.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 65536 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k kvarn5 --cache-type-v kvarn5 \
  --kv-tail-tokens 1024
```

`--kv-tail-tokens 1024` is worth stating explicitly: without it the precision tail is
not enabled. Add `--spec-draft-type-k kvarn2 --spec-draft-type-v kvarn2` for the compact
MTP draft cache and `--mmproj-host-weights` for the projector weights in host memory.
For the cache types themselves and how to choose between them, see
[README.upstream.md](README.upstream.md).

## Attribution

- Upstream: [`Anbeeld/beellama.cpp`](https://github.com/Anbeeld/beellama.cpp) — KVarN,
  the precision tail, low-bit cache types, the adaptive draft, reasoning-loop guards.
- Base project: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp).
- MIT licence, © 2023-2026 The ggml authors, Anbeeld. [LICENSE](LICENSE) and
  [AUTHORS](AUTHORS) are unmodified.
