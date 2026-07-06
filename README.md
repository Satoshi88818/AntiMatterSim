# AntimatterSim v10
### Planetary-Scale, Mathematically Verifiable, Zero-Latency Treatment Intelligence

AntimatterSim is a Monte Carlo particle-transport and treatment-planning simulation platform for particle therapy (proton/ion radiotherapy). Version 10 represents the transition from an **Autonomous Adaptive Treatment System** (v9) to a **Globally Distributed, Mathematically Verifiable, Zero-Latency Treatment Intelligence**.

This README documents the purpose, requirements, architecture, engineering decisions, and core innovations introduced in the v10 release.

---

## 1. Purpose

AntimatterSim simulates the physics of charged-particle beams passing through patient anatomy (protons, ions, and secondary fragments) in order to:

- Compute 3D/4D radiation dose distributions from first-principles Monte Carlo physics
- Optimize treatment plans against tumor-control and normal-tissue-toxicity objectives
- Model biological response (DNA damage, tumor control probability / normal tissue complication probability)
- Adapt treatment delivery in real time to patient motion
- Do all of the above at increasing scale, speed, and verifiability with each major version

Where v9 optimized a single treatment plan on a single machine with millisecond-scale adaptive control, **v10 reframes the problem globally**: a distributed system that learns across institutions without sharing patient data, produces a full space of optimal plans rather than one plan, proves the safety of its output cryptographically, and closes the control loop in microseconds at the edge.

---

## 2. The Seven Pillars of Level 10

| Pillar | v9 (Level 9) | v10 (Level 10) |
|---|---|---|
| Physics Engine | PyTorch surrogate (not differentiable through MC) | JAX end-to-end auto-diff Monte Carlo |
| Scaling Architecture | Kubernetes Jobs (slow cold-start) | Serverless Knative / Wasm warm-pool |
| Optimization | Single-objective IMAT (weighted cost) | Multi-objective Pareto frontier (NSGA-III / Bayesian Opt) |
| DIR Performance | B-spline via scipy.ndimage (CPU) | TensorRT + NVIDIA Optical Flow SDK (CUDA texture) |
| Control Latency | gRPC to centralized cloud (~10 ms) | FPGA + Jetson edge node (<1 ms) + cloud digital twin |
| Model Learning | Single-institution surrogate | Federated Learning (NVIDIA FLARE) — privacy-preserving |
| Auditability | None | zk-SNARK cryptographic proof-of-correctness |

---

## 3. Requirements

### 3.1 Functional Requirements

- Simulate hadronic, nuclear (evaporation, intra-nuclear cascade), and fragment transport physics
- Support static and 4D deformable (breathing-motion) patient geometry from DICOM/GDML/mesh sources
- Produce dose, PET-activation, and DNA-damage estimates
- Generate treatment plans that are inverse-optimized against clinical objectives
- Provide real-time adaptive beam control with sub-millisecond reaction to patient motion
- Train and improve dose-prediction surrogate models across multiple institutions without centralizing patient data
- Produce a machine-checkable proof that a delivered plan satisfies stated safety constraints
- Scale from a single workstation simulation to a globally distributed, multi-region execution

### 3.2 Non-Functional Requirements

- **Latency:** edge beam-gating loop under 1 ms; cloud re-planning loop ~100 ms
- **Differentiability:** gradients of dose with respect to beam parameters must be exact, not approximated
- **Privacy:** federated training must provide formal differential-privacy guarantees (ε, δ)
- **Verifiability:** safety constraints must be provable and independently verifiable in well under a second
- **Elastic scale:** system must tolerate loss of large blocks of cloud compute (e.g., spot-instance preemption) without losing simulation state
- **Backward compatibility:** all v7/v8/v9 physics modules remain unchanged and fully supported

### 3.3 Software & Hardware Dependencies

Core scientific stack (retained from v9): `numpy`, `scipy`, `numba`, `ray`, `cupy-cuda12x`, `torch`, `pybind11`, `pydicom`, `SimpleITK`, `grpcio`, `kubernetes`, among others.

New in v10:

| Capability | Key Packages |
|---|---|
| Differentiable physics (JAX/XLA) | `jax`, `jaxlib`, `optax`, `flax` |
| Multi-objective optimization | `pymoo` (NSGA-III), `botorch`, `gpytorch` |
| Population-based cloud search | `ray[tune]` |
| Federated learning | `nvflare`, `syft` (optional) |
| Cryptographic proofs | `py_ecc` (+ external `circom`/`snarkjs` toolchain for full zk-SNARK circuits) |
| Hardware-accelerated deformable registration | `tensorrt`, NVIDIA Optical Flow SDK (via CUDA/CuPy) |
| Serverless / edge runtime | `aws-lambda-powertools`, `wasmtime`, Knative (cluster infrastructure, no pip package) |

Hardware: NVIDIA GPU with CUDA 12.x for TensorRT/JAX-GPU paths; an FPGA (Xilinx/Intel) or NVIDIA Jetson device for the edge control loop; multi-region cloud accounts (e.g., AWS) for planetary deployment.

---

## 4. Architecture

### 4.1 High-Level Topology

```
                 ┌───────────────────────────────────────────┐
                 │           Planetary Cloud Layer            │
                 │  (Knative / Lambda / Wasm, multi-region)   │
                 │                                             │
                 │  ┌───────────┐   ┌────────────────────┐     │
                 │  │  JAX MC   │   │ Pareto Planner      │     │
                 │  │  Physics  │   │ (NSGA-III / Bayes)  │     │
                 │  └───────────┘   └────────────────────┘     │
                 │  ┌───────────┐   ┌────────────────────┐     │
                 │  │ Federated │   │ zk-SNARK Proof       │     │
                 │  │ Aggregator│   │ Generator            │     │
                 │  └───────────┘   └────────────────────┘     │
                 └───────────────────────┬─────────────────────┘
                                          │ gRPC / Wavelength routing
                          ┌───────────────┴────────────────┐
                          │        Cloud Digital Twin        │
                          │   (adaptive re-plan, ~100 ms)    │
                          └───────────────┬───────────────┘
                                          │
                          ┌───────────────┴────────────────┐
                          │         Edge Node (site)         │
                          │  Jetson ROM inference / FPGA      │
                          │  beam gating loop  (< 1 ms)       │
                          └───────────────┬───────────────┘
                                          │
                                  Treatment Machine
```

### 4.2 Project Structure (annotated)

```
antimatter_sim_v10/
├── requirements.txt
├── config.yaml
├── CMakeLists.txt
├── src/
│   ├── config_schema.py        # Extended v10: JAX, serverless, Pareto, edge, FL, ZKP configs
│   ├── physics.py, particles.py, fields.py, geometry*.py   # Unchanged core physics/geometry
│   ├── diff_physics/            # NEW v10: auto-differentiable physics
│   │   ├── jax_transport.py     # JAX auto-diff particle transport
│   │   └── xla_kernels.py       # Fused XLA physics kernels
│   ├── optimization/
│   │   ├── inverse_planner.py   # Extended: delegates to Pareto planner
│   │   ├── pareto_planner.py    # NEW: NSGA-III / Bayesian multi-objective search
│   │   └── cost_functions.py    # Extended: delivery-time objective added
│   ├── radiobiology/             # Unchanged (TCP/NTCP, MKM+TLK DNA damage models)
│   ├── surrogate/                # Extended: federated training hook
│   ├── cloud/                    # NEW v10: serverless entry points
│   │   ├── serverless_handler.py # Knative / Lambda handlers
│   │   └── spot_manager.py       # Predictive spot-instance migration
│   ├── edge/                     # NEW v10: edge-cloud hybrid
│   │   ├── fpga_control.v        # Verilog HLS sub-ms beam gating
│   │   └── jetson_inference.py   # TensorRT-optimized local ROM model
│   ├── security/                 # NEW v10: privacy + verifiability
│   │   ├── federated_aggregator.py # FedAvg / DP-SGD aggregation
│   │   └── proof_generator.py      # zk-SNARK proof generation/verification
│   ├── api/grpc_streamer.py      # Extended: edge-mode + Wavelength routing
│   ├── gpu_engine.py              # Extended: TensorRT 8 + Optical Flow SDK
│   ├── engine.py                  # Extended: JAX/edge/ZKP hooks
│   ├── distributed.py             # Extended: serverless + federated backends
│   ├── analysis.py, visualizer.py # Extended: Pareto/federated reporting
│   ├── api.py                     # Extended: /pareto_plan, /proof, /edge_status
│   └── cli.py                     # Extended: --jax, --pareto, --edge, --federated, --zkp
├── cpp/, cuda/                    # Native extensions; NEW texture_dir_kernel.cu
├── validation/                    # + JAX-vs-PyTorch, Pareto-vs-IMAT, ZKP timing benches
├── tests/                         # + new v10 unit tests
├── benchmarks/                    # + JAX-vs-surrogate, texture-DIR benchmarks
└── scripts/
    ├── deploy_k8s.sh              # v9, retained
    └── deploy_planetary.sh        # NEW v10: multi-region global deployment
```

### 4.3 Configuration Model

`config.yaml` is fully backward compatible: all v7–v9 sections (geometry, fields, transport, GPU, hadronic/nuclear physics, dose scoring, radiobiology, PINN, beam control, watchdog, distributed, visualizer) are retained unchanged. Six new top-level sections gate the Level 10 subsystems, each independently toggleable and **disabled by default**:

- `jax_physics` — backend (cpu/gpu/tpu), JIT compilation, batch size, forward vs. reverse-mode gradients, precision
- `serverless` — backend (knative/lambda/wasm), warm-pool size, checkpoint backend (NVMe-oF/Redis/S3), spot-preemption handling
- `pareto_planning` — algorithm (nsga3/bayesian/pbt), objective weights (TCP, NTCP, delivery time), population/generation sizes, Ray Tune worker counts
- `tensorrt_dir` — engine path, precision (fp32/fp16/int8), optical-flow preset, texture interpolation
- `edge_control` — edge backend (jetson/fpga/wavelength), edge vs. cloud update intervals, digital-twin sync rate
- `federated_learning` — backend (nvflare/pysyft), client count, aggregation strategy, differential-privacy budget (ε, δ), gradient compression
- `zkp` — proof system (groth16/plonk/stark), circuit/key paths, safety constraints encoded directly into the circuit (e.g., max spinal-cord dose)

This design lets an operator run pure v9 behavior, any single v10 capability, or the full "maximum physics mode" stack from one configuration file.

---

## 5. Engineering Details

### 5.1 Differentiable Physics (`diff_physics/`)

The core innovation replacing the v9 PyTorch surrogate is a **JAX-native Monte Carlo transport engine**. Rather than training a neural network to approximate the physics (which is not differentiable through the original simulator), v10 vectorizes particle transport with `vmap`/`lax.scan` and exposes exact gradients of dose with respect to beam parameters via `jax.value_and_grad`. Physics kernels (e.g., Bragg-peak deposition) are fused and JIT-compiled through XLA, and long-running simulation state can be checkpointed to NVMe-oF storage for resilience.

### 5.2 Multi-Objective Optimization (`optimization/pareto_planner.py`)

The v9 inverse planner minimized a single weighted cost function. v10 instead searches for a **Pareto frontier** across competing objectives (tumor control, normal-tissue toxicity, delivery time) using NSGA-III (via `pymoo`) or Bayesian expected-hypervolume-improvement search (via `BoTorch`/`GPyTorch`). Large-scale variant search is distributed across the cloud with Ray Tune population-based training. A hypervolume computation and knee-point selection routine help clinicians navigate the resulting set of plans rather than being handed a single answer.

### 5.3 Hardware-Accelerated Deformable Image Registration (`geometry/deformable_dir.py`)

4D deformable registration (needed to track breathing motion) moves from CPU B-spline interpolation (`scipy.ndimage`) to a TensorRT-serialized inference engine, with NVIDIA Optical Flow SDK for motion-field initialization and CUDA texture memory for hardware-accelerated trilinear interpolation.

### 5.4 Edge-Cloud Hybrid Control (`edge/`)

The v9 control loop closed over a centralized gRPC connection (~10 ms round trip). v10 adds a physically local control path: a distilled Reduced-Order Model runs on an NVIDIA Jetson device, and a Verilog HLS beam-gating controller runs on an FPGA with initiation interval 1, achieving sub-millisecond (down to microsecond-scale) beam-gating reaction times. The cloud continues to host a slower-cadence "digital twin" for adaptive re-planning (~100 ms).

### 5.5 Federated Learning (`security/federated_aggregator.py`)

Instead of training the dose-prediction surrogate on a single institution's data, v10 supports federated training across multiple hospital sites using NVIDIA FLARE (or PySyft as an alternate backend). Model updates — not patient data — leave each site. Differential-privacy is enforced via DP-SGD gradient clipping and noise injection with configurable privacy budget (ε, δ), and gradient updates are compressed (top-k sparsification) to reduce network egress by roughly 99%.

### 5.6 Verifiable Computing (`security/proof_generator.py`)

The final new subsystem produces a cryptographic **proof of correctness**: a zk-SNARK (Groth16/PLONK/STARK-compatible) circuit encodes safety constraints (e.g., maximum spinal-cord or brainstem dose, minimum tumor coverage) so that a treatment plan's compliance can be verified in under a millisecond by a third party, without re-running the simulation or exposing underlying patient data. A SHA-256 commitment scheme is available as a lightweight fallback when full zk-SNARK tooling (the external `circom`/`snarkjs` toolchain) is unavailable.

### 5.7 Serverless & Planetary Deployment (`cloud/`, `scripts/deploy_planetary.sh`)

Kubernetes Jobs (slow cold-start) are replaced with a Knative/Lambda/Wasm warm-pool architecture supporting up to 10,000 concurrent invocations, with predictive spot-instance-preemption handling that checkpoints and migrates state before interruption. `deploy_planetary.sh` deploys simulation workers as autoscaling Knative services across multiple cloud regions (e.g., primary compute, GDPR-compliant EU federated-learning nodes, and edge nodes at 5G cell-tower "Wavelength" zones), coordinating a global federated-learning server and a merge step for planetary-scale results.

---

## 6. Level 10 Capability Reference

| Capability | Implementation | File |
|---|---|---|
| JAX auto-diff transport | `JAXTransportEngine.simulate` (vmap + lax.scan) | `src/diff_physics/jax_transport.py` |
| Exact ∂Dose/∂Energy gradient | `grad_dose_wrt_energy` (jax.value_and_grad) | `src/diff_physics/jax_transport.py` |
| XLA kernel fusion | `fused_bragg_peak_kernel` (@jit, segment_sum) | `src/diff_physics/xla_kernels.py` |
| NVMe-oF checkpointing | `checkpoint_to_nvme_of` | `src/diff_physics/xla_kernels.py` |
| Serverless Knative handler | `KnativeHandler.handle` (async warm-pool) | `src/cloud/serverless_handler.py` |
| AWS Lambda handler | `AWSLambdaHandler.lambda_handler` | `src/cloud/serverless_handler.py` |
| Spot-preemption recovery | `SpotPreemptionHandler._on_sigterm` | `src/cloud/spot_manager.py` |
| NSGA-III Pareto planning | `ParetoPlanner._run_nsga3` | `src/optimization/pareto_planner.py` |
| Bayesian EHVI planning | `ParetoPlanner._run_bayesian` | `src/optimization/pareto_planner.py` |
| Pareto hypervolume | `_compute_hypervolume` | `src/optimization/pareto_planner.py` |
| Knee-point selection | `_select_knee_point` | `src/optimization/pareto_planner.py` |
| TensorRT DIR engine | `TensorRTDIREngine.register_accelerated` | `src/geometry/deformable_dir.py` |
| CUDA texture trilinear warp | `_texture_warp` (CuPy GPU) | `src/geometry/deformable_dir.py` |
| NVIDIA Optical Flow init | `init_optical_flow` | `src/geometry/deformable_dir.py` |
| Jetson ROM inference | `JetsonROMInference.infer` (< 1 ms) | `src/edge/jetson_inference.py` |
| Knowledge distillation | `JetsonROMInference.distill_from_surrogate` | `src/edge/jetson_inference.py` |
| FPGA beam gating | `beam_gate_controller` (Verilog, II=1) | `src/edge/fpga_control.v` |
| FedAvg aggregator | `FedAvgAggregator.aggregate` | `src/security/federated_aggregator.py` |
| DP-SGD clipping + noise | `_clip_gradient`, `_add_dp_noise` | `src/security/federated_aggregator.py` |
| Gradient compression (~99%) | `_compress_gradients` (top-k sparse) | `src/security/federated_aggregator.py` |
| zk-SNARK proof generation | `ZKSNARKProofGenerator.generate_proof` | `src/security/proof_generator.py` |
| Proof verification (< 1 ms) | `ZKSNARKProofGenerator.verify_proof` | `src/security/proof_generator.py` |
| Safety commitment (fallback) | `_generate_commitment_proof` (SHA-256) | `src/security/proof_generator.py` |
| CLI flags | `--jax`, `--pareto`, `--edge`, `--federated`, `--zkp`, `--serverless`, `--trt-dir` | `src/cli.py` |
| Planetary deployment | Multi-region Knative + autoscaling | `scripts/deploy_planetary.sh` |

---

## 7. Innovations Summary

1. **Exact gradients through Monte Carlo physics** — replacing an approximate learned surrogate with true auto-differentiation, so optimization is guided by real physics rather than a neural fit to it.
2. **Plans as a frontier, not a point** — clinicians are shown the full trade-off space between tumor control, toxicity, and delivery time (with automated hypervolume/knee-point guidance) instead of a single weighted-cost answer.
3. **Microsecond-scale physical control** — moving beam-gating logic onto an FPGA and Jetson device at the treatment site removes network round-trip time from the safety-critical control loop, while a cloud digital twin still handles slower adaptive re-planning.
4. **Cross-institution learning without data sharing** — federated learning with differential-privacy guarantees allows the dose-prediction model to improve using signal from many hospitals while keeping patient data local and provably resistant to memorization.
5. **Cryptographically checkable safety** — a zk-SNARK proof lets any third party (e.g., a regulator or auditor) verify that a delivered plan satisfies hard safety constraints in near-real time, without re-executing the simulation or exposing patient data.
6. **Elastic, planetary-scale execution** — a serverless warm-pool architecture with predictive spot-instance migration allows the system to absorb large-scale compute loss (e.g., cloud provider preemption) without losing simulation progress, and to scale treatment-planning search across global regions.
7. **Fully backward-compatible layering** — every v10 capability is an additive, independently toggleable subsystem; existing v7–v9 physics, geometry, and radiobiology modules are unchanged, so the platform can be run at any capability level from a single configuration file.

---

## 8. Safety & Validation Roadmap

| Level | Description | Status |
|---|---|---|
| 1 | Research-grade energy spectra, feasibility | ✅ Complete |
| 2 | Full photon tracking + Sternheimer corrections | ✅ Complete |
| 3 | Lorentz-force Boris pusher | ✅ Complete |
| 4 | Numba, 3D CSG, XS tables, NaN guardrails, PCG64, Ray, watchdog | ✅ Complete |
| 5 | CuPy GPU, C++ SIMD Boris, antiproton hadronic channel, ZMQ visualizer, STL/GDML, microdosimetry | ✅ Complete |
| 6 | Nuclear evaporation, adaptive RK4, GPU-resident cross-sections, π→μ→e decay, multi-GPU NCCL, space-charge | ✅ Complete |
| 7 | p/n/t channels + Gilbert-Cameron, Bertini INC, full GDML + BVH, LET variance reduction, Geant4/NIST validation, pybind11 + CUDA | ✅ Complete |
| 8 | Fragment transport, PET modeling, DICOM CT, 3D CNN surrogate, TCP/NTCP models | ✅ Complete |
| 9 | 4D deformable geometry, IMAT inverse planning, molecular DNA damage (MKM+TLK), PINNs, closed-loop gRPC, Kubernetes | ✅ Complete |
| 10 | JAX auto-diff physics, serverless warm-po
