# Hardware validation report

Validation run on a Linux host (kernel 7.0, amdgpu in-kernel driver) with two
AMD devices. No ROCm userspace installed — this doubles as proof the sysfs
backend works without ROCm.

## Devices found

| gpu_id | Device | VRAM | Backend |
|---|---|---|---|
| `amd-0000:03:00.0` | Radeon RX 6600 XT (Navi 23 XT, device id 0x73ff) | 8,573,157,376 B (7.98 GiB) | amd-sysfs |
| `amd-0000:0f:00.0` | Radeon Graphics (Cezanne iGPU, device id 0x1638) | 536,870,912 B (512 MiB) | amd-sysfs |

Discovery, VRAM, temperature, power and clocks for the iGPU matched native
sysfs values exactly (the backend reads the same kernel interfaces a vendor
CLI would; `rocm-smi`/`sensors` were not installed on this host, so no
third-party CLI comparison was possible — values were cross-checked against
`/sys/class/drm/card*/device/*` directly).

## Observed driver behavior (documented, not a bug)

On the RX 6600 XT while idle, `gpu_busy_percent` and the hwmon sensors
(`temp1_input`, `power1_average`, `freq1_input`) return **EBUSY** from the
amdgpu SMU. The server reports these as:

```json
{"available": false, "reason": "sensor busy — the GPU power management
 interface refused the read (common when the card is in a low-power state)"}
```

VRAM sensors keep working, so memory history still accumulates for the
dGPU. Under load these sensors typically answer; the same host's iGPU
reports all sensors fine at idle.

## History

With `GPU_MANAGER_SAMPLE_INTERVAL=2`, a 12-second run stored 6 samples per
available metric per GPU (30 rows for the iGPU, 18 for the dGPU). Row cap
and retention pruning are covered by unit tests.

## NVIDIA / Intel

No NVIDIA or Intel GPU was present on the validation host. Those backends
are exercised by mocked unit tests (NVML bindings, `nvidia-smi` CSV,
missing-driver paths, unsupported sensors). Hardware validation on
Tesla/Quadro/GeForce and Intel Arc is on the roadmap; contributions with
validation reports are welcome.
