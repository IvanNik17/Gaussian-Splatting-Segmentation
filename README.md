

# Semi-Automatic View-Based Segmentation of Gaussian Splat Scenes

This project builds on **Aras Pranckevičius’ Unity Gaussian Splatting renderer** and integrates it with:

* **SAM (Segment Anything Model)** – to get a 2D segmentation mask from a single camera view.
* **ZoeDepth** – to estimate depth in the same view and use that depth to prune the SAM selection in 3D.

The goal is **interactive, depth-aware selection of Gaussian splats in Unity**:
click on an object in the view → SAM selects everything in that 2D region → ZoeDepth finds the depth of the clicked surface → splats *behind* that depth get deselected, leaving a selection that approximates the intended 3D object.

---

## High-level pipeline

### 1. Gaussian splats (Aras’ renderer)

* A `.ply` file is converted (offline) into a Gaussian splat asset:

  * Positions and other attributes are packed into GPU buffers (`_SplatPos`, `_SplatColor`, etc.).
  * Chunk-local AABBs and encoding formats (11-11-10, etc.) are used for compression.
* At runtime, `GaussianSplatRenderer`:

  * Uploads the asset to GPU.
  * Uses compute shaders (`SplatUtilities.compute`) to build per-splat **view data** (`SplatViewData`) for the current camera.
  * Renders splats with `RenderGaussianSplats.shader`.

Important invariants:

* **Object/world/camera spaces** are standard Unity:

  * Local splat position → `localToWorld` → `worldToCamera` → `GL.GetGPUProjectionMatrix`.
  * Depth is `clip.z / clip.w`, then mapped to NDC [0,1] for gating.
* The position buffer (`_SplatPos` / `m_GpuPosData`) is a **packed ByteAddressBuffer**, not an array of `float3`. Do **not** treat it as `Vector3[]` on CPU; use the view buffer or decoded CPU data instead.

### 2. Central selection buffer: `_SplatSelectedBits`

All editing/selection flows through one GPU bitfield:

* `GaussianSplatRenderer` owns `m_GpuEditSelected`, exposed via `GpuEditSelected`.
* In shaders this is `_SplatSelectedBits`: a **bit-per-splat** buffer.
* If a bit is 1 → splat is “selected” (for highlighting, deleting, labeling, etc.).
* `UpdateEditCountsAndBounds()`:

  * Scans `_SplatSelectedBits` on GPU.
  * Computes:

    * `editSelectedSplats`
    * `editDeletedSplats`
    * `editCutSplats`
    * `editSelectedBounds` (AABB of selected splats).

Any tool (SAM, ZoeDepth, future segmentation) that wants to change selection must:

1. Set/clear bits in `_SplatSelectedBits`.
2. Call `UpdateEditCountsAndBounds()`.

This is the contract.

---

## SAM + ZoeDepth integration

### 3. Python sidecar (`cli_sam.py` + `SAM.py`)

A Python process runs alongside Unity and exposes a simple line-based protocol:

* Input JSON (one line per request) includes:

  * `image`: path to the captured RGB PNG.
  * `points`: positive click(s) in normalized image coordinates.
  * `out`: path to write the SAM mask PNG.
  * `depth_out`: path to write the ZoeDepth PNG.
  * `zoe_variant`, `zoe_root`, `zoe_max_dim`, etc.

For each request the sidecar:

1. Runs **SAM**:

   * Builds a binary mask (`mask.png`) from the positive points.
2. Runs **ZoeDepth**:

   * Computes a metric depth map `depth_raw` (meters).

   * Normalizes depth per-frame to `[0,1]`:

     ```python
     d_min = np.min(depth_raw[finite])
     d_max = np.max(depth_raw[finite])
     depth_norm = (depth_raw - d_min) / max(d_max - d_min, 1e-6)
     ```

   * Writes a 16-bit depth PNG (`depth_out`) with `depth_norm`.

   * Writes a `*_meta.json` with `depth_min`, `depth_max`, `depth_range`, plus optional stats for the masked region.

The sidecar does **not** know Unity units. It outputs:

* SAM mask: 0/1 in image space.
* Zoe depth: normalized [0,1] per frame, plus metadata with metric min/max.

### 4. Unity SAM client (`SAMBase.cs`)

`SAMBase` runs inside Unity and coordinates capture, SAM, ZoeDepth, and selection:

1. **Capture**

   * Grabs a downsampled render of `sourceCamera` at `captureSize`.
   * Saves it as a temporary PNG.
   * Builds a JSON payload with:

     * Image path.
     * Positive points (converted to top-left normalized coordinates).
     * Output paths for mask and depth.

2. **Call Python worker**

   * Starts `pythonExe` with `cli_sam.py --loop` once, then streams JSON requests over stdin.
   * Waits for a JSON response or times out.

3. **Load mask and depth**

   * `maskTex` = R8 PNG, `W × H`.
   * `depthTex` = Zoe depth PNG.
   * Keeps `m_LastMaskTex` and `m_LastDepthTex` for re-application.

4. **Apply mask to selection** – `ApplyMaskToSelection(maskTex, depthTex)`

   * Calls `gs.EditDeselectAll()` (clear `_SplatSelectedBits`).
   * Binds:

     * `_SplatViewData` (view buffer).
     * `_SplatPos` (raw positions).
     * `_MaskTex` (SAM mask).
     * `_DepthTex` (Zoe depth) if available.
   * Dispatches the `ApplyMaskSelection` kernel in `MaskSelect.compute`.

   The kernel:

   * Projects each splat into the mask image using `SplatViewData.pos` (clip space → NDC → pixels).
   * Samples the mask.
   * If mask ≥ threshold → sets bit in `_SplatSelectedBits`.
   * Optionally uses Zoe depth (`_DepthTex` + `_ZoeDepthCullOffset`) to reject splats too far behind the 2D mask (per-pixel).

   After dispatch, `GaussianSplatRenderer.UpdateEditCountsAndBounds()` runs, so the editor knows which splats are selected.

5. **Depth probe (plane for ZoeDepth)** – `RunProbeCull()`

   * Requires:

     * At least one `positivePoints[0]`.
     * A valid `m_LastDepthTex`.

   * Samples Zoe depth at that point:

     ```csharp
     float zoeDepthNorm = depthTex.GetPixel(px, py).r; // 0..1
     ```

   * Converts normalized Zoe depth to a probe ray and world point:

     ```csharp
     Ray   ray         = sourceCamera.ViewportPointToRay(new Vector3(pt.x, pt.y, 0f));
     float metricDepth = Mathf.Lerp(nearMeters, farMeters, zoeDepthNorm);
     Vector3 worldPos  = ray.GetPoint(metricDepth);
     float ndcDepth    = WorldToNdc01(worldPos); // 0..1
     ```

   * Computes a tolerance band in NDC using `ComputeProbeTolNdc` (metric meters → NDC delta via marching along the ray).

   * Stores:

     * `m_ZoeProbeWorld` – world position of the plane.
     * `m_ProbeClipDepthNdc` – NDC depth of the plane.
     * `m_ProbeClipTolNdc` – NDC slack for depth band.

   * Triggers the depth cull (CPU or GPU, depending on current implementation).

6. **Plane visualization**

   * `OnDrawGizmos` draws:

     * Blue plane at `m_ProbeForwardDepth` (Zoe depth).
     * Red plane at `m_LastProbeCullForwardDepth` (Zoe depth + tolerance).
   * Uses `Camera.CalculateFrustumCorners` at a given forward distance and transforms them to world space.
   * Optional: draws a small sphere at the probe point on the plane.

This gives immediate visual feedback that the Zoe depth and tolerance are mapped correctly before culling anything.

---

## Depth-aware culling

### 5. Depth band logic

Conceptually, depth culling works like this:

* Let `D_splat = depthNdc` of a splat (from `clip.z / clip.w` → NDC 0..1).
* Let `D_plane = ZoeDepthProbeNdc`.
* Let `Δ = toleranceNdc`.

Then:

* A splat is considered **behind** the plane if `D_splat > D_plane + Δ`.
* Those splats have their bit cleared in `_SplatSelectedBits`.

Two ways to implement:

#### A. GPU cull kernel (ideal)

* Compute shader `CSProbeDepthCull` reads:

  * `_SplatViewData` (clip positions).
  * `_SplatSelectedBits`.
  * `_ProbeDepthClip` and `_ProbeDepthClipTol`.
* Per splat:

  * Early out if bit not set.
  * `depth01 = saturate(clip.z / clip.w * 0.5 + 0.5)`.
  * `clipThreshold = _ProbeDepthClip + _ProbeDepthClipTol`.
  * If `depth01 > clipThreshold` → clear bit.

Optional debug:

* `_ProbeCullForceAll != 0` → clear all bits regardless of depth to sanity-check the pipeline.

#### B. CPU cull prototype

* Read `_SplatSelectedBits` to a `uint[]`.
* Read `_SplatViewData` to a `SplatViewDataCPU[]`.
* Loop over selected indices, use `view.pos` as clip position, convert to depth01, and apply the same `depth01 > clipThreshold` rule.
* Write back the bit array and call `UpdateEditCountsAndBounds()`.

The critical point: **never decode positions from `_SplatPos` on CPU**. Always use `SplatViewData` or decoded asset data; `_SplatPos` is packed and chunk-scaled.

---

## What this project is *not* doing

* It does **not** train SAM or ZoeDepth. Both are used as frozen pretrained models.
* It does **not** do view-consistent multi-view SAM fusion yet (one view at a time).
* It does **not** apply any heavy autograd or gradient-based optimization; all work is runtime compute shader + simple depth math.

---

## What this project is aiming toward

The current scope is:

* **Robust interactive selection** of Gaussian splats in Unity via SAM.
* **Depth-aware pruning** of that selection using ZoeDepth (keep the clicked object, drop background splats).
* A clean, documented pipeline so other tools (K-Means segmentation, label painting, export/import) can sit on top of the same selection mechanism.

Planned/possible extensions (not assumed implemented here):

* Training-free clustering over per-splat features (color, position, normals) to generate initial segments.
* Merging SAM-based 2D masks from multiple views into a consistent 3D label field.
* Simple open-vocabulary selection by associating per-segment text embeddings.

---

## Notes for anyone (or any LLM) hacking on this repo

* Selection is **only** real when it’s in `_SplatSelectedBits` and `UpdateEditCountsAndBounds()` has run.
* `_SplatPos` is a packed ByteAddressBuffer. Do not call `GetData<Vector3>` on it.
* Use:

  * `SplatViewData` for clip-space re-projection.
  * Asset-level decoded positions if you must work in CPU world space.
* Coordinate spaces:

  * Compute and CPU both use `worldToCameraMatrix` + `GL.GetGPUProjectionMatrix(projection, true)`.
  * Depth gating always works in NDC (`clip.z / clip.w` mapped to [0,1]).
* ZoeDepth’s PNG holds **normalized depth**; the actual meter scale lives in `*_meta.json` (`depth_min`, `depth_max`). If you need physically meaningful distances, use that metadata instead of a hardcoded 0.5–8 m mapping.

---

Use this as the canonical description for any future tooling/chat that needs to understand how SAM, ZoeDepth, and the Gaussian splat renderer fit together.
