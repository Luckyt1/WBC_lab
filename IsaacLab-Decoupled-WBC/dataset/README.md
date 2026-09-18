# Dataset

`g1/CMU/` contains a selected subset of the CMU motions from
[AMASS Retargeted for G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1)
(AMASS motions retargeted to the Unitree G1). During teacher training these
clips replay the upper-body arm motions while the policy controls the legs
and waist.

## Format

Files are `<subject_id>/<clip>_jpos.npy` with shape `(T, 36)`:
columns `0:3` root world position (m), `3:7` root quaternion (x, y, z, w),
`7:36` the 29 joint positions (rad) in Unitree G1 XML order. This repository
reads only the 14 arm columns (`22:36`).

Loaded by `legged_lab/utils/amass.py`. G1 tasks replay these targets on their
14 arm joints (`arm_motion_source=amass`). ELF3's `elf3_wbc` task uses
`amass_elf3`: named joint mapping with nominal shoulder mount compensation,
ELF3 soft limits, a reset ramp and target rate limiting. Its current default
`arm_motion_timing=legacy_frames` matches G1's one-source-frame-per-control-step
replay with last-frame padding. The earlier `source_fps` timing remains available:
it resamples from source FPS to control FPS and reflects short clips at endpoints.
This transfers joint motion, not Cartesian hand paths between different arm
lengths. See the [training guide](../../README.md).

Both training and play support the selected arm source; play restores it from
the saved run config. The original `elf3_flat` task holds default arm targets.
This backup includes simulation training and playback only.

## License and provenance

The motion files under `g1/CMU/` were manually selected from the CMU subset of
[AMASS Retargeted for G1](https://huggingface.co/datasets/ember-lab-berkeley/AMASS_Retargeted_for_G1), and are included only for users’ convenience.
They are not covered by this repository's BSD-3-Clause software license.
Use and redistribution remain subject to the applicable original dataset terms.
No additional rights to these files are granted by this repository.
