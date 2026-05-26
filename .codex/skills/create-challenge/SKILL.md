---
name: create-challenge
description: Create or modify local LeetCode-but-robotics challenges. Use when Codex needs to prompt through, design, or implement a new robotics or computer-vision coding challenge with injected Python APIs, composed MuJoCo/OpenCV/Rerun harnesses, frontend challenge specs and stubs, demo RobotPolicy starter code, scoring, tests, and browser verification.
---

# Create Challenge

## Overview

Create robotics and computer-vision coding challenges with a hidden simulator or dataset harness and a clear user-facing Python class API. Prefer composing existing robot/vision APIs, world builders, runners, replay exporters, frontend stub generators, and scoring patterns before adding new infrastructure.

Read `references/challenge-authoring-spec.md` before implementing a challenge in this repository. Use `assets/challenge-brief-template.md` when a persistent repo-local challenge brief would make the challenge easier to add or review.

## Workflow

1. Clarify the smallest set of missing facts.
   Ask at most three concise questions when the request does not specify the robot/environment, the exposed sensors and actuators, or what success means. If the intent is clear enough, proceed and write assumptions into the challenge brief or final notes.

2. Inspect the current app before editing.
   Check the frontend challenge model and stub generator, backend catalog/models, policy API and loader, existing simulation runners, MuJoCo world builders, Rerun export, tests, and sample policies. Treat the existing shape as the default style.

3. Choose a composition path.
   Reuse an existing runner when the requested task matches its embodiment, physics or dataset, action/submission space, replay shape, and scoring inputs. Extend the nearest runner/API when only sensors, stochasticity, scenario generation, or scoring differ. Add a new runner/world/API only when the embodiment, dynamics, or dataset modality require it, such as drones, motor fixtures, manipulators, image datasets, or signal-classification tasks that are not differential-drive planar control.

4. Keep the user code JSON-free.
   Expose an injected runtime object in `RobotPolicy.step(robot)`. Provide LeetCode-style commented stubs that describe classes, fields, methods, constants, and stochastic sensor behavior. Do not expose simulator implementation details, MuJoCo internals, raw JSON observations, or reset hooks unless the user explicitly asks for that surface.

5. Make physics challenges produce MuJoCo and Rerun output unless the user explicitly narrows scope. Make computer-vision challenges produce Rerun output and annotated image evidence, with MuJoCo intentionally disabled in the UI.
   A complete physics challenge should return render frames in the run response and write a Rerun `.rrd` artifact. A complete computer-vision challenge should provide seeded or fixture-backed images through an injected API, score submitted detections/classifications/estimates, and write a Rerun `.rrd` artifact with images, labels, predictions, and metrics. If MuJoCo, OpenCV, or Rerun is optional due local dependencies, keep the graceful fallback but report the missing artifact as residual risk.

6. Include non-solution demo code.
   Add a short `RobotPolicy` starter/demo that teaches the critical public API call and visibly exercises the replay path without serving as a solution. The frontend starter should be an API exercise, not a tuned controller, planner, estimator, or classifier. Examples: slow forward plus sinusoidal wheel differential for mobile robots, a placeholder `submit_odometry(Pose2d(...))` plus bounded wheel motion for odometry tasks, a neutral `submit_collision_decision(False, "none")` for classifier tasks, gentle sinusoidal current for a motor-scale fixture, or a conservative hover/forward command for drones. Keep stronger reference or sample policies in backend samples/tests, not in the frontend starter.

7. Validate end to end.
   Run backend tests, frontend build, and a browser run of the new challenge when it is runnable. Verify that the problem page shows the generated stub, the starter code imports no simulator internals, the run completes, MuJoCo frames render, and a Rerun recording is available.

## Implementation Checklist

- Add or update a challenge brief for nontrivial challenges using `assets/challenge-brief-template.md`.
- Add frontend interface members, stub types, problem frame, challenge contract, success conditions, replay capabilities, and starter implementation through the existing challenge composer.
- Add backend `ChallengeSpec` entries and shared robot/environment/vision primitives instead of duplicating literals.
- Add or extend injected policy API classes and policy loader globals for any user-visible types.
- Add or extend simulation runners and MuJoCo world builders only where composition is insufficient.
- Add or extend OpenCV-backed dataset/image runners for vision tasks; ensure `opencv-python` is a project dependency and `cv2` is available in submitted policies.
- Route runnable challenges in the API by robot or runner type; avoid hardcoded single-challenge gates.
- Write Rerun logs for static scene, robot state, controls, relevant sensors, and MuJoCo images.
- For computer-vision challenges, write Rerun logs for source images, ground truth, submitted outputs, matching diagnostics, and summary metrics; do not invent MuJoCo frames.
- Add focused tests for catalog detail, runner success/failure shape, bad policy validation, replay metadata, render frame presence, and artifact presence when dependencies exist.
- Keep frontend starter/demo code distinct from backend sample/reference policies; the starter must show API usage without solving the public task.
- Update sample policies and documentation only for the public challenge interface.

## Done Criteria

A physics challenge is not done until the user can open the problem, see the injected API stub and stochastic contract, run a visibly non-solution demo policy, inspect MuJoCo frames, and open or receive a Rerun recording. A computer-vision challenge is not done until the user can open the problem, see the injected image/submission API stub and stochastic/data contract, run a visibly non-solution demo policy, see MuJoCo unavailable by design, and open or receive a Rerun recording with image evidence. If the challenge is intentionally a draft/spec only, label it as such in the UI and do not present it as runnable.
