# Adding the flygym-1.0 tutorials/notebooks to flygym-2.0

In this fork, we want to re-implement the examples/notebooks of flygym-1.0 but, with the much faster/better flygym-2.0!

## Context

Will the tutorials from 1.x.x be ported to 2.0.0?

d3tk
on Apr 10
Summary
I'm trying to port the legacy hybrid turning / walking controller from flygym-gymnasium to flygym==2.0.0.

Question
I was previously using FlyGym 1.x.x, which had tutorials/examples for:

walking (hybrid controller)
olfaction
obstacle navigation
Are these still supported in FlyGym 2.0.0?

More specifically:

Is it still possible to implement these behaviors using the 2.0 API?
Are equivalent tutorials/examples planned for 2.0?
Can the old code be adapted using api1to2.py, or is a full rewrite expected?
Thanks!


sibocw - Maintainer
Hi,

You've hit a pain point—I'd really like to, but unfortunately I don't have the bandwidth to do it at the moment. (@tkclam has adapted some of these tutorials for a course—do you have any updates?)

Backporting the tutorials is conceptually straightforward, but it will take some time on the implementation level. As stated here, the two main differences between v1 and v2 APIs are:

FlyGym v1 interfaces with MuJoCo through dm_control; v2 uses the MuJoCo Python binding directly. Note that v2 still uses the dm_control.mjcf module for model composition, but this is only a nice API for creating/modifying the XML model file; there's no physics simulation in this (think of it as a XML parser/writer).
Reasons for the change:
dm_control doesn't support GPU simulation, so we'd have to use the MuJoCo Python binding for flygym.warp anyway
The code runs faster without the dm_control layer simply because there's less slow Python code running within the main simulation loop.
What code migration would entail:
The two most critical structs in the MuJoCo C implementation are MjModel (which defines model configuration, including DoF configs, etc) and MjData (which tracks the state of the model over time, including positions and forces, etc.). In v1, we first create a dm_control.Physics object. The structs are then exposed as my_physics_obj.model and my_physics_obj.data. In v2, these are simply mujoco.MjModel and mujoco.MjData objects (no Physics layer). The C structs that they bind to are exactly the same.
What's slightly more annoying is that when you read data from MjData, you typically use my_physics_obj.bind(composed_mjcf_element).variable, where composed_mjcf_element can be bodies, actuators, cameras, etc., and variable can be positions, forces etc.). In v2, we kept track of our own copy of body/actuator/camera indices and get/set data directly on my_mjdata.variable arrays. Therefore, backporting the old code would involving writing a bunch of masks/indices for these.
In the v1 codebase (including tutorials), there are 10 calls to physics.model, 4 calls to physics.data, and 51 calls to physics.bind. Changing these is not hard, just a bit tedious.
V1 implements the Gym interface, v2 doesn't.
Reasons for the change:
In v1, observation and action spaces are often defined as python objects (e.g. dicts). This is "user-friendly" but carries a lot of overhead. In v2, we simply used NumPy or Warp arrays to minimize native Python stuff in the simulation loop.
Ultimately, Gym is only a handy interface for the POMDP setup. The use of the Gym interface is ultimately only an implementation detail, not a fundamental feature. In any case, users can easily wrap the v2 API in a Gym environment and it probably takes <1 hour of coding.
What code migration would entail:
Implementing some higher-level observation/action interfaces, e.g. visual renderings, etc.
To be clear, I'm not blaming dm_control and Gym for these shortcomings. Things could have been implemented differently to reduce overhead (for example, in dm_control, we could have called physics.bind(...) less and cached Binding objects ourselves instead, rather than relying the internal caching mechanism within physics.bind(...). In Gym, we could have defined action/observation spaces using arrays instead of dicts). We simply took less performance-optimal design decisions, sometimes by choice (to make it easier to use for novice programmers), and other times out of foolishness. Regardless, now that we've released v2, it makes more sense to adapt the tutorials for v2 rather than improving their implementation using v1.

As you can see, none of these is actually hard. With some coding agent (especially with this "instruction" that I spent ~1hr writing), I really think one can get it done in 2 days or so. Unfortunately everybody on our team is completely tied up with other projects at the moment. I will probably get to it eventually, but the earliest that I can do it is early June (no guarantee though). If you need it on the short term, it'd be greatly appreciated if you want to do it yourself and contribute it back to this project.