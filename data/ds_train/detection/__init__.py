"""Detection / pose datasets used by PoseVLA's VLM 3D branch.

Modules
-------
- ``dataset_omni6d``   -- Omni6D consumer dataset + canonical helpers
                         (``_build_sparse_depth`` / ``_generate_rays`` / ...)
- ``dataset_omni3d``   -- Omni3D consumer dataset (DetAny3D pickle format)
- ``dataset_bop``      -- BOP-format consumer dataset
- ``dataset_clutter``  -- GraspClutter6D consumer dataset
- ``graspclutter6dAPI``-- low-level helpers used by ``dataset_clutter``
"""