"""PoseVLA dataset package.

The package is split into two sub-packages by responsibility:

- ``data.ds_train.detection`` -- 2D/3D detection & pose datasets
  (Omni6D / Omni3D / BOP / GraspClutter6D)
- ``data.ds_train.robot``     -- robot trajectory datasets
  (Agibot / LeRobot / HDF5 action)

Use full-path imports, e.g.::

    from data.ds_train.detection.dataset_omni6d import Omni6dConsumerDataset
    from data.ds_train.robot.dataset_hdf5_action import VLAConsumerDataset
"""