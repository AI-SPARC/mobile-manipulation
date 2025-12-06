
.. _file_src_add_collision_objects.cpp:

File add_collision_objects.cpp
==============================

|exhale_lsh| :ref:`Parent directory <dir_src>` (``src``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Nó ROS2 responsável por adicionar objetos de colisão no ambiente MoveIt com base em detecções 3D. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``src/add_collision_objects.cpp``)
----------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_src_add_collision_objects.cpp.rst



Detailed Description
--------------------

Este nó escuta mensagens de detecção 3D (topic ``/boxes_detection_array``), verifica se os objetos detectados estão autorizados de acordo com um arquivo YAML, e os adiciona (ou atualiza) como objetos de colisão na cena MoveIt.

O nó também adiciona um plano de chão e pode inicializar grupos de movimento (braço e garra).

1.0 

07-11-2025 

Lucas Momesso 






Includes
--------


- ``chrono``

- ``cmath``

- ``fstream``

- ``functional``

- ``geometry_msgs/msg/pose.hpp``

- ``iostream``

- ``memory``

- ``moveit/move_group_interface/move_group_interface.hpp``

- ``moveit/planning_scene_interface/planning_scene_interface.hpp``

- ``moveit/robot_model_loader/robot_model_loader.hpp``

- ``moveit/robot_state/robot_state.hpp``

- ``moveit_msgs/msg/collision_object.hpp``

- ``moveit_msgs/msg/move_it_error_codes.hpp``

- ``object_manipulation_interfaces/srv/object_collision.hpp``

- ``random``

- ``rclcpp/rclcpp.hpp``

- ``sensor_msgs/msg/point_cloud2.hpp``

- ``shape_msgs/msg/solid_primitive.hpp``

- ``tf2_geometry_msgs/tf2_geometry_msgs.hpp``

- ``tf2_ros/buffer.h``

- ``tf2_ros/transform_listener.h``

- ``trajectory_msgs/msg/joint_trajectory.hpp``

- ``trajectory_msgs/msg/joint_trajectory_point.hpp``

- ``tuple``

- ``unordered_map``

- ``unordered_set``

- ``vector``

- ``vision_msgs/msg/detection3_d_array.hpp``

- ``yaml-cpp/yaml.h``






Namespaces
----------


- :ref:`namespace_std`

- :ref:`namespace_std__chrono_literals`


Classes
-------


- :ref:`exhale_struct_structLabelRule`

- :ref:`exhale_struct_structpair__hash`

- :ref:`exhale_struct_structstd_1_1hash_3_01std_1_1tuple_3_01float_00_01float_00_01float_01_4_01_4`

- :ref:`exhale_struct_structstd_1_1hash_3_01std_1_1tuple_3_01std_1_1pair_3_01int_00_01int_01_4_00_01bool_01_4_01_4`

- :ref:`exhale_struct_structTupleEqual`

- :ref:`exhale_struct_structTupleHash`

- :ref:`exhale_class_classAddCollision`


Functions
---------


- :ref:`exhale_function_add__collision__objects_8cpp_1a3c04138a5bfe5d72780bb7e82a18e627`

- :ref:`exhale_function_add__collision__objects_8cpp_1acd1c775ffe27f83c098ab92b3a64965f`

