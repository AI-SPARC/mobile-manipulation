
.. _file_src_welding_with_circular_conveyor_belt.cpp:

File welding_with_circular_conveyor_belt.cpp
============================================

|exhale_lsh| :ref:`Parent directory <dir_src>` (``src``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Nó ROS2 responsável pelo controle automatizado de soldagem. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``src/welding_with_circular_conveyor_belt.cpp``)
------------------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_src_welding_with_circular_conveyor_belt.cpp.rst



Detailed Description
--------------------

Este arquivo implementa a classe ``WeldingWithCircularConveyorBelt``, responsável por controlar o braço robótico e a esteira no ambiente de simulação **Isaac Sim**, além de processar detecções 3D de objetos e executar operações de soldagem automatizadas. 


A arquitetura integra **ROS2**, **MoveIt2** e **YAML-CPP**, realizando o planejamento de trajetórias, controle da esteira e execução de poses de solda configuradas externamente via arquivo YAML.

1.0 

07-11-2025 

Lucas Momesso 






Includes
--------


- ``chrono``

- ``cmath``

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

- ``std_msgs/msg/float32.hpp``

- ``tf2/LinearMath/Matrix3x3.h``

- ``tf2/LinearMath/Quaternion.h``

- ``tf2_geometry_msgs/tf2_geometry_msgs.hpp``

- ``tf2_ros/buffer.h``

- ``tf2_ros/transform_listener.h``

- ``trajectory_msgs/msg/joint_trajectory.hpp``

- ``trajectory_msgs/msg/joint_trajectory_point.hpp``

- ``tuple``

- ``vector``

- ``vision_msgs/msg/detection3_d_array.hpp``

- ``yaml-cpp/yaml.h``






Namespaces
----------


- :ref:`namespace_std`


Classes
-------


- :ref:`exhale_struct_structpair__hash`

- :ref:`exhale_struct_structstd_1_1hash_3_01std_1_1tuple_3_01float_00_01float_00_01float_01_4_01_4`

- :ref:`exhale_struct_structstd_1_1hash_3_01std_1_1tuple_3_01std_1_1pair_3_01int_00_01int_01_4_00_01bool_01_4_01_4`

- :ref:`exhale_struct_structTupleEqual`

- :ref:`exhale_struct_structTupleHash`

- :ref:`exhale_class_classWeldingWithCircularConveyorBelt`


Functions
---------


- :ref:`exhale_function_welding__with__circular__conveyor__belt_8cpp_1a0ddf1224851353fc92bfbff6f499fa97`

- :ref:`exhale_function_welding__with__circular__conveyor__belt_8cpp_1acd1c775ffe27f83c098ab92b3a64965f`

