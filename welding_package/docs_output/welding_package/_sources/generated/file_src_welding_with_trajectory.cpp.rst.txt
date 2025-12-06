
.. _file_src_welding_with_trajectory.cpp:

File welding_with_trajectory.cpp
================================

|exhale_lsh| :ref:`Parent directory <dir_src>` (``src``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS



Nó ROS2 responsável pela soldagem automatizada com trajetórias complexas usando MoveIt2 e Isaac Sim. 



.. contents:: Contents
   :local:
   :backlinks: none

Definition (``src/welding_with_trajectory.cpp``)
------------------------------------------------


.. toctree::
   :maxdepth: 1

   program_listing_file_src_welding_with_trajectory.cpp.rst



Detailed Description
--------------------



- 


Este arquivo implementa o nó **WeldingWithTrajectory**, responsável por realizar o processo completo de soldagem de peças transportadas por uma **esteira circular** simulada no **Isaac Sim**, utilizando controle de trajetória do **MoveIt2** para o braço robótico.

- O nó coordena de forma autônoma a movimentação do robô e da esteira, realizando diferentes tipos de trajetórias de soldagem — **normais, lineares ou circulares** — de acordo com os parâmetros definidos em um arquivo YAML.

-  




Principais funcionalidades



- Integração direta com **Isaac Sim** para controle de velocidade da esteira circular;

- Planejamento e execução de **trajetórias cartesianas** com MoveIt2;

- Leitura de **poses e trajetórias de solda** a partir de um arquivo YAML configurável;

- Processamento de **detecções 3D** de objetos via tópico ``/bbox_3d_with_labels``;

- Controle sincronizado entre **detecção, parada da esteira e execução da solda**;

- Suporte a **três tipos de trajetória**:

- ``"normal"``: movimento até uma única pose de soldagem;

- ``"line"``: soldagem linear entre o ponto atual e um ponto alvo;

- ``"circle"``: soldagem circular (definida por raio e ângulos);

- Retorno automático do robô à posição inicial após a solda e retomada do movimento da esteira.

-  





Fluxo geral de operação



1. O nó é iniciado e declara o parâmetro ``yaml_file`` contendo o caminho do arquivo de configuração;

2. O arquivo YAML é carregado pelo método ``loadLocationsFromYaml()``, populando um mapa de operações de solda por objeto;

3. São criados publishers e subscribers ROS2 para comunicação com Isaac Sim e o sistema de visão;

4. Um temporizador (``init_timer_``) tenta inicializar o ``MoveGroupInterface`` de forma assíncrona;

5. Ao detectar uma peça via ``/bbox_3d_with_labels``, o callback ``detectionCallback()``:



- Identifica o tipo de trajetória associado à peça;

- Para a esteira quando a peça está na posição ideal;

- Executa a trajetória de solda correspondente (ponto, linha ou arco);

- Retoma o movimento da esteira após o término da solda.

-  





Tópicos ROS2

**Publishers**

- ``/conveyor_velocity`` (``std_msgs::msg::Float32``): Controla a **velocidade linear** da esteira.

- ``/conveyor_angular_velocity`` (``std_msgs::msg::Float32``): Controla a **velocidade angular** da esteira.

- **Subscriber**

- ``/bbox_3d_with_labels`` (``vision_msgs::msg::Detection3DArray``): Recebe detecções de objetos com pose e rótulo, disparando o processo de soldagem.

-  





Parâmetros



- ``yaml_file``: Caminho para o arquivo YAML contendo as poses e definições de trajetória para cada tipo de objeto.

-  





Integração com MoveIt2

O ``MoveGroupInterface`` para o grupo ``"denso_arm"`` é usado para planejar e executar movimentos. A função de planejamento **cartesiano** (``computeCartesianPath``) é fundamental para gerar trajetórias suaves e contínuas para as soldas do tipo ``"line"`` e ``"circle"``.

-  1.0 

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

- :ref:`exhale_struct_structWeldingWithTrajectory_1_1WeldingPoseData`

- :ref:`exhale_class_classWeldingWithTrajectory`


Functions
---------


- :ref:`exhale_function_welding__with__trajectory_8cpp_1a0ddf1224851353fc92bfbff6f499fa97`

- :ref:`exhale_function_welding__with__trajectory_8cpp_1acd1c775ffe27f83c098ab92b3a64965f`

