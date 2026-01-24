#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/mesh_operations.h>

#include <fcl/narrowphase/collision.h>
#include <fcl/geometry/bvh/BVH_model.h>

#include <Eigen/Geometry>
#include <memory>
#include <string>

// Tipo para Mesh FCL
typedef fcl::BVHModel<fcl::OBBRSSf> FCLMesh;
typedef std::shared_ptr<FCLMesh> FCLMeshPtr;

class PandaGripperModel
{
public:
    PandaGripperModel()
    {
        // 1. Resolve os caminhos absolutos usando ament_index
        // (Assume que você tem o pacote moveit_resources_panda_description instalado)
        std::string pkg_path = ament_index_cpp::get_package_share_directory("moveit_resources_panda_description");
        std::string hand_path = pkg_path + "/meshes/collision/hand.stl";
        std::string finger_path = pkg_path + "/meshes/collision/finger.stl";

        // 2. Carrega as Meshes (geometria pura na origem 0,0,0)
        hand_mesh_ = loadMesh(hand_path);
        left_finger_mesh_ = loadMesh(finger_path);
        right_finger_mesh_ = loadMesh(finger_path); // Reusa o arquivo, mas é outro objeto

        // 3. Define os Offsets EXATOS do URDF
        setupFixedTransforms();
    }

    // Verifica colisão da Mão Inteira contra um Hull (seu objeto mapeado)
    // hand_pose_world: A pose do link 'panda_hand' no mundo
    bool checkCollision(const Eigen::Affine3d& hand_pose_world, 
                        fcl::CollisionObjectf* object_hull)
    {
        // --- 1. Calcula Pose do Link da Mão no Mundo ---
        // Converte Eigen (double) para FCL (float)
        fcl::Transform3f tf_hand_world = toFCL(hand_pose_world);

        // --- 2. Calcula Pose dos Dedos baseada no URDF ---
        // Left: Hand * Joint_Offset
        fcl::Transform3f tf_left_world = tf_hand_world * tf_hand_to_left_joint_;
        
        // Right: Hand * Joint_Offset * Mesh_Rotation (origin rpy do XML)
        fcl::Transform3f tf_right_world = tf_hand_world * tf_hand_to_right_joint_ * tf_right_mesh_geom_;

        // --- 3. Instancia os Objetos de Colisão (Leve, apenas ponteiros) ---
        fcl::CollisionObjectf obj_hand(hand_mesh_, tf_hand_world);
        fcl::CollisionObjectf obj_left(left_finger_mesh_, tf_left_world);
        fcl::CollisionObjectf obj_right(right_finger_mesh_, tf_right_world);

        fcl::CollisionRequestf req;
        fcl::CollisionResultf res;

        // --- 4. Testa Colisões ---
        
        // Hand Base vs Objeto
        fcl::collide(object_hull, &obj_hand, req, res);
        if(res.isCollision()) return true;

        // Left Finger vs Objeto
        fcl::collide(object_hull, &obj_left, req, res);
        if(res.isCollision()) return true;

        // Right Finger vs Objeto
        fcl::collide(object_hull, &obj_right, req, res);
        if(res.isCollision()) return true;

        return false;
    }

private:
    FCLMeshPtr hand_mesh_;
    FCLMeshPtr left_finger_mesh_;
    FCLMeshPtr right_finger_mesh_;

    // Transformações Fixas (URDF)
    fcl::Transform3f tf_hand_to_left_joint_;
    fcl::Transform3f tf_hand_to_right_joint_;
    fcl::Transform3f tf_right_mesh_geom_;

    void setupFixedTransforms()
    {
        // --- BASEADO NO SEU XML ---
        
        // <joint name="panda_finger_joint1"> ... <origin xyz="0 0 0.0584" ... />
        tf_hand_to_left_joint_.setIdentity();
        tf_hand_to_left_joint_.translation() = fcl::Vector3f(0.0f, 0.0f, 0.0584f);

        // <joint name="panda_finger_joint2"> ... <origin xyz="0 0 0.0584" ... />
        tf_hand_to_right_joint_.setIdentity();
        tf_hand_to_right_joint_.translation() = fcl::Vector3f(0.0f, 0.0f, 0.0584f);

        // <link name="panda_rightfinger"> <collision> <origin rpy="0 0 3.14159..." />
        // Isso gira a malha 180 graus em Z DEPOIS de estar no link
        tf_right_mesh_geom_.setIdentity();
        
        Eigen::AngleAxisf rot_z(M_PI, Eigen::Vector3f::UnitZ()); 
        tf_right_mesh_geom_.linear() = rot_z.toRotationMatrix();
    }

    // Helper: Carrega STL usando geometric_shapes
    FCLMeshPtr loadMesh(const std::string& path)
    {
        shapes::Mesh* mesh = shapes::createMeshFromResource("file://" + path);
        if (!mesh) throw std::runtime_error("Erro ao carregar mesh: " + path);

        FCLMeshPtr fcl_mesh(new FCLMesh());
        fcl_mesh->beginModel();
        for (unsigned int i = 0; i < mesh->triangle_count; ++i)
        {
            unsigned int i1 = mesh->triangles[3 * i];
            unsigned int i2 = mesh->triangles[3 * i + 1];
            unsigned int i3 = mesh->triangles[3 * i + 2];

            fcl::Vector3f v1(mesh->vertices[3 * i1], mesh->vertices[3 * i1 + 1], mesh->vertices[3 * i1 + 2]);
            fcl::Vector3f v2(mesh->vertices[3 * i2], mesh->vertices[3 * i2 + 1], mesh->vertices[3 * i2 + 2]);
            fcl::Vector3f v3(mesh->vertices[3 * i3], mesh->vertices[3 * i3 + 1], mesh->vertices[3 * i3 + 2]);

            fcl_mesh->addTriangle(v1, v2, v3);
        }
        fcl_mesh->endModel();
        delete mesh; 
        return fcl_mesh;
    }

    // Helper: Eigen Double -> FCL Float
    fcl::Transform3f toFCL(const Eigen::Affine3d& e)
    {
        fcl::Transform3f t;
        t.translation() = e.translation().cast<float>();
        t.linear() = e.rotation().cast<float>();
        return t;
    }
};