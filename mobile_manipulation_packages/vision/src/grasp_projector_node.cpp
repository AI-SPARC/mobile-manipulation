#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_array.hpp>

// Sincronização
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

// OpenCV
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

// Eigen
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>

// Assimp
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

using std::placeholders::_1;
using std::placeholders::_2;

struct MeshFace {
    int v1, v2, v3;
    cv::Vec3b original_color; // Nova propriedade para armazenar a cor do .glb
};

struct RenderFace {
    int v1, v2, v3;
    float depth;
    cv::Point3f normal;
    cv::Vec3b color;
};

class GraspProjectorNode : public rclcpp::Node
{
public:
    GraspProjectorNode() : Node("grasp_projector_node")
    {
        // Parâmetros exatos que forneceu
        this->declare_parameter<std::string>("gripper_model_path", "/home/momesso/pcds/GLB_Foxglove/PandaHand.glb");
        this->declare_parameter<double>("gripper_mesh_scale", 1.0);
        this->declare_parameter<double>("mesh_offset_x", 0.025);
        this->declare_parameter<double>("mesh_offset_y", 0.0);
        this->declare_parameter<double>("mesh_offset_z", 0.0);
        this->declare_parameter<double>("mesh_rot_roll", 1.57);
        this->declare_parameter<double>("mesh_rot_pitch", 0.0);
        this->declare_parameter<double>("mesh_rot_yaw", 1.57);

        scale_ = this->get_parameter("gripper_mesh_scale").as_double();
        offset_x_ = this->get_parameter("mesh_offset_x").as_double();
        offset_y_ = this->get_parameter("mesh_offset_y").as_double();
        offset_z_ = this->get_parameter("mesh_offset_z").as_double();
        roll_ = this->get_parameter("mesh_rot_roll").as_double();
        pitch_ = this->get_parameter("mesh_rot_pitch").as_double();
        yaw_ = this->get_parameter("mesh_rot_yaw").as_double();

        std::string model_path = this->get_parameter("gripper_model_path").as_string();
        loadMesh(model_path);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
            "/best_grasps_poses", 10,
            std::bind(&GraspProjectorNode::pose_callback, this, _1));

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/camera/camera/color/image_with_grasps", 10);

        image_sub_.subscribe(this, "/camera/camera/color/image_raw", rmw_qos_profile_sensor_data);
        info_sub_.subscribe(this, "/camera/camera/color/camera_info", rmw_qos_profile_sensor_data);

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), image_sub_, info_sub_);
        sync_->registerCallback(std::bind(&GraspProjectorNode::camera_callback, this, _1, _2));

        RCLCPP_INFO(this->get_logger(), "Projetor 3D com Cores Originais Iniciado!");
    }

private:
    geometry_msgs::msg::PoseArray::SharedPtr latest_poses_;
    
    std::vector<cv::Point3f> mesh_vertices_base_;
    std::vector<MeshFace> mesh_faces_;

    double scale_, offset_x_, offset_y_, offset_z_, roll_, pitch_, yaw_;

    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> info_sub_;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image, sensor_msgs::msg::CameraInfo> SyncPolicy;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void loadMesh(const std::string& path)
    {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path, 
            aiProcess_Triangulate | 
            aiProcess_JoinIdenticalVertices |
            aiProcess_PreTransformVertices); 

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            RCLCPP_ERROR(this->get_logger(), "Erro Assimp: %s", importer.GetErrorString());
            return;
        }

        int vertex_offset = 0;
        for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
            aiMesh* mesh = scene->mMeshes[i];
            
            // --- LEITURA DA COR DO MATERIAL DO .GLB ---
            aiColor4D material_color(0.6f, 0.6f, 0.6f, 1.0f); // Cinzento padrão
            if (scene->HasMaterials()) {
                aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
                // .glb costuma usar PBR (BASE_COLOR) ou DIFFUSE tradicional
                if (AI_SUCCESS != material->Get(AI_MATKEY_BASE_COLOR, material_color)) {
                    material->Get(AI_MATKEY_COLOR_DIFFUSE, material_color);
                }
            }
            
            // O OpenCV usa BGR
            cv::Vec3b face_color(
                static_cast<uchar>(material_color.b * 255), 
                static_cast<uchar>(material_color.g * 255), 
                static_cast<uchar>(material_color.r * 255)
            );
            // ------------------------------------------

            for (unsigned int v = 0; v < mesh->mNumVertices; v++) {
                mesh_vertices_base_.push_back(cv::Point3f(
                    mesh->mVertices[v].x * scale_, 
                    mesh->mVertices[v].y * scale_, 
                    mesh->mVertices[v].z * scale_));
            }
            
            for (unsigned int f = 0; f < mesh->mNumFaces; f++) {
                aiFace face = mesh->mFaces[f];
                if (face.mNumIndices == 3) {
                    mesh_faces_.push_back({
                        (int)(vertex_offset + face.mIndices[0]), 
                        (int)(vertex_offset + face.mIndices[1]), 
                        (int)(vertex_offset + face.mIndices[2]),
                        face_color // Salva a cor real desta malha
                    });
                }
            }
            vertex_offset += mesh->mNumVertices;
        }
        RCLCPP_INFO(this->get_logger(), "Malha carregada: %lu vértices, %lu faces com materiais nativos.", mesh_vertices_base_.size(), mesh_faces_.size());
    }

    void pose_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
    {
        latest_poses_ = msg;
    }

    void camera_callback(const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                         const sensor_msgs::msg::CameraInfo::ConstSharedPtr& info_msg)
    {
        if (!latest_poses_ || latest_poses_->poses.empty() || mesh_vertices_base_.empty()) return;

        std::string target_frame = info_msg->header.frame_id;
        if (target_frame.find("optical") == std::string::npos) {
            target_frame = "camera_color_optical_frame";
        }

        geometry_msgs::msg::TransformStamped tf_map_to_cam;
        try {
            tf_map_to_cam = tf_buffer_->lookupTransform(target_frame, "map", tf2::TimePointZero);
        } catch (const tf2::TransformException & ex) {
            return;
        }

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) { return; }

        cv::Mat K(3, 3, CV_64F, 0.0);
        for (int i = 0; i < 9; i++) K.at<double>(i / 3, i % 3) = info_msg->k[i];
        std::vector<double> D(info_msg->d.begin(), info_msg->d.end());

        Eigen::Affine3d eigen_map_to_cam = tf2::transformToEigen(tf_map_to_cam);

        // --- CONSTRUÇÃO DO OFFSET GEOMÉTRICO ---
        Eigen::Affine3d tf_geometry_to_tcp = Eigen::Affine3d::Identity();
        tf_geometry_to_tcp.translation() = Eigen::Vector3d(offset_x_, offset_y_, offset_z_);
        
        Eigen::Matrix3d rot_geom;
        rot_geom = Eigen::AngleAxisd(roll_, Eigen::Vector3d::UnitX())
                 * Eigen::AngleAxisd(pitch_, Eigen::Vector3d::UnitY())
                 * Eigen::AngleAxisd(yaw_, Eigen::Vector3d::UnitZ());

        // AQUI ESTÁ A CORREÇÃO DA ROTAÇÃO ERRADA DE 180 GRAUS NO Z:
        Eigen::Matrix3d z_flip_correction = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        
        // Multiplicamos o offset original pela correção de 180 graus
        tf_geometry_to_tcp.linear() = rot_geom * z_flip_correction;
        // ---------------------------------------

        for (size_t g = 0; g < latest_poses_->poses.size(); g++)
        {
            Eigen::Affine3d eigen_tcp_to_map;
            tf2::fromMsg(latest_poses_->poses[g], eigen_tcp_to_map);

            Eigen::Affine3d eigen_mesh_to_cam = eigen_map_to_cam * eigen_tcp_to_map * tf_geometry_to_tcp;

            std::vector<cv::Point3f> cam_vertices(mesh_vertices_base_.size());
            for (size_t i = 0; i < mesh_vertices_base_.size(); ++i) {
                Eigen::Vector3d pt(mesh_vertices_base_[i].x, mesh_vertices_base_[i].y, mesh_vertices_base_[i].z);
                Eigen::Vector3d pt_cam = eigen_mesh_to_cam * pt;
                cam_vertices[i] = cv::Point3f(pt_cam.x(), pt_cam.y(), pt_cam.z());
            }

            cv::Mat rvec_zero = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat tvec_zero = cv::Mat::zeros(3, 1, CV_64F);

            std::vector<cv::Point2f> img_vertices;
            cv::projectPoints(cam_vertices, rvec_zero, tvec_zero, K, D, img_vertices);

            std::vector<RenderFace> faces_to_render;
            faces_to_render.reserve(mesh_faces_.size());

            for (const auto& f : mesh_faces_) {
                cv::Point3f p1 = cam_vertices[f.v1];
                cv::Point3f p2 = cam_vertices[f.v2];
                cv::Point3f p3 = cam_vertices[f.v3];
                
                float depth = (p1.z + p2.z + p3.z) / 3.0f;

                cv::Point3f u = p2 - p1;
                cv::Point3f v = p3 - p1;
                cv::Point3f n = u.cross(v);
                float norm = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
                if (norm > 0) { n.x/=norm; n.y/=norm; n.z/=norm; }

                faces_to_render.push_back({f.v1, f.v2, f.v3, depth, n, f.original_color});
            }

            // Z-Sorting (desenha do fundo para a frente)
            std::sort(faces_to_render.begin(), faces_to_render.end(),
                [](const RenderFace& a, const RenderFace& b) {
                    return a.depth > b.depth;
                });

            cv::Point3f light_dir(0, -0.5, -1);
            float light_norm = std::sqrt(light_dir.x*light_dir.x + light_dir.y*light_dir.y + light_dir.z*light_dir.z);
            light_dir.x /= light_norm; light_dir.y /= light_norm; light_dir.z /= light_norm;

            for (const auto& f : faces_to_render) {
                if (f.depth < 0.05) continue;

                cv::Point pts[3];
                pts[0] = img_vertices[f.v1];
                pts[1] = img_vertices[f.v2];
                pts[2] = img_vertices[f.v3];
                
                float dot = std::abs(f.normal.x * light_dir.x + f.normal.y * light_dir.y + f.normal.z * light_dir.z);
                float intensity = 0.5f + 0.5f * dot; 

                // Aplica a iluminação sobre a cor real da malha
                cv::Scalar color = CV_RGB(
                    std::min(255.0f, f.color[2] * intensity), 
                    std::min(255.0f, f.color[1] * intensity), 
                    std::min(255.0f, f.color[0] * intensity)
                );
                
                const cv::Point* ppt[1] = { pts };
                int npt[] = { 3 };
                
                cv::fillPoly(cv_ptr->image, ppt, npt, 1, color, cv::LINE_AA);
                
                // Desenha bordas sutis com a mesma cor para evitar fissuras na malha
                cv::polylines(cv_ptr->image, ppt, npt, 1, true, color, 1, cv::LINE_AA);
            }
        }
        image_pub_->publish(*cv_ptr->toImageMsg());
    }
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GraspProjectorNode>());
    rclcpp::shutdown();
    return 0;
}