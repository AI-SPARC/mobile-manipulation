#include "llms/WorldStateNode.hpp"
#include <filesystem>
#include <sstream> 
#include "rclcpp_components/register_node_macro.hpp"

namespace llms
{

WorldStateNode::WorldStateNode(const rclcpp::NodeOptions & options)
: Node("world_state_node", options), db_(nullptr)
{
    this->declare_parameter<std::string>("database_path", "/home/momesso/pibic/src/mobile_manipulation_packages/llms/db/robot_world_data.db");
    this->declare_parameter<int>("num_cameras", 1);

    std::string db_path = this->get_parameter("database_path").as_string();
    int num_cameras = this->get_parameter("num_cameras").as_int();
    
    std::filesystem::path path_obj(db_path);
    
    if (path_obj.has_parent_path()) 
    {
        if (!std::filesystem::exists(path_obj.parent_path())) 
        {
            RCLCPP_WARN(this->get_logger(), "Diretório não existe. Criando: %s", path_obj.parent_path().c_str());
            std::filesystem::create_directories(path_obj.parent_path()); 
        }
    }

    if (sqlite3_open(db_path.c_str(), &db_) != SQLITE_OK) 
    {
        RCLCPP_FATAL(this->get_logger(), "Falha ao abrir DB: %s", sqlite3_errmsg(db_));
    } 
    else 
    {
        init_database();
        RCLCPP_INFO(this->get_logger(), "DB conectado com sucesso em: %s", db_path.c_str());
    }

    for (int i = 0; i < num_cameras; ++i)
    {
        std::string topic_name = "/bbox_3d_with_labels";

        auto sub = this->create_subscription<vision_msgs::msg::Detection3DArray>(
            topic_name, 10, 
            std::bind(&WorldStateNode::handle_detections, this, std::placeholders::_1)
        );

        subscriptions_.push_back(sub);
        RCLCPP_INFO(this->get_logger(), "Inscrito no tópico: %s", topic_name.c_str());
    }
}

WorldStateNode::~WorldStateNode()
{
    if (db_) 
    {
        sqlite3_close(db_);
    }
}

void WorldStateNode::init_database()
{
    std::lock_guard<std::mutex> lock(db_mutex_);

    const char* sql_create = R"(
        CREATE TABLE IF NOT EXISTS objects (
            id TEXT PRIMARY KEY,
            pose TEXT,
            size TEXT,
            last_update INTEGER
        );
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
    )";

    char* err_msg = 0;
    if (sqlite3_exec(db_, sql_create, 0, 0, &err_msg) != SQLITE_OK) 
    {
        RCLCPP_ERROR(this->get_logger(), "SQL Init Error: %s", err_msg);
        sqlite3_free(err_msg);
    }
}

void WorldStateNode::handle_detections(const vision_msgs::msg::Detection3DArray::SharedPtr msg)
{
    for (const auto& detection : msg->detections)
    {
        if (detection.results.empty()) 
        {
            continue;
        }

        std::string obj_id = detection.results[0].hypothesis.class_id;
        if (obj_id.empty()) 
        {
            continue;
        }

        std::stringstream ss_pose;
        ss_pose << detection.bbox.center.position.x << ";" 
                << detection.bbox.center.position.y << ";" 
                << detection.bbox.center.position.z << ";"
                << detection.bbox.center.orientation.x << ";"
                << detection.bbox.center.orientation.y << ";"
                << detection.bbox.center.orientation.z << ";"
                << detection.bbox.center.orientation.w;
        
        std::stringstream ss_size;
        ss_size << detection.bbox.size.x << ";" 
                << detection.bbox.size.y << ";" 
                << detection.bbox.size.z;

        if (upsert_object(obj_id, ss_pose.str(), ss_size.str())) 
        {
            RCLCPP_DEBUG(this->get_logger(), "Update: %s", obj_id.c_str());
        }
    }
}

bool WorldStateNode::upsert_object(const std::string& id, 
    const std::string& pose, const std::string& size)
{
    std::lock_guard<std::mutex> lock(db_mutex_);

    std::string sql = R"(
        INSERT INTO objects (id, pose, size, last_update)
        VALUES (?, ?, ?, strftime('%s','now'))
        ON CONFLICT(id) DO UPDATE SET
            pose=excluded.pose,
            size=excluded.size,
            last_update=excluded.last_update;
    )";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, 0) != SQLITE_OK) 
    {
        RCLCPP_ERROR(this->get_logger(), "SQL Prepare Error: %s", sqlite3_errmsg(db_));
        return false;
    }

    sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, pose.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, size.c_str(), -1, SQLITE_STATIC);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    
    if (!success) 
    {
        RCLCPP_ERROR(this->get_logger(), "SQL Step Error: %s", sqlite3_errmsg(db_));
    }

    sqlite3_finalize(stmt);
    return success;
}

} 

RCLCPP_COMPONENTS_REGISTER_NODE(llms::WorldStateNode)