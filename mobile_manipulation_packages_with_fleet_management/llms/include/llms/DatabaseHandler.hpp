#pragma once

#include <sqlite3.h>
#include <string>
#include <optional>
#include <mutex>
#include <stdexcept>


struct ObjectProperties {
    std::string id;         // Chave primária (ex: "box_01", "storage_02")
    std::string pose_str;   // Pose como string: "x;y;z"
    std::string size_str;   // Tamanho como string: "width;height;depth"
    int64_t last_update;    // Timestamp da última atualização
};

class DatabaseHandler {
public:
    explicit DatabaseHandler(const std::string& db_path) : db_(nullptr) 
    {
        int rc = sqlite3_open(db_path.c_str(), &db_);
        if (rc != SQLITE_OK) {
            throw std::runtime_error("Erro ao abrir banco de dados: " + 
                                    std::string(sqlite3_errmsg(db_)));
        }
    }
    
    ~DatabaseHandler() 
    {
        if (db_) {
            sqlite3_close(db_);
        }
    }

   
    DatabaseHandler(const DatabaseHandler&) = delete;
    DatabaseHandler& operator=(const DatabaseHandler&) = delete;
    
    
    std::optional<ObjectProperties> get_object_data(const std::string& id) 
    {
        std::lock_guard<std::mutex> lock(db_mutex_);

        const char* sql = "SELECT id, pose, size, last_update FROM objects WHERE id = ?";
        
        sqlite3_stmt* stmt;
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            return std::nullopt;
        }
        
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);
        
        std::optional<ObjectProperties> result = std::nullopt;
        
        if (sqlite3_step(stmt) == SQLITE_ROW) 
        {
            ObjectProperties props;
            
            
            const char* id_ptr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            const char* pose_ptr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            const char* size_ptr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            
            props.id = id_ptr ? id_ptr : "";
            props.pose_str = pose_ptr ? pose_ptr : "";
            props.size_str = size_ptr ? size_ptr : "";
            props.last_update = sqlite3_column_int64(stmt, 3);
            
            result = props;
        }
        
        sqlite3_finalize(stmt);
        return result;
    }

   
    bool object_exists(const std::string& id)
    {
        std::lock_guard<std::mutex> lock(db_mutex_);

        const char* sql = "SELECT 1 FROM objects WHERE id = ? LIMIT 1";
        
        sqlite3_stmt* stmt;
        
        if (sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr) != SQLITE_OK) 
        {
            return false;
        }
        
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_STATIC);
        
        bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
        sqlite3_finalize(stmt);
        return exists;
    }

private:
    sqlite3* db_;
    std::mutex db_mutex_;
};