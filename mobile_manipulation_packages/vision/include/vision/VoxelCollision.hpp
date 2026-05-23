#pragma once

#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Precisa existir no mesmo escopo — ajuste o include se LocalBox estiver no seu header
// struct LocalBox {
//     Eigen::Vector3f min_pt, max_pt, center, dimensions;
// };

class VoxelCollisionChecker
{
public:
    void build(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float resolution = 0.003f)
    {
        ready_ = false;
        if (!cloud || cloud->empty()) return;

        res_ = resolution;
        inv_res_ = 1.0f / resolution;

        // Bounding box da nuvem
        Eigen::Vector3f cmin(1e9f, 1e9f, 1e9f), cmax(-1e9f, -1e9f, -1e9f);
        for (const auto& pt : cloud->points) {
            if (pt.x < cmin.x()) cmin.x() = pt.x;
            if (pt.y < cmin.y()) cmin.y() = pt.y;
            if (pt.z < cmin.z()) cmin.z() = pt.z;
            if (pt.x > cmax.x()) cmax.x() = pt.x;
            if (pt.y > cmax.y()) cmax.y() = pt.y;
            if (pt.z > cmax.z()) cmax.z() = pt.z;
        }

        float pad = 0.05f;
        origin_ = cmin.array() - pad;
        Eigen::Vector3f extent = (cmax - cmin).array() + 2.0f * pad;

        nx_ = static_cast<int>(std::ceil(extent.x() * inv_res_)) + 1;
        ny_ = static_cast<int>(std::ceil(extent.y() * inv_res_)) + 1;
        nz_ = static_cast<int>(std::ceil(extent.z() * inv_res_)) + 1;

        grid_.assign(static_cast<size_t>(nx_) * ny_ * nz_, 0);

        for (const auto& pt : cloud->points) {
            int ix = static_cast<int>((pt.x - origin_.x()) * inv_res_);
            int iy = static_cast<int>((pt.y - origin_.y()) * inv_res_);
            int iz = static_cast<int>((pt.z - origin_.z()) * inv_res_);
            if (ix >= 0 && ix < nx_ && iy >= 0 && iy < ny_ && iz >= 0 && iz < nz_)
                grid_[idx(ix, iy, iz)] = 1;
        }
        ready_ = true;
    }

    bool isReady() const { return ready_; }

    // Checa se UMA box local (no frame do TCP) colide com o grid,
    // dada a transformada TCP→world e sua inversa.
    bool boxCollides(const Eigen::Affine3f& tf_tcp_to_world,
                     const Eigen::Affine3f& tf_world_to_tcp,
                     const Eigen::Vector3f& box_min_local,
                     const Eigen::Vector3f& box_max_local,
                     float margin) const
    {
        // 8 cantos da box com margem, no frame local
        const Eigen::Vector3f bmin = box_min_local.array() - margin;
        const Eigen::Vector3f bmax = box_max_local.array() + margin;

        Eigen::Vector3f corners[8];
        int c = 0;
        for (int a = 0; a < 2; a++)
            for (int b = 0; b < 2; b++)
                for (int d = 0; d < 2; d++)
                    corners[c++] = Eigen::Vector3f(
                        a ? bmax.x() : bmin.x(),
                        b ? bmax.y() : bmin.y(),
                        d ? bmax.z() : bmin.z());

        // AABB no world para limitar varredura no grid
        Eigen::Vector3f wmin(1e9f, 1e9f, 1e9f), wmax(-1e9f, -1e9f, -1e9f);
        for (int k = 0; k < 8; k++) {
            Eigen::Vector3f pw = tf_tcp_to_world * corners[k];
            wmin = wmin.cwiseMin(pw);
            wmax = wmax.cwiseMax(pw);
        }

        int ix0 = std::max(0, static_cast<int>((wmin.x() - origin_.x()) * inv_res_));
        int iy0 = std::max(0, static_cast<int>((wmin.y() - origin_.y()) * inv_res_));
        int iz0 = std::max(0, static_cast<int>((wmin.z() - origin_.z()) * inv_res_));
        int ix1 = std::min(nx_ - 1, static_cast<int>((wmax.x() - origin_.x()) * inv_res_));
        int iy1 = std::min(ny_ - 1, static_cast<int>((wmax.y() - origin_.y()) * inv_res_));
        int iz1 = std::min(nz_ - 1, static_cast<int>((wmax.z() - origin_.z()) * inv_res_));

        // Pré-computa colunas da rotação world→tcp para evitar chamar Affine3f * Vector3f no loop interno
        const Eigen::Matrix3f R = tf_world_to_tcp.linear();
        const Eigen::Vector3f t = tf_world_to_tcp.translation();

        for (int ix = ix0; ix <= ix1; ix++) {
            float wx = origin_.x() + ix * res_ + res_ * 0.5f;
            for (int iy = iy0; iy <= iy1; iy++) {
                float wy = origin_.y() + iy * res_ + res_ * 0.5f;
                // Stride contínuo em Z → cache-friendly
                size_t base = idx(ix, iy, iz0);
                for (int iz = iz0; iz <= iz1; iz++, base++) {
                    if (!grid_[base]) continue;

                    float wz = origin_.z() + iz * res_ + res_ * 0.5f;

                    // world→local inline (evita overhead de Eigen Affine3f op)
                    float lx = R(0,0)*wx + R(0,1)*wy + R(0,2)*wz + t.x();
                    if (lx < bmin.x() || lx > bmax.x()) continue;
                    float ly = R(1,0)*wx + R(1,1)*wy + R(1,2)*wz + t.y();
                    if (ly < bmin.y() || ly > bmax.y()) continue;
                    float lz = R(2,0)*wx + R(2,1)*wy + R(2,2)*wz + t.z();
                    if (lz < bmin.z() || lz > bmax.z()) continue;

                    return true; // COLISÃO
                }
            }
        }
        return false;
    }

    // Checa todas as boxes do gripper. Retorna true se QUALQUER uma colidir.
    // Template para aceitar qualquer struct com .min_pt e .max_pt
    template<typename BoxT>
    bool gripperCollides(const Eigen::Affine3f& tf_tcp_to_world,
                         const Eigen::Affine3f& tf_world_to_tcp,
                         const std::vector<BoxT>& boxes,
                         float margin = 0.002f) const
    {
        for (const auto& box : boxes) {
            if (boxCollides(tf_tcp_to_world, tf_world_to_tcp, box.min_pt, box.max_pt, margin))
                return true;
        }
        return false;
    }

private:
    inline size_t idx(int x, int y, int z) const {
        return static_cast<size_t>(x) * static_cast<size_t>(ny_) * nz_ + y * nz_ + z;
    }

    std::vector<uint8_t> grid_;
    Eigen::Vector3f origin_;
    float res_ = 0.003f, inv_res_ = 1.0f / 0.003f;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    bool ready_ = false;
};