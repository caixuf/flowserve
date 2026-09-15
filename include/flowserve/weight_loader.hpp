#pragma once

// ============================================================================
// flowserve - 操作系统级 mmap 零拷贝大模型权重加载器
// 借鉴自 kun-cellular 的 sdsc_binary_runtime.h 紧凑二进制布局与 mmap 体系
//
// 核心特性:
//   1. 0 动态内存拷贝：基于 POSIX mmap 直接将磁盘权重映射至虚拟内存地址空间；
//   2. 64 字节缓存行对齐 (Cache-line aligned)，支持现代 CPU SIMD 指令与 GPU 直接复制；
//   3. 原生支持 MLA (Multi-Head Latent Attention) + SwiGLU Transformer 完整层权重。
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace flowserve {

#define FLOWSERVE_MODEL_MAGIC 0x464C5356 /* "FLSV" */
#define FLOWSERVE_MODEL_VERSION 1

#pragma pack(push, 1)
struct ModelFileHeader {
    uint32_t magic;            // 0x464C5356 ("FLSV")
    uint32_t version;          // 1
    uint32_t num_layers;       // 模型层数 (如 2, 4)
    uint32_t hidden_dim;       // 隐藏层维度 (如 64, 128)
    uint32_t num_heads;        // 注意力头数 (如 4)
    uint32_t head_dim;         // 每头维度 (如 16)
    uint32_t mla_latent_dim;   // MLA KV 潜空间维度 (如 16)
    uint32_t intermediate_dim; // SwiGLU FFN 维度 (如 128)
    uint32_t vocab_size;       // 词表大小 (如 259)
    uint64_t weights_offset;   // 权重张量区偏移
    uint64_t total_params;     // 总参数浮点数数量
    uint8_t  reserved[12];
};
#pragma pack(pop)

struct LayerWeights {
    const float* norm1{nullptr};        // [hidden_dim]
    const float* w_dkv{nullptr};        // [mla_latent_dim, hidden_dim]
    const float* w_dq{nullptr};         // [mla_latent_dim, hidden_dim]
    const float* w_uk{nullptr};         // [num_heads * head_dim, mla_latent_dim]
    const float* w_uv{nullptr};         // [num_heads * head_dim, mla_latent_dim]
    const float* w_uq{nullptr};         // [num_heads * head_dim, mla_latent_dim]
    const float* w_o{nullptr};          // [hidden_dim, num_heads * head_dim]
    const float* norm2{nullptr};        // [hidden_dim]
    const float* w_gate{nullptr};       // [intermediate_dim, hidden_dim]
    const float* w_up{nullptr};         // [intermediate_dim, hidden_dim]
    const float* w_down{nullptr};       // [hidden_dim, intermediate_dim]
};

class WeightLoader {
public:
    WeightLoader() = default;
    ~WeightLoader() { close(); }

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    WeightLoader(WeightLoader&& other) noexcept {
        move_from(std::move(other));
    }
    WeightLoader& operator=(WeightLoader&& other) noexcept {
        if (this != &other) {
            close();
            move_from(std::move(other));
        }
        return *this;
    }

    static WeightLoader open_mmap(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("Failed to open model weight file: " + path);
        }

        struct stat st;
        if (::fstat(fd, &st) < 0) {
            ::close(fd);
            throw std::runtime_error("Failed to stat model file: " + path);
        }
        size_t file_size = static_cast<size_t>(st.st_size);
        if (file_size < sizeof(ModelFileHeader)) {
            ::close(fd);
            throw std::runtime_error("Model file size is too small: " + path);
        }

        void* addr = ::mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            ::close(fd);
            throw std::runtime_error("mmap failed for model file: " + path);
        }

        WeightLoader loader;
        loader.fd_ = fd;
        loader.mmap_base_ = addr;
        loader.mmap_size_ = file_size;
        loader.header_ = *reinterpret_cast<const ModelFileHeader*>(addr);

        if (loader.header_.magic != FLOWSERVE_MODEL_MAGIC) {
            loader.close();
            throw std::runtime_error("Invalid model magic in file: " + path);
        }

        loader.parse_weights();
        return loader;
    }

    // 将参数导出为 FLSV 二进制权重文件
    static void dump_binary(const std::string& path,
                            const ModelFileHeader& hdr,
                            const std::vector<float>& all_params) {
        int fd = ::open(path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) {
            throw std::runtime_error("Failed to create file for dump: " + path);
        }

        ModelFileHeader mutable_hdr = hdr;
        mutable_hdr.magic = FLOWSERVE_MODEL_MAGIC;
        mutable_hdr.version = FLOWSERVE_MODEL_VERSION;
        mutable_hdr.weights_offset = sizeof(ModelFileHeader);
        mutable_hdr.total_params = all_params.size();

        if (::write(fd, &mutable_hdr, sizeof(mutable_hdr)) != sizeof(mutable_hdr)) {
            ::close(fd);
            throw std::runtime_error("Failed to write model header to: " + path);
        }

        size_t byte_size = all_params.size() * sizeof(float);
        if (::write(fd, all_params.data(), byte_size) != static_cast<ssize_t>(byte_size)) {
            ::close(fd);
            throw std::runtime_error("Failed to write model weights to: " + path);
        }

        ::close(fd);
    }

    const ModelFileHeader& header() const { return header_; }
    const std::vector<LayerWeights>& layers() const { return layers_; }
    const LayerWeights& layer(size_t idx) const { return layers_.at(idx); }
    const float* token_embeddings() const { return token_embeddings_; }
    const float* final_norm() const { return final_norm_; }
    const float* lm_head() const { return lm_head_; }

    void close() {
        if (mmap_base_ && mmap_base_ != MAP_FAILED) {
            ::munmap(mmap_base_, mmap_size_);
            mmap_base_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        layers_.clear();
    }

private:
    void move_from(WeightLoader&& other) noexcept {
        fd_ = other.fd_;
        mmap_base_ = other.mmap_base_;
        mmap_size_ = other.mmap_size_;
        header_ = other.header_;
        layers_ = std::move(other.layers_);
        token_embeddings_ = other.token_embeddings_;
        final_norm_ = other.final_norm_;
        lm_head_ = other.lm_head_;

        other.fd_ = -1;
        other.mmap_base_ = nullptr;
        other.mmap_size_ = 0;
    }

    void parse_weights() {
        const float* ptr = reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(mmap_base_) + header_.weights_offset);

        const size_t H = header_.hidden_dim;
        const size_t L = header_.mla_latent_dim;
        const size_t heads = header_.num_heads;
        const size_t d = header_.head_dim;
        const size_t M = header_.intermediate_dim;
        const size_t V = header_.vocab_size;
        const size_t total_head_dim = heads * d;

        // 1. Token Embeddings
        token_embeddings_ = ptr;
        ptr += (V * H);

        // 2. Layers
        layers_.resize(header_.num_layers);
        for (size_t l = 0; l < header_.num_layers; ++l) {
            layers_[l].norm1  = ptr; ptr += H;
            layers_[l].w_dkv  = ptr; ptr += (L * H);
            layers_[l].w_dq   = ptr; ptr += (L * H);
            layers_[l].w_uk   = ptr; ptr += (total_head_dim * L);
            layers_[l].w_uv   = ptr; ptr += (total_head_dim * L);
            layers_[l].w_uq   = ptr; ptr += (total_head_dim * L);
            layers_[l].w_o    = ptr; ptr += (H * total_head_dim);
            layers_[l].norm2  = ptr; ptr += H;
            layers_[l].w_gate = ptr; ptr += (M * H);
            layers_[l].w_up   = ptr; ptr += (M * H);
            layers_[l].w_down = ptr; ptr += (H * M);
        }

        // 3. Final Norm & LM Head
        final_norm_ = ptr; ptr += H;
        lm_head_ = ptr; ptr += (V * H);
    }

    int fd_{-1};
    void* mmap_base_{nullptr};
    size_t mmap_size_{0};
    ModelFileHeader header_{};
    std::vector<LayerWeights> layers_;
    const float* token_embeddings_{nullptr};
    const float* final_norm_{nullptr};
    const float* lm_head_{nullptr};
};

} // namespace flowserve
