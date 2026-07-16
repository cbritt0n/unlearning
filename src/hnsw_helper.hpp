#pragma once

/**
 * hnsw_helper.hpp
 *
 * In-memory HNSW index proxy for Latent Space Erasure & Graph Healing.
 *
 * Concurrency
 * -----------
 * Neighborhood-striped shared/exclusive locks (see neighborhood_locks.hpp):
 *   - Readers (search, get_neighbors) take shared locks with hand-over-hand
 *     coupling along the visited path.
 *   - Healers take exclusive locks only on the local region
 *     {q} ∪ N(q) ∪ N(N(q)) (2-hop), never a global index lock.
 *   - Timed acquisition failures throw LockTimeoutError for Python retry.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "neighborhood_locks.hpp"

namespace hnsw {

using layer_t = std::int32_t;

/** Result of a single node erasure + optional graph-side cleanup. */
struct ErasureResult {
    bool success{false};
    labeltype node_id{-1};
    std::size_t bytes_wiped{0};
    std::string message;
};

/**
 * Metrics returned by heal_graph_structure after repairing HNSW adjacency
 * following a hard delete of node q.
 */
struct HealingMetrics {
    bool success{false};
    int edges_removed{0};
    int edges_added{0};
    double repair_duration_ms{0.0};
};

/** One hit from a concurrent k-NN / greedy search. */
struct SearchHit {
    labeltype node_id{-1};
    double distance{0.0};
};

/**
 * Securely overwrite a raw byte region with zeros.
 *
 * Uses a volatile pointer walk so the compiler cannot elide the stores
 * under the as-if rule (critical for unlearning / residual-data guarantees).
 */
inline void secure_wipe_bytes(void* data, std::size_t nbytes) {
    if (data == nullptr || nbytes == 0) {
        return;
    }
    volatile unsigned char* p = static_cast<volatile unsigned char*>(data);
    for (std::size_t i = 0; i < nbytes; ++i) {
        p[i] = static_cast<unsigned char>(0);
    }
}

/**
 * Write physical 0.0f into a float region (same anti-elision discipline).
 */
inline void secure_wipe_floats(float* data, std::size_t count) {
    if (data == nullptr || count == 0) {
        return;
    }
    volatile float* p = data;
    for (std::size_t i = 0; i < count; ++i) {
        p[i] = 0.0f;
    }
}

/** Euclidean (L2) distance between two dim-vectors. */
inline double l2_distance(const float* a, const float* b, std::size_t dim) {
    double acc = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        const double d =
            static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
    }
    return std::sqrt(acc);
}

/**
 * Proxy over an in-memory HNSW-like index with neighborhood-striped locking.
 */
class HNSWIndexProxy {
public:
    explicit HNSWIndexProxy(std::size_t lock_pool_size = 64,
                            int lock_timeout_ms = 50)
        : locks_(std::make_unique<NeighborhoodLockManager>(
              lock_pool_size,
              std::chrono::milliseconds(
                  lock_timeout_ms > 0 ? lock_timeout_ms : 50))) {}

    ~HNSWIndexProxy() {
        // Exclusive structure barrier so no readers observe free.
        std::unique_lock<std::shared_timed_mutex> structure(structure_mu_);
        release_data();
    }

    HNSWIndexProxy(const HNSWIndexProxy&) = delete;
    HNSWIndexProxy& operator=(const HNSWIndexProxy&) = delete;

    HNSWIndexProxy(HNSWIndexProxy&& other) noexcept {
        std::unique_lock<std::shared_timed_mutex> lk(other.structure_mu_);
        move_from(std::move(other));
    }

    HNSWIndexProxy& operator=(HNSWIndexProxy&& other) noexcept {
        if (this != &other) {
            std::unique_lock<std::shared_timed_mutex> a(structure_mu_,
                                                        std::defer_lock);
            std::unique_lock<std::shared_timed_mutex> b(other.structure_mu_,
                                                        std::defer_lock);
            std::lock(a, b);
            release_data();
            move_from(std::move(other));
        }
        return *this;
    }

    void set_lock_timeout_ms(int timeout_ms) {
        if (timeout_ms <= 0) {
            throw std::invalid_argument("lock timeout must be positive");
        }
        locks_->set_timeout(std::chrono::milliseconds(timeout_ms));
    }

    int lock_timeout_ms() const {
        return static_cast<int>(locks_->timeout().count());
    }

    std::size_t lock_pool_size() const { return locks_->pool_size(); }

    // ------------------------------------------------------------------
    // Loading
    // ------------------------------------------------------------------

    void load_index(const float* source,
                    std::size_t dimensions,
                    std::size_t num_elements) {
        if (num_elements > 0 && source == nullptr) {
            throw std::invalid_argument(
                "load_index: source pointer is null but num_elements > 0");
        }
        if (num_elements > 0 && dimensions == 0) {
            throw std::invalid_argument(
                "load_index: dimensions must be > 0 when num_elements > 0");
        }

        std::unique_lock<std::shared_timed_mutex> structure(structure_mu_);
        release_data();
        adjacency_.clear();

        dim_ = dimensions;
        num_elements_ = num_elements;
        owns_data_ = true;

        const std::size_t total = dim_ * num_elements_;
        if (total == 0) {
            data_ = nullptr;
            return;
        }

        data_ = new float[total];
        std::memcpy(data_, source, total * sizeof(float));
        adjacency_.assign(num_elements_, {});
        entrypoint_ = num_elements_ > 0 ? 0 : -1;
    }

    void attach_index(float* source,
                      std::size_t dimensions,
                      std::size_t num_elements) {
        if (num_elements > 0 && source == nullptr) {
            throw std::invalid_argument(
                "attach_index: source pointer is null but num_elements > 0");
        }
        if (num_elements > 0 && dimensions == 0) {
            throw std::invalid_argument(
                "attach_index: dimensions must be > 0 when num_elements > 0");
        }

        std::unique_lock<std::shared_timed_mutex> structure(structure_mu_);
        release_data();
        adjacency_.clear();

        data_ = source;
        dim_ = dimensions;
        num_elements_ = num_elements;
        owns_data_ = false;
        adjacency_.assign(num_elements_, {});
        entrypoint_ = num_elements_ > 0 ? 0 : -1;
    }

    // ------------------------------------------------------------------
    // Persistence (atomic commit target for index.bin)
    // ------------------------------------------------------------------

    /**
     * Serialize the live HNSW structure to a binary file.
     *
     * Format (little-endian):
     *   magic[4] = "HNSW"
     *   version  u32 = 1
     *   N u64, dim u64, entrypoint i64
     *   vectors: float32[N * dim]
     *   for each node:
     *     n_layers u32
     *     for each layer: degree u32, neighbors i64[degree]
     *
     * Intended write path: save to ``index.bin.tmp`` then ``os.replace``
     * onto ``index.bin`` from Python for crash-safe commits.
     */
    void save_to_file(const std::string& path) const {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        if (num_elements_ == 0 || data_ == nullptr) {
            throw std::logic_error(
                "save_to_file: no index loaded; call load_index first");
        }

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            throw std::runtime_error("save_to_file: cannot open " + path);
        }

        auto write_pod = [&](const auto& value) {
            out.write(reinterpret_cast<const char*>(&value),
                      static_cast<std::streamsize>(sizeof(value)));
            if (!out) {
                throw std::runtime_error("save_to_file: write failed on " + path);
            }
        };

        const char magic[4] = {'H', 'N', 'S', 'W'};
        out.write(magic, 4);
        const std::uint32_t version = 1;
        write_pod(version);
        const std::uint64_t n64 = static_cast<std::uint64_t>(num_elements_);
        const std::uint64_t d64 = static_cast<std::uint64_t>(dim_);
        const std::int64_t ep = entrypoint_;
        write_pod(n64);
        write_pod(d64);
        write_pod(ep);

        const std::size_t total = num_elements_ * dim_;
        out.write(reinterpret_cast<const char*>(data_),
                  static_cast<std::streamsize>(total * sizeof(float)));
        if (!out) {
            throw std::runtime_error(
                "save_to_file: vector payload write failed on " + path);
        }

        for (std::size_t node = 0; node < num_elements_; ++node) {
            const std::uint32_t n_layers =
                node < adjacency_.size()
                    ? static_cast<std::uint32_t>(adjacency_[node].size())
                    : 0u;
            write_pod(n_layers);
            for (std::uint32_t L = 0; L < n_layers; ++L) {
                const auto& nbrs = adjacency_[node][L];
                const std::uint32_t degree =
                    static_cast<std::uint32_t>(nbrs.size());
                write_pod(degree);
                for (labeltype nb : nbrs) {
                    const std::int64_t v = static_cast<std::int64_t>(nb);
                    write_pod(v);
                }
            }
        }

        out.flush();
        if (!out) {
            throw std::runtime_error("save_to_file: flush failed on " + path);
        }
    }

    /**
     * Load a binary index previously written by ``save_to_file``.
     * Replaces any in-memory index under the structure exclusive lock.
     */
    void load_from_file(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("load_from_file: cannot open " + path);
        }

        auto read_pod = [&](auto& value) {
            in.read(reinterpret_cast<char*>(&value),
                    static_cast<std::streamsize>(sizeof(value)));
            if (!in) {
                throw std::runtime_error(
                    "load_from_file: truncated or corrupt file " + path);
            }
        };

        char magic[4] = {};
        in.read(magic, 4);
        if (magic[0] != 'H' || magic[1] != 'N' || magic[2] != 'S'
            || magic[3] != 'W') {
            throw std::runtime_error(
                "load_from_file: bad magic (not an HNSW index blob)");
        }

        std::uint32_t version = 0;
        read_pod(version);
        if (version != 1) {
            throw std::runtime_error(
                "load_from_file: unsupported version "
                + std::to_string(version));
        }

        std::uint64_t n64 = 0;
        std::uint64_t d64 = 0;
        std::int64_t ep = -1;
        read_pod(n64);
        read_pod(d64);
        read_pod(ep);

        if (n64 > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())
            || d64 > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("load_from_file: dimensions too large");
        }

        const std::size_t n = static_cast<std::size_t>(n64);
        const std::size_t d = static_cast<std::size_t>(d64);
        if (n > 0 && d == 0) {
            throw std::runtime_error(
                "load_from_file: dim == 0 with non-zero element count");
        }

        // Overflow guard: n * d * sizeof(float)
        if (d != 0 && n > (std::numeric_limits<std::size_t>::max() / d)) {
            throw std::runtime_error("load_from_file: vector buffer overflow");
        }

        std::unique_ptr<float[]> vectors;
        const std::size_t total = n * d;
        if (total > 0) {
            vectors.reset(new float[total]);
            in.read(reinterpret_cast<char*>(vectors.get()),
                    static_cast<std::streamsize>(total * sizeof(float)));
            if (!in) {
                throw std::runtime_error(
                    "load_from_file: truncated vector payload");
            }
        }

        std::vector<std::vector<std::vector<labeltype>>> adj(n);
        for (std::size_t node = 0; node < n; ++node) {
            std::uint32_t n_layers = 0;
            read_pod(n_layers);
            adj[node].resize(n_layers);
            for (std::uint32_t L = 0; L < n_layers; ++L) {
                std::uint32_t degree = 0;
                read_pod(degree);
                adj[node][L].resize(degree);
                for (std::uint32_t i = 0; i < degree; ++i) {
                    std::int64_t nb = 0;
                    read_pod(nb);
                    if (nb < 0 || static_cast<std::uint64_t>(nb) >= n64) {
                        throw std::runtime_error(
                            "load_from_file: neighbor out of range");
                    }
                    adj[node][L][i] = static_cast<labeltype>(nb);
                }
            }
        }

        std::unique_lock<std::shared_timed_mutex> structure(structure_mu_);
        release_data();
        adjacency_.clear();

        dim_ = d;
        num_elements_ = n;
        owns_data_ = true;
        if (total > 0) {
            data_ = vectors.release();
        } else {
            data_ = nullptr;
        }
        adjacency_ = std::move(adj);
        entrypoint_ = ep;
        if (entrypoint_ >= 0
            && static_cast<std::size_t>(entrypoint_) >= num_elements_) {
            entrypoint_ = num_elements_ > 0 ? 0 : -1;
        }
    }

    void load_adjacency(
        std::vector<std::vector<std::vector<labeltype>>> adjacency) {
        std::unique_lock<std::shared_timed_mutex> structure(structure_mu_);
        if (num_elements_ == 0) {
            throw std::logic_error(
                "load_adjacency: no index loaded; call load_index first");
        }
        if (adjacency.size() != num_elements_) {
            throw std::invalid_argument(
                "load_adjacency: outer length must equal num_elements ("
                + std::to_string(num_elements_) + "), got "
                + std::to_string(adjacency.size()));
        }

        for (std::size_t node = 0; node < adjacency.size(); ++node) {
            for (std::size_t layer = 0; layer < adjacency[node].size();
                 ++layer) {
                for (labeltype nb : adjacency[node][layer]) {
                    if (nb < 0
                        || static_cast<std::size_t>(nb) >= num_elements_) {
                        throw std::out_of_range(
                            "load_adjacency: neighbor label "
                            + std::to_string(nb) + " out of range [0, "
                            + std::to_string(num_elements_) + ")");
                    }
                }
            }
        }

        adjacency_ = std::move(adjacency);
    }

    void set_neighbors(labeltype node_id,
                       layer_t layer,
                       std::vector<labeltype> neighbors) {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        ensure_node(node_id);
        if (layer < 0) {
            throw std::out_of_range(
                "set_neighbors: layer must be non-negative, got "
                + std::to_string(layer));
        }

        // Exclusive on the node itself; neighbor labels are only validated.
        auto excl = locks_->acquire_exclusive_one(node_id);

        const std::size_t n = static_cast<std::size_t>(node_id);
        const std::size_t L = static_cast<std::size_t>(layer);
        ensure_layer_slot(n, L);

        for (labeltype nb : neighbors) {
            if (nb < 0 || static_cast<std::size_t>(nb) >= num_elements_) {
                throw std::out_of_range(
                    "set_neighbors: neighbor label " + std::to_string(nb)
                    + " out of range [0, " + std::to_string(num_elements_)
                    + ")");
            }
        }

        adjacency_[n][L] = std::move(neighbors);
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    bool is_loaded() const noexcept {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        return num_elements_ > 0 && data_ != nullptr;
    }

    std::size_t dimensions() const noexcept {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        return dim_;
    }

    std::size_t num_elements() const noexcept {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        return num_elements_;
    }

    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }

    float* vector_ptr(labeltype node_id) {
        ensure_node(node_id);
        return data_ + static_cast<std::size_t>(node_id) * dim_;
    }

    const float* vector_ptr(labeltype node_id) const {
        ensure_node(node_id);
        return data_ + static_cast<std::size_t>(node_id) * dim_;
    }

    std::vector<float> get_vector(labeltype node_id) const {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        ensure_node(node_id);
        auto shared = locks_->acquire_shared(node_id);
        const float* src =
            data_ + static_cast<std::size_t>(node_id) * dim_;
        return std::vector<float>(src, src + dim_);
    }

    std::vector<labeltype> get_neighbors(labeltype node_id,
                                         layer_t layer) const {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        ensure_node(node_id);
        if (layer < 0) {
            throw std::out_of_range(
                "get_neighbors: layer must be non-negative, got "
                + std::to_string(layer));
        }

        auto shared = locks_->acquire_shared(node_id);

        const std::size_t n = static_cast<std::size_t>(node_id);
        const std::size_t L = static_cast<std::size_t>(layer);

        if (n >= adjacency_.size() || L >= adjacency_[n].size()) {
            throw std::out_of_range(
                "get_neighbors: layer " + std::to_string(layer)
                + " is out of range for node_id " + std::to_string(node_id)
                + " (node has "
                + std::to_string(n < adjacency_.size() ? adjacency_[n].size()
                                                       : 0)
                + " layer(s))");
        }

        return adjacency_[n][L];
    }

    std::size_t num_layers(labeltype node_id) const {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        ensure_node(node_id);
        auto shared = locks_->acquire_shared(node_id);
        const std::size_t n = static_cast<std::size_t>(node_id);
        if (n >= adjacency_.size()) {
            return 0;
        }
        return adjacency_[n].size();
    }

    // ------------------------------------------------------------------
    // Concurrent search (shared locks + lock coupling)
    // ------------------------------------------------------------------

    /**
     * Greedy descent / base-layer search with hand-over-hand shared locks.
     *
     * @param query         Query vector of length dimensions()
     * @param k             Number of nearest neighbors to return
     * @param entry_node    Optional entrypoint (-1 → index entrypoint_)
     * @return Sorted hits (ascending distance). May throw LockTimeoutError.
     */
    std::vector<SearchHit> search_knn(const float* query,
                                      std::size_t k,
                                      labeltype entry_node = -1) const {
        if (query == nullptr) {
            throw std::invalid_argument("search_knn: query is null");
        }
        if (k == 0) {
            return {};
        }

        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        if (num_elements_ == 0 || data_ == nullptr) {
            throw std::logic_error(
                "search_knn: no index loaded; call load_index first");
        }

        labeltype curr = entry_node;
        if (curr < 0) {
            curr = entrypoint_;
        }
        if (curr < 0 || static_cast<std::size_t>(curr) >= num_elements_) {
            curr = 0;
        }

        // Highest layer present on entry; greedy descend.
        std::size_t top_layer = 0;
        {
            auto shared = locks_->acquire_shared(curr);
            const std::size_t c = static_cast<std::size_t>(curr);
            if (c < adjacency_.size() && !adjacency_[c].empty()) {
                top_layer = adjacency_[c].size() - 1;
            }
        }

        NeighborhoodLockManager::SharedLockCursor cursor(locks_.get());
        cursor.enter(curr);

        for (std::size_t lc = top_layer; lc > 0; --lc) {
            curr = greedy_update_locked(query, curr, lc, cursor);
        }

        // Base layer: small ef-style exploration under lock coupling.
        return base_layer_search_locked(query, curr, k, cursor);
    }

    // ------------------------------------------------------------------
    // Mutation / erasure
    // ------------------------------------------------------------------

    void overwrite_vector(labeltype node_id) {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        ensure_node(node_id);
        auto excl = locks_->acquire_exclusive_one(node_id);
        float* dest = data_ + static_cast<std::size_t>(node_id) * dim_;
        secure_wipe_floats(dest, dim_);
    }

    void prune_adjacency(labeltype node_id) {
        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);
        ensure_node(node_id);

        // Conservative: exclusive on full 2-hop around node_id.
        auto region = collect_heal_region_unlocked(node_id);
        auto excl = locks_->acquire_exclusive(region);

        const labeltype target = node_id;
        for (auto& per_node : adjacency_) {
            for (auto& layer_list : per_node) {
                layer_list.erase(
                    std::remove(layer_list.begin(), layer_list.end(), target),
                    layer_list.end());
            }
        }
        const std::size_t n = static_cast<std::size_t>(node_id);
        if (n < adjacency_.size()) {
            for (auto& layer_list : adjacency_[n]) {
                layer_list.clear();
            }
        }
    }

    /**
     * Core MN-RU graph healing under exclusive locks on
     * {q} ∪ N(q) ∪ N(N(q)) only — no global graph lock.
     */
    HealingMetrics heal_graph_structure(std::size_t node_id, int max_m) {
        using clock = std::chrono::steady_clock;
        const auto t0 = clock::now();

        HealingMetrics metrics;

        if (max_m <= 0) {
            throw std::invalid_argument(
                "heal_graph_structure: max_m must be positive, got "
                + std::to_string(max_m));
        }

        std::shared_lock<std::shared_timed_mutex> structure(structure_mu_);

        if (num_elements_ == 0 || data_ == nullptr) {
            throw std::logic_error(
                "heal_graph_structure: no index loaded; call load_index first");
        }
        if (node_id >= num_elements_) {
            throw std::out_of_range(
                "heal_graph_structure: node_id " + std::to_string(node_id)
                + " out of range [0, " + std::to_string(num_elements_) + ")");
        }

        const labeltype q = static_cast<labeltype>(node_id);

        // --- Localized exclusive region (2-hop) -------------------------
        // Snapshot region under a brief exclusive on q + provisional 1-hop,
        // then expand and re-acquire (may retry once if topology changes).
        std::vector<labeltype> region;
        NeighborhoodLockManager::ExclusiveGuard excl;
        {
            // First pass: lock q alone to read N(q).
            auto q_only = locks_->acquire_exclusive_one(q);
            region = collect_heal_region_unlocked(q);
            q_only.release();

            excl = locks_->acquire_exclusive(region);

            // Recompute under full exclusive; expand if 2-hop grew.
            auto region2 = collect_heal_region_unlocked(q);
            if (region2.size() != region.size()
                || !std::is_permutation(
                    region.begin(), region.end(), region2.begin())) {
                excl.release();
                region = std::move(region2);
                excl = locks_->acquire_exclusive(region);
            }
        }

        if (adjacency_.size() < num_elements_) {
            adjacency_.resize(num_elements_);
        }

        std::size_t max_layer_global = adjacency_[node_id].size();
        for (labeltype n : region) {
            const std::size_t ni = static_cast<std::size_t>(n);
            if (ni < adjacency_.size()
                && adjacency_[ni].size() > max_layer_global) {
                max_layer_global = adjacency_[ni].size();
            }
        }

        for (std::size_t L = 0; L < max_layer_global; ++L) {
            // ----------------------------------------------------------
            // 1. Neighborhood isolation (locked region only)
            // ----------------------------------------------------------
            std::vector<labeltype> orphans;
            if (L < adjacency_[node_id].size()) {
                orphans = adjacency_[node_id][L];
            }

            {
                std::vector<labeltype> uniq;
                uniq.reserve(orphans.size());
                for (labeltype u : orphans) {
                    if (u == q) {
                        continue;
                    }
                    if (u < 0
                        || static_cast<std::size_t>(u) >= num_elements_) {
                        continue;
                    }
                    if (std::find(uniq.begin(), uniq.end(), u) == uniq.end()) {
                        uniq.push_back(u);
                    }
                }
                orphans.swap(uniq);
            }

            for (labeltype u : orphans) {
                const std::size_t ui = static_cast<std::size_t>(u);
                ensure_layer_slot(ui, L);
                metrics.edges_removed +=
                    remove_label(adjacency_[ui][L], q);
            }

            if (L < adjacency_[node_id].size()) {
                metrics.edges_removed +=
                    static_cast<int>(adjacency_[node_id][L].size());
                adjacency_[node_id][L].clear();
            }

            // Residual inbound edges only from locked region (localized).
            for (labeltype vlab : region) {
                const std::size_t v = static_cast<std::size_t>(vlab);
                if (v == node_id) {
                    continue;
                }
                if (L < adjacency_[v].size()) {
                    metrics.edges_removed +=
                        remove_label(adjacency_[v][L], q);
                }
            }

            if (orphans.size() < 2) {
                continue;
            }

            // ----------------------------------------------------------
            // 2. MN-RU healing among orphans
            // ----------------------------------------------------------
            struct Candidate {
                labeltype u;
                labeltype w;
                double dist;
            };
            std::vector<Candidate> candidates;
            candidates.reserve(orphans.size() * (orphans.size() - 1) / 2);

            for (std::size_t i = 0; i < orphans.size(); ++i) {
                for (std::size_t j = i + 1; j < orphans.size(); ++j) {
                    const labeltype u = orphans[i];
                    const labeltype w = orphans[j];
                    candidates.push_back(Candidate{u, w, distance(u, w)});
                }
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate& a, const Candidate& b) {
                          if (a.dist != b.dist) {
                              return a.dist < b.dist;
                          }
                          if (a.u != b.u) {
                              return a.u < b.u;
                          }
                          return a.w < b.w;
                      });

            for (const Candidate& c : candidates) {
                try_insert_healed_edge(c.u, c.w, L, max_m, metrics);
            }
        }

        for (auto& layer_list : adjacency_[node_id]) {
            if (!layer_list.empty()) {
                metrics.edges_removed +=
                    static_cast<int>(layer_list.size());
                layer_list.clear();
            }
        }

        // Retire entrypoint if it was deleted.
        if (entrypoint_ == q) {
            entrypoint_ = -1;
            for (std::size_t i = 0; i < num_elements_; ++i) {
                if (i == node_id) {
                    continue;
                }
                entrypoint_ = static_cast<labeltype>(i);
                break;
            }
        }

        const auto t1 = clock::now();
        metrics.repair_duration_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        metrics.success = true;
        return metrics;
    }

    ErasureResult erase_node(labeltype node_id, int max_m = 16) {
        // overwrite_vector takes its own exclusive stripe lock and releases.
        overwrite_vector(node_id);
        const HealingMetrics hm =
            heal_graph_structure(static_cast<std::size_t>(node_id), max_m);

        ErasureResult result;
        result.success = hm.success;
        result.node_id = node_id;
        result.bytes_wiped = dim_ * sizeof(float);
        result.message =
            "vector zeroed; graph healed (removed="
            + std::to_string(hm.edges_removed) + ", added="
            + std::to_string(hm.edges_added) + ", "
            + std::to_string(hm.repair_duration_ms) + " ms)";
        return result;
    }

private:
    float* data_{nullptr};
    bool owns_data_{false};
    std::size_t dim_{0};
    std::size_t num_elements_{0};
    labeltype entrypoint_{-1};

    // adjacency_[node_id][layer] -> neighbor labels
    std::vector<std::vector<std::vector<labeltype>>> adjacency_;

    // Structure lock: load/resize vs concurrent readers of metadata.
    mutable std::shared_timed_mutex structure_mu_;
    std::unique_ptr<NeighborhoodLockManager> locks_;

    void release_data() noexcept {
        if (owns_data_ && data_ != nullptr) {
            const std::size_t total = dim_ * num_elements_;
            secure_wipe_floats(data_, total);
            delete[] data_;
        }
        data_ = nullptr;
        owns_data_ = false;
        dim_ = 0;
        num_elements_ = 0;
        entrypoint_ = -1;
    }

    void move_from(HNSWIndexProxy&& other) noexcept {
        data_ = other.data_;
        owns_data_ = other.owns_data_;
        dim_ = other.dim_;
        num_elements_ = other.num_elements_;
        entrypoint_ = other.entrypoint_;
        adjacency_ = std::move(other.adjacency_);
        locks_ = std::move(other.locks_);
        if (!locks_) {
            locks_ = std::make_unique<NeighborhoodLockManager>();
        }

        other.data_ = nullptr;
        other.owns_data_ = false;
        other.dim_ = 0;
        other.num_elements_ = 0;
        other.entrypoint_ = -1;
        other.adjacency_.clear();
        if (!other.locks_) {
            other.locks_ = std::make_unique<NeighborhoodLockManager>();
        }
    }

    void ensure_node(labeltype node_id) const {
        if (num_elements_ == 0 || data_ == nullptr) {
            throw std::logic_error(
                "HNSWIndexProxy: no index loaded; call load_index first");
        }
        if (node_id < 0
            || static_cast<std::size_t>(node_id) >= num_elements_) {
            throw std::out_of_range(
                "node_id " + std::to_string(node_id)
                + " out of range [0, " + std::to_string(num_elements_) + ")");
        }
    }

    void ensure_layer_slot(std::size_t node, std::size_t layer) {
        if (adjacency_.size() < num_elements_) {
            adjacency_.resize(num_elements_);
        }
        if (adjacency_[node].size() <= layer) {
            adjacency_[node].resize(layer + 1);
        }
    }

    static int remove_label(std::vector<labeltype>& list, labeltype target) {
        const auto before = list.size();
        list.erase(std::remove(list.begin(), list.end(), target), list.end());
        return static_cast<int>(before - list.size());
    }

    static bool contains_label(const std::vector<labeltype>& list,
                               labeltype target) {
        return std::find(list.begin(), list.end(), target) != list.end();
    }

    double distance(labeltype a, labeltype b) const {
        const float* pa = data_ + static_cast<std::size_t>(a) * dim_;
        const float* pb = data_ + static_cast<std::size_t>(b) * dim_;
        return l2_distance(pa, pb, dim_);
    }

    double distance_query(const float* query, labeltype node) const {
        const float* pv = data_ + static_cast<std::size_t>(node) * dim_;
        return l2_distance(query, pv, dim_);
    }

    /**
     * Build exclusive-lock region: {q} ∪ 1-hop ∪ 2-hop (all layers).
     * Caller must already hold locks sufficient to read adjacency safely,
     * or accept a best-effort snapshot (used under q-only then revalidated).
     */
    std::vector<labeltype> collect_heal_region_unlocked(labeltype q) const {
        std::vector<labeltype> region;
        region.push_back(q);

        auto add_unique = [&](labeltype n) {
            if (n < 0 || static_cast<std::size_t>(n) >= num_elements_) {
                return;
            }
            if (std::find(region.begin(), region.end(), n) == region.end()) {
                region.push_back(n);
            }
        };

        const std::size_t qi = static_cast<std::size_t>(q);
        if (qi < adjacency_.size()) {
            for (const auto& layer_list : adjacency_[qi]) {
                for (labeltype nb : layer_list) {
                    add_unique(nb);
                }
            }
        }

        // 2-hop: neighbors of every 1-hop member (snapshot size).
        const std::size_t one_hop_end = region.size();
        for (std::size_t i = 1; i < one_hop_end; ++i) {
            const std::size_t ui = static_cast<std::size_t>(region[i]);
            if (ui >= adjacency_.size()) {
                continue;
            }
            for (const auto& layer_list : adjacency_[ui]) {
                for (labeltype nb : layer_list) {
                    add_unique(nb);
                }
            }
        }

        return region;
    }

    double navigability_score(labeltype node, std::size_t layer) const {
        const std::size_t n = static_cast<std::size_t>(node);
        if (n >= adjacency_.size() || layer >= adjacency_[n].size()) {
            return 0.0;
        }
        double score = 0.0;
        for (labeltype nb : adjacency_[n][layer]) {
            if (nb < 0 || static_cast<std::size_t>(nb) >= num_elements_) {
                continue;
            }
            score += 1.0 / (1.0 + distance(node, nb));
        }
        return score;
    }

    bool ensure_capacity_for(labeltype node,
                             labeltype cand,
                             std::size_t layer,
                             int max_m,
                             HealingMetrics& metrics) {
        const std::size_t n = static_cast<std::size_t>(node);
        ensure_layer_slot(n, layer);
        auto& nbrs = adjacency_[n][layer];

        if (contains_label(nbrs, cand)) {
            return true;
        }
        if (static_cast<int>(nbrs.size()) < max_m) {
            return true;
        }

        std::size_t far_idx = 0;
        double far_dist = -1.0;
        for (std::size_t i = 0; i < nbrs.size(); ++i) {
            const double d = distance(node, nbrs[i]);
            if (d > far_dist) {
                far_dist = d;
                far_idx = i;
            }
        }

        const double dist_cand = distance(node, cand);
        if (!(dist_cand < far_dist)) {
            return false;
        }

        const labeltype far = nbrs[far_idx];
        const double score_before = navigability_score(node, layer);
        const double score_after =
            score_before
            - (1.0 / (1.0 + far_dist))
            + (1.0 / (1.0 + dist_cand));

        if (!(score_after > score_before)) {
            return false;
        }

        nbrs.erase(nbrs.begin() + static_cast<std::ptrdiff_t>(far_idx));
        metrics.edges_removed += 1;

        const std::size_t far_i = static_cast<std::size_t>(far);
        // Reverse cleanup only if far is still a valid index; heal region
        // locking (2-hop) guarantees far is exclusively held when it was
        // a neighbor of a locked orphan.
        if (far_i < num_elements_) {
            ensure_layer_slot(far_i, layer);
            metrics.edges_removed +=
                remove_label(adjacency_[far_i][layer], node);
        }

        return true;
    }

    bool commit_half_edge(labeltype node,
                          labeltype cand,
                          std::size_t layer,
                          int max_m,
                          HealingMetrics& metrics) {
        const std::size_t n = static_cast<std::size_t>(node);
        ensure_layer_slot(n, layer);
        auto& nbrs = adjacency_[n][layer];

        if (contains_label(nbrs, cand)) {
            return true;
        }
        if (!ensure_capacity_for(node, cand, layer, max_m, metrics)) {
            return false;
        }
        if (contains_label(nbrs, cand)) {
            return true;
        }
        if (static_cast<int>(nbrs.size()) >= max_m) {
            return false;
        }
        nbrs.push_back(cand);
        metrics.edges_added += 1;
        return true;
    }

    void try_insert_healed_edge(labeltype u,
                                labeltype w,
                                std::size_t layer,
                                int max_m,
                                HealingMetrics& metrics) {
        if (u == w) {
            return;
        }
        const std::size_t ui = static_cast<std::size_t>(u);
        const std::size_t wi = static_cast<std::size_t>(w);
        if (ui >= num_elements_ || wi >= num_elements_) {
            return;
        }

        ensure_layer_slot(ui, layer);
        ensure_layer_slot(wi, layer);

        const bool u_has = contains_label(adjacency_[ui][layer], w);
        const bool w_has = contains_label(adjacency_[wi][layer], u);

        if (u_has && w_has) {
            return;
        }

        if (!u_has && !w_has) {
            if (!ensure_capacity_for(u, w, layer, max_m, metrics)) {
                return;
            }
            if (!ensure_capacity_for(w, u, layer, max_m, metrics)) {
                return;
            }
        }

        if (!u_has) {
            commit_half_edge(u, w, layer, max_m, metrics);
        }
        if (!w_has) {
            commit_half_edge(w, u, layer, max_m, metrics);
        }
    }

    // ---- Search helpers (caller holds structure shared lock) ------------

    /**
     * Greedy move on layer ``layer`` with lock coupling via ``cursor``.
     * Cursor must already hold a shared lock on ``curr``.
     */
    labeltype greedy_update_locked(
        const float* query,
        labeltype curr,
        std::size_t layer,
        NeighborhoodLockManager::SharedLockCursor& cursor) const {
        bool improved = true;
        while (improved) {
            improved = false;
            double best_dist = distance_query(query, curr);

            // Snapshot neighbors under current shared lock.
            std::vector<labeltype> nbrs;
            {
                const std::size_t c = static_cast<std::size_t>(curr);
                if (c < adjacency_.size() && layer < adjacency_[c].size()) {
                    nbrs = adjacency_[c][layer];
                }
            }

            for (labeltype nb : nbrs) {
                if (nb < 0 || static_cast<std::size_t>(nb) >= num_elements_) {
                    continue;
                }
                // Hand-over-hand: lock neighbor before reading its vector.
                cursor.advance(nb);
                const double d = distance_query(query, nb);
                if (d < best_dist) {
                    best_dist = d;
                    curr = nb;
                    improved = true;
                    // Stay on nb (cursor already there) and restart.
                    break;
                }
                // Not better: couple back to curr for the next candidate.
                cursor.advance(curr);
            }
        }
        return curr;
    }

    std::vector<SearchHit> base_layer_search_locked(
        const float* query,
        labeltype entry,
        std::size_t k,
        NeighborhoodLockManager::SharedLockCursor& cursor) const {
        // Max-heap of size k by distance (worst of the best on top).
        auto cmp = [](const SearchHit& a, const SearchHit& b) {
            return a.distance < b.distance;  // max-heap
        };
        std::priority_queue<SearchHit, std::vector<SearchHit>, decltype(cmp)>
            best(cmp);

        std::vector<char> visited(num_elements_, 0);
        std::queue<labeltype> frontier;

        cursor.advance(entry);
        visited[static_cast<std::size_t>(entry)] = 1;
        frontier.push(entry);
        {
            SearchHit h;
            h.node_id = entry;
            h.distance = distance_query(query, entry);
            best.push(h);
        }

        const std::size_t ef = std::max<std::size_t>(k, 16);

        while (!frontier.empty()) {
            const labeltype u = frontier.front();
            frontier.pop();
            cursor.advance(u);

            std::vector<labeltype> nbrs;
            {
                const std::size_t ui = static_cast<std::size_t>(u);
                if (ui < adjacency_.size() && !adjacency_[ui].empty()) {
                    // Layer 0
                    if (!adjacency_[ui][0].empty() || adjacency_[ui].size() > 0) {
                        nbrs = adjacency_[ui][0];
                    }
                }
            }

            for (labeltype nb : nbrs) {
                if (nb < 0 || static_cast<std::size_t>(nb) >= num_elements_) {
                    continue;
                }
                const std::size_t nbi = static_cast<std::size_t>(nb);
                if (visited[nbi]) {
                    continue;
                }
                visited[nbi] = 1;

                cursor.advance(nb);
                const double d = distance_query(query, nb);

                if (best.size() < ef || d < best.top().distance) {
                    SearchHit h;
                    h.node_id = nb;
                    h.distance = d;
                    best.push(h);
                    if (best.size() > ef) {
                        best.pop();
                    }
                    frontier.push(nb);
                }
                cursor.advance(u);
            }
        }

        std::vector<SearchHit> out;
        out.reserve(best.size());
        while (!best.empty()) {
            out.push_back(best.top());
            best.pop();
        }
        std::sort(out.begin(), out.end(),
                  [](const SearchHit& a, const SearchHit& b) {
                      return a.distance < b.distance;
                  });
        if (out.size() > k) {
            out.resize(k);
        }
        return out;
    }
};

}  // namespace hnsw
