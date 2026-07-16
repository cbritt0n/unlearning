#pragma once

/**
 * neighborhood_locks.hpp
 *
 * Striped reader/writer locks for localized HNSW neighborhoods.
 *
 * Uses std::shared_timed_mutex (same shared/unique model as shared_mutex,
 * plus try_lock_*_for) so heal/search can bound wait time and surface
 * LockTimeoutError to Python for safe retry.
 *
 * Stripe mapping:  node_id  ->  stripe = node_id % pool_size
 * Multi-node acquires always lock stripes in ascending order to avoid
 * deadlock between concurrent heal regions.
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hnsw {

using labeltype = std::int64_t;

/** Thrown when a timed lock acquisition fails (Python: LockContentionError). */
class LockTimeoutError : public std::runtime_error {
public:
    explicit LockTimeoutError(const std::string& what)
        : std::runtime_error(what) {}
};

/**
 * Pool of shared_timed_mutex stripes covering graph neighborhoods.
 */
class NeighborhoodLockManager {
public:
    using clock = std::chrono::steady_clock;
    using duration = std::chrono::milliseconds;

    explicit NeighborhoodLockManager(std::size_t pool_size = 64,
                                     duration timeout = duration(50))
        : stripes_(pool_size > 0 ? pool_size : 1)
        , timeout_(timeout) {}

    NeighborhoodLockManager(const NeighborhoodLockManager&) = delete;
    NeighborhoodLockManager& operator=(const NeighborhoodLockManager&) = delete;

    std::size_t pool_size() const noexcept { return stripes_.size(); }

    duration timeout() const noexcept { return timeout_; }

    void set_timeout(duration timeout) noexcept { timeout_ = timeout; }

    std::size_t stripe_of(labeltype node_id) const noexcept {
        if (node_id < 0) {
            node_id = -node_id;
        }
        return static_cast<std::size_t>(node_id) % stripes_.size();
    }

    // ------------------------------------------------------------------
    // Unique (exclusive) multi-stripe guard — heal / mutate
    // ------------------------------------------------------------------

    class ExclusiveGuard {
    public:
        ExclusiveGuard() = default;

        ExclusiveGuard(NeighborhoodLockManager* mgr,
                       std::vector<std::size_t> stripes)
            : mgr_(mgr), stripes_(std::move(stripes)) {
            if (mgr_ == nullptr || stripes_.empty()) {
                return;
            }
            // stripes_ already sorted unique.
            locked_.reserve(stripes_.size());
            for (std::size_t s : stripes_) {
                auto& mu = mgr_->stripes_[s];
                if (!mu.try_lock_for(mgr_->timeout_)) {
                    // Roll back any stripes already taken.
                    release();
                    throw LockTimeoutError(
                        "exclusive lock contention timeout on stripe "
                        + std::to_string(s) + " (timeout_ms="
                        + std::to_string(mgr_->timeout_.count()) + ")");
                }
                locked_.push_back(s);
            }
        }

        ~ExclusiveGuard() { release(); }

        ExclusiveGuard(const ExclusiveGuard&) = delete;
        ExclusiveGuard& operator=(const ExclusiveGuard&) = delete;

        ExclusiveGuard(ExclusiveGuard&& other) noexcept {
            move_from(std::move(other));
        }

        ExclusiveGuard& operator=(ExclusiveGuard&& other) noexcept {
            if (this != &other) {
                release();
                move_from(std::move(other));
            }
            return *this;
        }

        bool owns_lock() const noexcept { return !locked_.empty(); }

        void release() noexcept {
            // Unlock in reverse order of acquisition.
            for (auto it = locked_.rbegin(); it != locked_.rend(); ++it) {
                if (mgr_ != nullptr) {
                    mgr_->stripes_[*it].unlock();
                }
            }
            locked_.clear();
            stripes_.clear();
            mgr_ = nullptr;
        }

    private:
        NeighborhoodLockManager* mgr_{nullptr};
        std::vector<std::size_t> stripes_;
        std::vector<std::size_t> locked_;

        void move_from(ExclusiveGuard&& other) noexcept {
            mgr_ = other.mgr_;
            stripes_ = std::move(other.stripes_);
            locked_ = std::move(other.locked_);
            other.mgr_ = nullptr;
            other.stripes_.clear();
            other.locked_.clear();
        }
    };

    // ------------------------------------------------------------------
    // Shared single-stripe guard — search step / read
    // ------------------------------------------------------------------

    class SharedGuard {
    public:
        SharedGuard() = default;

        SharedGuard(NeighborhoodLockManager* mgr, std::size_t stripe)
            : mgr_(mgr), stripe_(stripe) {
            if (mgr_ == nullptr) {
                return;
            }
            auto& mu = mgr_->stripes_[stripe_];
            if (!mu.try_lock_shared_for(mgr_->timeout_)) {
                mgr_ = nullptr;
                throw LockTimeoutError(
                    "shared lock contention timeout on stripe "
                    + std::to_string(stripe_) + " (timeout_ms="
                    + std::to_string(mgr->timeout_.count()) + ")");
            }
            held_ = true;
        }

        ~SharedGuard() { release(); }

        SharedGuard(const SharedGuard&) = delete;
        SharedGuard& operator=(const SharedGuard&) = delete;

        SharedGuard(SharedGuard&& other) noexcept { move_from(std::move(other)); }

        SharedGuard& operator=(SharedGuard&& other) noexcept {
            if (this != &other) {
                release();
                move_from(std::move(other));
            }
            return *this;
        }

        bool owns_lock() const noexcept { return held_; }

        void release() noexcept {
            if (held_ && mgr_ != nullptr) {
                mgr_->stripes_[stripe_].unlock_shared();
            }
            held_ = false;
            mgr_ = nullptr;
        }

    private:
        NeighborhoodLockManager* mgr_{nullptr};
        std::size_t stripe_{0};
        bool held_{false};

        void move_from(SharedGuard&& other) noexcept {
            mgr_ = other.mgr_;
            stripe_ = other.stripe_;
            held_ = other.held_;
            other.held_ = false;
            other.mgr_ = nullptr;
        }
    };

    /**
     * Hand-over-hand (lock coupling) cursor for search traversal.
     *
     * Holds a shared lock on the current node; advance(next) acquires next
     * before releasing current so the edge current→next cannot be rewired
     * out from under the walker mid-step.
     */
    class SharedLockCursor {
    public:
        explicit SharedLockCursor(NeighborhoodLockManager* mgr) : mgr_(mgr) {}

        /** Acquire shared lock on the entry node (no previous hold). */
        void enter(labeltype node_id) {
            release();
            current_ = SharedGuard(mgr_, mgr_->stripe_of(node_id));
            current_node_ = node_id;
        }

        /**
         * Lock-coupling step: shared-lock ``next`` then drop ``current``.
         * If next maps to the same stripe as current, the existing hold is
         * retained (stripe already covers both labels).
         */
        void advance(labeltype next) {
            if (mgr_ == nullptr) {
                return;
            }
            const std::size_t next_stripe = mgr_->stripe_of(next);
            if (current_.owns_lock()
                && mgr_->stripe_of(current_node_) == next_stripe) {
                current_node_ = next;
                return;
            }
            SharedGuard next_guard(mgr_, next_stripe);
            // Hand-over: next is held; safe to drop previous.
            current_ = std::move(next_guard);
            current_node_ = next;
        }

        void release() {
            current_.release();
            current_node_ = -1;
        }

        labeltype current() const noexcept { return current_node_; }

        bool holds() const noexcept { return current_.owns_lock(); }

    private:
        NeighborhoodLockManager* mgr_{nullptr};
        SharedGuard current_;
        labeltype current_node_{-1};
    };

    /** Exclusive locks covering every stripe touched by ``nodes``. */
    ExclusiveGuard acquire_exclusive(const std::vector<labeltype>& nodes) {
        return ExclusiveGuard(this, unique_stripes(nodes));
    }

    /** Shared lock for a single node (its stripe). */
    SharedGuard acquire_shared(labeltype node_id) {
        return SharedGuard(this, stripe_of(node_id));
    }

    /** Exclusive lock for a single node (its stripe). */
    ExclusiveGuard acquire_exclusive_one(labeltype node_id) {
        return acquire_exclusive(std::vector<labeltype>{node_id});
    }

private:
    // Note: shared_timed_mutex is not movable; vector default-constructs them.
    std::vector<std::shared_timed_mutex> stripes_;
    duration timeout_;

    std::vector<std::size_t> unique_stripes(
        const std::vector<labeltype>& nodes) const {
        std::vector<std::size_t> out;
        out.reserve(nodes.size());
        for (labeltype n : nodes) {
            out.push_back(stripe_of(n));
        }
        std::sort(out.begin(), out.end());
        out.erase(std::unique(out.begin(), out.end()), out.end());
        return out;
    }
};

}  // namespace hnsw
