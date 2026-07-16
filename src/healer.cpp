/**
 * healer.cpp
 *
 * Pybind11 module entrypoint for the HNSW graph-healing native extension.
 *
 * Concurrency surface:
 *   - LockContentionError  ← LockTimeoutError (timed stripe lock failure)
 *   - search_knn           ← shared locks + hand-over-hand coupling
 *   - heal_graph_structure ← exclusive locks on local 2-hop neighborhood
 */

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hnsw_helper.hpp"

namespace py = pybind11;

namespace {

/** Process-local default index used by free-function API. */
std::shared_ptr<hnsw::HNSWIndexProxy>& default_index() {
    static std::shared_ptr<hnsw::HNSWIndexProxy> instance =
        std::make_shared<hnsw::HNSWIndexProxy>();
    return instance;
}

hnsw::HNSWIndexProxy& require_default_index() {
    auto& idx = default_index();
    if (!idx) {
        idx = std::make_shared<hnsw::HNSWIndexProxy>();
    }
    return *idx;
}

void proxy_load_index_buffer(hnsw::HNSWIndexProxy& self,
                             const py::buffer& float_array,
                             std::size_t dimensions,
                             std::size_t num_elements) {
    py::buffer_info info = float_array.request();

    if (info.ndim != 1 && info.ndim != 2) {
        throw std::invalid_argument(
            "load_index: float array must be 1-D (flat) or 2-D (N x D)");
    }

    const std::size_t expected = dimensions * num_elements;
    const std::size_t got = static_cast<std::size_t>(info.size);

    if (info.ndim == 2) {
        const auto rows = static_cast<std::size_t>(info.shape[0]);
        const auto cols = static_cast<std::size_t>(info.shape[1]);
        if (rows != num_elements || cols != dimensions) {
            throw std::invalid_argument(
                "load_index: 2-D shape (" + std::to_string(rows) + ", "
                + std::to_string(cols) + ") does not match num_elements="
                + std::to_string(num_elements) + ", dimensions="
                + std::to_string(dimensions));
        }
    } else if (got != expected) {
        throw std::invalid_argument(
            "load_index: flat length " + std::to_string(got)
            + " does not equal dimensions * num_elements ("
            + std::to_string(expected) + ")");
    }

    py::array_t<float, py::array::c_style | py::array::forcecast> contiguous(
        float_array);
    py::buffer_info cinfo = contiguous.request();
    const float* src = static_cast<const float*>(cinfo.ptr);

    self.load_index(src, dimensions, num_elements);
}

void free_load_index(const py::buffer& float_array,
                     std::size_t dimensions,
                     std::size_t num_elements) {
    proxy_load_index_buffer(
        require_default_index(), float_array, dimensions, num_elements);
}

/**
 * Zero-copy attach of a writable float32 NumPy buffer into a proxy.
 *
 * The caller MUST keep ``float_array`` alive for the lifetime of the proxy
 * attachment (see integrations.vendor_attach.InPlaceVendorSession).
 */
void proxy_attach_index_buffer(hnsw::HNSWIndexProxy& self,
                               py::array_t<float> float_array,
                               std::size_t dimensions,
                               std::size_t num_elements) {
    py::buffer_info info = float_array.request(/*writable=*/true);
    if (info.ndim != 1 && info.ndim != 2) {
        throw std::invalid_argument(
            "attach_index: float array must be 1-D (flat) or 2-D (N x D)");
    }
    const std::size_t expected = dimensions * num_elements;
    if (static_cast<std::size_t>(info.size) != expected) {
        throw std::invalid_argument(
            "attach_index: size mismatch vs dimensions * num_elements");
    }
    if (info.ndim == 2) {
        if (static_cast<std::size_t>(info.shape[0]) != num_elements
            || static_cast<std::size_t>(info.shape[1]) != dimensions) {
            throw std::invalid_argument("attach_index: 2-D shape mismatch");
        }
        // Require C-contiguous layout (no silent copy — this is zero-copy attach).
        const auto s0 = static_cast<std::size_t>(info.strides[0]);
        const auto s1 = static_cast<std::size_t>(info.strides[1]);
        if (s1 != sizeof(float)
            || s0 != dimensions * sizeof(float)) {
            throw std::invalid_argument(
                "attach_index: array must be C-contiguous float32");
        }
    } else if (info.ndim == 1) {
        if (static_cast<std::size_t>(info.strides[0]) != sizeof(float)) {
            throw std::invalid_argument(
                "attach_index: flat array must be contiguous float32");
        }
    }
    float* ptr = static_cast<float*>(info.ptr);
    self.attach_index(ptr, dimensions, num_elements);
}

void free_attach_index_buffer(py::array_t<float> float_array,
                              std::size_t dimensions,
                              std::size_t num_elements) {
    proxy_attach_index_buffer(
        require_default_index(), float_array, dimensions, num_elements);
}

std::vector<hnsw::labeltype> free_get_neighbors(hnsw::labeltype node_id,
                                                hnsw::layer_t layer) {
    return require_default_index().get_neighbors(node_id, layer);
}

void free_overwrite_vector(hnsw::labeltype node_id) {
    require_default_index().overwrite_vector(node_id);
}

hnsw::ErasureResult free_erase_node(hnsw::labeltype node_id, int max_m) {
    return require_default_index().erase_node(node_id, max_m);
}

hnsw::HealingMetrics free_heal_graph_structure(std::size_t node_id,
                                               int max_m) {
    return require_default_index().heal_graph_structure(node_id, max_m);
}

std::vector<hnsw::SearchHit> proxy_search_knn(
    hnsw::HNSWIndexProxy& self,
    const py::buffer& query,
    std::size_t k,
    hnsw::labeltype entry_node) {
    py::array_t<float, py::array::c_style | py::array::forcecast> q(query);
    py::buffer_info info = q.request();
    if (info.size == 0) {
        throw std::invalid_argument("search_knn: empty query");
    }
    const float* ptr = static_cast<const float*>(info.ptr);
    return self.search_knn(ptr, k, entry_node);
}

std::vector<hnsw::SearchHit> free_search_knn(const py::buffer& query,
                                             std::size_t k,
                                             hnsw::labeltype entry_node) {
    return proxy_search_knn(require_default_index(), query, k, entry_node);
}

}  // namespace

PYBIND11_MODULE(hnsw_healer, m) {
    m.doc() =
        "Latent Space Erasure & HNSW Graph Healing — neighborhood-locked "
        "index proxy with MN-RU repair and lock-coupled search";

    // ------------------------------------------------------------------
    // Exceptions
    // ------------------------------------------------------------------
    // LockContentionError: timed shared/exclusive stripe acquisition failed.
    // Python callers should retry (see api.main.with_lock_retry).
    static py::exception<hnsw::LockTimeoutError> lock_contention_exc(
        m, "LockContentionError", PyExc_RuntimeError);

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) {
                std::rethrow_exception(p);
            }
        } catch (const hnsw::LockTimeoutError& e) {
            lock_contention_exc(e.what());
        } catch (const std::out_of_range& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        }
    });

    // ------------------------------------------------------------------
    // ErasureResult / HealingMetrics / SearchHit
    // ------------------------------------------------------------------
    py::class_<hnsw::ErasureResult>(m, "ErasureResult")
        .def_readonly("success", &hnsw::ErasureResult::success)
        .def_readonly("node_id", &hnsw::ErasureResult::node_id)
        .def_readonly("bytes_wiped", &hnsw::ErasureResult::bytes_wiped)
        .def_readonly("message", &hnsw::ErasureResult::message)
        .def("__repr__", [](const hnsw::ErasureResult& r) {
            return "<ErasureResult success="
                 + std::string(r.success ? "True" : "False")
                 + " node_id=" + std::to_string(r.node_id)
                 + " bytes_wiped=" + std::to_string(r.bytes_wiped)
                 + " message='" + r.message + "'>";
        });

    py::class_<hnsw::HealingMetrics>(m, "HealingMetrics")
        .def_readonly("success", &hnsw::HealingMetrics::success)
        .def_readonly("edges_removed", &hnsw::HealingMetrics::edges_removed)
        .def_readonly("edges_added", &hnsw::HealingMetrics::edges_added)
        .def_readonly(
            "repair_duration_ms", &hnsw::HealingMetrics::repair_duration_ms)
        .def("__repr__", [](const hnsw::HealingMetrics& h) {
            return "<HealingMetrics success="
                 + std::string(h.success ? "True" : "False")
                 + " edges_removed=" + std::to_string(h.edges_removed)
                 + " edges_added=" + std::to_string(h.edges_added)
                 + " repair_duration_ms="
                 + std::to_string(h.repair_duration_ms) + ">";
        });

    py::class_<hnsw::SearchHit>(m, "SearchHit")
        .def_readonly("node_id", &hnsw::SearchHit::node_id)
        .def_readonly("distance", &hnsw::SearchHit::distance)
        .def("__repr__", [](const hnsw::SearchHit& h) {
            return "<SearchHit node_id=" + std::to_string(h.node_id)
                 + " distance=" + std::to_string(h.distance) + ">";
        });

    // ------------------------------------------------------------------
    // HNSWIndexProxy
    // ------------------------------------------------------------------
    py::class_<hnsw::HNSWIndexProxy, std::shared_ptr<hnsw::HNSWIndexProxy>>(
        m, "HNSWIndexProxy",
        R"pbdoc(
            Thread-safe HNSW memory proxy.

            Readers take shared stripe locks with hand-over-hand coupling.
            Heal takes exclusive locks only on the local 2-hop neighborhood
            of q. Timed lock failures raise ``LockContentionError``.
        )pbdoc")
        .def(py::init<std::size_t, int>(),
             py::arg("lock_pool_size") = 64,
             py::arg("lock_timeout_ms") = 50,
             R"pbdoc(
                Parameters
                ----------
                lock_pool_size : int
                    Number of shared_timed_mutex stripes.
                lock_timeout_ms : int
                    try_lock timeout before LockContentionError.
             )pbdoc")
        .def(
            "set_lock_timeout_ms",
            &hnsw::HNSWIndexProxy::set_lock_timeout_ms,
            py::arg("timeout_ms"))
        .def_property_readonly(
            "lock_timeout_ms", &hnsw::HNSWIndexProxy::lock_timeout_ms)
        .def_property_readonly(
            "lock_pool_size", &hnsw::HNSWIndexProxy::lock_pool_size)
        .def(
            "load_index",
            &proxy_load_index_buffer,
            py::arg("float_array"),
            py::arg("dimensions"),
            py::arg("num_elements"))
        .def(
            "attach_index",
            &proxy_attach_index_buffer,
            py::arg("float_array"),
            py::arg("dimensions"),
            py::arg("num_elements"),
            R"pbdoc(
                Zero-copy attach of a writable C-contiguous float32 buffer.

                The array MUST remain alive while the proxy uses it.
                Mutations such as overwrite_vector / erase_node write through
                to the caller's memory (in-process vendor attach).
            )pbdoc")
        .def(
            "save_to_file",
            &hnsw::HNSWIndexProxy::save_to_file,
            py::arg("path"),
            "Serialize live HNSW memory to a binary file (e.g. index.bin.tmp).")
        .def(
            "load_from_file",
            &hnsw::HNSWIndexProxy::load_from_file,
            py::arg("path"),
            "Load a binary HNSW blob previously written by save_to_file.")
        .def(
            "load_adjacency",
            &hnsw::HNSWIndexProxy::load_adjacency,
            py::arg("adjacency"))
        .def(
            "set_neighbors",
            &hnsw::HNSWIndexProxy::set_neighbors,
            py::arg("node_id"),
            py::arg("layer"),
            py::arg("neighbors"))
        .def(
            "get_neighbors",
            &hnsw::HNSWIndexProxy::get_neighbors,
            py::arg("node_id"),
            py::arg("layer"))
        .def(
            "overwrite_vector",
            &hnsw::HNSWIndexProxy::overwrite_vector,
            py::arg("node_id"))
        .def(
            "get_vector",
            &hnsw::HNSWIndexProxy::get_vector,
            py::arg("node_id"))
        .def(
            "search_knn",
            &proxy_search_knn,
            py::arg("query"),
            py::arg("k") = 10,
            py::arg("entry_node") = -1,
            R"pbdoc(
                Concurrent k-NN search with shared lock coupling.

                Raises
                ------
                LockContentionError
                    If a shared stripe lock times out (safe to retry).
            )pbdoc")
        .def(
            "heal_graph_structure",
            &hnsw::HNSWIndexProxy::heal_graph_structure,
            py::arg("node_id"),
            py::arg("max_m"),
            R"pbdoc(
                MN-RU heal under exclusive locks on {q}∪N(q)∪N(N(q)).

                Raises
                ------
                LockContentionError
                    If exclusive neighborhood acquisition times out.
            )pbdoc")
        .def(
            "erase_node",
            &hnsw::HNSWIndexProxy::erase_node,
            py::arg("node_id"),
            py::arg("max_m") = 16)
        .def(
            "prune_adjacency",
            &hnsw::HNSWIndexProxy::prune_adjacency,
            py::arg("node_id"))
        .def(
            "num_layers",
            &hnsw::HNSWIndexProxy::num_layers,
            py::arg("node_id"))
        .def_property_readonly(
            "dimensions", &hnsw::HNSWIndexProxy::dimensions)
        .def_property_readonly(
            "num_elements", &hnsw::HNSWIndexProxy::num_elements)
        .def_property_readonly(
            "is_loaded", &hnsw::HNSWIndexProxy::is_loaded);

    // ------------------------------------------------------------------
    // Free functions
    // ------------------------------------------------------------------
    m.def("load_index", &free_load_index,
          py::arg("float_array"), py::arg("dimensions"),
          py::arg("num_elements"));

    m.def(
        "attach_index_buffer",
        &free_attach_index_buffer,
        py::arg("float_array"),
        py::arg("dimensions"),
        py::arg("num_elements"),
        "Zero-copy attach into the default proxy (caller keeps array alive).");

    m.def(
        "save_index",
        [](const std::string& path) {
            require_default_index().save_to_file(path);
        },
        py::arg("path"),
        "Serialize the default index to a binary path.");

    m.def(
        "load_index_file",
        [](const std::string& path) {
            require_default_index().load_from_file(path);
        },
        py::arg("path"),
        "Load the default index from a binary path (index.bin).");

    m.def("get_neighbors", &free_get_neighbors,
          py::arg("node_id"), py::arg("layer"));

    m.def("overwrite_vector", &free_overwrite_vector, py::arg("node_id"));

    m.def("heal_graph_structure", &free_heal_graph_structure,
          py::arg("node_id"), py::arg("max_m"));

    m.def("erase_node", &free_erase_node,
          py::arg("node_id"), py::arg("max_m") = 16);

    m.def("search_knn", &free_search_knn,
          py::arg("query"), py::arg("k") = 10, py::arg("entry_node") = -1,
          R"pbdoc(
              k-NN on the default index (shared lock coupling).

              Raises LockContentionError on stripe timeout — retry in Python.
          )pbdoc");

    m.def("default_index", []() { return default_index(); });

    m.def(
        "set_lock_timeout_ms",
        [](int ms) { require_default_index().set_lock_timeout_ms(ms); },
        py::arg("timeout_ms"),
        "Set default-index stripe lock timeout.");

    m.attr("__version__") = "0.1.0";
}
