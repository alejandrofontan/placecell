/**
 * placecell Python bindings (nanobind).
 *
 * Core only (no MegaLoc / OpenCV): the store, the kernel (descriptor-backed through add(),
 * kernel-only through set_kernel() + similarity_from_distance(), or item-backed through
 * set_items()), the information queries (unexplained_information for a descriptor,
 * unexplained_information_items for an item set) and the three managers' Python-side surface — verbosity, profile report,
 * recorder access and dump() — so the offline visualizer (tools/plot_placecell.py), a
 * notebook, or VSLAM-LAB's rgb_placecell capability can drive placecell on precomputed
 * descriptors or a precomputed pairwise matrix.
 */
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <placecell/kernel_io.h>
#include <placecell/placecell.h>

namespace nb = nanobind;
using namespace nb::literals;
using placecell::PlaceCell;

NB_MODULE(_placecell, m)
{
    m.doc() = "Keyframe lifecycle management for VSLAM and 3D reconstruction";

    // ---- Logger --------------------------------------------------------------------------
    nb::enum_<placecell::LogLevel>(m, "LogLevel")
        .value("off", placecell::LogLevel::off)
        .value("error", placecell::LogLevel::error)
        .value("warn", placecell::LogLevel::warn)
        .value("info", placecell::LogLevel::info)
        .value("debug", placecell::LogLevel::debug)
        .value("trace", placecell::LogLevel::trace);
    m.def("set_verbosity", [](placecell::LogLevel level) { placecell::Logger::instance().set_level(level); }, "level"_a,
          "Process-wide log verbosity (PLACECELL_VERBOSITY in the environment applies at startup)");
    m.def("set_verbosity", [](const std::string& level) {
        const auto parsed = placecell::Logger::parse(level);
        if(!parsed)
            throw nb::value_error("unknown level (off|error|warn|info|debug|trace or 0-5)");
        placecell::Logger::instance().set_level(*parsed);
    }, "level"_a);
    m.def("verbosity", []() { return placecell::Logger::instance().level(); });

    // ---- kernel_io -----------------------------------------------------------------------
    m.def("similarity_from_distance",
          [](const Eigen::MatrixXf& distance, const std::string& kind) {
              const auto parsed = placecell::parse_distance_kind(kind);
              if(!parsed)
                  throw nb::value_error("unknown distance kind (similarity|squared-euclidean|cosine-distance|euclidean)");
              return placecell::similarity_from_distance(distance, *parsed);
          },
          "distance"_a, "kind"_a,
          "Pairwise distance matrix of unit-norm descriptors -> similarity kernel for PlaceCell.set_kernel: "
          "'similarity' (identity), 'squared-euclidean' (S = 1 - D/2; faiss / VPR-LAB D.npy), "
          "'cosine-distance' (S = 1 - D), 'euclidean' (S = 1 - D^2/2)");

    // ---- Profiler ------------------------------------------------------------------------
    nb::class_<placecell::Profiler> profiler(m, "Profiler");
    nb::class_<placecell::Profiler::Stats>(profiler, "Stats")
        .def_ro("count", &placecell::Profiler::Stats::count)
        .def_ro("last_ms", &placecell::Profiler::Stats::last_ms)
        .def_ro("median_ms", &placecell::Profiler::Stats::median_ms)
        .def_ro("p95_ms", &placecell::Profiler::Stats::p95_ms)
        .def_ro("max_ms", &placecell::Profiler::Stats::max_ms)
        .def_ro("total_ms", &placecell::Profiler::Stats::total_ms);
    profiler.def_prop_rw("enabled", &placecell::Profiler::enabled, &placecell::Profiler::set_enabled)
        .def("functions", &placecell::Profiler::functions)
        .def("stats", &placecell::Profiler::stats, "function"_a)
        .def("report", &placecell::Profiler::report)
        .def("dump_csv", &placecell::Profiler::dump_csv, "path"_a)
        .def("clear", &placecell::Profiler::clear);

    // ---- Recorder ------------------------------------------------------------------------
    nb::class_<placecell::Recorder> recorder(m, "Recorder");
    nb::class_<placecell::Recorder::Query>(recorder, "Query")
        .def_ro("index", &placecell::Recorder::Query::index)
        .def_ro("time_s", &placecell::Recorder::Query::time_s)
        .def_ro("unexplained", &placecell::Recorder::Query::unexplained)
        .def_ro("explainers", &placecell::Recorder::Query::explainers)
        .def_ro("best_explainer", &placecell::Recorder::Query::best_explainer)
        .def_ro("best_similarity", &placecell::Recorder::Query::best_similarity)
        .def_ro("centred", &placecell::Recorder::Query::centred)
        .def_ro("stored", &placecell::Recorder::Query::stored)
        .def_ro("window", &placecell::Recorder::Query::window);
    nb::class_<placecell::Recorder::Cull>(recorder, "Cull")
        .def_ro("call_index", &placecell::Recorder::Cull::call_index)
        .def_ro("time_s", &placecell::Recorder::Cull::time_s)
        .def_ro("id", &placecell::Recorder::Cull::id)
        .def_ro("unique_information", &placecell::Recorder::Cull::unique_information)
        .def_ro("worst_unexplained_after", &placecell::Recorder::Cull::worst_unexplained_after)
        .def_ro("alive_after", &placecell::Recorder::Cull::alive_after)
        .def_ro("tau", &placecell::Recorder::Cull::tau);
    nb::class_<placecell::Recorder::Decision>(recorder, "Decision")
        .def_ro("query_index", &placecell::Recorder::Decision::query_index)
        .def_ro("time_s", &placecell::Recorder::Decision::time_s)
        .def_ro("id", &placecell::Recorder::Decision::id)
        .def_ro("inserted", &placecell::Recorder::Decision::inserted)
        .def_ro("unexplained", &placecell::Recorder::Decision::unexplained)
        .def_ro("reason", &placecell::Recorder::Decision::reason);
    recorder.def_prop_rw("enabled", &placecell::Recorder::enabled, &placecell::Recorder::set_enabled)
        .def("record_decision", &placecell::Recorder::record_decision, "id"_a, "inserted"_a,
             "unexplained"_a = std::numeric_limits<float>::quiet_NaN(), "reason"_a = std::string())
        .def("set_thresholds", &placecell::Recorder::set_thresholds, "tau"_a, "min_information"_a)
        .def("query_count", &placecell::Recorder::query_count)
        .def("cull_count", &placecell::Recorder::cull_count)
        .def("queries", &placecell::Recorder::queries)
        .def("culls", &placecell::Recorder::culls)
        .def("decisions", &placecell::Recorder::decisions)
        .def("dump_csv", &placecell::Recorder::dump_csv, "directory"_a)
        .def("clear", &placecell::Recorder::clear);

    // ---- PlaceCell -----------------------------------------------------------------------
    nb::class_<PlaceCell> cell(m, "PlaceCell");
    nb::class_<PlaceCell::Options>(cell, "Options")
        .def(nb::init<>())
        .def_rw("verbosity", &PlaceCell::Options::verbosity)
        .def_rw("profile", &PlaceCell::Options::profile)
        .def_rw("record", &PlaceCell::Options::record)
        .def_rw("report_on_destruction", &PlaceCell::Options::report_on_destruction)
        .def_rw("name", &PlaceCell::Options::name);
    nb::class_<PlaceCell::Information>(cell, "Information")
        .def_ro("unexplained", &PlaceCell::Information::unexplained)
        .def_ro("explainers", &PlaceCell::Information::explainers)
        .def_ro("best_explainer", &PlaceCell::Information::best_explainer)
        .def_ro("best_similarity", &PlaceCell::Information::best_similarity);
    nb::class_<PlaceCell::CullParameters>(cell, "CullParameters")
        .def(nb::init<>())
        .def_rw("method", &PlaceCell::CullParameters::method)
        .def_rw("max_unexplained", &PlaceCell::CullParameters::max_unexplained)
        .def_rw("centred", &PlaceCell::CullParameters::centred)
        .def_rw("min_keyframes", &PlaceCell::CullParameters::min_keyframes)
        .def_rw("protect_last", &PlaceCell::CullParameters::protect_last)
        .def_rw("max_per_call", &PlaceCell::CullParameters::max_per_call)
        .def_rw("protect_first", &PlaceCell::CullParameters::protect_first)
        .def_rw("target_alive", &PlaceCell::CullParameters::target_alive);
    nb::class_<PlaceCell::KernelOptions>(cell, "KernelOptions")
        .def(nb::init<>())
        .def_rw("symmetrise", &PlaceCell::KernelOptions::symmetrise)
        .def_rw("asymmetry_warn", &PlaceCell::KernelOptions::asymmetry_warn)
        .def_rw("unit_diagonal", &PlaceCell::KernelOptions::unit_diagonal)
        .def_rw("psd_check", &PlaceCell::KernelOptions::psd_check)
        .def_rw("clip_to_psd", &PlaceCell::KernelOptions::clip_to_psd);
    nb::class_<PlaceCell::KernelReport>(cell, "KernelReport")
        .def_ro("views", &PlaceCell::KernelReport::views)
        .def_ro("max_asymmetry", &PlaceCell::KernelReport::max_asymmetry)
        .def_ro("max_diagonal_deviation", &PlaceCell::KernelReport::max_diagonal_deviation)
        .def_ro("min_eigenvalue", &PlaceCell::KernelReport::min_eigenvalue)
        .def_ro("max_eigenvalue", &PlaceCell::KernelReport::max_eigenvalue)
        .def_ro("negative_eigenvalues", &PlaceCell::KernelReport::negative_eigenvalues)
        .def_ro("clipped", &PlaceCell::KernelReport::clipped)
        .def_ro("ms", &PlaceCell::KernelReport::ms);
    nb::class_<PlaceCell::CullReport> report(cell, "CullReport");
    nb::class_<PlaceCell::CullReport::CulledView>(report, "CulledView")
        .def_ro("id", &PlaceCell::CullReport::CulledView::id)
        .def_ro("unique_information", &PlaceCell::CullReport::CulledView::unique_information)
        .def_ro("worst_unexplained_after", &PlaceCell::CullReport::CulledView::worst_unexplained_after)
        .def_ro("alive_after", &PlaceCell::CullReport::CulledView::alive_after);
    report.def_ro("culled", &PlaceCell::CullReport::culled)
        .def_ro("views_total", &PlaceCell::CullReport::views_total)
        .def_ro("candidates", &PlaceCell::CullReport::candidates)
        .def_ro("alive_after", &PlaceCell::CullReport::alive_after)
        .def_ro("worst_history", &PlaceCell::CullReport::worst_history)
        .def_ro("history_over_budget", &PlaceCell::CullReport::history_over_budget)
        .def_ro("reached_max_per_call", &PlaceCell::CullReport::reached_max_per_call)
        .def_ro("alive_ids", &PlaceCell::CullReport::alive_ids)
        .def_ro("alive_unique_information", &PlaceCell::CullReport::alive_unique_information);

    cell.attr("invalid_id") = PlaceCell::invalid_id;
    cell.def(nb::init<>())
        .def(nb::init<const PlaceCell::Options&>(), "options"_a)
        .def("add", &PlaceCell::add, "id"_a, "descriptor"_a)
        .def("set_kernel",
             [](PlaceCell& self, const Eigen::MatrixXf& similarity,
                const std::optional<std::vector<PlaceCell::ExternalId>>& ids,
                const std::optional<PlaceCell::KernelOptions>& options) {
                 return self.set_kernel(similarity, ids ? *ids : std::vector<PlaceCell::ExternalId>{},
                                        options ? *options : PlaceCell::KernelOptions{});
             },
             "similarity"_a, "ids"_a = nb::none(), "options"_a = nb::none(),
             "Initialise an EMPTY store from an n x n similarity (kernel-only: no descriptors, add() refused); "
             "row i -> ids[i] (or i). Raises ValueError on a bad matrix, RuntimeError on a non-empty store.")
        .def("kernel_only", &PlaceCell::kernel_only)
        .def("set_items",
             [](PlaceCell& self, const PlaceCell::ExternalId id, std::vector<PlaceCell::ItemId> items) {
                 return self.set_items(id, std::move(items));
             },
             "id"_a, "items"_a,
             "Store or refresh the item set (e.g. map-point ids) of a view: K_ij = |P_i & P_j| / sqrt(|P_i| |P_j|). "
             "First call on an EMPTY store makes it item-backed (add() then refused, descriptor() None); "
             "refused (invalid_id) on a descriptor or kernel-only store. A culled view's set is frozen; "
             "an empty set gives a NaN (unusable) row. Returns the internal id (kernel row).")
        .def("items",
             [](const PlaceCell& self, const PlaceCell::ExternalId id) -> std::optional<std::vector<PlaceCell::ItemId>> {
                 const std::vector<PlaceCell::ItemId>* items = self.items(id);
                 if(!items)
                     return std::nullopt;
                 return *items;
             },
             "id"_a, "Sorted item set of a view (a copy), None for an unknown id or a store that is not item-backed")
        .def("item_mode", &PlaceCell::item_mode)
        .def("unexplained_information_items",
             [](const PlaceCell& self, const std::vector<PlaceCell::ItemId>& items,
                const std::optional<std::vector<PlaceCell::ExternalId>>& window, const bool centred) {
                 return self.unexplained_information(items, window ? &*window : nullptr, centred);
             },
             "items"_a, "window"_a = nb::none(), "centred"_a = false,
             "Unexplained information of a view NOT in the store given by its item set (item-backed stores only; "
             "NaN otherwise or for an empty set). Same maths as unexplained_information; centring defaults to off "
             "because the covisibility cosine has no common-mode floor.")
        .def("has", &PlaceCell::has, "id"_a)
        .def("internal_id", &PlaceCell::internal_id, "id"_a)
        .def("size", &PlaceCell::size)
        .def("__len__", &PlaceCell::size)
        .def("kernel", &PlaceCell::kernel)
        .def("centred_kernel", &PlaceCell::centred_kernel)
        .def("external_ids", &PlaceCell::external_ids)
        .def("clear", &PlaceCell::clear)
        .def("unexplained_information",
             [](const PlaceCell& self, const Eigen::VectorXf& descriptor,
                const std::optional<std::vector<PlaceCell::ExternalId>>& window, const bool centred) {
                 return self.unexplained_information(descriptor, window ? &*window : nullptr, centred);
             },
             "descriptor"_a, "window"_a = nb::none(), "centred"_a = true)
        .def("set_protected", &PlaceCell::set_protected, "id"_a, "value"_a = true)
        .def("is_protected", &PlaceCell::is_protected, "id"_a)
        .def("set_culled", &PlaceCell::set_culled, "id"_a)
        .def("is_culled", &PlaceCell::is_culled, "id"_a)
        .def("cull_keyframes",
             [](PlaceCell& self, const PlaceCell::CullParameters& parameters, const PlaceCell::CullCallback& try_cull,
                const std::optional<std::vector<PlaceCell::ExternalId>>& local_window) {
                 return self.cull_keyframes(parameters, try_cull, local_window ? &*local_window : nullptr);
             },
             "parameters"_a, "try_cull"_a, "local_window"_a = nb::none())
        .def("profiler", nb::overload_cast<>(&PlaceCell::profiler), nb::rv_policy::reference_internal)
        .def("recorder", nb::overload_cast<>(&PlaceCell::recorder), nb::rv_policy::reference_internal)
        .def("print_profile", &PlaceCell::print_profile)
        .def("dump", &PlaceCell::dump, "directory"_a)
        .def("__repr__", [](const PlaceCell& self) {
            return "PlaceCell(" + std::to_string(self.size()) + " views)";
        });
}
