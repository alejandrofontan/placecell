/**
 * placecell Python bindings (nanobind).
 *
 * Core only (no MegaLoc / OpenCV): the store, the kernel, the two information queries
 * and the three managers' Python-side surface — verbosity, profile report, recorder
 * access and dump() — so the offline visualizer (tools/plot_placecell.py) and a
 * notebook can drive placecell on precomputed descriptors.
 */
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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
        .def_rw("protect_first", &PlaceCell::CullParameters::protect_first);
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

    cell.def(nb::init<>())
        .def(nb::init<const PlaceCell::Options&>(), "options"_a)
        .def("add", &PlaceCell::add, "id"_a, "descriptor"_a)
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
