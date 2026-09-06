/**
 * Module: placecell - synthetic_demo.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-05
 * - License: Apache-2.0
 *
 * GPU-free exercise of the whole core through the three managers: a synthetic
 * trajectory visits a handful of "places" (random unit centres in R^64, each view =
 * centre + noise, renormalised), every view is queried for its unexplained information,
 * the demo plays host (insert when novel, or every k-th view while not redundant), the
 * culler runs every few insertions, and at the end the profile table is printed, the
 * store is dumped (kernel .npy + CSVs) and, when built with placecell::viz, the three
 * plots are written as PNGs. Also the smoke test for those managers: exit code 1 when
 * the store, the recorder or the culler disagree with what the script did.
 *
 * Usage:
 *   synthetic_demo [<output_dir>] [--views N] [--places P] [--tau T] [--min-info M]
 *                  [--verbosity off|error|warn|info|debug|trace] [--windows]
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <placecell/placecell.h>
#ifdef PLACECELL_HAS_VIZ
#include <opencv2/highgui.hpp>
#include <placecell/viz.h>
#endif

int main(int argc, char** argv)
{
    std::string output_dir = "placecell_demo_out";
    int num_views = 400, num_places = 8;
    float tau = 0.3f, min_information = 0.05f;
    bool windows = false;
    placecell::PlaceCell::Options options;
    options.name = "synthetic";
    for(int i = 1; i < argc; i++)
    {
        auto next = [&](const char* flag) -> const char* {
            if(i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(1); }
            return argv[++i];
        };
        if(std::strcmp(argv[i], "--views") == 0) num_views = std::atoi(next("--views"));
        else if(std::strcmp(argv[i], "--places") == 0) num_places = std::atoi(next("--places"));
        else if(std::strcmp(argv[i], "--tau") == 0) tau = float(std::atof(next("--tau")));
        else if(std::strcmp(argv[i], "--min-info") == 0) min_information = float(std::atof(next("--min-info")));
        else if(std::strcmp(argv[i], "--windows") == 0) windows = true;
        else if(std::strcmp(argv[i], "--verbosity") == 0)
        {
            const auto level = placecell::Logger::parse(next("--verbosity"));
            if(!level) { std::fprintf(stderr, "unknown verbosity\n"); return 1; }
            options.verbosity = *level;
        }
        else output_dir = argv[i];
    }

    placecell::Logger::instance().set_show_elapsed(true);
    placecell::PlaceCell cell(options);
    cell.recorder().set_thresholds(tau, min_information);

    // Synthetic world: places as random unit centres; the trajectory dwells at a place
    // for a while, moves on, and revisits earlier places (loop closures) at the end.
    std::mt19937 rng(7);
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int dim = 64;
    std::vector<Eigen::VectorXf> centres;
    for(int p = 0; p < num_places; p++)
    {
        Eigen::VectorXf c(dim);
        for(int d = 0; d < dim; d++) c(d) = gauss(rng);
        centres.push_back(c.normalized());
    }
    auto view_at = [&](const int place, const float noise) {
        Eigen::VectorXf v = centres[std::size_t(place)];
        for(int d = 0; d < dim; d++) v(d) += noise * gauss(rng);
        return v.normalized();
    };

    placecell::PlaceCell::CullParameters cull_parameters;
    cull_parameters.max_unexplained = tau;
    cull_parameters.centred = true;
    cull_parameters.min_keyframes = 6;
    cull_parameters.protect_last = 2;
    cull_parameters.max_per_call = 3;

    int inserted = 0, culled_total = 0;
    const int dwell = std::max(1, num_views / (num_places + num_places / 2));
    for(int t = 0; t < num_views; t++)
    {
        // Place schedule: forward through every place, then revisit the first half
        const int step = t / dwell;
        const int place = step < num_places ? step : (step - num_places) % std::max(1, num_places / 2);
        const Eigen::VectorXf view = view_at(place, 0.35f);
        const placecell::PlaceCell::ExternalId id = placecell::PlaceCell::ExternalId(t);

        const placecell::PlaceCell::Information info = cell.unexplained_information(view, nullptr, true);
        const bool redundant = info.unexplained < min_information;
        const bool novel = info.unexplained > tau;
        const bool periodic = (t % 5 == 0);
        const bool insert = !redundant && (novel || periodic);
        cell.recorder().record_decision(id, insert, info.unexplained, novel ? "novel" : (periodic ? "periodic" : "skip"));
        if(!insert)
            continue;
        cell.add(id, view);
        inserted++;

        if(inserted % 4 == 0)
        {
            const auto report = cell.cull_keyframes(cull_parameters, [](placecell::PlaceCell::ExternalId) { return true; });
            culled_total += int(report.culled.size());
        }
    }

    // ---- Checks ------------------------------------------------------------------------
    bool ok = true;
    const auto& recorder = cell.recorder();
    if(int(cell.size()) != inserted) { std::fprintf(stderr, "store size %zu != inserted %d\n", cell.size(), inserted); ok = false; }
    if(int(recorder.query_count()) != num_views) { std::fprintf(stderr, "queries %zu != views %d\n", recorder.query_count(), num_views); ok = false; }
    if(int(recorder.cull_count()) != culled_total) { std::fprintf(stderr, "recorded culls %zu != %d\n", recorder.cull_count(), culled_total); ok = false; }
    if(recorder.decisions().size() != std::size_t(num_views)) { std::fprintf(stderr, "decisions mismatch\n"); ok = false; }
    int culled_flags = 0;
    for(const auto id : cell.external_ids()) culled_flags += cell.is_culled(id);
    if(culled_flags != culled_total) { std::fprintf(stderr, "is_culled flags %d != culls %d\n", culled_flags, culled_total); ok = false; }
    const auto stats_add = cell.profiler().stats("add");
    const auto stats_query = cell.profiler().stats("unexplained_information");
    const auto stats_cull = cell.profiler().stats("cull_keyframes");
    if(int(stats_add.count) != inserted || int(stats_query.count) != num_views || stats_cull.count != std::size_t(inserted / 4))
    {
        std::fprintf(stderr, "profiler counts add=%zu query=%zu cull=%zu (expected %d, %d, %d)\n", stats_add.count,
                     stats_query.count, stats_cull.count, inserted, num_views, inserted / 4);
        ok = false;
    }
    const Eigen::MatrixXf centred = cell.centred_kernel();
    if(centred.rows() != Eigen::Index(cell.size()) || (cell.size() >= 3 && std::abs(centred(0, 0) - 1.0f) > 1e-4f))
    {
        std::fprintf(stderr, "centred kernel: bad shape or diagonal\n");
        ok = false;
    }

    std::cout << "synthetic demo: " << num_views << " views, " << inserted << " inserted, " << culled_total
              << " culled, " << cell.size() - std::size_t(culled_total) << " alive" << std::endl;
    cell.print_profile();
    cell.dump(output_dir);

#ifdef PLACECELL_HAS_VIZ
    placecell::viz::Visualizer::Options viz_options;
    viz_options.windows = windows;
    viz_options.kernel.centred = true;
    placecell::viz::Visualizer visualizer(cell, viz_options);
    visualizer.update(true);
    visualizer.save(output_dir);
    if(windows)
    {
        std::cout << "press any key in a window to exit" << std::endl;
        cv::waitKey(0);
    }
#else
    (void)windows;
#endif

    std::cout << (ok ? "synthetic demo OK" : "synthetic demo FAILED") << std::endl;
    return ok ? 0 : 1;
}
