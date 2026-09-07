/**
 * Module: placecell - kernel_demo.cpp
 * - Author: Alejandro Fontan Villacampa
 * - Assisted by: Claude (Fable 5)
 * - Version: 1.0
 * - Created: 2026-09-07
 * - License: Apache-2.0
 *
 * Kernel-only store from a precomputed pairwise matrix: load a NumPy .npy (e.g.
 * VPR-LAB's <sequence>/vpr-lab/D.npy — faiss squared-L2 distances between MegaLoc
 * descriptors, min over four image rotations), convert it to a similarity with the
 * stated convention, initialise a PlaceCell with set_kernel (row i -> id i, all views
 * alive), print what set_kernel had to fix, run the information culler offline over
 * every view (the offline keyframe selection: which frames survive at a given tau),
 * dump the store for tools/plot_placecell.py, and write the surviving frames as
 * <out>/rgb.csv (the rows of the sequence's rgb.csv whose matrix row survived — row i of
 * the matrix is data row i of the csv it was computed on). GPU-free.
 *
 * Usage:
 *   kernel_demo <matrix.npy> [--kind similarity|squared-euclidean|cosine-distance|euclidean]
 *               [--tau T] [--min-keyframes N] [--raw] [--clip] [--no-psd-check]
 *               [--out <dir>] [--rgb-csv <file>] [--verbosity off|error|warn|info|debug|trace]
 *
 * Defaults: --kind squared-euclidean (VPR-LAB), --tau 0.3, --min-keyframes 5, centred
 * kernel, no clipping, PSD check on, --out placecell_kernel_out; --rgb-csv defaults to
 * <sequence>/rgb.csv for a VSLAM-LAB <sequence>/vpr-lab/<matrix>.npy (then rgb_raw.csv,
 * if rgb.csv was already downsampled and its row count no longer matches). Exit code 1
 * when the store disagrees with the matrix (size, ids, kernel round trip), a cull result
 * is inconsistent, or an explicit --rgb-csv cannot be used.
 *
 * Note (issue #5): with every view alive and no history, the centred kernel is rank
 * n-1 and the very first cull's unique-information score is jitter-scale; the culler
 * warns once and recovers from the second cull on. --raw avoids the centring.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <placecell/kernel_io.h>
#include <placecell/placecell.h>
#ifdef PLACECELL_HAS_VIZ
#include <placecell/viz.h>
#endif

int main(int argc, char** argv)
{
    std::string matrix_path;
    std::string output_dir = "placecell_kernel_out";
    std::string rgb_csv_path;   // empty: look next to the matrix (<sequence>/rgb.csv, rgb_raw.csv)
    placecell::DistanceKind kind = placecell::DistanceKind::squared_euclidean;
    float tau = 0.3f;
    int min_keyframes = 5;
    bool centred = true;
    placecell::PlaceCell::KernelOptions kernel_options;
    placecell::PlaceCell::Options options;
    options.name = "kernel_demo";
    for(int i = 1; i < argc; i++)
    {
        auto next = [&](const char* flag) -> const char* {
            if(i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", flag); std::exit(1); }
            return argv[++i];
        };
        if(std::strcmp(argv[i], "--kind") == 0)
        {
            const auto parsed = placecell::parse_distance_kind(next("--kind"));
            if(!parsed) { std::fprintf(stderr, "unknown --kind (similarity|squared-euclidean|cosine-distance|euclidean)\n"); return 1; }
            kind = *parsed;
        }
        else if(std::strcmp(argv[i], "--tau") == 0) tau = float(std::atof(next("--tau")));
        else if(std::strcmp(argv[i], "--min-keyframes") == 0) min_keyframes = std::atoi(next("--min-keyframes"));
        else if(std::strcmp(argv[i], "--raw") == 0) centred = false;
        else if(std::strcmp(argv[i], "--clip") == 0) kernel_options.clip_to_psd = true;
        else if(std::strcmp(argv[i], "--no-psd-check") == 0) kernel_options.psd_check = false;
        else if(std::strcmp(argv[i], "--out") == 0) output_dir = next("--out");
        else if(std::strcmp(argv[i], "--rgb-csv") == 0) rgb_csv_path = next("--rgb-csv");
        else if(std::strcmp(argv[i], "--verbosity") == 0)
        {
            const auto level = placecell::Logger::parse(next("--verbosity"));
            if(!level) { std::fprintf(stderr, "unknown verbosity\n"); return 1; }
            options.verbosity = *level;
        }
        else if(matrix_path.empty()) matrix_path = argv[i];
        else { std::fprintf(stderr, "unexpected argument %s\n", argv[i]); return 1; }
    }
    if(matrix_path.empty())
    {
        std::fprintf(stderr, "usage: kernel_demo <matrix.npy> [--kind K] [--tau T] [--min-keyframes N] [--raw] [--clip] "
                             "[--no-psd-check] [--out dir] [--rgb-csv file] [--verbosity L]\n");
        return 1;
    }
    if(!options.verbosity)
        options.verbosity = placecell::LogLevel::info;   // set_kernel's report is the point of this demo

    // ---- Load + convert -----------------------------------------------------------------
    Eigen::MatrixXf distance;
    try
    {
        distance = placecell::load_npy(matrix_path);
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    const Eigen::MatrixXf similarity = placecell::similarity_from_distance(distance, kind);
    const Eigen::Index n = similarity.rows();
    std::cout << "kernel demo: " << matrix_path << " -> " << distance.rows() << " x " << distance.cols()
              << " (" << placecell::distance_kind_name(kind) << ")" << std::endl;
    if(n > 1)
    {
        // A few similarity statistics before set_kernel touches anything
        std::vector<float> values(similarity.data(), similarity.data() + similarity.size());
        std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
        std::vector<float> consecutive;
        for(Eigen::Index i = 0; i + 1 < n; i++)
            consecutive.push_back(similarity(i, i + 1));
        std::nth_element(consecutive.begin(), consecutive.begin() + consecutive.size() / 2, consecutive.end());
        std::printf("  similarity: min %.3f  median %.3f  max %.3f  consecutive median %.3f\n",
                    similarity.minCoeff(), values[values.size() / 2], similarity.maxCoeff(), consecutive[consecutive.size() / 2]);
    }

    // ---- set_kernel ----------------------------------------------------------------------
    placecell::Logger::instance().set_show_elapsed(true);
    placecell::PlaceCell cell(options);
    placecell::PlaceCell::KernelReport report;
    try
    {
        report = cell.set_kernel(similarity, {}, kernel_options);
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "set_kernel: %s\n", e.what());
        return 1;
    }
    std::printf("  set_kernel: %d views in %.1f ms | max asymmetry %.4f | max |diag-1| %.2e | eigenvalues [%.4f, %.2f], %d negative%s\n",
                report.views, report.ms, report.max_asymmetry, report.max_diagonal_deviation, report.min_eigenvalue,
                report.max_eigenvalue, report.negative_eigenvalues, report.clipped ? " (clipped)" : "");

    // ---- Checks: the store mirrors the matrix ------------------------------------------
    bool ok = true;
    if(Eigen::Index(cell.size()) != n) { std::fprintf(stderr, "store size %zu != %ld\n", cell.size(), long(n)); ok = false; }
    if(!cell.kernel_only()) { std::fprintf(stderr, "store is not kernel-only\n"); ok = false; }
    const std::vector<placecell::PlaceCell::ExternalId> ids = cell.external_ids();
    for(std::size_t i = 0; i < ids.size(); i++)
        if(ids[i] != i || cell.internal_id(i) != i) { std::fprintf(stderr, "id mapping broken at row %zu\n", i); ok = false; break; }
    {
        const Eigen::MatrixXf stored = cell.kernel();
        const Eigen::MatrixXf expected = (0.5f * (similarity + similarity.transpose())).eval();
        Eigen::MatrixXf difference = (stored - expected).cwiseAbs();
        difference.diagonal().setZero();   // the diagonal was set to 1
        if(!report.clipped && difference.maxCoeff() > 1e-5f)
        {
            std::fprintf(stderr, "stored kernel differs from the symmetrised input by %.2e\n", difference.maxCoeff());
            ok = false;
        }
        if((stored.diagonal().array() - 1.0f).abs().maxCoeff() > 0.0f) { std::fprintf(stderr, "diagonal not 1\n"); ok = false; }
    }
    if(cell.descriptor(0) != nullptr) { std::fprintf(stderr, "descriptor() must be nullptr on a kernel-only store\n"); ok = false; }
    if(cell.add(ids.size(), Eigen::VectorXf::Ones(8)) != placecell::PlaceCell::invalid_id)
    { std::fprintf(stderr, "add() must be refused on a kernel-only store\n"); ok = false; }
    if(!std::isnan(cell.unexplained_information(Eigen::VectorXf::Ones(8)).unexplained))
    { std::fprintf(stderr, "descriptor query must return NaN on a kernel-only store\n"); ok = false; }

    // ---- Offline keyframe selection ---------------------------------------------------
    placecell::PlaceCell::CullParameters cull_parameters;
    cull_parameters.max_unexplained = tau;
    cull_parameters.centred = centred;
    cull_parameters.min_keyframes = min_keyframes;
    cull_parameters.protect_last = 1;
    cull_parameters.max_per_call = 0;
    cell.recorder().set_thresholds(tau, 0.0f);
    const auto cull_report = cell.cull_keyframes(cull_parameters, [](placecell::PlaceCell::ExternalId) { return true; });
    int alive = 0;
    for(const auto id : ids)
        alive += !cell.is_culled(id);
    if(alive != cull_report.alive_after || int(cull_report.culled.size()) + alive != int(n))
    {
        std::fprintf(stderr, "cull bookkeeping: %d alive + %zu culled != %ld (report says %d alive)\n", alive,
                     cull_report.culled.size(), long(n), cull_report.alive_after);
        ok = false;
    }
    std::printf("  cull (tau %.2f, %s, min %d): %zu of %d candidates culled -> %d keyframes survive; worst history %.3f\n",
                tau, centred ? "centred" : "raw", min_keyframes, cull_report.culled.size(), cull_report.candidates,
                alive, cull_report.worst_history);
    std::cout << "  surviving ids:";
    int printed = 0;
    for(const auto id : ids)
    {
        if(cell.is_culled(id)) continue;
        if(printed++ < 40) std::cout << " " << id;
    }
    if(printed > 40) std::cout << " ... (" << printed << " total)";
    std::cout << std::endl;

    cell.print_profile();
    cell.dump(output_dir);

    // ---- Surviving frames as an rgb.csv ------------------------------------------------
    // Row i of the matrix is data row i of the rgb.csv the matrix was computed on, so the
    // surviving views map straight onto csv rows. Source: --rgb-csv, or (VSLAM-LAB layout
    // <sequence>/vpr-lab/D.npy) <sequence>/rgb.csv, falling back to rgb_raw.csv when
    // rgb.csv was already downsampled and no longer has one row per matrix row.
    {
        namespace fs = std::filesystem;
        std::vector<std::string> candidates;
        if(!rgb_csv_path.empty())
            candidates.push_back(rgb_csv_path);
        else
        {
            const fs::path sequence = fs::absolute(matrix_path).parent_path().parent_path();
            candidates.push_back((sequence / "rgb.csv").string());
            candidates.push_back((sequence / "rgb_raw.csv").string());
        }
        std::string header, used;
        std::vector<std::string> rows;
        for(const std::string& candidate : candidates)
        {
            std::ifstream in(candidate);
            if(!in)
                continue;
            std::string line;
            header.clear();
            rows.clear();
            if(!std::getline(in, header))
                continue;
            while(std::getline(in, line))
                if(!line.empty() && line.find_first_not_of(" \t\r") != std::string::npos)
                    rows.push_back(line);
            if(Eigen::Index(rows.size()) == n) { used = candidate; break; }
            std::fprintf(stderr, "  %s has %zu data rows, the matrix %ld: not the csv this matrix was computed on\n",
                         candidate.c_str(), rows.size(), long(n));
        }
        if(used.empty())
        {
            if(!rgb_csv_path.empty()) { std::fprintf(stderr, "rgb.csv not written: --rgb-csv %s unusable\n", rgb_csv_path.c_str()); ok = false; }
            else std::cout << "  no rgb.csv with " << n << " rows found next to the matrix (pass --rgb-csv); surviving csv not written" << std::endl;
        }
        else
        {
            const fs::path out_path = fs::path(output_dir) / "rgb.csv";
            std::ofstream out(out_path);
            out << header << "\n";
            int written = 0;
            for(Eigen::Index i = 0; i < n; i++)
                if(!cell.is_culled(ids[std::size_t(i)])) { out << rows[std::size_t(i)] << "\n"; written++; }
            if(!out || written != alive) { std::fprintf(stderr, "failed writing %s\n", out_path.c_str()); ok = false; }
            std::cout << "  surviving frames: " << written << " of " << n << " rows of " << used << " -> " << out_path.string() << std::endl;
        }
    }
#ifdef PLACECELL_HAS_VIZ
    placecell::viz::Visualizer::Options viz_options;
    viz_options.kernel.centred = centred;
    placecell::viz::Visualizer visualizer(cell, viz_options);
    visualizer.update(true);
    visualizer.save(output_dir);
#endif
    std::cout << (ok ? "kernel demo OK" : "kernel demo FAILED") << " (dump in " << output_dir << ")" << std::endl;
    return ok ? 0 : 1;
}
