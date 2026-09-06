/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 *
 * viz: placecell's visualization manager (optional module, PLACECELL_WITH_VIZ, OpenCV).
 * Renders what a PlaceCell and its Recorder hold into cv::Mat images the host can blit
 * into its own viewer, or show in standalone windows:
 *
 *   render_kernel                 the similarity kernel as a heatmap (raw or centred;
 *                                 history rows dimmed, protected views ticked)
 *   render_information_history    unexplained information of every query over time,
 *                                 with tau / min-information lines, the host's keyframe
 *                                 insertions and the culls
 *   render_alive_information      unique information of every alive view after the
 *                                 latest cull_keyframes call, against tau
 *
 * Visualizer bundles the three: update() re-renders only when the store or the history
 * changed and at most `max_hz` times per second, shows them when `windows` is on
 * (cv::imshow — call update() from one thread, the host's viewer thread), save() writes
 * PNGs. The core library never depends on this module.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <string>

#include <opencv2/core.hpp>

#include "placecell/placecell.h"

namespace placecell
{
namespace viz
{

struct KernelStyle
{
    bool centred{false};          // render PlaceCell::centred_kernel() instead of the raw kernel
    int colormap{2};              // cv::ColormapTypes (2 = COLORMAP_JET; 16 = VIRIDIS on OpenCV >= 4.1)
    int target_size{768};         // heatmap side in pixels (area-downsampled or nearest-upsampled)
    bool dim_culled{true};        // history rows/columns at 35% brightness
    bool mark_protected{true};    // ticks along the top edge for protected views
    bool colorbar{true};
    bool title{true};
    bool auto_range{true};        // raw: [min,1]; centred: [-1,1]; else use vmin/vmax
    float vmin{0.0f};
    float vmax{1.0f};
};

struct HistoryStyle
{
    int width{1200};
    int height{420};
    std::size_t last_n{0};        // show only the last n queries (0 = all)
    bool show_decisions{true};    // host insertions (Recorder::record_decision)
    bool show_culls{true};
};

struct AliveStyle
{
    int width{1200};
    int height{240};
};

cv::Mat render_kernel(const PlaceCell::Snapshot& snapshot, const KernelStyle& style = {});
cv::Mat render_kernel(const PlaceCell& cell, const KernelStyle& style = {});
cv::Mat render_information_history(const Recorder& recorder, const HistoryStyle& style = {});
cv::Mat render_alive_information(const Recorder& recorder, const AliveStyle& style = {});

class Visualizer
{
public:
    struct Options
    {
        bool windows{true};                     // cv::imshow the three images on update()
        double max_hz{2.0};                     // re-render at most this often (0 = every update)
        std::string window_prefix{"placecell"};
        KernelStyle kernel;
        HistoryStyle history;
        AliveStyle alive;
    };

    explicit Visualizer(const PlaceCell& cell);
    Visualizer(const PlaceCell& cell, Options options);
    ~Visualizer();
    Visualizer(const Visualizer&) = delete;
    Visualizer& operator=(const Visualizer&) = delete;

    // Re-render if the store or the recorded history changed (and the throttle allows);
    // returns true when new images were produced. `force` ignores both checks.
    bool update(bool force = false);
    // kernel.png, information.png, alive.png into `directory` (renders first if needed)
    void save(const std::string& directory);

    const cv::Mat& kernel_image() const { return kernel_; }
    const cv::Mat& information_image() const { return information_; }
    const cv::Mat& alive_image() const { return alive_; }

    Options& options() { return options_; }

private:
    void render();

    const PlaceCell& cell_;
    Options options_;
    std::size_t last_size_{0};
    std::size_t last_queries_{0};
    std::size_t last_culls_{0};
    std::size_t last_decisions_{0};
    bool rendered_{false};
    std::chrono::steady_clock::time_point last_render_{};
    cv::Mat kernel_;
    cv::Mat information_;
    cv::Mat alive_;
    bool windows_open_{false};
};

} // namespace viz
} // namespace placecell
