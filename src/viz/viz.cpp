/**
 * placecell — keyframe lifecycle management for VSLAM and 3D reconstruction.
 *
 * Author:  Alejandro Fontan
 * Assisted by: Claude (Fable 5)
 * Created: 2026-09-05
 * License: Apache-2.0
 */
#include "placecell/viz.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace placecell
{
namespace viz
{

namespace
{
// Dark palette (BGR)
const cv::Scalar kBackground(28, 28, 28);
const cv::Scalar kPanel(38, 38, 38);
const cv::Scalar kGrid(64, 64, 64);
const cv::Scalar kText(220, 220, 220);
const cv::Scalar kTextDim(150, 150, 150);
const cv::Scalar kInformation(255, 210, 60);     // cyan-ish
const cv::Scalar kTau(80, 80, 255);              // red
const cv::Scalar kMinInformation(80, 220, 80);   // green
const cv::Scalar kDecision(0, 220, 255);         // yellow
const cv::Scalar kCull(255, 80, 255);            // magenta
const cv::Scalar kNaN(110, 110, 110);

constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;
constexpr double kFontScale = 0.45;
constexpr int kTitleHeight = 26;

std::string fixed(const double x, const int digits = 2)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(digits) << x;
    return s.str();
}

void put_text(cv::Mat& image, const std::string& text, const cv::Point& at, const cv::Scalar& color = kText,
              const double scale = kFontScale)
{
    cv::putText(image, text, at, kFont, scale, color, 1, cv::LINE_AA);
}

void dashed_line(cv::Mat& image, cv::Point a, cv::Point b, const cv::Scalar& color, const int dash = 6)
{
    const double length = cv::norm(b - a);
    if(length < 1.0)
        return;
    const cv::Point2d dir((b.x - a.x) / length, (b.y - a.y) / length);
    for(double t = 0.0; t < length; t += 2.0 * dash)
    {
        const cv::Point p0(int(a.x + dir.x * t), int(a.y + dir.y * t));
        const double t1 = std::min(length, t + dash);
        const cv::Point p1(int(a.x + dir.x * t1), int(a.y + dir.y * t1));
        cv::line(image, p0, p1, color, 1, cv::LINE_AA);
    }
}

// First query index whose time stamp is >= t (for placing time-stamped events on the
// query axis); the last index when t is beyond every query.
std::size_t query_index_at(const std::vector<Recorder::Query>& queries, const double t)
{
    const auto it = std::lower_bound(queries.begin(), queries.end(), t,
                                     [](const Recorder::Query& q, const double time) { return q.time_s < time; });
    if(it == queries.end())
        return queries.empty() ? 0 : queries.size() - 1;
    return std::size_t(it - queries.begin());
}

struct Axes
{
    cv::Rect plot;      // pixel rectangle of the plotting area
    double x0, x1;      // data range on x
    double y0, y1;      // data range on y
    int x(const double v) const
    {
        return plot.x + int(std::lround((v - x0) / std::max(1e-12, x1 - x0) * (plot.width - 1)));
    }
    int y(const double v) const
    {
        return plot.y + plot.height - 1 - int(std::lround((v - y0) / std::max(1e-12, y1 - y0) * (plot.height - 1)));
    }
};

void draw_axes(cv::Mat& image, const Axes& axes, const std::string& x_label, const std::string& y_label)
{
    cv::rectangle(image, axes.plot, kPanel, cv::FILLED);
    // y grid: 0, 0.25, 0.5, 0.75, 1
    for(int k = 0; k <= 4; k++)
    {
        const double v = axes.y0 + (axes.y1 - axes.y0) * k / 4.0;
        const int y = axes.y(v);
        cv::line(image, cv::Point(axes.plot.x, y), cv::Point(axes.plot.x + axes.plot.width, y), kGrid, 1);
        put_text(image, fixed(v, 2), cv::Point(4, y + 4), kTextDim, 0.4);
    }
    // x ticks: 5
    for(int k = 0; k <= 5; k++)
    {
        const double v = axes.x0 + (axes.x1 - axes.x0) * k / 5.0;
        const int x = axes.x(v);
        cv::line(image, cv::Point(x, axes.plot.y), cv::Point(x, axes.plot.y + axes.plot.height), kGrid, 1);
        put_text(image, std::to_string(long(std::lround(v))), cv::Point(x - 10, axes.plot.y + axes.plot.height + 14),
                 kTextDim, 0.4);
    }
    put_text(image, x_label, cv::Point(axes.plot.x + axes.plot.width - 8 * int(x_label.size()) - 4,
                                       axes.plot.y + axes.plot.height + 28), kTextDim, 0.4);
    // y label: right-aligned above the tick column (the title names the quantity too)
    if(!y_label.empty())
        put_text(image, y_label, cv::Point(axes.plot.x + axes.plot.width - 7 * int(y_label.size()) - 6,
                                           axes.plot.y + axes.plot.height - 6), kTextDim, 0.4);
}

cv::Mat empty_panel(const int width, const int height, const std::string& message)
{
    cv::Mat image(height, width, CV_8UC3, kBackground);
    put_text(image, message, cv::Point(12, height / 2), kTextDim);
    return image;
}
} // namespace

// ---- Kernel heatmap ----------------------------------------------------------------

cv::Mat render_kernel(const PlaceCell::Snapshot& snapshot, const KernelStyle& style)
{
    const int n = int(snapshot.ids.size());
    const int side = std::max(64, style.target_size);
    const int colorbar_width = style.colorbar ? 56 : 0;
    const int title_height = style.title ? kTitleHeight : 0;
    cv::Mat canvas(side + title_height, side + colorbar_width, CV_8UC3, kBackground);
    if(n == 0)
    {
        put_text(canvas, "kernel: empty store", cv::Point(12, title_height + side / 2), kTextDim);
        return canvas;
    }

    // Value range
    float vmin = style.vmin, vmax = style.vmax;
    if(style.auto_range)
    {
        if(snapshot.centred)
        {
            vmin = -1.0f;
            vmax = 1.0f;
        }
        else
        {
            vmin = 1.0f;
            vmax = 1.0f;
            for(int i = 0; i < n; i++)
                for(int j = 0; j < n; j++)
                    if(!std::isnan(snapshot.kernel(i, j)))
                        vmin = std::min(vmin, snapshot.kernel(i, j));
            if(vmax - vmin < 1e-3f)
                vmin = vmax - 1.0f;
        }
    }

    // Normalised 8-bit heat values + weights (dim history rows/cols, grey NaN cells)
    cv::Mat heat(n, n, CV_8UC1);
    cv::Mat weight(n, n, CV_32FC1, cv::Scalar(1.0f));
    cv::Mat nan_mask(n, n, CV_8UC1, cv::Scalar(0));
    int alive = 0;
    for(int i = 0; i < n; i++)
    {
        const bool row_culled = snapshot.culled[std::size_t(i)] != 0;
        if(!row_culled)
            alive++;
        for(int j = 0; j < n; j++)
        {
            const float v = snapshot.kernel(i, j);
            if(std::isnan(v))
            {
                heat.at<unsigned char>(i, j) = 0;
                nan_mask.at<unsigned char>(i, j) = 255;
            }
            else
            {
                const float t = std::min(1.0f, std::max(0.0f, (v - vmin) / (vmax - vmin)));
                heat.at<unsigned char>(i, j) = static_cast<unsigned char>(std::lround(t * 255.0f));
            }
            if(style.dim_culled && (row_culled || snapshot.culled[std::size_t(j)] != 0))
                weight.at<float>(i, j) = 0.35f;
        }
    }
    cv::Mat colored;
    cv::applyColorMap(heat, colored, style.colormap);
    colored.setTo(kNaN, nan_mask);

    // Resize: area when shrinking (n > side), nearest when enlarging (crisp cells)
    const int interpolation = n > side ? cv::INTER_AREA : cv::INTER_NEAREST;
    cv::Mat colored_resized, weight_resized;
    cv::resize(colored, colored_resized, cv::Size(side, side), 0, 0, interpolation);
    cv::resize(weight, weight_resized, cv::Size(side, side), 0, 0, interpolation);
    cv::Mat colored_float;
    colored_resized.convertTo(colored_float, CV_32FC3);
    cv::Mat weight3;
    cv::merge(std::vector<cv::Mat>{weight_resized, weight_resized, weight_resized}, weight3);
    cv::multiply(colored_float, weight3, colored_float);
    cv::Mat heatmap;
    colored_float.convertTo(heatmap, CV_8UC3);
    heatmap.copyTo(canvas(cv::Rect(0, title_height, side, side)));

    // Protected ticks along the top edge
    if(style.mark_protected)
    {
        const double scale = double(side) / n;
        for(int i = 0; i < n; i++)
        {
            if(snapshot.protected_views[std::size_t(i)] == 0)
                continue;
            const int x = int((i + 0.5) * scale);
            cv::line(canvas, cv::Point(x, title_height), cv::Point(x, title_height + 8), cv::Scalar(255, 255, 255), 2);
        }
    }

    // Colourbar
    if(style.colorbar)
    {
        const int x0 = side + 12, width = 14;
        cv::Mat ramp(side, 1, CV_8UC1);
        for(int y = 0; y < side; y++)
            ramp.at<unsigned char>(y, 0) = static_cast<unsigned char>(255 - std::lround(255.0 * y / (side - 1)));
        cv::Mat ramp_colored;
        cv::applyColorMap(ramp, ramp_colored, style.colormap);
        cv::resize(ramp_colored, ramp_colored, cv::Size(width, side), 0, 0, cv::INTER_NEAREST);
        ramp_colored.copyTo(canvas(cv::Rect(x0, title_height, width, side)));
        put_text(canvas, fixed(vmax), cv::Point(x0 + width + 3, title_height + 12), kText, 0.38);
        put_text(canvas, fixed(vmin), cv::Point(x0 + width + 3, title_height + side - 4), kText, 0.38);
    }

    if(style.title)
    {
        std::ostringstream title;
        title << "kernel (" << (snapshot.centred ? "centred" : "raw") << ")  n=" << n << "  alive=" << alive
              << "  history=" << (n - alive);
        put_text(canvas, title.str(), cv::Point(8, 18));
    }
    return canvas;
}

cv::Mat render_kernel(const PlaceCell& cell, const KernelStyle& style)
{
    return render_kernel(cell.snapshot(style.centred), style);
}

// ---- Unexplained-information history ------------------------------------------------

cv::Mat render_information_history(const Recorder& recorder, const HistoryStyle& style)
{
    const std::vector<Recorder::Query> queries = recorder.queries();
    if(queries.empty())
        return empty_panel(style.width, style.height, "unexplained information: no queries yet");

    cv::Mat image(style.height, style.width, CV_8UC3, kBackground);
    const std::size_t first = (style.last_n > 0 && queries.size() > style.last_n) ? queries.size() - style.last_n : 0;
    Axes axes;
    axes.plot = cv::Rect(46, kTitleHeight + 6, style.width - 46 - 12, style.height - kTitleHeight - 6 - 34);
    axes.x0 = double(first);
    axes.x1 = double(std::max(first + 1, queries.size() - 1));
    axes.y0 = 0.0;
    axes.y1 = 1.0;
    draw_axes(image, axes, "query", "unexplained information");

    // Thresholds as step lines (history mapped onto the query axis by time)
    const std::vector<Recorder::Thresholds> thresholds = recorder.threshold_history();
    auto draw_threshold_steps = [&](const bool tau_not_min, const cv::Scalar& color)
    {
        for(std::size_t k = 0; k < thresholds.size(); k++)
        {
            const float value = tau_not_min ? thresholds[k].tau : thresholds[k].min_information;
            if(std::isnan(value))
                continue;
            const std::size_t from = std::max(first, k == 0 ? first : query_index_at(queries, thresholds[k].time_s));
            const std::size_t to = k + 1 < thresholds.size()
                                       ? query_index_at(queries, thresholds[k + 1].time_s)
                                       : queries.size() - 1;
            if(to < from)
                continue;
            dashed_line(image, cv::Point(axes.x(double(from)), axes.y(value)),
                        cv::Point(axes.x(double(to)), axes.y(value)), color);
        }
    };
    draw_threshold_steps(true, kTau);
    draw_threshold_steps(false, kMinInformation);

    // The information series
    std::vector<cv::Point> polyline;
    polyline.reserve(queries.size() - first);
    for(std::size_t i = first; i < queries.size(); i++)
    {
        if(std::isnan(queries[i].unexplained))
        {
            if(!polyline.empty())
                cv::polylines(image, polyline, false, kInformation, 1, cv::LINE_AA);
            polyline.clear();
            continue;
        }
        polyline.emplace_back(axes.x(double(i)), axes.y(queries[i].unexplained));
    }
    if(polyline.size() >= 2)
        cv::polylines(image, polyline, false, kInformation, 1, cv::LINE_AA);
    else if(polyline.size() == 1)
        cv::circle(image, polyline.front(), 2, kInformation, cv::FILLED);

    // Host insertions: triangles at the query they were decided on
    std::size_t insertions = 0;
    if(style.show_decisions)
    {
        for(const Recorder::Decision& decision : recorder.decisions())
        {
            if(!decision.inserted || decision.query_index < first)
                continue;
            insertions++;
            const std::size_t qi = std::min(decision.query_index, queries.size() - 1);
            const float v = std::isnan(decision.unexplained) ? queries[qi].unexplained : decision.unexplained;
            const cv::Point p(axes.x(double(decision.query_index)), axes.y(std::isnan(v) ? 0.0f : v));
            const std::vector<cv::Point> triangle{p + cv::Point(0, -7), p + cv::Point(-5, 2), p + cv::Point(5, 2)};
            cv::fillConvexPoly(image, triangle, kDecision, cv::LINE_AA);
        }
    }

    // Culls: tick at the bottom + dot at the culled view's unique information
    std::size_t culls = 0;
    if(style.show_culls)
    {
        for(const Recorder::Cull& cull : recorder.culls())
        {
            const std::size_t qi = query_index_at(queries, cull.time_s);
            if(qi < first)
                continue;
            culls++;
            const int x = axes.x(double(qi));
            cv::line(image, cv::Point(x, axes.plot.y + axes.plot.height - 10), cv::Point(x, axes.plot.y + axes.plot.height),
                     kCull, 1);
            cv::circle(image, cv::Point(x, axes.y(cull.unique_information)), 2, kCull, cv::FILLED, cv::LINE_AA);
        }
    }

    // Title + legend
    const Recorder::Thresholds latest = recorder.thresholds();
    std::ostringstream title;
    title << "unexplained information  queries=" << queries.size() << "  last=" << fixed(queries.back().unexplained, 3)
          << "  explainers=" << queries.back().explainers;
    if(!std::isnan(latest.tau))
        title << "  tau=" << fixed(latest.tau);
    if(!std::isnan(latest.min_information))
        title << "  min=" << fixed(latest.min_information);
    put_text(image, title.str(), cv::Point(8, 18));
    int lx = axes.plot.x + 8, ly = axes.plot.y + 14;
    auto legend = [&](const cv::Scalar& color, const std::string& text)
    {
        cv::line(image, cv::Point(lx, ly - 4), cv::Point(lx + 14, ly - 4), color, 2);
        put_text(image, text, cv::Point(lx + 18, ly), kTextDim, 0.4);
        lx += 22 + 7 * int(text.size());
    };
    legend(kInformation, "v (query)");
    if(!std::isnan(latest.tau)) legend(kTau, "tau");
    if(!std::isnan(latest.min_information)) legend(kMinInformation, "min info");
    if(style.show_decisions) legend(kDecision, "inserted (" + std::to_string(insertions) + ")");
    if(style.show_culls) legend(kCull, "culled (" + std::to_string(culls) + ")");
    return image;
}

// ---- Alive unique information ---------------------------------------------------------

cv::Mat render_alive_information(const Recorder& recorder, const AliveStyle& style)
{
    const std::optional<Recorder::CullCall> call = recorder.latest_cull_call();
    if(!call || call->alive_ids.empty())
        return empty_panel(style.width, style.height, "alive information: no cull_keyframes call yet");

    cv::Mat image(style.height, style.width, CV_8UC3, kBackground);
    const int n = int(call->alive_ids.size());
    Axes axes;
    axes.plot = cv::Rect(46, kTitleHeight + 6, style.width - 46 - 12, style.height - kTitleHeight - 6 - 34);
    axes.x0 = 0.0;
    axes.x1 = double(n);
    axes.y0 = 0.0;
    axes.y1 = 1.0;
    draw_axes(image, axes, "alive view (insertion order)", "unique information");

    const double bar = double(axes.plot.width) / n;
    int over = 0;
    for(int i = 0; i < n; i++)
    {
        const float v = call->alive_unique_information[std::size_t(i)];
        const int x0 = axes.plot.x + int(i * bar);
        const int x1 = std::max(x0 + 1, axes.plot.x + int((i + 1) * bar) - (bar > 3 ? 1 : 0));
        if(std::isnan(v))
        {
            cv::rectangle(image, cv::Point(x0, axes.plot.y), cv::Point(x1, axes.plot.y + axes.plot.height), kNaN, cv::FILLED);
            continue;
        }
        const bool above = v > call->tau;
        over += above;
        cv::rectangle(image, cv::Point(x0, axes.y(std::min(1.0f, v))), cv::Point(x1, axes.plot.y + axes.plot.height - 1),
                      above ? kInformation : kTextDim, cv::FILLED);
    }
    dashed_line(image, cv::Point(axes.plot.x, axes.y(call->tau)), cv::Point(axes.plot.x + axes.plot.width, axes.y(call->tau)), kTau);

    std::ostringstream title;
    title << "unique information of alive views after cull #" << call->index << "  alive=" << n << "  above tau=" << over
          << "  tau=" << fixed(call->tau) << "  culled=" << call->culled << "/" << call->candidates
          << (call->local ? "  [local]" : "  [map]") << (call->centred ? " centred" : " raw");
    put_text(image, title.str(), cv::Point(8, 18));
    return image;
}

// ---- Visualizer ----------------------------------------------------------------------

Visualizer::Visualizer(const PlaceCell& cell) : Visualizer(cell, Options{}) {}

Visualizer::Visualizer(const PlaceCell& cell, Options options) : cell_(cell), options_(std::move(options)) {}

Visualizer::~Visualizer()
{
    if(windows_open_)
    {
        cv::destroyWindow(options_.window_prefix + ": kernel");
        cv::destroyWindow(options_.window_prefix + ": information");
        cv::destroyWindow(options_.window_prefix + ": alive");
    }
}

void Visualizer::render()
{
    kernel_ = render_kernel(cell_, options_.kernel);
    information_ = render_information_history(cell_.recorder(), options_.history);
    alive_ = render_alive_information(cell_.recorder(), options_.alive);
    rendered_ = true;
    last_render_ = std::chrono::steady_clock::now();
}

bool Visualizer::update(const bool force)
{
    const Recorder& recorder = cell_.recorder();
    const std::size_t size = cell_.size();
    const std::size_t queries = recorder.query_count();
    const std::size_t culls = recorder.cull_count();
    const std::size_t decisions = recorder.decisions().size();
    const bool changed = size != last_size_ || queries != last_queries_ || culls != last_culls_ || decisions != last_decisions_;
    if(!force)
    {
        if(rendered_ && !changed)
            return false;
        if(rendered_ && options_.max_hz > 0.0)
        {
            const double since = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_render_).count();
            if(since < 1.0 / options_.max_hz)
                return false;
        }
    }
    last_size_ = size;
    last_queries_ = queries;
    last_culls_ = culls;
    last_decisions_ = decisions;
    render();
    if(options_.windows)
    {
        cv::imshow(options_.window_prefix + ": kernel", kernel_);
        cv::imshow(options_.window_prefix + ": information", information_);
        cv::imshow(options_.window_prefix + ": alive", alive_);
        cv::waitKey(1);
        windows_open_ = true;
    }
    return true;
}

void Visualizer::save(const std::string& directory)
{
    if(!rendered_)
        render();
    std::filesystem::create_directories(directory);
    const std::filesystem::path dir(directory);
    cv::imwrite((dir / "kernel.png").string(), kernel_);
    cv::imwrite((dir / "information.png").string(), information_);
    cv::imwrite((dir / "alive.png").string(), alive_);
    PLACECELL_INFO("viz", "kernel.png, information.png, alive.png written to " << directory);
}

} // namespace viz
} // namespace placecell
