#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "hyphy/core/types.hpp"
#include "hyphy/core/console.hpp"

namespace hyphy::analyses {

using namespace hyphy::core;

/**
 * @brief Selection classification category for a codon site.
 */
enum class SiteSelectionType : uint8_t {
    Pending = 0,     // Not yet evaluated
    Neutral = 1,     // Evaluated, no statistically significant selection detected
    Positive = 2,    // Positive / Diversifying / Episodic (beta > alpha, beta+ > alpha, dN > dS)
    Negative = 3,    // Negative / Purifying / Conserved (alpha > beta, dS > dN)
    Significant = 4  // General significant contrast / property shift (e.g. Contrast-FEL, PRIME omnibus)
};

/**
 * @brief Configuration settings for SiteVisualizer.
 */
struct SiteVisualizerConfig {
    std::string method_name = "Site Analysis";
    Scalar p_value_threshold = 0.1;
    std::string pos_label = "positive";
    std::string neg_label = "negative";
    std::string sig_label = "significant";
    bool quiet = false;
    bool force_tty = false;
    bool enable_chart = true;
};

/**
 * @brief Top hit record for significant site discoveries.
 */
struct SiteTopHit {
    size_t site_index = 0;
    Scalar p_value = 1.0;
    SiteSelectionType type = SiteSelectionType::Neutral;
    std::string detail;
};

/**
 * @brief Real-time responsive visualizer and dynamic terminal dashboard for site-level selection analyses.
 *
 * Provides:
 * - 1:1 or binned proportional alignment coordinate ruler spanning codon sites (1 .. N).
 * - Real-time alignment selection track showing evaluated sites:
 *     ▲ Positive / Diversifying selection (bright green / orange)
 *     ▼ Negative / Purifying selection (cyan / blue)
 *     ★ / ◆ Significant contrast / omnibus (accent / amber)
 *     ▪ Evaluated neutral sites (muted gray)
 *     · Pending sites (dim gray)
 * - Live statistical evidence sparkline (-log10 p-value) across alignment bins (  ▂▃▄▅▆▇█).
 * - Live status line with completion counts, percentage, elapsed time, processing speed (sites/s), and ETA.
 * - Live discovery ticker and top hits display showcasing the most significant site findings.
 * - Smooth in-place ANSI terminal updates (~25 FPS) with lock-free atomic counters for OpenMP parallel loops.
 * - Clean, non-destructive plain-text fallback for non-TTY / cluster log environments.
 */
class SiteVisualizer {
public:
    size_t num_sites = 0;
    size_t num_patterns = 0;
    SiteVisualizerConfig config;

    bool is_tty = false;
    int chart_width = 65;

    // Per-site state tracking
    std::vector<std::atomic<uint8_t>> site_states;
    // Stored as -log10(p) * 1000 for lock-free atomic updates
    std::vector<std::atomic<uint32_t>> site_neg_log_p_milli;

    // Aggregate counters
    std::atomic<size_t> sites_completed{0};
    std::atomic<size_t> patterns_completed{0};
    std::atomic<size_t> count_positive{0};
    std::atomic<size_t> count_negative{0};
    std::atomic<size_t> count_significant{0};

    // Time tracking
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_render_time;
    std::chrono::steady_clock::time_point last_log_time;

    // Ticker & top hits
    mutable std::mutex hits_mutex;
    std::string latest_hit_text;
    std::vector<SiteTopHit> top_hits;

    // Rendering
    std::mutex render_mutex;
    int lines_rendered = 0;

    static constexpr const char* kSparklineGlyphs[9] = {
        " ", " ", "▂", "▃", "▄", "▅", "▆", "▇", "█"
    };

    SiteVisualizer(size_t total_sites, size_t total_patterns, SiteVisualizerConfig cfg = SiteVisualizerConfig{})
        : num_sites(total_sites), num_patterns(total_patterns), config(std::move(cfg)) {
        is_tty = config.force_tty || core::Console::is_terminal();

        int term_w = core::Console::terminal_width(80);
        int target_w = std::clamp(term_w - 14, 30, 80);
        if (num_sites > 0 && num_sites < static_cast<size_t>(target_w)) {
            chart_width = static_cast<int>(num_sites);
        } else {
            chart_width = target_w;
        }

        site_states = std::vector<std::atomic<uint8_t>>(num_sites);
        for (auto& s : site_states) s.store(static_cast<uint8_t>(SiteSelectionType::Pending));

        site_neg_log_p_milli = std::vector<std::atomic<uint32_t>>(num_sites);
        for (auto& p : site_neg_log_p_milli) p.store(0);

        start_time = std::chrono::steady_clock::now();
        last_render_time = start_time;
        last_log_time = start_time;

        if (!config.quiet) {
            if (is_tty && config.enable_chart) {
                try_render(true);
            } else if (!is_tty) {
                std::cout << core::Console::bold("[" + config.method_name + "] ")
                          << "Analyzing " << num_sites << " codon sites ("
                          << num_patterns << " unique patterns)...\n" << std::flush;
            }
        }
    }

    /**
     * @brief Record result for a single codon site.
     */
    void record_site(
        size_t site_idx,
        SiteSelectionType type,
        Scalar p_value,
        Scalar metric_value = 0.0,
        const std::string& detail_text = ""
    ) {
        if (site_idx >= num_sites) return;
        (void)metric_value;

        site_states[site_idx].store(static_cast<uint8_t>(type), std::memory_order_relaxed);

        Scalar neg_log_p = 0.0;
        if (p_value > 0.0) {
            neg_log_p = -std::log10(std::clamp(p_value, 1e-15, 1.0));
        } else {
            neg_log_p = 15.0;
        }
        uint32_t milli_p = static_cast<uint32_t>(std::clamp(neg_log_p * 1000.0, 0.0, 15000.0));
        site_neg_log_p_milli[site_idx].store(milli_p, std::memory_order_relaxed);

        if (type == SiteSelectionType::Positive) {
            count_positive.fetch_add(1, std::memory_order_relaxed);
        } else if (type == SiteSelectionType::Negative) {
            count_negative.fetch_add(1, std::memory_order_relaxed);
        } else if (type == SiteSelectionType::Significant) {
            count_significant.fetch_add(1, std::memory_order_relaxed);
        }

        if (type != SiteSelectionType::Pending && type != SiteSelectionType::Neutral) {
            std::lock_guard<std::mutex> lock(hits_mutex);
            std::ostringstream ss;
            ss << "Site " << (site_idx + 1) << " (";
            if (!detail_text.empty()) {
                ss << detail_text;
            } else {
                if (type == SiteSelectionType::Positive) ss << "▲ " << config.pos_label;
                else if (type == SiteSelectionType::Negative) ss << "▼ " << config.neg_label;
                else ss << "★ " << config.sig_label;
                ss << ", p=" << format_pvalue(p_value);
            }
            ss << ")";
            latest_hit_text = ss.str();

            // Insert into top_hits sorted by ascending p_value (strongest first)
            SiteTopHit hit{site_idx, p_value, type, latest_hit_text};
            auto it = std::lower_bound(top_hits.begin(), top_hits.end(), hit,
                [](const SiteTopHit& a, const SiteTopHit& b) { return a.p_value < b.p_value; });
            top_hits.insert(it, hit);
            if (top_hits.size() > 4) {
                top_hits.pop_back();
            }
        }

        sites_completed.fetch_add(1, std::memory_order_relaxed);
        tick_progress();
    }

    /**
     * @brief Record result for a compressed pattern and all associated sites.
     */
    void record_pattern(
        size_t pattern_idx,
        const std::vector<size_t>& sites,
        SiteSelectionType type,
        Scalar p_value,
        Scalar metric_value = 0.0,
        const std::string& detail_text = ""
    ) {
        (void)pattern_idx;
        (void)metric_value;
        Scalar neg_log_p = 0.0;
        if (p_value > 0.0) {
            neg_log_p = -std::log10(std::clamp(p_value, 1e-15, 1.0));
        } else {
            neg_log_p = 15.0;
        }
        uint32_t milli_p = static_cast<uint32_t>(std::clamp(neg_log_p * 1000.0, 0.0, 15000.0));

        size_t n_sites = sites.size();
        for (size_t s : sites) {
            if (s < num_sites) {
                site_states[s].store(static_cast<uint8_t>(type), std::memory_order_relaxed);
                site_neg_log_p_milli[s].store(milli_p, std::memory_order_relaxed);
            }
        }

        if (type == SiteSelectionType::Positive) {
            count_positive.fetch_add(n_sites, std::memory_order_relaxed);
        } else if (type == SiteSelectionType::Negative) {
            count_negative.fetch_add(n_sites, std::memory_order_relaxed);
        } else if (type == SiteSelectionType::Significant) {
            count_significant.fetch_add(n_sites, std::memory_order_relaxed);
        }

        if (type != SiteSelectionType::Pending && type != SiteSelectionType::Neutral && !sites.empty()) {
            std::lock_guard<std::mutex> lock(hits_mutex);
            std::ostringstream ss;
            ss << "Site " << (sites[0] + 1);
            if (sites.size() > 1) {
                ss << "(+" << (sites.size() - 1) << ")";
            }
            ss << " (";
            if (!detail_text.empty()) {
                ss << detail_text;
            } else {
                if (type == SiteSelectionType::Positive) ss << "▲ " << config.pos_label;
                else if (type == SiteSelectionType::Negative) ss << "▼ " << config.neg_label;
                else ss << "★ " << config.sig_label;
                ss << ", p=" << format_pvalue(p_value);
            }
            ss << ")";
            latest_hit_text = ss.str();

            for (size_t s : sites) {
                SiteTopHit hit{s, p_value, type, "Site " + std::to_string(s + 1) + " (" + (!detail_text.empty() ? detail_text : ("p=" + format_pvalue(p_value))) + ")"};
                auto it = std::lower_bound(top_hits.begin(), top_hits.end(), hit,
                    [](const SiteTopHit& a, const SiteTopHit& b) { return a.p_value < b.p_value; });
                top_hits.insert(it, hit);
                if (top_hits.size() > 4) {
                    top_hits.pop_back();
                }
            }
        }

        sites_completed.fetch_add(n_sites, std::memory_order_relaxed);
        patterns_completed.fetch_add(1, std::memory_order_relaxed);
        tick_progress();
    }

    /**
     * @brief Finish visualizer and render final status line.
     */
    void finish() {
        if (config.quiet) return;

        if (is_tty && config.enable_chart) {
            force_render_final();
            lines_rendered = 0;
        }

        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
        size_t total_done = sites_completed.load();
        double speed = (elapsed_sec > 1e-3) ? (total_done / elapsed_sec) : 0.0;

        size_t pos = count_positive.load();
        size_t neg = count_negative.load();
        size_t sig = count_significant.load();

        std::ostringstream ss;
        ss << core::Console::success("  ✔ [" + config.method_name + "] ")
           << core::Console::bold("Completed " + std::to_string(total_done) + " sites")
           << " in " << core::Console::bold(format_duration(elapsed_sec))
           << " (" << std::fixed << std::setprecision(1) << speed << " sites/s) — ";

        bool has_tally = false;
        if (pos > 0) {
            ss << core::Console::bold(core::Console::success(std::to_string(pos) + " " + config.pos_label));
            has_tally = true;
        }
        if (neg > 0) {
            if (has_tally) ss << ", ";
            ss << core::Console::bold(core::Console::info(std::to_string(neg) + " " + config.neg_label));
            has_tally = true;
        }
        if (sig > 0) {
            if (has_tally) ss << ", ";
            ss << core::Console::bold(core::Console::brand(std::to_string(sig) + " " + config.sig_label));
            has_tally = true;
        }
        if (!has_tally) {
            ss << core::Console::muted("0 selected sites");
        }
        ss << core::Console::muted(" (p ≤ " + format_scalar(config.p_value_threshold, 2) + ")\n\n");

        std::cout << ss.str() << std::flush;
    }

private:
    void tick_progress() {
        if (config.quiet) return;

        if (!is_tty) {
            auto now = std::chrono::steady_clock::now();
            auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - last_log_time).count();
            if (sec >= 5) {
                last_log_time = now;
                print_non_tty_status();
            }
        } else if (config.enable_chart) {
            try_render(false);
        }
    }

    void try_render(bool force = false) {
        if (config.quiet || !config.enable_chart || !is_tty) return;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_render_time).count();
        if (!force && elapsed_ms < 40) {
            return;
        }

        if (!render_mutex.try_lock()) {
            return;
        }

        last_render_time = now;
        render_in_place();
        render_mutex.unlock();
    }

    void force_render_final() {
        std::lock_guard<std::mutex> lock(render_mutex);
        render_in_place();
        std::cout << "\n" << std::flush;
    }

    std::vector<std::string> format_dashboard_lines() const {
        size_t done_sites = sites_completed.load(std::memory_order_relaxed);
        size_t done_pats = patterns_completed.load(std::memory_order_relaxed);
        size_t pos_sites = count_positive.load(std::memory_order_relaxed);
        size_t neg_sites = count_negative.load(std::memory_order_relaxed);
        size_t sig_sites = count_significant.load(std::memory_order_relaxed);

        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
        double speed = (elapsed_sec > 1e-3) ? (done_sites / elapsed_sec) : 0.0;

        double eta_sec = 0.0;
        if (speed > 1e-3 && done_sites < num_sites) {
            eta_sec = (num_sites - done_sites) / speed;
        }

        int term_w = core::Console::terminal_width(80);
        int chart_w = chart_width;

        // --- Line 0: Status & Speed Line ---
        std::ostringstream status_ss;
        double pct = (num_sites > 0) ? (100.0 * done_sites / num_sites) : 0.0;
        status_ss << core::Console::accent("◆ [" + config.method_name + "] ");
        if (num_patterns > 0 && num_patterns < num_sites) {
            status_ss << core::Console::bold(std::to_string(done_pats) + "/" + std::to_string(num_patterns) + " pat")
                      << " (" << std::to_string(done_sites) + "/" + std::to_string(num_sites) + " sites, "
                      << std::fixed << std::setprecision(1) << pct << "%) ";
        } else {
            status_ss << core::Console::bold(std::to_string(done_sites) + "/" + std::to_string(num_sites))
                      << " (" << std::fixed << std::setprecision(1) << pct << "%) ";
        }
        status_ss << core::Console::muted("[" + format_duration(elapsed_sec) + ", " +
                     format_scalar(speed, 0) + " sites/s");
        if (done_sites < num_sites && eta_sec > 0.0) {
            status_ss << core::Console::muted(", ETA: " + format_duration(eta_sec));
        }
        status_ss << core::Console::muted("]");

        // Append live selection tallies
        status_ss << " • ";
        bool has_tallies = false;
        if (pos_sites > 0) {
            status_ss << core::Console::bold(core::Console::success("▲ " + std::to_string(pos_sites) + " " + config.pos_label));
            has_tallies = true;
        }
        if (neg_sites > 0) {
            if (has_tallies) status_ss << " | ";
            status_ss << core::Console::bold(core::Console::info("▼ " + std::to_string(neg_sites) + " " + config.neg_label));
            has_tallies = true;
        }
        if (sig_sites > 0) {
            if (has_tallies) status_ss << " | ";
            status_ss << core::Console::bold(core::Console::brand("★ " + std::to_string(sig_sites) + " " + config.sig_label));
            has_tallies = true;
        }
        if (!has_tallies) {
            status_ss << core::Console::muted("scanning sites");
        }

        // --- Line 1: Ruler Coordinates ---
        std::vector<std::string> ruler_chars(chart_w, " ");
        ruler_chars[0] = core::Console::muted("1");

        std::string end_str = std::to_string(num_sites);
        int end_start = chart_w - static_cast<int>(end_str.length());
        for (int i = 0; i < static_cast<int>(end_str.length()); ++i) {
            if (end_start + i < chart_w && end_start + i >= 0) {
                ruler_chars[end_start + i] = core::Console::muted(std::string(1, end_str[i]));
            }
        }

        // Place midpoint tick / label if wide enough
        if (chart_w >= 40 && num_sites >= 10) {
            size_t mid_site = num_sites / 2;
            std::string mid_str = std::to_string(mid_site);
            int mid_col = chart_w / 2 - static_cast<int>(mid_str.length()) / 2;
            if (mid_col > 3 && mid_col + static_cast<int>(mid_str.length()) < end_start - 3) {
                for (size_t i = 0; i < mid_str.length(); ++i) {
                    ruler_chars[mid_col + i] = core::Console::muted(std::string(1, mid_str[i]));
                }
            }
        }

        // Subtle quarter ticks if space allows
        if (chart_w >= 60) {
            int q1_col = chart_w / 4;
            int q3_col = (chart_w * 3) / 4;
            if (ruler_chars[q1_col] == " ") ruler_chars[q1_col] = core::Console::muted("·");
            if (ruler_chars[q3_col] == " ") ruler_chars[q3_col] = core::Console::muted("·");
        }

        std::ostringstream ruler_ss;
        ruler_ss << core::Console::muted("Ruler:   ");
        for (const auto& ch : ruler_chars) ruler_ss << ch;

        // --- Line 2: Alignment Selection Track ---
        std::ostringstream track_ss;
        track_ss << core::Console::border("Track:   ") << core::Console::border("├");

        // Map sites into chart_w - 2 interior columns
        int track_inner_w = std::max(1, chart_w - 2);
        for (int col = 0; col < track_inner_w; ++col) {
            size_t s_start = static_cast<size_t>(std::floor(static_cast<double>(col) * num_sites / track_inner_w));
            size_t s_end = static_cast<size_t>(std::floor(static_cast<double>(col + 1) * num_sites / track_inner_w));
            if (s_end <= s_start) s_end = s_start + 1;
            if (s_end > num_sites) s_end = num_sites;

            bool any_pos = false;
            bool any_neg = false;
            bool any_sig = false;
            bool any_neutral = false;
            bool all_pending = true;

            for (size_t s = s_start; s < s_end; ++s) {
                uint8_t st = site_states[s].load(std::memory_order_relaxed);
                if (st != static_cast<uint8_t>(SiteSelectionType::Pending)) {
                    all_pending = false;
                }
                if (st == static_cast<uint8_t>(SiteSelectionType::Positive)) any_pos = true;
                else if (st == static_cast<uint8_t>(SiteSelectionType::Negative)) any_neg = true;
                else if (st == static_cast<uint8_t>(SiteSelectionType::Significant)) any_sig = true;
                else if (st == static_cast<uint8_t>(SiteSelectionType::Neutral)) any_neutral = true;
            }

            if (any_pos) {
                track_ss << core::Console::bold(core::Console::success("▲"));
            } else if (any_neg) {
                track_ss << core::Console::bold(core::Console::info("▼"));
            } else if (any_sig) {
                track_ss << core::Console::bold(core::Console::brand("★"));
            } else if (all_pending) {
                track_ss << core::Console::muted("·");
            } else if (any_neutral) {
                track_ss << core::Console::muted("▪");
            } else {
                track_ss << core::Console::muted("▫");
            }
        }
        track_ss << core::Console::border("┤");

        // --- Line 3: Statistical Evidence Sparkline (-log10 p-value) ---
        std::ostringstream spark_ss;
        spark_ss << core::Console::info("P-val:   ");

        for (int col = 0; col < chart_w; ++col) {
            size_t s_start = static_cast<size_t>(std::floor(static_cast<double>(col) * num_sites / chart_w));
            size_t s_end = static_cast<size_t>(std::floor(static_cast<double>(col + 1) * num_sites / chart_w));
            if (s_end <= s_start) s_end = s_start + 1;
            if (s_end > num_sites) s_end = num_sites;

            uint32_t max_milli = 0;
            bool evaluated = false;
            for (size_t s = s_start; s < s_end; ++s) {
                uint8_t st = site_states[s].load(std::memory_order_relaxed);
                if (st != static_cast<uint8_t>(SiteSelectionType::Pending)) {
                    evaluated = true;
                    uint32_t val = site_neg_log_p_milli[s].load(std::memory_order_relaxed);
                    if (val > max_milli) max_milli = val;
                }
            }

            if (!evaluated) {
                spark_ss << core::Console::muted("·");
            } else {
                Scalar neg_log_p = max_milli / 1000.0;
                int lvl = 0;
                if (neg_log_p < 0.301) {       // p > 0.50
                    lvl = 0;
                } else if (neg_log_p < 0.699) { // 0.20 < p <= 0.50
                    lvl = 1;
                } else if (neg_log_p < 1.0) {   // 0.10 < p <= 0.20
                    lvl = 2;
                } else if (neg_log_p < 1.301) { // 0.05 < p <= 0.10 (Marginal)
                    lvl = 3;
                } else if (neg_log_p < 2.0) {   // 0.01 < p <= 0.05 (Significant)
                    lvl = 4;
                } else if (neg_log_p < 3.0) {   // 0.001 < p <= 0.01
                    lvl = 5;
                } else if (neg_log_p < 4.0) {   // 1e-4 < p <= 0.001
                    lvl = 6;
                } else if (neg_log_p < 5.0) {   // 1e-5 < p <= 1e-4
                    lvl = 7;
                } else {                        // p <= 1e-5
                    lvl = 8;
                }

                if (lvl == 0) {
                    spark_ss << core::Console::muted(" ");
                } else {
                    std::string g = kSparklineGlyphs[lvl];
                    if (lvl <= 2) {
                        spark_ss << core::Console::muted(g);
                    } else if (lvl <= 3) {
                        spark_ss << core::Console::info(g);
                    } else if (lvl <= 5) {
                        spark_ss << core::Console::brand(g);
                    } else if (lvl <= 7) {
                        spark_ss << core::Console::warning(g);
                    } else {
                        spark_ss << core::Console::bold(core::Console::success(g));
                    }
                }
            }
        }

        // --- Line 4: Live Discovery Ticker / Top Hits ---
        std::ostringstream ticker_ss;
        ticker_ss << core::Console::bold("Latest:  ");
        {
            std::lock_guard<std::mutex> lock(hits_mutex);
            if (!latest_hit_text.empty()) {
                ticker_ss << latest_hit_text;
                if (!top_hits.empty() && top_hits.size() > 1) {
                    ticker_ss << core::Console::muted(" • Top: ")
                              << "Site " << (top_hits[0].site_index + 1)
                              << " (p=" << format_pvalue(top_hits[0].p_value) << ")";
                }
            } else {
                ticker_ss << core::Console::muted("Testing codon sites across phylogeny...");
            }
        }

        // Truncate ticker to terminal width if necessary
        std::string ticker_str = ticker_ss.str();
        int max_line_len = std::max(60, term_w - 2);
        (void)max_line_len;

        return {
            status_ss.str(),
            ruler_ss.str(),
            track_ss.str(),
            spark_ss.str(),
            ticker_str
        };
    }

    void render_in_place() {
        auto lines = format_dashboard_lines();
        if (lines_rendered > 0) {
            std::cout << "\033[" << lines_rendered << "A";
        }
        for (const auto& line : lines) {
            std::cout << "\r\033[K" << line << "\n";
        }
        std::cout << std::flush;
        lines_rendered = static_cast<int>(lines.size());
    }

    void print_non_tty_status() {
        size_t done_sites = sites_completed.load();
        double pct = (num_sites > 0) ? (100.0 * done_sites / num_sites) : 0.0;
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - start_time).count();
        double speed = (elapsed_sec > 1e-3) ? (done_sites / elapsed_sec) : 0.0;

        size_t pos = count_positive.load();
        size_t neg = count_negative.load();
        size_t sig = count_significant.load();

        std::ostringstream ss;
        ss << "  [" << config.method_name << "] "
           << done_sites << "/" << num_sites << " sites (" << std::fixed << std::setprecision(1) << pct << "%) "
           << "[" << format_duration(elapsed_sec) << ", " << std::fixed << std::setprecision(0) << speed << " sites/s] | ";
        if (pos > 0) ss << pos << " " << config.pos_label << ", ";
        if (neg > 0) ss << neg << " " << config.neg_label << ", ";
        if (sig > 0) ss << sig << " " << config.sig_label << ", ";
        ss << "(p <= " << std::fixed << std::setprecision(2) << config.p_value_threshold << ")";

        {
            std::lock_guard<std::mutex> lock(hits_mutex);
            if (!latest_hit_text.empty()) {
                ss << " | Latest: " << latest_hit_text;
            }
        }

        std::cout << ss.str() << "\n" << std::flush;
    }

public:
    static std::string format_scalar(Scalar val, int precision = 2) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << val;
        return ss.str();
    }

    static std::string format_pvalue(Scalar p) {
        std::ostringstream ss;
        if (p < 0.0001) ss << "<0.0001";
        else if (p < 0.001) ss << std::scientific << std::setprecision(2) << p;
        else ss << std::fixed << std::setprecision(4) << p;
        return ss.str();
    }

    static std::string format_duration(double sec) {
        int s = static_cast<int>(sec);
        int m = s / 60;
        s %= 60;
        int h = m / 60;
        m %= 60;
        std::ostringstream ss;
        if (h > 0) {
            ss << std::setfill('0') << std::setw(2) << h << ":"
               << std::setfill('0') << std::setw(2) << m << ":"
               << std::setfill('0') << std::setw(2) << s;
        } else {
            ss << std::setfill('0') << std::setw(2) << m << ":"
               << std::setfill('0') << std::setw(2) << s;
        }
        return ss.str();
    }
};

} // namespace hyphy::analyses
