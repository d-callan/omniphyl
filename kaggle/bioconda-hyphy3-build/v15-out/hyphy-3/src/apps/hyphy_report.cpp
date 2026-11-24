#include "hyphy/core/console.hpp"
#include "hyphy/report/html_report.hpp"

#include <iostream>
#include <string>

using namespace hyphy::core;

static void print_report_usage(const char* prog) {
    std::cout << Console::bold("Usage:") << " " << prog << " " << Console::accent("<result.json>") << " "
              << Console::brand("--html") << " <file>\n\n"
              << "Write a self-contained interactive HTML report for an existing result JSON\n"
              << "(hyphy3 v3 or legacy HyPhy 2.5 / Datamonkey format). The report needs no network\n"
              << "access and opens directly from disk in any browser.\n\n"
              << Console::bold("Supported analyses:") << " " << hyphy::report::supported_analyses_text() << "\n\n"
              << Console::bold("Options:") << "\n"
              << "  " << Console::brand("--html") << " <file>   Output HTML file (default: <result.json> with .html extension)\n"
              << "  " << Console::brand("--help, -h") << "      Show this help message\n";
}

int run_report(int argc, char* argv[]) {
    std::string json_file;
    std::string html_file;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_report_usage(argv[0]);
            return 0;
        } else if (arg == "--html" && i + 1 < argc) {
            html_file = argv[++i];
        } else if (!arg.empty() && arg[0] != '-' && json_file.empty()) {
            json_file = arg;
        } else {
            std::cerr << Console::danger("Unknown argument: ") << arg << "\n\n";
            print_report_usage(argv[0]);
            return 1;
        }
    }
    if (json_file.empty()) {
        print_report_usage(argv[0]);
        return 1;
    }
    if (html_file.empty()) {
        auto dot = json_file.find_last_of('.');
        auto slash = json_file.find_last_of("/\\");
        bool has_ext = dot != std::string::npos && (slash == std::string::npos || dot > slash);
        html_file = (has_ext ? json_file.substr(0, dot) : json_file) + ".html";
    }

    try {
        hyphy::report::write_html_report(json_file, html_file);
    } catch (const std::exception& e) {
        std::cerr << Console::danger("Error: ") << e.what() << "\n";
        return 1;
    }
    std::cout << Console::success("✔") << " " << Console::bold("Saved HTML report to: ") << Console::brand(html_file)
              << "\n";
    return 0;
}
