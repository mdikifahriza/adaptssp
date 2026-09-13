#include "adaptsspCore.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

void print_help_menu(const char* prog_name) {
    std::cout << "========================================================================================================\n";
    std::cout << "                         ADAPT-SSP SOLVER v4 - COMMAND LINE INTERFACE (CLI)                             \n";
    std::cout << "========================================================================================================\n\n";
    std::cout << "USAGE:\n";
    std::cout << "  " << prog_name << " \"<elements_list>\" <target> [mode] [max_solutions] [time_limit_ms] [--json]\n\n";
    std::cout << "ARGUMENTS:\n";
    std::cout << "  <elements_list>  : Comma or space separated positive integers or decimals (e.g. \"10, 20, 30\" or \"12.5, 30.25\").\n";
    std::cout << "  <target>         : Target value (integer or decimal, e.g. 60 or 42.75).\n";
    std::cout << "  [mode]           : Search mode / strategy policy:\n";
    std::cout << "                     1. findone             : Find single exact solution witness and halt (Default / Fastest).\n";
    std::cout << "                     2. findall-zero / zero : Find solution and extract variations via Zero-Sum Swap.\n";
    std::cout << "                     3. findall-dfs  / dfs  : Exhaustive 100% full-tree DFS search for all solutions.\n";
    std::cout << "                     4. countall     / count: Count total number of valid subset combinations.\n";
    std::cout << "                     5. decision     / decide: Pure decision problem test (SATISFIABLE or PROVABLY UNSAT).\n";
    std::cout << "  [max_solutions]  : Max witness solutions stored in memory for FindAll (Default: 5000).\n";
    std::cout << "  [time_limit_ms]  : Execution time limit in milliseconds (Default: 120000.0 ms / 2 minutes).\n";
    std::cout << "                     Use 0 / none / unlimited / inf for NO time limit (run until solved).\n";
    std::cout << "  [--json]         : Output execution result as valid JSON to stdout (default: ASCII report).\n\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  1. Find One Mode (Fast Single Witness):\n";
    std::cout << "     " << prog_name << " \"123, 456, 789, 101112\" 579 findone\n\n";
    std::cout << "  2. JSON Output Mode:\n";
    std::cout << "     " << prog_name << " \"10, 20, 30, 40, 50, 60\" 60 findone --json\n\n";
    std::cout << "  3. Decimal / Fraction Mode:\n";
    std::cout << "     " << prog_name << " \"12.5, 30.25, 45.75, 11.5\" 57.75 findone\n";
    std::cout << "========================================================================================================\n";
}

void print_solution_details(const std::string& label, const Instance& inst, const ExecutionStats& stats, SolveMode mode, bool exhaustive) {
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "             ADAPT-SSP SOLVER - TERMINAL EXECUTION REPORT                     " << std::endl;
    std::cout << "================================================================================" << std::endl;
    std::cout << "Test Label         : " << label << std::endl;
    std::cout << "Elements Count (N) : " << inst.raw_elements.size() << " raw (" << inst.A.size() << " positive active)" << std::endl;
    std::cout << "Target Value (T)   : " << inst.target;
    if (inst.decimal_scale_factor > 1) {
        std::cout << " (Scaled from Original Decimal " << format_scaled_val(inst.target, inst.decimal_scale_factor)
                  << ", Scale Factor: " << inst.decimal_scale_factor << ")";
    }
    std::cout << std::endl;
    std::cout << "Total Sum (Sigma)  : " << (u64)inst.total_sum;
    if (inst.decimal_scale_factor > 1) {
        std::cout << " (Original: " << format_scaled_val((u64)inst.total_sum, inst.decimal_scale_factor) << ")";
    }
    std::cout << std::endl;
    std::cout << "Effective Target   : " << inst.effective_target << (inst.complement_applied ? " (Dual Complement Active)" : "") << std::endl;
    std::cout << "Feasible K Range   : [" << inst.k_min << " .. " << inst.k_max << "] (" << inst.feasible_k_count << " cardinality window, " << inst.window_ratio_pct << "%)" << std::endl;
    std::cout << "GCD Value          : " << inst.gcd_val << std::endl;
    std::cout << "Density Score      : " << std::fixed << std::setprecision(4) << inst.density << std::endl;
    std::cout << "Structure Profile  : " << (inst.strong_structure ? "STRONG STRUCTURE" : "FLAT / NON-STRUCTURAL") << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Strategy Chosen    : " << strategy_to_string(stats.strategy_chosen) << std::endl;

    if (stats.boundary_swap_applied) {
        std::cout << "Boundary Swap Hit  : " << stats.boundary_swap_details << std::endl;
    }
    if (stats.residue_primes_checked > 0) {
        std::cout << "Residue Sieve (L2.5): " << stats.residue_primes_checked << " primes checked, "
                  << stats.residue_elements_eliminated << " elements eliminated" << std::endl;
    }
    if (stats.strategy_chosen == StrategyType::HybridTailTable) {
        std::cout << "BlockBound Cache   : " << stats.block_bound_prunes << " prunes | Hits: " << stats.block_bound_hits 
                  << ", Misses: " << stats.block_bound_misses 
                  << (stats.block_bound_disabled ? " [AUTO-DISABLED: low hit-rate]" : " [ACTIVE]") << std::endl;
    }

    std::string mode_str = "Find One (Early Exit)";
    if (mode == SolveMode::FindAll) mode_str = exhaustive ? "Find All (Exhaustive DFS Traversal)" : "Find All (Zero-Sum Swap Fast Expansion)";
    else if (mode == SolveMode::CountAll) mode_str = "Count All (Total Combinations Counter)";
    else if (mode == SolveMode::DecisionOnly) mode_str = "Decision Only (Existence SAT Proof)";
    std::cout << "Solve Mode         : " << mode_str << std::endl;

    std::string status_str = "FAILED / TIMEOUT";
    if (stats.status == SolverStatus::UnknownTimeout) {
        status_str = "FAILED / TIMEOUT";
    } else if (stats.status == SolverStatus::StoppedByUser) {
        status_str = "STOPPED BY USER";
    } else if (stats.has_solution) {
        status_str = "SOLVED (SATISFIABLE)";
    } else {
        status_str = "PROVABLY UNSAT";
    }
    std::cout << "Status             : " << status_str << std::endl;
    std::cout << "Total Runtime      : " << std::fixed << std::setprecision(4) << stats.runtime_ms << " ms (Preprocess: " << stats.preprocess_ms << " ms, Solve: " << stats.solve_ms << " ms)" << std::endl;
    std::cout << "States Evaluated   : " << stats.states_evaluated << std::endl;
    std::cout << "States Pruned      : " << stats.states_pruned << std::endl;
    std::cout << "Oracle Calls/Prune : " << stats.oracle_calls << " / " << stats.oracle_pruned << std::endl;
    std::cout << "Table Lookups      : " << stats.table_lookups << std::endl;
    std::cout << "Peak RAM Memory    : " << std::fixed << std::setprecision(2) << stats.peak_ram_mb << " MB" << std::endl;
    std::cout << "L7 Verifier        : " << (stats.verified ? "[100% INDEPENDENTLY VERIFIED VALID]" : "[VERIFICATION FAILED]") << std::endl;
    std::cout << "L7 Detail Message  : " << stats.verification_message << std::endl;
    std::cout << "Engine Message     : " << stats.message << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;

    if (stats.has_solution) {
        if (mode == SolveMode::FindOne) {
            std::cout << "Exact Solution Found (" << stats.sample_solution.values.size() << " elements):" << std::endl;
            std::cout << "Values : [";
            u64 sum = 0;
            for (size_t i = 0; i < stats.sample_solution.values.size(); ++i) {
                std::cout << stats.sample_solution.values[i];
                sum += stats.sample_solution.values[i];
                if (i + 1 < stats.sample_solution.values.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            if (inst.decimal_scale_factor > 1) {
                std::cout << "Values (Decimal): [";
                for (size_t i = 0; i < stats.sample_solution.values.size(); ++i) {
                    std::cout << format_scaled_val(stats.sample_solution.values[i], inst.decimal_scale_factor);
                    if (i + 1 < stats.sample_solution.values.size()) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
            std::cout << "Indices: [";
            for (size_t i = 0; i < stats.sample_solution.original_indices.size(); ++i) {
                std::cout << stats.sample_solution.original_indices[i];
                if (i + 1 < stats.sample_solution.original_indices.size()) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "Sum Check: " << sum << " == Target " << inst.target << " (" << (sum == inst.target ? "EXACT MATCH 100% VALID" : "MISMATCH ERROR") << ")" << std::endl;
        } else if (mode == SolveMode::DecisionOnly) {
            std::cout << "DECISION RESULT    : SATISFIABLE (YES, AT LEAST ONE SOLUTION EXISTS)" << std::endl;
            std::cout << "Witness Solution   : " << stats.sample_solution.values.size() << " elements -> Sum = " << inst.target << std::endl;
        } else if (mode == SolveMode::FindAll) {
            std::cout << "Total Solutions Counted: " << (u64)stats.solution_count << std::endl;
            std::cout << "Total Solutions Stored : " << stats.all_solutions.size() << std::endl;
            size_t disp = std::min(stats.all_solutions.size(), (size_t)30);
            for (size_t idx = 0; idx < disp; ++idx) {
                const auto& sol = stats.all_solutions[idx];
                u64 sum = 0;
                std::cout << "  #" << std::setw(3) << (idx + 1) << " (" << std::setw(2) << sol.values.size() << " elements): [";
                for (size_t i = 0; i < sol.values.size(); ++i) {
                    std::cout << sol.values[i];
                    sum += sol.values[i];
                    if (i + 1 < sol.values.size()) std::cout << ", ";
                }
                std::cout << "] -> Sum = " << sum << std::endl;
            }
            if (stats.all_solutions.size() > disp) {
                std::cout << "  ... (" << (stats.all_solutions.size() - disp) << " more solutions stored in memory)" << std::endl;
            }
        } else if (mode == SolveMode::CountAll) {
            std::cout << "TOTAL EXACT SOLUTION COUNT: " << (u64)stats.solution_count << " valid subset combinations equal to " << inst.target << "." << std::endl;
        }
    } else {
        std::cout << "NO EXACT SUBSET FOUND (PROVABLY UNSAT)" << std::endl;
    }
    std::cout << "================================================================================\n" << std::endl;
}

inline std::string escape_json(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\b') out += "\\b";
        else if (c == '\f') out += "\\f";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

void print_solution_json(const std::string& label, const Instance& inst, const ExecutionStats& stats, SolveMode mode, bool exhaustive) {
    std::string status_str = "FAILED / TIMEOUT";
    if (stats.status == SolverStatus::UnknownTimeout) {
        status_str = "FAILED / TIMEOUT";
    } else if (stats.status == SolverStatus::StoppedByUser) {
        status_str = "STOPPED BY USER";
    } else if (stats.has_solution) {
        status_str = "SOLVED (SATISFIABLE)";
    } else {
        status_str = "PROVABLY UNSAT";
    }

    std::string mode_str = "findone";
    if (mode == SolveMode::FindAll) mode_str = exhaustive ? "findall-dfs" : "findall-zero";
    else if (mode == SolveMode::CountAll) mode_str = "countall";
    else if (mode == SolveMode::DecisionOnly) mode_str = "decision";

    std::cout << "{\n";
    std::cout << "  \"test_label\": \"" << escape_json(label) << "\",\n";
    std::cout << "  \"elements_count\": " << inst.raw_elements.size() << ",\n";
    std::cout << "  \"active_elements_count\": " << inst.A.size() << ",\n";
    std::cout << "  \"target\": " << inst.target << ",\n";
    std::cout << "  \"total_sum\": " << (u64)inst.total_sum << ",\n";
    std::cout << "  \"effective_target\": " << inst.effective_target << ",\n";
    std::cout << "  \"complement_applied\": " << (inst.complement_applied ? "true" : "false") << ",\n";
    std::cout << "  \"k_min\": " << inst.k_min << ",\n";
    std::cout << "  \"k_max\": " << inst.k_max << ",\n";
    std::cout << "  \"feasible_k_count\": " << inst.feasible_k_count << ",\n";
    std::cout << "  \"window_ratio_pct\": " << inst.window_ratio_pct << ",\n";
    std::cout << "  \"gcd_val\": " << inst.gcd_val << ",\n";
    std::cout << "  \"density\": " << std::fixed << std::setprecision(6) << inst.density << ",\n";
    std::cout << "  \"structure_profile\": \"" << (inst.strong_structure ? "STRONG STRUCTURE" : "FLAT / NON-STRUCTURAL") << "\",\n";
    std::cout << "  \"strategy_chosen\": \"" << escape_json(strategy_to_string(stats.strategy_chosen)) << "\",\n";
    std::cout << "  \"boundary_swap_hit\": " << (stats.boundary_swap_applied ? "true" : "false") << ",\n";
    std::cout << "  \"boundary_swap_details\": \"" << escape_json(stats.boundary_swap_details) << "\",\n";
    std::cout << "  \"residue_primes_checked\": " << stats.residue_primes_checked << ",\n";
    std::cout << "  \"residue_elements_eliminated\": " << stats.residue_elements_eliminated << ",\n";
    std::cout << "  \"block_bound_prunes\": " << stats.block_bound_prunes << ",\n";
    std::cout << "  \"block_bound_hits\": " << stats.block_bound_hits << ",\n";
    std::cout << "  \"block_bound_misses\": " << stats.block_bound_misses << ",\n";
    std::cout << "  \"block_bound_disabled\": " << (stats.block_bound_disabled ? "true" : "false") << ",\n";
    std::cout << "  \"solve_mode\": \"" << mode_str << "\",\n";
    std::cout << "  \"status\": \"" << status_str << "\",\n";
    std::cout << "  \"has_solution\": " << (stats.has_solution ? "true" : "false") << ",\n";
    std::cout << "  \"solution_count\": " << (u64)stats.solution_count << ",\n";
    std::cout << "  \"runtime_ms\": " << std::fixed << std::setprecision(4) << stats.runtime_ms << ",\n";
    std::cout << "  \"preprocess_ms\": " << std::fixed << std::setprecision(4) << stats.preprocess_ms << ",\n";
    std::cout << "  \"solve_ms\": " << std::fixed << std::setprecision(4) << stats.solve_ms << ",\n";
    std::cout << "  \"states_evaluated\": " << stats.states_evaluated << ",\n";
    std::cout << "  \"states_pruned\": " << stats.states_pruned << ",\n";
    std::cout << "  \"oracle_calls\": " << stats.oracle_calls << ",\n";
    std::cout << "  \"oracle_pruned\": " << stats.oracle_pruned << ",\n";
    std::cout << "  \"table_lookups\": " << stats.table_lookups << ",\n";
    std::cout << "  \"peak_ram_mb\": " << std::fixed << std::setprecision(2) << stats.peak_ram_mb << ",\n";
    std::cout << "  \"verified\": " << (stats.verified ? "true" : "false") << ",\n";
    std::cout << "  \"verification_message\": \"" << escape_json(stats.verification_message) << "\",\n";
    std::cout << "  \"engine_message\": \"" << escape_json(stats.message) << "\",\n";
    std::cout << "  \"decimal_scale_factor\": " << inst.decimal_scale_factor << ",\n";

    if (inst.decimal_scale_factor > 1) {
        std::cout << "  \"original_target\": \"" << format_scaled_val(inst.target, inst.decimal_scale_factor) << "\",\n";
    }

    std::cout << "  \"sample_solution\": {\n";
    std::cout << "    \"count\": " << stats.sample_solution.values.size() << ",\n";
    std::cout << "    \"values\": [";
    for (size_t i = 0; i < stats.sample_solution.values.size(); ++i) {
        std::cout << stats.sample_solution.values[i];
        if (i + 1 < stats.sample_solution.values.size()) std::cout << ", ";
    }
    std::cout << "],\n";

    if (inst.decimal_scale_factor > 1) {
        std::cout << "    \"original_values\": [";
        for (size_t i = 0; i < stats.sample_solution.values.size(); ++i) {
            std::cout << "\"" << format_scaled_val(stats.sample_solution.values[i], inst.decimal_scale_factor) << "\"";
            if (i + 1 < stats.sample_solution.values.size()) std::cout << ", ";
        }
        std::cout << "],\n";
    }

    std::cout << "    \"original_indices\": [";
    for (size_t i = 0; i < stats.sample_solution.original_indices.size(); ++i) {
        std::cout << stats.sample_solution.original_indices[i];
        if (i + 1 < stats.sample_solution.original_indices.size()) std::cout << ", ";
    }
    std::cout << "],\n";
    std::cout << "    \"sum\": " << (u64)stats.sample_solution.sum << "\n";
    std::cout << "  },\n";

    std::cout << "  \"all_solutions\": [\n";
    size_t disp = std::min(stats.all_solutions.size(), (size_t)30);
    for (size_t idx = 0; idx < disp; ++idx) {
        const auto& sol = stats.all_solutions[idx];
        std::cout << "    {\n";
        std::cout << "      \"index\": " << (idx + 1) << ",\n";
        std::cout << "      \"count\": " << sol.values.size() << ",\n";
        std::cout << "      \"values\": [";
        for (size_t i = 0; i < sol.values.size(); ++i) {
            std::cout << sol.values[i];
            if (i + 1 < sol.values.size()) std::cout << ", ";
        }
        std::cout << "],\n";
        std::cout << "      \"sum\": " << (u64)sol.sum << "\n";
        std::cout << "    }" << (idx + 1 < disp ? "," : "") << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

int main(int argc, char* argv[]) {
    bool json_output = false;
    std::vector<std::string> raw_args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--json") {
            json_output = true;
        } else {
            raw_args.push_back(a);
        }
    }

    if (raw_args.size() == 1 && (raw_args[0] == "--help" || raw_args[0] == "-h" || raw_args[0] == "/?")) {
        print_help_menu(argv[0]);
        return 0;
    }

    if (raw_args.size() < 2) {
        if (!json_output) print_help_menu(argv[0]);
        if (!json_output) std::cout << "\n[INFO]: Running automated built-in benchmark demo...\n\n";
        
        std::string default_elements = 
            "75872066500, 68562112744, 19339160129, 24156275768, 11525390137, 34580469918, "
            "87752813318, 25906232742, 20636790211, 47170921689, 84559604264, 23643831465, "
            "33227966252, 76687960064, 39654715033, 65528900292, 79579105284, 38705012015, "
            "20401157240, 23386976450, 31066475967, 59591231930, 54278830146, 11050175502, "
            "14436155101, 38734149918, 61482157040, 31440878278, 60036394637, 89364113560, "
            "13381395447, 54381459751";
        std::string default_target = "135205864112";
        Instance inst = Instance::from_string(default_elements, default_target);
        AdaptiveExactSolver solver;
        ExecutionStats stats = solver.run(inst, SolveMode::FindOne, 4096, false, 120000.0, 5000);
        if (json_output) {
            print_solution_json("Demo Default Dataset (32 Elements) [Find One]", inst, stats, SolveMode::FindOne, false);
        } else {
            print_solution_details("Demo Default Dataset (32 Elements) [Find One]", inst, stats, SolveMode::FindOne, false);
        }
        return 0;
    }

    std::string elem_arg = raw_args[0];
    std::string tgt_arg_str = raw_args[1];
    Instance inst = Instance::from_string(elem_arg, tgt_arg_str);

    SolveMode mode = SolveMode::FindOne;
    bool exhaustive = false;

    if (raw_args.size() >= 3) {
        std::string m = raw_args[2];
        std::transform(m.begin(), m.end(), m.begin(), ::tolower);

        if (m == "findone" || m == "1" || m == "one") {
            mode = SolveMode::FindOne;
            exhaustive = false;
        } else if (m == "findall-zero" || m == "zero" || m == "findall" || m == "all" || m == "2") {
            mode = SolveMode::FindAll;
            exhaustive = false;
        } else if (m == "findall-dfs" || m == "dfs" || m == "exhaustive" || m == "full" || m == "3") {
            mode = SolveMode::FindAll;
            exhaustive = true;
        } else if (m == "countall" || m == "count" || m == "4") {
            mode = SolveMode::CountAll;
            exhaustive = false;
        } else if (m == "decision" || m == "decide" || m == "5") {
            mode = SolveMode::DecisionOnly;
            exhaustive = false;
        } else {
            if (!json_output) std::cerr << "[WARNING]: Mode '" << raw_args[2] << "' unrecognized, falling back to default 'findone'.\n";
        }
    }

    size_t max_solutions = (raw_args.size() >= 4) ? (size_t)std::strtoull(raw_args[3].c_str(), nullptr, 10) : 5000;
    double time_limit_ms = 120000.0;
    if (raw_args.size() >= 5) {
        std::string tl = raw_args[4];
        std::string tl_lower = tl;
        std::transform(tl_lower.begin(), tl_lower.end(), tl_lower.begin(), ::tolower);
        if (tl_lower == "0" || tl_lower == "none" || tl_lower == "unlimited" || tl_lower == "inf" || tl_lower == "infinite") {
            time_limit_ms = 0.0;
        } else {
            time_limit_ms = std::atof(raw_args[4].c_str());
        }
    }

    AdaptiveExactSolver solver;
    ExecutionStats stats = solver.run(inst, mode, 4096, exhaustive, time_limit_ms, max_solutions);

    if (json_output) {
        print_solution_json("CLI Execution Result", inst, stats, mode, exhaustive);
    } else {
        print_solution_details("CLI Execution Result", inst, stats, mode, exhaustive);
    }
    return 0;
}
