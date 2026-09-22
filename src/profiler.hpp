#pragma once

#include <iostream>
#include <chrono>
#include <unordered_map>
#include <string>

class ScopedProfiler
{
public:
    ScopedProfiler(const std::string &section) : section_name(section), start_time(std::chrono::high_resolution_clock::now()) {}

    ~ScopedProfiler()
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        total_times[section_name] += elapsed.count();
        // std::cout << section_name << " measured " << elapsed.count() << "s\n";
    }

    double elapsed() const
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        return elapsed.count();
    }

    static void report()
    {
        for (const auto &[section, time] : total_times)
            printf("%s\t total time %8.2fs\n", section.c_str(), time);
    }

private:
    std::string section_name;
    std::chrono::high_resolution_clock::time_point start_time;
    static inline std::unordered_map<std::string, double> total_times;
};
