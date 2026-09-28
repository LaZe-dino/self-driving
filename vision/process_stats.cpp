#include "process_stats.hpp"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

double process_memory_mib() {
    PROCESS_MEMORY_COUNTERS pmc;
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return 0.0;
    }
    return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
}
#elif defined(__APPLE__)
#include <mach/mach.h>

double process_memory_mib() {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
        KERN_SUCCESS) {
        return 0.0;
    }
    return static_cast<double>(info.resident_size) / (1024.0 * 1024.0);
}
#elif defined(__linux__)
#include <cstdio>
#include <unistd.h>

double process_memory_mib() {
    // /proc/self/statm: total and resident size, in pages.
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) {
        return 0.0;
    }
    long total = 0, resident = 0;
    int n = std::fscanf(f, "%ld %ld", &total, &resident);
    std::fclose(f);
    if (n != 2) {
        return 0.0;
    }
    return static_cast<double>(resident) * static_cast<double>(sysconf(_SC_PAGESIZE)) / (1024.0 * 1024.0);
}
#else
double process_memory_mib() {
    return 0.0;
}
#endif
