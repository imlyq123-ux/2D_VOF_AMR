// main.cpp - command-line entry point.
//
//   vof2d [config.cfg] [key=value ...]
//
// The case is chosen by caseName (from the file or the command line); its defaults are
// applied first, then the config file, then the command-line overrides.
#include <cstdio>
#include <cstring>
#include <exception>
#include "cases.hpp"
#include "io.hpp"
#include "simulation.hpp"
#ifdef __GLIBC__
#include <malloc.h>
#endif

static void usage() {
    std::printf("usage: vof2d [config.cfg] [key=value ...]\n  cases:");
    for (const auto& n : cases::caseNames()) std::printf(" %s", n.c_str());
    std::printf("\n  example: vof2d cases/rising_bubble.cfg maxLevel=8 outDir=out_l8\n");
}

int main(int argc, char** argv) {
#ifdef __GLIBC__
    // The time step allocates and frees many field-sized arrays. By default glibc serves
    // each of them with a fresh mmap and returns it on free, so every step pays ~10^5
    // page faults, which are serial and dominate the threaded loops on large meshes.
    // Keep freed memory in the heap instead (the footprint stays at its peak).
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
#endif
    try {
        int first = 1;
        io::ConfigMap map;
        if (argc > 1 && (!std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help"))) {
            usage();
            return 0;
        }
        if (argc > 1 && !std::strchr(argv[1], '=')) {
            map = io::readConfigFile(argv[1]);
            first = 2;
        }
        io::parseOverrides(argc, argv, first, map);
        if (map.empty()) {
            usage();
            return 1;
        }

        State s;
        cases::CaseSetup cs;
        const std::string name = map.count("caseName") ? map["caseName"] : s.prm.caseName;
        if (!cases::setupCase(name, s.prm, cs)) {
            std::fprintf(stderr, "unknown case '%s'\n", name.c_str());
            usage();
            return 1;
        }
        io::applyConfig(map, s.prm);
        sim::initialize(s, cs);
        sim::run(s, cs);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
