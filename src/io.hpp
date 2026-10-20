// io.hpp - configuration, output and diagnostics.
//
// Owner: I/O module (src/io.cpp).
#pragma once
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>
#include "state.hpp"

namespace io {

using ConfigMap = std::map<std::string, std::string>;

// Reads "key = value" lines; '#' starts a comment; blank lines ignored.
ConfigMap readConfigFile(const std::string& path);

// Parses "key=value" command-line tokens into map (later tokens override earlier ones).
void parseOverrides(int argc, char** argv, int first, ConfigMap& map);

// Applies the map to prm. Keys are the Params member names (caseName, outDir, nx0,
// maxLevel, rho1, sigma, gy, tEnd, cfl, implicitViscosity, ...). Boundary conditions:
//   bc_left = wall|slip|inflow|outflow|periodic, bc_left_u, bc_left_v, bc_left_f
//   (same for bc_right, bc_bottom, bc_top). Booleans accept 1/0/true/false.
// Throws std::runtime_error on unknown keys or bad values.
void applyConfig(const ConfigMap& map, Params& prm);

// Human-readable dump of all parameters (written to outDir/params.log by the driver).
void printParams(const Params& prm, std::ostream& os);

// VTK XML unstructured grid of the leaf cells (quads) with cell data
// f, u, v, p, rho, mu, kappa, level, vorticity.
void writeVTU(const State& s, const std::string& filename);

// VTK XML polydata with the PLIC interface segments (vof::interfaceSegments).
void writeInterfaceVTP(const State& s, const std::string& filename);

// ParaView collection file listing (time, relative file name) entries.
void writePVD(const std::string& filename, const std::vector<std::pair<double, std::string>>& entries);

// Integral diagnostics (phase 2 = f == 0, e.g. the bubble in the Hysing benchmark).
struct Diagnostics {
    double t = 0.0, dt = 0.0;
    int step = 0, nLeaves = 0, poissonIters = 0;
    double vol1 = 0.0, vol2 = 0.0;            // volumes of phase 1 / phase 2
    double xc2 = 0.0, yc2 = 0.0;              // centroid of phase 2
    double uc2 = 0.0, vc2 = 0.0;              // mean velocity of phase 2
    double xc1 = 0.0, yc1 = 0.0;              // centroid of phase 1
    double uc1 = 0.0, vc1 = 0.0;              // mean velocity of phase 1
    double ekin = 0.0;                        // sum 0.5 rho |u|^2 V
    double umax = 0.0;                        // max |u| over leaves
    double circularity = 0.0;                 // perimeter of area-equivalent circle of the
                                              // smaller phase / PLIC interface length
};
Diagnostics computeDiagnostics(const State& s);
void writeDiagnosticsHeader(std::ostream& os);   // one '#'-prefixed CSV-style header line
void writeDiagnosticsLine(std::ostream& os, const Diagnostics& d);

}  // namespace io
