// simulation.cpp - driver: initialization, time step sequence, regrid, output loop.
#include "simulation.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include "amr.hpp"
#include "curvature.hpp"
#include "io.hpp"
#include "navier_stokes.hpp"
#include "vof.hpp"

namespace sim {

namespace {

bool twoPhaseFlow(const State& s) { return !s.prm.vofOnly; }

// Wall time per phase of the time step, accumulated over the run (printed by run()).
enum Phase { kAdvect, kProperties, kPredictor, kMomentum, kProjection, kRegrid, kNPhases };
const char* const kPhaseName[kNPhases] = {"advection", "properties+curvature", "MAC predictor",
                                          "momentum+viscosity", "projection", "regrid"};
double gPhaseTime[kNPhases] = {};

class PhaseTimer {
public:
    explicit PhaseTimer(Phase p) : p_(p), t0_(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() { gPhaseTime[p_] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count(); }
private:
    Phase p_;
    std::chrono::steady_clock::time_point t0_;
};

void setInitialFields(State& s, const cases::CaseSetup& cs) {
    Mesh& m = s.m();
    auto& f = s.F(s.id.f);
    auto& u = s.F(s.id.u);
    auto& v = s.F(s.id.v);
#pragma omp parallel for schedule(dynamic, 64)
    for (int k = 0; k < m.nLeaves(); ++k) {
        const int c = m.leaves()[k];
        f[c] = cases::cellFraction(m, c, cs.phi);
        const Vec2 x = m.center(c);
        const Vec2 u0 = cs.u0 ? cs.u0(x.x, x.y) : Vec2();
        u[c] = u0.x;
        v[c] = u0.y;
    }
}

// Momentum-conserving advection: the momentum of each phase (f u and (1-f) u per unit
// phase density) is transported with the same geometric fluxes as f, then the velocity
// is recovered as u = (rho1 q1 + rho2 q2) / (rho1 f + rho2 (1-f)). This keeps kinetic
// energy from being created when cells change phase at large density ratios.
void advectConserving(State& s, double dt) {
    Mesh& m = s.m();
    const Params& prm = s.prm;
    std::vector<double>& f = s.F(s.id.f);
    std::vector<double>* vel[2] = {&s.F(s.id.u), &s.F(s.id.v)};
    std::vector<double> q[2][2];   // [phase][component]
    std::vector<vof::Tracer> tr;
    for (int ph = 0; ph < 2; ++ph)
        for (int a = 0; a < 2; ++a) {
            q[ph][a].assign(m.nCells(), 0.0);
            for (int c : m.leaves()) q[ph][a][c] = (ph == 0 ? f[c] : 1.0 - f[c]) * (*vel[a])[c];
            vof::Tracer t;
            t.data = &q[ph][a];
            t.phase2 = ph == 1;
            t.fallback = vel[a];
            for (int side = 0; side < 4; ++side) t.inflow[side] = a == 0 ? prm.bc[side].u : prm.bc[side].v;
            tr.push_back(t);
        }
    vof::advect(m, f, s.uf, dt, s.step, prm, tr);
    for (int c : m.leaves()) {
        const double rho = prm.rho1 * f[c] + prm.rho2 * (1.0 - f[c]);
        for (int a = 0; a < 2; ++a) (*vel[a])[c] = (prm.rho1 * q[0][a][c] + prm.rho2 * q[1][a][c]) / rho;
    }
}

// After any change of mesh or f: properties, curvature, face velocity.
void refreshDerived(State& s, const cases::CaseSetup& cs, bool project) {
    if (twoPhaseFlow(s)) {
        ns::updateProperties(s);
        if (s.prm.sigma != 0.0)
            curvature::compute(s.m(), s.F(s.id.f), s.F(s.id.kappa), s.prm.rejectUnresolvedCurvature);
        if (project) ns::reproject(s);
    } else {
        faceVelocityFromStreamFunction(s.m(), cs, s.t, s.uf);
    }
}

}  // namespace

void faceVelocityFromStreamFunction(const Mesh& m, const cases::CaseSetup& cs, double t,
                                    std::vector<double>& uf) {
    if (!cs.psi) throw std::runtime_error("vofOnly case without a stream function");
    uf.assign(m.nFaces(), 0.0);
#pragma omp parallel for
    for (int k = 0; k < m.nFaces(); ++k) {
        const Mesh::Face& F = m.faces()[k];
        const double a = 0.5 * F.area;
        uf[k] = F.dir == 0 ? (cs.psi(F.x.x, F.x.y + a, t) - cs.psi(F.x.x, F.x.y - a, t)) / F.area
                           : -(cs.psi(F.x.x + a, F.x.y, t) - cs.psi(F.x.x - a, F.x.y, t)) / F.area;
    }
}

void initialize(State& s, const cases::CaseSetup& cs) {
    const Params& prm = s.prm;
    if (!cs.phi) throw std::runtime_error("case has no initial interface (phi)");
    const bool perX = prm.bc[LEFT].type == BCType::Periodic;
    const bool perY = prm.bc[BOTTOM].type == BCType::Periodic;
    if (perX != (prm.bc[RIGHT].type == BCType::Periodic) || perY != (prm.bc[TOP].type == BCType::Periodic))
        throw std::runtime_error("periodic boundary conditions must be set on both opposite sides");

    Mesh::Config cfg;
    cfg.nx0 = prm.nx0; cfg.ny0 = prm.ny0;
    cfg.x0 = prm.x0; cfg.y0 = prm.y0; cfg.L0 = prm.L0;
    cfg.minLevel = prm.minLevel; cfg.maxLevel = prm.maxLevel;
    cfg.periodicX = perX; cfg.periodicY = perY;
    s.mesh = std::make_unique<Mesh>(cfg);
    Mesh& m = s.m();
    s.id.f = m.addField("f", &vof::prolong);
    s.id.u = m.addField("u");
    s.id.v = m.addField("v");
    s.id.p = m.addField("p");
    s.id.rho = m.addField("rho", &Mesh::prolongInject);
    s.id.mu = m.addField("mu", &Mesh::prolongInject);
    s.id.kappa = m.addField("kappa", &Mesh::prolongInject);

    // refine around the interface level by level, re-evaluating exact fractions each pass
    for (int pass = 0; pass <= prm.maxLevel + 2; ++pass) {
        setInitialFields(s, cs);
        s.uf.assign(m.nFaces(), 0.0);
        if (!m.adapt(amr::computeFlags(s))) break;
    }
    setInitialFields(s, cs);
    s.uf.assign(m.nFaces(), 0.0);
    refreshDerived(s, cs, true);
    s.t = 0.0;
    s.step = 0;
}

bool regrid(State& s, const cases::CaseSetup& cs) {
    PhaseTimer timer(kRegrid);
    if (!s.m().adapt(amr::computeFlags(s))) return false;
    refreshDerived(s, cs, true);
    return true;
}

void step(State& s, const cases::CaseSetup& cs, double dtLimit) {
    Mesh& m = s.m();
    const Params& prm = s.prm;
    double dt;
    if (!twoPhaseFlow(s)) {
        // CFL on the prescribed velocity at t, then advect with the mid-point velocity
        PhaseTimer timer(kAdvect);
        faceVelocityFromStreamFunction(m, cs, s.t, s.uf);
        double rate = 0.0;
        for (int k = 0; k < m.nFaces(); ++k) rate = std::max(rate, std::fabs(s.uf[k]) / m.faces()[k].area);
        dt = std::min(prm.dtMax, rate > 0.0 ? prm.cfl / rate : prm.dtMax);
        dt = std::min(dt, dtLimit);
        faceVelocityFromStreamFunction(m, cs, s.t + 0.5 * dt, s.uf);
        vof::advect(m, s.F(s.id.f), s.uf, dt, s.step, prm);
    } else {
        {
            PhaseTimer timer(kAdvect);
            dt = std::min(ns::stableDt(s), dtLimit);
            if (prm.conservingMomentum) advectConserving(s, dt);
            else vof::advect(m, s.F(s.id.f), s.uf, dt, s.step, prm);
        }
        {
            PhaseTimer timer(kProperties);
            ns::updateProperties(s);
            if (prm.sigma != 0.0)
                curvature::compute(m, s.F(s.id.f), s.F(s.id.kappa), prm.rejectUnresolvedCurvature);
        }
        // MAC-projected, time-centred advection velocity for the momentum (the momentum is
        // already transported by the VOF fluxes in conserving mode)
        if (prm.macPredictor && !prm.conservingMomentum) {
            PhaseTimer timer(kPredictor);
            const poisson::Result r = ns::predictFaceVelocity(s, dt);
            if (!r.converged)
                std::fprintf(stderr, "warning: step %d MAC predictor solve not converged (%d its, res %.2e)\n",
                             s.step, r.iterations, r.residual);
        }
        {
            PhaseTimer timer(kMomentum);
            ns::advanceMomentum(s, dt);
        }
        PhaseTimer timer(kProjection);
        const poisson::Result r = ns::project(s, dt);
        s.lastPoissonIters = r.iterations;
        if (!r.converged)
            std::fprintf(stderr, "warning: step %d pressure solve not converged (%d its, res %.2e)\n",
                         s.step, r.iterations, r.residual);
    }
    s.dt = dt;
    s.t += dt;
    s.step++;
    if (prm.adaptEvery > 0 && s.step % prm.adaptEvery == 0) regrid(s, cs);
}

void run(State& s, const cases::CaseSetup& cs) {
    const Params& prm = s.prm;
    namespace fs = std::filesystem;
    fs::create_directories(prm.outDir);
    {
        std::ofstream pl(prm.outDir + "/params.log");
        io::printParams(prm, pl);
    }
    std::ofstream diag(prm.outDir + "/diagnostics.dat");
    io::writeDiagnosticsHeader(diag);
    std::vector<std::pair<double, std::string>> vtus, itfs;

    int frame = 0;
    auto output = [&]() {
        char name[64];
        std::snprintf(name, sizeof name, "fields_%05d.vtu", frame);
        io::writeVTU(s, prm.outDir + "/" + name);
        vtus.emplace_back(s.t, name);
        std::snprintf(name, sizeof name, "interface_%05d.vtp", frame);
        io::writeInterfaceVTP(s, prm.outDir + "/" + name);
        itfs.emplace_back(s.t, name);
        io::writePVD(prm.outDir + "/fields.pvd", vtus);
        io::writePVD(prm.outDir + "/interface.pvd", itfs);
        ++frame;
    };
    auto diagnostics = [&]() {
        io::Diagnostics d = io::computeDiagnostics(s);
        io::writeDiagnosticsLine(diag, d);
        diag.flush();
        return d;
    };

    const io::Diagnostics d0 = diagnostics();
    const double vol0 = d0.vol1;
    const bool doOutput = prm.outputEvery > 0.0;
    if (doOutput) output();
    double nextOut = doOutput ? prm.outputEvery : prm.tEnd;

    const std::string gpuDev = prm.gpu && !prm.vofOnly ? poisson::gpuDevice() : "";  // initializes CUDA

    using clock = std::chrono::steady_clock;
    const auto tStart = clock::now();
    const poisson::Timing pt0 = poisson::timing();  // excludes the solves of initialize()
    auto tLast = tStart;
    std::printf("case %s: %d leaves (levels %d..%d), tEnd = %g\n", prm.caseName.c_str(),
                s.m().nLeaves(), prm.minLevel, prm.maxLevel, prm.tEnd);
    if (prm.gpu && !prm.vofOnly) {
        if (!gpuDev.empty())
            std::printf("linear solver on the GPU: %s\n", gpuDev.c_str());
        else
            std::fprintf(stderr, "warning: gpu = true, but %s; linear solves run on the CPU\n",
#ifdef VOF2D_CUDA
                         "no usable CUDA device was found"
#else
                         "this is a CPU-only build (make CUDA=1)"
#endif
            );
    }
    while (s.t < prm.tEnd - 1e-12 && s.step < prm.maxSteps) {
        const double target = std::min(nextOut, prm.tEnd);
        step(s, cs, target - s.t);
        if (s.dt < prm.dtMin) throw std::runtime_error("time step collapsed below dtMin");
        const bool hitOut = doOutput && s.t >= nextOut - 1e-12;
        if (hitOut || s.step % std::max(1, prm.diagEvery) == 0) {
            const io::Diagnostics d = diagnostics();
            const double wall = std::chrono::duration<double>(clock::now() - tStart).count();
            if (hitOut || std::chrono::duration<double>(clock::now() - tLast).count() > 5.0) {
                tLast = clock::now();
                std::printf("step %7d  t = %-10.5g dt = %-10.3e leaves = %-8d its = %-4d "
                            "umax = %-10.3e dvol1 = %+.2e  wall = %.1fs\n",
                            s.step, s.t, s.dt, s.m().nLeaves(), s.lastPoissonIters, d.umax,
                            vol0 > 0 ? (d.vol1 - vol0) / vol0 : 0.0, wall);
                std::fflush(stdout);
            }
            if (!std::isfinite(d.umax)) throw std::runtime_error("solution diverged (non-finite velocity)");
        }
        if (hitOut) {
            output();
            nextOut += prm.outputEvery;
        }
    }
    diagnostics();
    const double wall = std::chrono::duration<double>(clock::now() - tStart).count();
    std::printf("done: %d steps, t = %g, wall %.1f s (%.2f ms/step)\n", s.step, s.t, wall,
                1e3 * wall / std::max(1, s.step));
    const double perStep = 1e3 / std::max(1, s.step);
    std::printf("time per step (ms):");
    for (int k = 0; k < kNPhases; ++k)
        if (gPhaseTime[k] > 0.0) std::printf("  %s %.2f", kPhaseName[k], gPhaseTime[k] * perStep);
    std::printf("\n");
    const poisson::Timing pt = poisson::timing();
    if (pt.solves > pt0.solves)
        std::printf("  of which linear solves: %d solves, setup %.2f, iterations %.2f\n", pt.solves - pt0.solves,
                    (pt.setup - pt0.setup) * perStep, (pt.iterate - pt0.iterate) * perStep);
}

}  // namespace sim
