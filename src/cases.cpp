// cases.cpp - predefined test cases and exact initial volume fractions (see cases.hpp).
#include "cases.hpp"

#include <algorithm>
#include <cmath>

namespace cases {

namespace {

// Sets every case-relevant parameter (so that switching cases never leaks values).
struct Defaults {
    int nx0 = 1, ny0 = 1;
    double x0 = 0.0, y0 = 0.0, L0 = 1.0;
    int minLevel = 3, maxLevel = 7;
    double rho1 = 1.0, rho2 = 1.0, mu1 = 0.0, mu2 = 0.0, sigma = 0.0, gx = 0.0, gy = 0.0;
    double tEnd = 1.0, cfl = 0.25, dtMax = 1e-2, outputEvery = 0.1;
    BCType bc[4] = {BCType::Wall, BCType::Wall, BCType::Wall, BCType::Wall};
    double uTol = 1e-2;
    bool adaptVelocity = true;
    bool conservingMomentum = false;
    bool vofOnly = false;
    double period = 0.0;
};

void apply(const std::string& name, const Defaults& d, Params& p) {
    p.caseName = name;
    p.outDir = "output/" + name;
    p.nx0 = d.nx0; p.ny0 = d.ny0; p.x0 = d.x0; p.y0 = d.y0; p.L0 = d.L0;
    p.minLevel = d.minLevel; p.maxLevel = d.maxLevel;
    p.rho1 = d.rho1; p.rho2 = d.rho2; p.mu1 = d.mu1; p.mu2 = d.mu2;
    p.sigma = d.sigma; p.gx = d.gx; p.gy = d.gy;
    p.tEnd = d.tEnd; p.cfl = d.cfl; p.dtMax = d.dtMax; p.outputEvery = d.outputEvery;
    for (int s = 0; s < 4; ++s) p.bc[s] = SideBC{d.bc[s], 0.0, 0.0, 0.0};
    p.uTol = d.uTol; p.adaptVelocity = d.adaptVelocity;
    p.conservingMomentum = d.conservingMomentum;
    p.vofOnly = d.vofOnly; p.period = d.period;
}

// Level set > 0 inside the circle (cx, cy, R).
std::function<double(double, double)> disk(double cx, double cy, double R) {
    return [=](double x, double y) { return R - std::hypot(x - cx, y - cy); };
}

// Positive fraction of a triangle with linear phi given by its vertex values.
double triFraction(double a, double b, double c) {
    const bool pa = a > 0, pb = b > 0, pc = c > 0;
    const int np = pa + pb + pc;
    if (np == 0) return 0.0;
    if (np == 3) return 1.0;
    // lone vertex (the one whose sign differs from the other two)
    double l, o1, o2;
    if (pa != pb && pa != pc) { l = a; o1 = b; o2 = c; }
    else if (pb != pa && pb != pc) { l = b; o1 = a; o2 = c; }
    else { l = c; o1 = a; o2 = b; }
    const double corner = (l * l) / ((l - o1) * (l - o2));   // area fraction cut at the lone vertex
    return l > 0 ? corner : 1.0 - corner;
}

constexpr int kSampleN = 8;    // initial uniform sampling: features narrower than h/8 may be missed
constexpr int kMaxDepth = 7;   // further bisections of mixed sub-cells (finest h/1024): per-cell
                               // error ~1e-8 for smooth interfaces, <~5e-7 at corners (Zalesak)

// Fraction of {phi > 0} in the square [x, x+s] x [y, y+s] given its corner values.
double subFraction(const std::function<double(double, double)>& phi, double x, double y, double s, double p00,
                   double p10, double p01, double p11, int depth) {
    const bool a = p00 > 0, b = p10 > 0, c = p01 > 0, d = p11 > 0;
    if (a == b && b == c && c == d) return a ? 1.0 : 0.0;
    if (depth >= kMaxDepth) {
        // two triangles with linear interpolation (exact for linear phi)
        return 0.5 * (triFraction(p00, p10, p11) + triFraction(p00, p11, p01));
    }
    const double h = 0.5 * s;
    const double pm0 = phi(x + h, y), pm2 = phi(x + h, y + s);
    const double p0m = phi(x, y + h), p2m = phi(x + s, y + h), pmm = phi(x + h, y + h);
    return 0.25 * (subFraction(phi, x, y, h, p00, pm0, p0m, pmm, depth + 1) +
                   subFraction(phi, x + h, y, h, pm0, p10, pmm, p2m, depth + 1) +
                   subFraction(phi, x, y + h, h, p0m, pmm, p01, pm2, depth + 1) +
                   subFraction(phi, x + h, y + h, h, pmm, p2m, pm2, p11, depth + 1));
}

}  // namespace

double cellFraction(const Mesh& m, int c, const std::function<double(double, double)>& phi) {
    constexpr int N = kSampleN;
    const double hc = m.size(c), s = hc / N;
    const Vec2 xc = m.center(c);
    const double xl = xc.x - 0.5 * hc, yl = xc.y - 0.5 * hc;
    double g[N + 1][N + 1];
    int npos = 0;
    for (int j = 0; j <= N; ++j)
        for (int i = 0; i <= N; ++i) {
            g[j][i] = phi(xl + i * s, yl + j * s);
            npos += g[j][i] > 0;
        }
    if (npos == 0) return 0.0;
    if (npos == (N + 1) * (N + 1)) return 1.0;
    double sum = 0.0;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
            sum += subFraction(phi, xl + i * s, yl + j * s, s, g[j][i], g[j][i + 1], g[j + 1][i], g[j + 1][i + 1], 0);
    return std::clamp(sum / (N * N), 0.0, 1.0);
}

std::vector<std::string> caseNames() {
    return {"rising_bubble", "rising_bubble2", "static_drop",   "oscillating_drop",
            "rayleigh_taylor", "dam_break",    "breaking_wave", "zalesak",
            "single_vortex"};
}

bool setupCase(const std::string& name, Params& prm, CaseSetup& cs) {
    cs = CaseSetup();
    Defaults d;
    if (name == "rising_bubble" || name == "rising_bubble2") {
        // Hysing et al., Int. J. Numer. Meth. Fluids 60 (2009) 1259. Domain [0,1]x[0,2],
        // bubble (phase 2) of radius 0.25 at (0.5, 0.5) in liquid (phase 1).
        // TC1: rho 1000/100, mu 10/1, sigma 24.5 (Re 35, Eo 10);
        // TC2: rho 1000/1,   mu 10/0.1, sigma 1.96 (Re 35, Eo 125).
        const bool tc2 = name == "rising_bubble2";
        d.ny0 = 2;
        d.minLevel = 3; d.maxLevel = 7;   // h = 1/128
        d.rho1 = 1000.0; d.mu1 = 10.0;
        d.rho2 = tc2 ? 1.0 : 100.0;
        d.mu2 = tc2 ? 0.1 : 1.0;
        d.sigma = tc2 ? 1.96 : 24.5;
        d.gy = -0.98;
        d.bc[LEFT] = d.bc[RIGHT] = BCType::Slip;
        d.bc[BOTTOM] = d.bc[TOP] = BCType::Wall;
        d.tEnd = 3.0; d.outputEvery = 0.1;
        d.uTol = 2e-3;   // ~1% of the rise velocity (AMR matches uniform 128^2 at 1e-3)
        const auto in = disk(0.5, 0.5, 0.25);
        cs.phi = [in](double x, double y) { return -in(x, y); };
    } else if (name == "static_drop") {
        // Laplace equilibrium (Popinet 2009): drop of phase 1, D = 0.4, rho = sigma = 1,
        // La = sigma rho D / mu^2 = 12000 -> mu = sqrt(0.4 / 12000). Capillary time
        // sqrt(rho D^3 / sigma) = 0.253; tEnd = 5 ~ 20 capillary times.
        d.minLevel = 3; d.maxLevel = 6;   // D / h = 25.6 as in Popinet (2009)
        d.rho1 = d.rho2 = 1.0;
        d.mu1 = d.mu2 = std::sqrt(0.4 / 12000.0);
        d.sigma = 1.0;
        d.tEnd = 5.0; d.outputEvery = 0.25;
        d.adaptVelocity = false;   // spurious currents only: keep the mesh driven by the interface
        cs.phi = disk(0.5, 0.5, 0.2);
    } else if (name == "oscillating_drop") {
        // Capillary oscillation of a 2D drop (mode n = 2): r(theta) = R (1 + eps cos 2 theta),
        // R = 0.2, eps = 0.1, centred in [0,1]^2. Linear (Lamb) theory in 2D:
        //   omega^2 = n (n^2 - 1) sigma / ((rho1 + rho2) R^3) -> omega = 27.25, T = 0.2306,
        //   amplitude damping exp(-beta t), beta ~ 2 n (n - 1) nu / R^2 = 0.2 (nu1 = 2e-3).
        // Kinetic energy oscillates with period T/2. Equal kinematic viscosities, density
        // ratio 100, no gravity, walls; the velocity criterion lets the mesh follow the flow.
        d.minLevel = 3; d.maxLevel = 7;   // R / h = 25.6
        d.rho1 = 1.0; d.rho2 = 0.01;
        d.mu1 = 2e-3; d.mu2 = 2e-5;
        d.sigma = 1.0;
        d.tEnd = 2.5; d.outputEvery = 0.02;
        d.uTol = 5e-3;   // ~1% of the peak velocity eps R omega ~ 0.5
        constexpr double R = 0.2, eps = 0.1;
        cs.phi = [](double x, double y) {
            const double dx = x - 0.5, dy = y - 0.5, r = std::hypot(dx, dy);
            const double c2 = r > 0.0 ? (dx * dx - dy * dy) / (r * r) : 1.0;   // cos 2 theta
            return R * (1.0 + eps * c2) - r;
        };
    } else if (name == "breaking_wave") {
        // Plunging breaker of Deike, Popinet & Melville, J. Fluid Mech. 769 (2015) 541-569,
        // in units k = 1, g = 1 (wavelength lambda = 2 pi): domain [-pi, pi]^2, periodic in x,
        // depth h = lambda / 2 = pi, free-slip top and bottom. Steepness eps = a0 k = 0.55,
        // Re = sqrt(g lambda^3) / nu_w = 40000, Bo = (rho_w - rho_a) g / (sigma k^2) = 1000,
        // rho_a / rho_w = 1/850, mu_a / mu_w = 17.4e-6 / 8.9e-4.
        // Initial condition: third-order Stokes wave of Lamb (1932), their eqs. (2.3)-(2.4),
        // with chi = 1 / tanh(kh); velocity from the potential in the water, air at rest (the
        // paper adds a vortex sheet in the air; Basilisk's wave.c by default does not).
        // Linear period T = 2 pi / sqrt(g k) = 2 pi; run to 4 T. Reference (their figs. 8-9):
        // ~70% of the wave energy dissipated after 4 periods.
        constexpr double k = 1.0, g = 1.0, eps = 0.55, a0 = eps / k, h = M_PI;
        constexpr double lambda = 2.0 * M_PI / k;
        const double chi = 1.0 / std::tanh(k * h), chi2 = chi * chi;
        const double omega = std::sqrt(g * k * std::tanh(k * h) * (1.0 + eps * eps * (9.0 / 8.0 * sq(chi2 - 1.0) + chi2)));
        d.L0 = lambda; d.x0 = -M_PI; d.y0 = -h;
        d.minLevel = 4; d.maxLevel = 9;   // lambda / 512, as in Deike et al. at LMAX = 9
        d.rho1 = 1.0; d.rho2 = 1.0 / 850.0;
        d.mu1 = std::sqrt(g * lambda * lambda * lambda) / 40000.0;
        d.mu2 = d.mu1 * 17.4e-6 / 8.9e-4;
        d.sigma = (d.rho1 - d.rho2) * g / (1000.0 * k * k);
        d.gy = -g;
        d.bc[LEFT] = d.bc[RIGHT] = BCType::Periodic;
        d.bc[BOTTOM] = d.bc[TOP] = BCType::Slip;
        d.tEnd = 4.0 * 2.0 * M_PI / std::sqrt(g * k); d.outputEvery = 0.1;   // 4 periods
        d.uTol = 1e-2;                         // ~2.5% of the phase speed sqrt(g/k)
        d.conservingMomentum = true;           // density ratio 850 + violent interface motion
        auto eta = [=](double x) {
            return a0 * std::cos(k * x) + eps * a0 / 4.0 * chi * (3.0 * chi2 - 1.0) * std::cos(2 * k * x) +
                   eps * eps * a0 * (-3.0 / 8.0 * (chi2 * chi2 - 3.0 * chi2 + 3.0) * std::cos(k * x) +
                                     3.0 / 64.0 * (8.0 * chi2 * chi2 * chi2 + sq(chi2 - 1.0)) * std::cos(3 * k * x));
        };
        cs.phi = [eta](double x, double y) { return eta(x) - y; };
        // gradient of phi = sum_n C_n cosh(n k (y + h)) / cosh(n k h) sin(n k x), eq. (2.4)
        const double C[3] = {a0 * g / omega, eps * 3.0 * a0 * g / omega * sq(chi2 - 1.0) / (8.0 * chi),
                             eps * eps * a0 * g / (64.0 * omega) * (chi2 - 1.0) * (chi2 + 3.0) * (9.0 * chi2 - 13.0)};
        cs.u0 = [=](double x, double y) {
            if (y > eta(x)) return Vec2();
            Vec2 u;
            for (int n = 1; n <= 3; ++n) {
                const double c = C[n - 1] * n * k / std::cosh(n * k * h);
                u.x += c * std::cosh(n * k * (y + h)) * std::cos(n * k * x);
                u.y += c * std::sinh(n * k * (y + h)) * std::sin(n * k * x);
            }
            return u;
        };
    } else if (name == "rayleigh_taylor") {
        // Tryggvason (1988) / Guermond & Quartapelle (2000): d = 1, domain [0,1]x[0,4],
        // heavy fluid (phase 1, rho 3) above light fluid (phase 2, rho 1): Atwood 0.5;
        // interface y = 2 + 0.05 cos(2 pi x); g = 1 downward; equal viscosities with
        // Re = rho1 d^1.5 g^0.5 / mu = 3000 -> mu = 1e-3; no surface tension.
        // tEnd = 2.5 / sqrt(At) = 3.5355, i.e. t_Tryg = t sqrt(At g / d) = 2.5.
        d.ny0 = 4;
        d.minLevel = 3; d.maxLevel = 7;   // 128 x 512 at the finest level
        d.rho1 = 3.0; d.rho2 = 1.0;
        d.mu1 = d.mu2 = 1e-3;
        d.gy = -1.0;
        d.bc[LEFT] = d.bc[RIGHT] = BCType::Slip;
        d.bc[BOTTOM] = d.bc[TOP] = BCType::Wall;
        d.tEnd = 2.5 / std::sqrt(0.5); d.outputEvery = 0.1;
        d.uTol = 2e-2;
        cs.phi = [](double x, double y) { return y - (2.0 + 0.05 * std::cos(2.0 * M_PI * x)); };
    } else if (name == "dam_break") {
        // Martin & Moyce (1952): water column a = 0.05715 m wide, 2a high, in the lower
        // left corner of a closed square box of side 5a = 0.28575 m (the front reaches
        // the right wall at T = t sqrt(2g/a) ~ 3.7, after the M&M data range Z = x/a <= 4).
        // Water (phase 1) rho 1000, mu 1e-3; air rho 1.2, mu 1.8e-5; sigma 0.0728; g 9.81.
        const double a = 0.05715;
        d.L0 = 5.0 * a;
        d.minLevel = 4; d.maxLevel = 8;   // h = 5a/256 ~ a/51
        d.rho1 = 1000.0; d.mu1 = 1.0e-3;
        d.rho2 = 1.2; d.mu2 = 1.8e-5;
        d.sigma = 0.0728;
        d.gy = -9.81;
        d.tEnd = 0.25; d.outputEvery = 0.01; d.dtMax = 1e-3;
        d.uTol = 2e-2;   // ~2% of sqrt(2 g a) = 1.06 m/s
        cs.phi = [a](double x, double y) { return std::min(a - x, 2.0 * a - y); };
    } else if (name == "zalesak") {
        // Zalesak (1979) slotted disk in the unit square (Rudman 1997 dimensions): R 0.15
        // at (0.5, 0.75), slot 0.05 wide and 0.25 deep (open at the bottom, top of the slot
        // at y = 0.85); counter-clockwise rigid rotation about (0.5, 0.5), period 1:
        // psi = -(pi/T) ((x-0.5)^2 + (y-0.5)^2) -> u = -2 pi/T (y-0.5), v = 2 pi/T (x-0.5).
        d.minLevel = 3; d.maxLevel = 7;
        d.tEnd = 1.0; d.outputEvery = 0.05;
        d.adaptVelocity = false;
        d.vofOnly = true; d.period = 1.0;
        const auto in = disk(0.5, 0.75, 0.15);
        cs.phi = [in](double x, double y) {
            const double outsideSlot = std::max(std::fabs(x - 0.5) - 0.025, y - 0.85);
            return std::min(in(x, y), outsideSlot);
        };
        const double* T = &prm.period;   // honour a user override of `period` (see below)
        cs.psi = [T](double x, double y, double) {
            const double Tp = *T > 0.0 ? *T : 1.0;
            return -(M_PI / Tp) * (sq(x - 0.5) + sq(y - 0.5));
        };
    } else if (name == "single_vortex") {
        // Bell-Colella-Glaz (1989) / Rider & Kothe (1998) reversible vortex: disk R 0.15
        // at (0.5, 0.75), psi = sin^2(pi x) sin^2(pi y) cos(pi t / T) / pi, T = 8: maximum
        // stretching at t = T/2, exact return to the initial shape at t = T.
        d.minLevel = 3; d.maxLevel = 7;
        d.tEnd = 8.0; d.outputEvery = 0.25;
        d.adaptVelocity = false;
        d.vofOnly = true; d.period = 8.0;
        cs.phi = disk(0.5, 0.75, 0.15);
        // NOTE: psi reads prm.period at call time, so the Params object passed to
        // setupCase must outlive cs (true for State::prm in the driver).
        const double* T = &prm.period;
        cs.psi = [T](double x, double y, double t) {
            const double Tp = *T > 0.0 ? *T : 8.0;
            return sq(std::sin(M_PI * x)) * sq(std::sin(M_PI * y)) * std::cos(M_PI * t / Tp) / M_PI;
        };
    } else {
        return false;
    }
    apply(name, d, prm);
    return true;
}

}  // namespace cases
