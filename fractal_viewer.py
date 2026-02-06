#!/usr/bin/env python3
"""GPU-Accelerated Mandelbrot Fractal Viewer using Pygame + PyCUDA.

Two-pass rendering:
  1. Compute pass — expensive iteration, writes smooth iter values to a float buffer.
     Only runs when the view changes (pan, zoom, iteration count).
  2. Colorize pass — cheap palette lookup from the float buffer.
     Runs every frame during color cycling at full speed regardless of zoom depth.

Two compute kernels: fast standard-double for normal zoom, double-double for deep zoom.
8 selectable color palettes, on-screen control panel, color cycling with speed control.
Center coordinates tracked in Python Decimal for lossless accumulation.
"""

import json
import os
import time
import math
from collections import deque
from decimal import Decimal, getcontext

import numpy as np
import pygame
import pygame.surfarray

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

getcontext().prec = 50

DD_ZOOM_THRESHOLD = 1e13
LOCATIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locations")
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

PALETTE_NAMES = [
    "Classic",
    "Inferno",
    "Ocean",
    "Electric",
    "Emerald",
    "Twilight",
    "Grayscale",
    "Rainbow",
]

# Fractal types
# Format: (name, formula_id)
FRACTAL_TYPES = [
    ("Mandelbrot", 0),
    ("Burning Ship", 1),
    ("Tricorn", 2),
    ("Julia: Dragon", 3),
    ("Julia: Dendrite", 4),
    ("Julia: Spiral", 5),
    ("Julia: San Marco", 6),
    ("Julia: Douady", 7),
    ("Multibrot z^3", 8),
    ("Multibrot z^4", 9),
    ("Phoenix", 10),
]

# Aspect ratios: (name, width_ratio, height_ratio)
ASPECT_RATIOS = [
    ("16:9", 16, 9),
    ("21:9", 21, 9),
    ("32:9", 32, 9),
    ("48:9", 48, 9),
    ("4:3", 4, 3),
    ("1:1", 1, 1),
    ("3:2", 3, 2),
    ("16:10", 16, 10),
]

# Resolution presets keyed by aspect name
RESOLUTION_PRESETS = {
    "16:9":  [(1920, 1080), (2560, 1440), (3840, 2160), (7680, 4320)],
    "21:9":  [(2560, 1080), (3440, 1440), (5120, 2160)],
    "32:9":  [(3840, 1080), (5120, 1440), (7680, 2160)],
    "48:9":  [(5760, 1080), (7680, 1440), (11520, 2160)],
    "4:3":   [(1600, 1200), (2048, 1536), (3200, 2400)],
    "1:1":   [(1080, 1080), (2160, 2160), (4320, 4320)],
    "3:2":   [(2160, 1440), (3240, 2160), (4320, 2880)],
    "16:10": [(1920, 1200), (2560, 1600), (3840, 2400)],
}

# Default view for each fractal type: (center_x, center_y, zoom, iter_offset)
FRACTAL_DEFAULTS = {
    0: ("-0.26810170634920634894882274351822159257723058694218", "0.056267777777777815295764226350661802688530433904766", 0.35012779664577565, 0),
    1: ("-0.27716215079365079317564050491252832676385248421592", "-0.56993576091269838512981976364474714955070041857701", 0.35012779664577565, 100),
    2: ("0.1158900793650794005919354060172860889428918478661", "0.00737988095238100367036183897741664908513060658075", 0.2693290743429043, 50),
    3: ("0.10203831018518517761096358230894453589459663162644", "0.013817592592592609497586271499271427541957853600986", 0.3231948892114852, 50),
    4: ("0.10203831018518517761096358230894453589459663162644", "0.013817592592592609497586271499271427541957853600986", 0.3231948892114852, 50),
    5: ("0.10203831018518517761096358230894453589459663162644", "0.013817592592592609497586271499271427541957853600986", 0.3231948892114852, 50),
    6: ("0.10203831018518517761096358230894453589459663162644", "0.013817592592592609497586271499271427541957853600986", 0.3231948892114852, 50),
    7: ("0.10203831018518517761096358230894453589459663162644", "0.013817592592592609497586271499271427541957853600986", 0.3231948892114852, 50),
    8: ("0.10629497003968257195600202755845435177255299165414", "0.016411609457671956526630818772511962720872176101914", 0.4039936115143565, 50),
    9: ("0.14631660180555559192148422729312915470768486330834", "0.013644879965277797322876377459160706452458002140234", 0.4143524220660067, 50),
    10: ("0.09215930357142862058936394227472728376637419787360", "0.00432279067460318581569185286353633229477135238307", 0.41435242206600664, 50),
}

# Interesting locations - now includes fractal type
# Format: (name, center_x, center_y, zoom, iter_offset, fractal_type)
PRESET_LOCATIONS = [
    ("Mandelbrot", "-0.5", "0.0", 1.0, 0, 0),
    ("Seahorse Valley", "-0.75", "0.1", 100.0, 50, 0),
    ("Double Spiral", "-0.7269", "0.1889", 5000.0, 100, 0),
    ("Elephant Valley", "0.2549", "0.0005", 5000.0, 100, 0),
    ("Triple Spiral", "-0.0883", "0.6549", 10000.0, 150, 0),
    ("Burning Ship", "-0.5", "-0.5", 1.0, 100, 1),
    ("Tricorn", "-0.3", "0.0", 1.0, 50, 2),
    ("Julia: Dragon", "0.0", "0.0", 1.2, 50, 3),
    ("Julia: Dendrite", "0.0", "0.0", 1.2, 50, 4),
    ("Julia: Spiral", "0.0", "0.0", 1.2, 50, 5),
    ("Julia: San Marco", "0.0", "0.0", 1.2, 50, 6),
    ("Julia: Douady", "0.0", "0.0", 1.2, 50, 7),
    ("Multibrot z^3", "0.0", "0.0", 1.5, 50, 8),
    ("Multibrot z^4", "0.0", "0.0", 2.0, 50, 9),
    ("Phoenix", "0.0", "0.0", 2.0, 50, 10),
]

CUDA_KERNEL = r"""
#include <math.h>

// ============================================================
// Compute kernels — write smooth iteration count to float buffer.
// Interior points (iter >= max_iter) stored as -1.0.
// ============================================================

// Fast compute — standard double precision with multiple fractal formulas
__global__ void compute_fast(
    float *iter_buf,
    int width, int height,
    double center_x, double center_y,
    double zoom, int max_iter,
    int fractal_type,
    double rot_cos, double rot_sin
)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    double aspect = (double)width / (double)height;
    double raw_x = (((double)px / (double)width - 0.5) * aspect) / zoom;
    double raw_y = (((double)py / (double)height - 0.5)) / zoom;
    double x0 = center_x + raw_x * rot_cos - raw_y * rot_sin;
    double y0 = center_y + raw_x * rot_sin + raw_y * rot_cos;

    double zx = 0.0, zy = 0.0, zx2 = 0.0, zy2 = 0.0;
    double cx = x0, cy = y0;
    double temp, zx_old = 0.0, zy_old = 0.0;
    int iter = 0;

    // Julia set parameters: z starts at pixel coord, c is fixed constant
    if (fractal_type == 3) { // Julia Dragon
        cx = -0.8; cy = 0.156;
        zx = x0; zy = y0; zx2 = zx*zx; zy2 = zy*zy;
    } else if (fractal_type == 4) { // Julia Dendrite
        cx = 0.0; cy = 1.0;
        zx = x0; zy = y0; zx2 = zx*zx; zy2 = zy*zy;
    } else if (fractal_type == 5) { // Julia Spiral
        cx = -0.4; cy = 0.6;
        zx = x0; zy = y0; zx2 = zx*zx; zy2 = zy*zy;
    } else if (fractal_type == 6) { // Julia San Marco
        cx = -0.75; cy = 0.0;
        zx = x0; zy = y0; zx2 = zx*zx; zy2 = zy*zy;
    } else if (fractal_type == 7) { // Julia Douady Rabbit
        cx = -0.123; cy = 0.745;
        zx = x0; zy = y0; zx2 = zx*zx; zy2 = zy*zy;
    }

    while (zx2 + zy2 <= 65536.0 && iter < max_iter) {
        if (fractal_type == 1) { // Burning Ship: abs BEFORE squaring
            double ax = fabs(zx), ay = fabs(zy);
            zy = 2.0 * ax * ay + cy;
            zx = ax * ax - ay * ay + cx;
        } else if (fractal_type == 2) { // Tricorn: conjugate z
            zy = -2.0 * zx * zy + cy;
            zx = zx2 - zy2 + cx;
        } else if (fractal_type == 8) { // Multibrot z^3
            temp = zx * (zx2 - 3.0 * zy2) + cx;
            zy = zy * (3.0 * zx2 - zy2) + cy;
            zx = temp;
        } else if (fractal_type == 9) { // Multibrot z^4
            temp = zx2 * zx2 - 6.0 * zx2 * zy2 + zy2 * zy2 + cx;
            zy = 4.0 * zx * zy * (zx2 - zy2) + cy;
            zx = temp;
        } else if (fractal_type == 10) { // Phoenix
            temp = zx2 - zy2 + cx + 0.5 * zx_old;
            zy = 2.0 * zx * zy + cy - 0.5 * zy_old;
            zx_old = zx;
            zy_old = zy;
            zx = temp;
        } else { // Mandelbrot (0) and all Julia sets (3-7)
            zy = 2.0 * zx * zy + cy;
            zx = zx2 - zy2 + cx;
        }

        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
    }

    int pidx = py * width + px;
    if (iter >= max_iter) {
        iter_buf[pidx] = -1.0f;
    } else {
        double mod2 = zx2 + zy2;
        double log_zn = log(mod2) / 2.0;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        iter_buf[pidx] = (float)((double)iter + 1.0 - nu);
    }
}

// ============================================================
// Double-double arithmetic helpers
// ============================================================
__device__ __forceinline__ void two_sum(double a, double b, double *s, double *e) {
    *s = a + b;
    double v = *s - a;
    *e = (a - (*s - v)) + (b - v);
}

__device__ __forceinline__ void two_prod(double a, double b, double *p, double *e) {
    *p = a * b;
    *e = fma(a, b, -(*p));
}

__device__ __forceinline__ void dd_add(double ah, double al, double bh, double bl,
                                       double *rh, double *rl) {
    double s, e;
    two_sum(ah, bh, &s, &e);
    e += al + bl;
    two_sum(s, e, rh, rl);
}

__device__ __forceinline__ void dd_mul(double ah, double al, double bh, double bl,
                                       double *rh, double *rl) {
    double p, e;
    two_prod(ah, bh, &p, &e);
    e += ah * bl + al * bh;
    two_sum(p, e, rh, rl);
}

// Deep-zoom compute — double-double precision with multiple fractal formulas
__global__ void compute_deep(
    float *iter_buf,
    int width, int height,
    double center_x_hi, double center_x_lo,
    double center_y_hi, double center_y_lo,
    double zoom, int max_iter,
    int fractal_type,
    double rot_cos, double rot_sin
)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    double aspect = (double)width / (double)height;
    double raw_x = (((double)px / (double)width - 0.5) * aspect) / zoom;
    double raw_y = (((double)py / (double)height - 0.5)) / zoom;
    double off_x = raw_x * rot_cos - raw_y * rot_sin;
    double off_y = raw_x * rot_sin + raw_y * rot_cos;

    double cx_hi, cx_lo, cy_hi, cy_lo;
    dd_add(center_x_hi, center_x_lo, off_x, 0.0, &cx_hi, &cx_lo);
    dd_add(center_y_hi, center_y_lo, off_y, 0.0, &cy_hi, &cy_lo);

    double zx_hi = 0.0, zx_lo = 0.0;
    double zy_hi = 0.0, zy_lo = 0.0;
    double zx2_hi, zx2_lo, zy2_hi, zy2_lo;
    int iter = 0;

    // For Julia sets, swap initial z and c
    if (fractal_type == 3) { // Dragon
        zx_hi = cx_hi; zx_lo = cx_lo;
        zy_hi = cy_hi; zy_lo = cy_lo;
        cx_hi = -0.8; cx_lo = 0.0;
        cy_hi = 0.156; cy_lo = 0.0;
    } else if (fractal_type == 4) { // Dendrite
        zx_hi = cx_hi; zx_lo = cx_lo;
        zy_hi = cy_hi; zy_lo = cy_lo;
        cx_hi = 0.0; cx_lo = 0.0;
        cy_hi = 1.0; cy_lo = 0.0;
    } else if (fractal_type == 5) { // Spiral
        zx_hi = cx_hi; zx_lo = cx_lo;
        zy_hi = cy_hi; zy_lo = cy_lo;
        cx_hi = -0.4; cx_lo = 0.0;
        cy_hi = 0.6; cy_lo = 0.0;
    } else if (fractal_type == 6) { // San Marco
        zx_hi = cx_hi; zx_lo = cx_lo;
        zy_hi = cy_hi; zy_lo = cy_lo;
        cx_hi = -0.75; cx_lo = 0.0;
        cy_hi = 0.0; cy_lo = 0.0;
    } else if (fractal_type == 7) { // Douady Rabbit
        zx_hi = cx_hi; zx_lo = cx_lo;
        zy_hi = cy_hi; zy_lo = cy_lo;
        cx_hi = -0.123; cx_lo = 0.0;
        cy_hi = 0.745; cy_lo = 0.0;
    }

    while (iter < max_iter) {
        // Burning Ship: take abs of z components before squaring
        double ax_hi = zx_hi, ax_lo = zx_lo;
        double ay_hi = zy_hi, ay_lo = zy_lo;
        if (fractal_type == 1) {
            if (ax_hi < 0.0) { ax_hi = -ax_hi; ax_lo = -ax_lo; }
            if (ay_hi < 0.0) { ay_hi = -ay_hi; ay_lo = -ay_lo; }
        }

        dd_mul(ax_hi, ax_lo, ax_hi, ax_lo, &zx2_hi, &zx2_lo);
        dd_mul(ay_hi, ay_lo, ay_hi, ay_lo, &zy2_hi, &zy2_lo);

        if (zx2_hi + zy2_hi > 65536.0) break;

        double txy_hi, txy_lo;
        dd_mul(ax_hi, ax_lo, ay_hi, ay_lo, &txy_hi, &txy_lo);
        dd_add(txy_hi, txy_lo, txy_hi, txy_lo, &zy_hi, &zy_lo);

        if (fractal_type == 2) { // Tricorn (conjugate)
            dd_add(-zy_hi, -zy_lo, cy_hi, cy_lo, &zy_hi, &zy_lo);
        } else {
            dd_add(zy_hi, zy_lo, cy_hi, cy_lo, &zy_hi, &zy_lo);
        }

        dd_add(zx2_hi, zx2_lo, -zy2_hi, -zy2_lo, &zx_hi, &zx_lo);
        dd_add(zx_hi, zx_lo, cx_hi, cx_lo, &zx_hi, &zx_lo);

        iter++;
    }

    int pidx = py * width + px;
    if (iter >= max_iter) {
        iter_buf[pidx] = -1.0f;
    } else {
        double mod2 = zx_hi * zx_hi + zy_hi * zy_hi;
        double log_zn = log(mod2) / 2.0;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        iter_buf[pidx] = (float)((double)iter + 1.0 - nu);
    }
}

// ============================================================
// Colorize kernel — palette lookup from cached iteration buffer.
// This is extremely cheap: just cosines, no iteration loop.
// ============================================================
__device__ void get_palette_color(
    double t, int palette_id,
    double *r, double *g, double *b
)
{
    double ar, ag, ab, br, bg, bb, cr, cg, cb, dr, dg, db;

    switch (palette_id) {
    case 1: // Inferno
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=0.7;  cb=0.4;
        dr=0.00; dg=0.15; db=0.20;
        break;
    case 2: // Ocean
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=1.0;  cb=1.0;
        dr=0.50; dg=0.60; db=0.70;
        break;
    case 3: // Electric
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=1.0;  cb=1.0;
        dr=0.00; dg=0.10; db=0.67;
        break;
    case 4: // Emerald
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=1.0;  cb=1.0;
        dr=0.35; dg=0.00; db=0.67;
        break;
    case 5: // Twilight
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=1.0;  cb=0.5;
        dr=0.80; dg=0.90; db=0.30;
        break;
    case 6: // Grayscale
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=1.0;  cb=1.0;
        dr=0.00; dg=0.00; db=0.00;
        break;
    case 7: // Rainbow
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=1.0;  cb=1.0;
        dr=0.00; dg=0.167; db=0.333;
        break;
    default: // Classic (case 0)
        ar=0.5;  ag=0.5;  ab=0.5;
        br=0.5;  bg=0.5;  bb=0.5;
        cr=1.0;  cg=1.0;  cb=1.0;
        dr=0.00; dg=0.33; db=0.67;
        break;
    }

    double pi2 = 2.0 * 3.14159265358979323846;
    *r = ar + br * cos(pi2 * (cr * t + dr));
    *g = ag + bg * cos(pi2 * (cg * t + dg));
    *b = ab + bb * cos(pi2 * (cb * t + db));

    *r = fmin(fmax(*r, 0.0), 1.0);
    *g = fmin(fmax(*g, 0.0), 1.0);
    *b = fmin(fmax(*b, 0.0), 1.0);
}

__global__ void colorize(
    unsigned char *output,
    const float *iter_buf,
    int width, int height,
    double color_offset,
    int palette_id
)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int pidx = py * width + px;
    float val = iter_buf[pidx];
    int idx = pidx * 3;

    if (val < 0.0f) {
        output[idx]     = 0;
        output[idx + 1] = 0;
        output[idx + 2] = 0;
    } else {
        double t = (double)val / 64.0 + color_offset;
        double r, g, b;
        get_palette_color(t, palette_id, &r, &g, &b);
        output[idx]     = (unsigned char)(r * 255.0);
        output[idx + 1] = (unsigned char)(g * 255.0);
        output[idx + 2] = (unsigned char)(b * 255.0);
    }
}
"""

BLOCK_SIZE = (16, 16, 1)

# ── UI constants ──────────────────────────────────────────────
PANEL_W = 210
PANEL_PAD = 10
BTN_H = 28
BTN_GAP = 6
SECTION_GAP = 14

COL_BG = (18, 18, 28, 210)
COL_BTN = (55, 55, 75)
COL_BTN_HOVER = (80, 80, 110)
COL_BTN_ACTIVE = (100, 130, 200)
COL_TEXT = (220, 220, 230)
COL_HEADING = (140, 160, 200)
COL_ACCENT = (100, 180, 255)


class MandelbrotViewer:
    """Real-time GPU-accelerated Mandelbrot set viewer."""

    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # Fractal state
        self.base_iter = 200
        self.fractal_type = 0
        cx, cy, zoom, iter_off = FRACTAL_DEFAULTS[0]
        self.center_x = Decimal(cx)
        self.center_y = Decimal(cy)
        self.zoom = zoom
        self.iter_offset = iter_off
        self.max_iter = 200
        self.rotation = 0.0
        self.color_offset = 0.0
        self.color_cycling = False
        self.cycle_speed = 1.0
        self.palette_id = 0
        self.current_preset = 0
        self.aspect_id = 0
        self.save_res_idx = 0

        # Interaction state
        self.dragging = False
        self.drag_start = None
        self.drag_center_start = None
        self.rotating = False
        self.rotate_start = None
        self.rotate_angle_start = 0.0
        self.last_click_time = 0.0
        self.last_click_pos = (0, 0)

        # Smooth zoom animation
        self.animating = False
        self.anim_start_time = 0.0
        self.anim_duration = 0.6
        self.anim_start_cx = Decimal("0")
        self.anim_start_cy = Decimal("0")
        self.anim_target_cx = Decimal("0")
        self.anim_target_cy = Decimal("0")
        self.anim_start_zoom = 1.0
        self.anim_target_zoom = 1.0

        # Two-pass rendering flags
        self.needs_compute = True   # rerun iteration kernel
        self.needs_colorize = True  # rerun colorize kernel
        self.running = True
        self.panel_visible = True

        # FPS tracking
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0

        # UI buttons
        self.buttons = []

        # Modal dialog state: None, "save", or "load"
        self.modal = None
        self.modal_text = ""
        self.load_list = []
        self.load_scroll = 0
        self.load_hover = -1

        self._init_pygame()
        self._init_cuda()
        self._build_buttons()

    def _update_render_area(self):
        """Calculate render area dimensions to fit the current aspect ratio within the window."""
        _name, wr, hr = ASPECT_RATIOS[self.aspect_id]
        target_aspect = wr / hr
        window_aspect = self.width / self.height
        if window_aspect > target_aspect:
            # Window wider than target — pillarbox
            self.render_h = self.height
            self.render_w = int(self.height * target_aspect)
        else:
            # Window taller than target — letterbox
            self.render_w = self.width
            self.render_h = int(self.width / target_aspect)
        self.render_w = max(self.render_w, 64)
        self.render_h = max(self.render_h, 64)
        self.render_x = (self.width - self.render_w) // 2
        self.render_y = (self.height - self.render_h) // 2

    def _init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Mandelbrot Fractal Viewer (GPU)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 13)
        self.font_head = pygame.font.SysFont("monospace", 13, bold=True)
        self._update_render_area()
        self.surface = pygame.Surface((self.render_w, self.render_h))

    def _init_cuda(self):
        self.module = SourceModule(CUDA_KERNEL)
        self.k_compute_fast = self.module.get_function("compute_fast")
        self.k_compute_deep = self.module.get_function("compute_deep")
        self.k_colorize = self.module.get_function("colorize")
        self._alloc_gpu_buffers()

    def _alloc_gpu_buffers(self):
        self.host_buf = np.zeros((self.render_h, self.render_w, 3), dtype=np.uint8)
        self.gpu_rgb = cuda.mem_alloc(self.host_buf.nbytes)
        # Float buffer for cached smooth iteration counts
        iter_bytes = self.render_w * self.render_h * np.dtype(np.float32).itemsize
        self.gpu_iter = cuda.mem_alloc(iter_bytes)

    # ── UI button layout ──────────────────────────────────────

    def _build_buttons(self):
        self.buttons = []
        px = self.width - PANEL_W + PANEL_PAD
        bw = PANEL_W - 2 * PANEL_PAD
        hw = (bw - BTN_GAP) // 2

        y = PANEL_PAD + 20

        # Fractal Type
        y += SECTION_GAP + 4
        self.buttons.append((pygame.Rect(px, y, hw, BTN_H), "<", "fractal_prev"))
        self.buttons.append((pygame.Rect(px + hw + BTN_GAP, y, hw, BTN_H), ">", "fractal_next"))
        y += BTN_H + BTN_GAP
        self._fractal_label_y = y
        y += 18

        # Preset Locations
        y += SECTION_GAP + 4
        self.buttons.append((pygame.Rect(px, y, hw, BTN_H), "<", "preset_prev"))
        self.buttons.append((pygame.Rect(px + hw + BTN_GAP, y, hw, BTN_H), ">", "preset_next"))
        y += BTN_H + BTN_GAP
        self._preset_label_y = y
        y += 18

        # Iterations
        y += SECTION_GAP + 4
        self.buttons.append((pygame.Rect(px, y, hw, BTN_H), "-100", "iter_down"))
        self.buttons.append((pygame.Rect(px + hw + BTN_GAP, y, hw, BTN_H), "+100", "iter_up"))
        y += BTN_H + BTN_GAP
        self._iter_label_y = y
        y += 18

        # Palette
        y += SECTION_GAP + 4
        self.buttons.append((pygame.Rect(px, y, hw, BTN_H), "<", "palette_prev"))
        self.buttons.append((pygame.Rect(px + hw + BTN_GAP, y, hw, BTN_H), ">", "palette_next"))
        y += BTN_H + BTN_GAP
        self._palette_label_y = y
        y += 18

        # Aspect Ratio
        y += SECTION_GAP + 4
        self.buttons.append((pygame.Rect(px, y, hw, BTN_H), "<", "aspect_prev"))
        self.buttons.append((pygame.Rect(px + hw + BTN_GAP, y, hw, BTN_H), ">", "aspect_next"))
        y += BTN_H + BTN_GAP
        self._aspect_label_y = y
        y += 18

        # Color Cycle
        y += SECTION_GAP + 4
        self.buttons.append((pygame.Rect(px, y, bw, BTN_H), "Toggle Cycle", "cycle_toggle"))
        y += BTN_H + BTN_GAP
        self._cycle_label_y = y
        y += 20

        self.buttons.append((pygame.Rect(px, y, hw, BTN_H), "Slower", "speed_down"))
        self.buttons.append((pygame.Rect(px + hw + BTN_GAP, y, hw, BTN_H), "Faster", "speed_up"))
        y += BTN_H + BTN_GAP
        self._speed_label_y = y
        y += 20

        # Save / Load
        y += SECTION_GAP + 4
        self.buttons.append((pygame.Rect(px, y, hw, BTN_H), "Save", "save"))
        self.buttons.append((pygame.Rect(px + hw + BTN_GAP, y, hw, BTN_H), "Load", "load"))
        y += BTN_H + BTN_GAP
        self.buttons.append((pygame.Rect(px, y, bw, BTN_H), "Save Image", "save_image"))
        y += BTN_H + BTN_GAP

        # Reset
        y += SECTION_GAP
        self.buttons.append((pygame.Rect(px, y, bw, BTN_H), "Reset View", "reset"))
        y += BTN_H + BTN_GAP + 20

        self._panel_h = y + PANEL_PAD

    def _point_in_panel(self, x, y):
        if not self.panel_visible:
            return False
        return x >= self.width - PANEL_W and y <= self._panel_h

    # ── Button actions ────────────────────────────────────────

    def _do_action(self, action):
        if action == "fractal_prev":
            self.fractal_type = (self.fractal_type - 1) % len(FRACTAL_TYPES)
            self._reset_view_for_fractal()
        elif action == "fractal_next":
            self.fractal_type = (self.fractal_type + 1) % len(FRACTAL_TYPES)
            self._reset_view_for_fractal()
        elif action == "preset_prev":
            self.current_preset = (self.current_preset - 1) % len(PRESET_LOCATIONS)
            self._load_preset(self.current_preset)
        elif action == "preset_next":
            self.current_preset = (self.current_preset + 1) % len(PRESET_LOCATIONS)
            self._load_preset(self.current_preset)
        elif action == "iter_up":
            self.iter_offset += 100
            self._auto_iter()
            self.needs_compute = True
        elif action == "iter_down":
            self.iter_offset -= 100
            self._auto_iter()
            self.needs_compute = True
        elif action == "palette_prev":
            self.palette_id = (self.palette_id - 1) % len(PALETTE_NAMES)
            self.needs_colorize = True
        elif action == "palette_next":
            self.palette_id = (self.palette_id + 1) % len(PALETTE_NAMES)
            self.needs_colorize = True
        elif action == "aspect_prev":
            self.aspect_id = (self.aspect_id - 1) % len(ASPECT_RATIOS)
            self._apply_aspect_ratio()
        elif action == "aspect_next":
            self.aspect_id = (self.aspect_id + 1) % len(ASPECT_RATIOS)
            self._apply_aspect_ratio()
        elif action == "cycle_toggle":
            self.color_cycling = not self.color_cycling
        elif action == "speed_down":
            self.cycle_speed = max(0.25, round(self.cycle_speed - 0.25, 2))
        elif action == "speed_up":
            self.cycle_speed = min(8.0, round(self.cycle_speed + 0.25, 2))
        elif action == "save":
            self.modal = "save"
            self.modal_text = ""
        elif action == "save_image":
            self.modal = "save_image"
            self.modal_text = ""
        elif action == "load":
            self._refresh_saved_list()
            if self.load_list:
                self.modal = "load"
                self.load_scroll = 0
                self.load_hover = -1
        elif action == "reset":
            self.color_offset = 0.0
            self.color_cycling = False
            self.cycle_speed = 1.0
            self.palette_id = 0
            self.current_preset = 0
            self.fractal_type = 0
            self._reset_view_for_fractal()

    def _reset_view_for_fractal(self):
        """Reset center, zoom, iterations, and rotation to defaults for the current fractal type."""
        cx, cy, zoom, iter_off = FRACTAL_DEFAULTS[self.fractal_type]
        self.center_x = Decimal(cx)
        self.center_y = Decimal(cy)
        self.zoom = zoom
        self.iter_offset = iter_off
        self.rotation = 0.0
        self._auto_iter()
        self.needs_compute = True

    def _apply_aspect_ratio(self):
        """Update render area to match the selected aspect ratio within the current window."""
        self._update_render_area()
        self.surface = pygame.Surface((self.render_w, self.render_h))
        self._alloc_gpu_buffers()
        self.needs_compute = True
        self.save_res_idx = 0

    def _load_preset(self, preset_idx):
        """Load a preset location."""
        name, cx, cy, zoom, iter_off, ftype = PRESET_LOCATIONS[preset_idx]
        self.center_x = Decimal(cx)
        self.center_y = Decimal(cy)
        self.zoom = zoom
        self.iter_offset = iter_off
        self.fractal_type = ftype
        self._auto_iter()
        self.needs_compute = True

    # ── Save / Load locations ────────────────────────────────

    def _refresh_saved_list(self):
        """Scan the locations folder and build list of (display_name, filename)."""
        os.makedirs(LOCATIONS_DIR, exist_ok=True)
        self.load_list = []
        for f in sorted(os.listdir(LOCATIONS_DIR)):
            if not f.endswith(".json"):
                continue
            filepath = os.path.join(LOCATIONS_DIR, f)
            try:
                with open(filepath, "r") as fh:
                    data = json.load(fh)
                name = data.get("name", f.replace(".json", ""))
            except (json.JSONDecodeError, OSError):
                name = f.replace(".json", "")
            self.load_list.append((name, f))

    def _save_location(self, name):
        """Save all current settings to a named JSON file."""
        os.makedirs(LOCATIONS_DIR, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
        if not safe:
            safe = "unnamed"
        filename = f"{safe}.json"
        data = {
            "name": name,
            "center_x": str(self.center_x),
            "center_y": str(self.center_y),
            "zoom": self.zoom,
            "iter_offset": self.iter_offset,
            "max_iter": self.max_iter,
            "color_offset": self.color_offset,
            "color_cycling": self.color_cycling,
            "cycle_speed": self.cycle_speed,
            "palette_id": self.palette_id,
            "fractal_type": self.fractal_type,
            "rotation": self.rotation,
        }
        filepath = os.path.join(LOCATIONS_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _load_saved_location(self, filename):
        """Load settings from a saved file."""
        filepath = os.path.join(LOCATIONS_DIR, filename)
        with open(filepath, "r") as f:
            data = json.load(f)
        self.center_x = Decimal(data["center_x"])
        self.center_y = Decimal(data["center_y"])
        self.zoom = data["zoom"]
        self.iter_offset = data["iter_offset"]
        self.color_offset = data.get("color_offset", 0.0)
        self.color_cycling = data.get("color_cycling", False)
        self.cycle_speed = data.get("cycle_speed", 1.0)
        self.palette_id = data.get("palette_id", 0)
        self.fractal_type = data.get("fractal_type", 0)
        self.rotation = data.get("rotation", 0.0)
        self._auto_iter()
        self.needs_compute = True

    def _delete_saved_location(self, filename):
        """Delete a saved file and refresh the list."""
        filepath = os.path.join(LOCATIONS_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
        self._refresh_saved_list()
        if not self.load_list:
            self.modal = None

    # ── Save image ─────────────────────────────────────────────

    def _save_image(self, name):
        """Render the current view at the selected resolution and save as PNG."""
        os.makedirs(IMAGES_DIR, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
        if not safe:
            safe = "unnamed"
        if not safe.lower().endswith(".png"):
            safe += ".png"
        filepath = os.path.join(IMAGES_DIR, safe)

        aspect_name = ASPECT_RATIOS[self.aspect_id][0]
        presets = RESOLUTION_PRESETS[aspect_name]
        idx = min(self.save_res_idx, len(presets) - 1)
        img_w, img_h = presets[idx]
        img_buf = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        gpu_rgb_img = cuda.mem_alloc(img_buf.nbytes)
        iter_bytes = img_w * img_h * np.dtype(np.float32).itemsize
        gpu_iter_img = cuda.mem_alloc(iter_bytes)

        grid_x = math.ceil(img_w / BLOCK_SIZE[0])
        grid_y = math.ceil(img_h / BLOCK_SIZE[1])
        grid = (grid_x, grid_y, 1)

        rc = math.cos(self.rotation)
        rs = math.sin(self.rotation)

        if self.zoom >= DD_ZOOM_THRESHOLD:
            cx_hi, cx_lo = self._split_double(self.center_x)
            cy_hi, cy_lo = self._split_double(self.center_y)
            self.k_compute_deep(
                gpu_iter_img,
                np.int32(img_w), np.int32(img_h),
                np.float64(cx_hi), np.float64(cx_lo),
                np.float64(cy_hi), np.float64(cy_lo),
                np.float64(self.zoom),
                np.int32(self.max_iter),
                np.int32(self.fractal_type),
                np.float64(rc), np.float64(rs),
                block=BLOCK_SIZE, grid=grid,
            )
        else:
            self.k_compute_fast(
                gpu_iter_img,
                np.int32(img_w), np.int32(img_h),
                np.float64(float(self.center_x)),
                np.float64(float(self.center_y)),
                np.float64(self.zoom),
                np.int32(self.max_iter),
                np.int32(self.fractal_type),
                np.float64(rc), np.float64(rs),
                block=BLOCK_SIZE, grid=grid,
            )

        self.k_colorize(
            gpu_rgb_img,
            gpu_iter_img,
            np.int32(img_w), np.int32(img_h),
            np.float64(self.color_offset),
            np.int32(self.palette_id),
            block=BLOCK_SIZE, grid=grid,
        )

        cuda.memcpy_dtoh(img_buf, gpu_rgb_img)
        img_surface = pygame.Surface((img_w, img_h))
        pygame.surfarray.blit_array(img_surface, np.transpose(img_buf, (1, 0, 2)))
        pygame.image.save(img_surface, filepath)

        gpu_rgb_img.free()
        gpu_iter_img.free()

    # ── Core helpers ──────────────────────────────────────────

    @staticmethod
    def _split_double(val):
        hi = float(val)
        lo = float(val - Decimal(hi))
        return hi, lo

    def _pixel_to_fractal(self, px, py):
        rx = px - self.render_x
        ry = py - self.render_y
        aspect = Decimal(self.render_w) / Decimal(self.render_h)
        zoom_d = Decimal(self.zoom)
        raw_x = (Decimal(rx) / Decimal(self.render_w) - Decimal("0.5")) * aspect / zoom_d
        raw_y = (Decimal(ry) / Decimal(self.render_h) - Decimal("0.5")) / zoom_d
        rc = Decimal(math.cos(self.rotation))
        rs = Decimal(math.sin(self.rotation))
        fx = self.center_x + raw_x * rc - raw_y * rs
        fy = self.center_y + raw_x * rs + raw_y * rc
        return fx, fy

    def _auto_iter(self):
        self.max_iter = max(
            50,
            int(self.base_iter + 100 * math.log10(max(self.zoom, 1.0))) + self.iter_offset,
        )

    # ── GPU two-pass render ───────────────────────────────────

    def _run_compute(self):
        """Pass 1: run the expensive iteration kernel, cache results in gpu_iter."""
        grid_x = math.ceil(self.render_w / BLOCK_SIZE[0])
        grid_y = math.ceil(self.render_h / BLOCK_SIZE[1])
        grid = (grid_x, grid_y, 1)

        rc = math.cos(self.rotation)
        rs = math.sin(self.rotation)

        if self.zoom >= DD_ZOOM_THRESHOLD:
            cx_hi, cx_lo = self._split_double(self.center_x)
            cy_hi, cy_lo = self._split_double(self.center_y)
            self.k_compute_deep(
                self.gpu_iter,
                np.int32(self.render_w), np.int32(self.render_h),
                np.float64(cx_hi), np.float64(cx_lo),
                np.float64(cy_hi), np.float64(cy_lo),
                np.float64(self.zoom),
                np.int32(self.max_iter),
                np.int32(self.fractal_type),
                np.float64(rc), np.float64(rs),
                block=BLOCK_SIZE, grid=grid,
            )
        else:
            self.k_compute_fast(
                self.gpu_iter,
                np.int32(self.render_w), np.int32(self.render_h),
                np.float64(float(self.center_x)),
                np.float64(float(self.center_y)),
                np.float64(self.zoom),
                np.int32(self.max_iter),
                np.int32(self.fractal_type),
                np.float64(rc), np.float64(rs),
                block=BLOCK_SIZE, grid=grid,
            )

    def _run_colorize(self):
        """Pass 2: cheap palette lookup from cached iteration buffer."""
        grid_x = math.ceil(self.render_w / BLOCK_SIZE[0])
        grid_y = math.ceil(self.render_h / BLOCK_SIZE[1])
        grid = (grid_x, grid_y, 1)

        self.k_colorize(
            self.gpu_rgb,
            self.gpu_iter,
            np.int32(self.render_w), np.int32(self.render_h),
            np.float64(self.color_offset),
            np.int32(self.palette_id),
            block=BLOCK_SIZE, grid=grid,
        )

    def render(self):
        """Two-pass render: compute (if needed) then colorize."""
        if self.needs_compute:
            self._run_compute()
            self.needs_compute = False
            self.needs_colorize = True  # must recolor after new iteration data

        if self.needs_colorize:
            self._run_colorize()
            self.needs_colorize = False

            cuda.memcpy_dtoh(self.host_buf, self.gpu_rgb)
            arr = np.transpose(self.host_buf, (1, 0, 2))
            pygame.surfarray.blit_array(self.surface, arr)

    # ── HUD overlay (top-left info) ──────────────────────────

    def draw_overlay(self):
        mx, my = pygame.mouse.get_pos()
        mouse_fx, mouse_fy = self._pixel_to_fractal(mx, my)
        mode = "deep" if self.zoom >= DD_ZOOM_THRESHOLD else "fast"

        rot_deg = math.degrees(self.rotation) % 360
        lines = [
            f"Center: ({float(self.center_x):.15g}, {float(self.center_y):.15g})",
            f"Zoom: {self.zoom:.6e}   Kernel: {mode}   Rot: {rot_deg:.1f}\u00b0",
            f"Mouse: ({float(mouse_fx):.15g}, {float(mouse_fy):.15g})",
            f"FPS: {self.fps:.1f}",
        ]

        padding = 6
        lh = self.font.get_linesize()
        box_h = padding * 2 + lh * len(lines)
        box_w = padding * 2 + max(self.font.size(l)[0] for l in lines)

        overlay = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        for i, line in enumerate(lines):
            overlay.blit(self.font.render(line, True, COL_TEXT), (padding, padding + i * lh))
        self.screen.blit(overlay, (8, 8))

    # ── Control panel (right side) ───────────────────────────

    def draw_panel(self):
        if not self.panel_visible:
            hint = self.font.render("[Tab] Controls", True, (180, 180, 180))
            hint_bg = pygame.Surface((hint.get_width() + 12, hint.get_height() + 8), pygame.SRCALPHA)
            hint_bg.fill((0, 0, 0, 120))
            hint_bg.blit(hint, (6, 4))
            self.screen.blit(hint_bg, (self.width - hint_bg.get_width() - 8, 8))
            return

        panel_x = self.width - PANEL_W
        panel = pygame.Surface((PANEL_W, self._panel_h), pygame.SRCALPHA)
        panel.fill(COL_BG)
        self.screen.blit(panel, (panel_x, 0))

        mx, my = pygame.mouse.get_pos()

        # Title
        self.screen.blit(self.font_head.render("CONTROLS", True, COL_ACCENT),
                         (panel_x + PANEL_PAD, PANEL_PAD))

        # Section: FRACTAL TYPE
        sec_y = self.buttons[0][0].y - 18
        self.screen.blit(self.font_head.render("FRACTAL", True, COL_HEADING), (panel_x + PANEL_PAD, sec_y))
        fractal_name = FRACTAL_TYPES[self.fractal_type][0]
        self.screen.blit(self.font.render(fractal_name, True, COL_ACCENT),
                         (panel_x + PANEL_PAD, self._fractal_label_y))

        # Section: PRESET LOCATIONS
        sec_y = self.buttons[2][0].y - 18
        self.screen.blit(self.font_head.render("LOCATION", True, COL_HEADING), (panel_x + PANEL_PAD, sec_y))
        preset_name = PRESET_LOCATIONS[self.current_preset][0]
        self.screen.blit(self.font.render(preset_name, True, COL_ACCENT),
                         (panel_x + PANEL_PAD, self._preset_label_y))

        # Section: ITERATIONS
        sec_y = self.buttons[4][0].y - 18
        self.screen.blit(self.font_head.render("ITERATIONS", True, COL_HEADING), (panel_x + PANEL_PAD, sec_y))
        val = f"{self.max_iter}  (offset {self.iter_offset:+d})"
        self.screen.blit(self.font.render(val, True, COL_TEXT), (panel_x + PANEL_PAD, self._iter_label_y))

        # Section: PALETTE
        sec_y = self.buttons[6][0].y - 18
        self.screen.blit(self.font_head.render("PALETTE", True, COL_HEADING), (panel_x + PANEL_PAD, sec_y))
        self.screen.blit(self.font.render(PALETTE_NAMES[self.palette_id], True, COL_ACCENT),
                         (panel_x + PANEL_PAD, self._palette_label_y))

        # Section: ASPECT RATIO
        sec_y = self.buttons[8][0].y - 18
        self.screen.blit(self.font_head.render("ASPECT", True, COL_HEADING), (panel_x + PANEL_PAD, sec_y))
        aspect_name = ASPECT_RATIOS[self.aspect_id][0]
        self.screen.blit(self.font.render(aspect_name, True, COL_ACCENT),
                         (panel_x + PANEL_PAD, self._aspect_label_y))

        # Section: COLOR CYCLE
        sec_y = self.buttons[10][0].y - 18
        self.screen.blit(self.font_head.render("COLOR CYCLE", True, COL_HEADING), (panel_x + PANEL_PAD, sec_y))
        state = "ON" if self.color_cycling else "OFF"
        self.screen.blit(self.font.render(f"Cycling: {state}", True, COL_TEXT),
                         (panel_x + PANEL_PAD, self._cycle_label_y))
        self.screen.blit(self.font.render(f"Speed: {self.cycle_speed:.2f}x", True, COL_TEXT),
                         (panel_x + PANEL_PAD, self._speed_label_y))

        # Section: SAVE / LOAD
        sec_y = self.buttons[13][0].y - 18
        self.screen.blit(self.font_head.render("SAVE / LOAD", True, COL_HEADING), (panel_x + PANEL_PAD, sec_y))

        # Draw buttons
        for i, (rect, label, action) in enumerate(self.buttons):
            hovered = rect.collidepoint(mx, my)
            if action == "cycle_toggle" and self.color_cycling:
                color = COL_BTN_ACTIVE
            else:
                color = COL_BTN_HOVER if hovered else COL_BTN
            pygame.draw.rect(self.screen, color, rect, border_radius=4)
            pygame.draw.rect(self.screen, (80, 80, 100), rect, 1, border_radius=4)
            txt = self.font.render(label, True, COL_TEXT)
            tx = rect.x + (rect.w - txt.get_width()) // 2
            ty = rect.y + (rect.h - txt.get_height()) // 2
            self.screen.blit(txt, (tx, ty))

        # Tab hint at bottom
        hint = self.font.render("[Tab] Hide", True, (120, 120, 140))
        self.screen.blit(hint, (panel_x + PANEL_PAD, self._panel_h - 20))

    # ── Modal dialogs ──────────────────────────────────────────

    def draw_modal(self):
        if self.modal is None:
            return

        # Dim background
        dim = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        dim.fill((0, 0, 0, 140))
        self.screen.blit(dim, (0, 0))

        if self.modal == "save":
            self._draw_save_dialog()
        elif self.modal == "save_image":
            self._draw_save_image_dialog()
        elif self.modal == "load":
            self._draw_load_dialog()

    def _draw_save_dialog(self):
        dw, dh = 360, 120
        dx = (self.width - dw) // 2
        dy = (self.height - dh) // 2
        box = pygame.Surface((dw, dh), pygame.SRCALPHA)
        box.fill((30, 30, 45, 240))
        self.screen.blit(box, (dx, dy))
        pygame.draw.rect(self.screen, COL_ACCENT, (dx, dy, dw, dh), 1, border_radius=6)

        self.screen.blit(self.font_head.render("Save Location", True, COL_ACCENT),
                         (dx + 12, dy + 12))
        self.screen.blit(self.font.render("Enter a name and press Enter:", True, COL_TEXT),
                         (dx + 12, dy + 34))

        # Text input box
        input_rect = pygame.Rect(dx + 12, dy + 56, dw - 24, 28)
        pygame.draw.rect(self.screen, (50, 50, 70), input_rect, border_radius=3)
        pygame.draw.rect(self.screen, COL_ACCENT, input_rect, 1, border_radius=3)

        display_text = self.modal_text
        cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
        self.screen.blit(self.font.render(display_text + cursor, True, (255, 255, 255)),
                         (input_rect.x + 6, input_rect.y + 7))

        self.screen.blit(self.font.render("[Esc] Cancel", True, (120, 120, 140)),
                         (dx + 12, dy + dh - 22))

    def _draw_save_image_dialog(self):
        dw, dh = 400, 200
        dx = (self.width - dw) // 2
        dy = (self.height - dh) // 2
        box = pygame.Surface((dw, dh), pygame.SRCALPHA)
        box.fill((30, 30, 45, 240))
        self.screen.blit(box, (dx, dy))
        pygame.draw.rect(self.screen, COL_ACCENT, (dx, dy, dw, dh), 1, border_radius=6)

        self.screen.blit(self.font_head.render("Save Image", True, COL_ACCENT),
                         (dx + 12, dy + 12))
        self.screen.blit(self.font.render("Enter filename and press Enter:", True, COL_TEXT),
                         (dx + 12, dy + 34))

        # Filename input
        input_rect = pygame.Rect(dx + 12, dy + 56, dw - 24, 28)
        pygame.draw.rect(self.screen, (50, 50, 70), input_rect, border_radius=3)
        pygame.draw.rect(self.screen, COL_ACCENT, input_rect, 1, border_radius=3)

        display_text = self.modal_text
        cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
        self.screen.blit(self.font.render(display_text + cursor, True, (255, 255, 255)),
                         (input_rect.x + 6, input_rect.y + 7))

        # Resolution selector
        res_y = dy + 96
        self.screen.blit(self.font.render("Resolution:", True, COL_TEXT), (dx + 12, res_y))

        aspect_name = ASPECT_RATIOS[self.aspect_id][0]
        presets = RESOLUTION_PRESETS[aspect_name]
        self.save_res_idx = min(self.save_res_idx, len(presets) - 1)
        rw, rh = presets[self.save_res_idx]
        res_label = f"{rw} x {rh}"

        # < > buttons for resolution
        btn_w = 28
        btn_h = 24
        left_x = dx + 12
        right_x = dx + 12 + btn_w + 6
        btn_y = res_y + 20

        self._save_res_left_rect = pygame.Rect(left_x, btn_y, btn_w, btn_h)
        self._save_res_right_rect = pygame.Rect(right_x, btn_y, btn_w, btn_h)

        mx, my = pygame.mouse.get_pos()
        for rect, label in [(self._save_res_left_rect, "<"), (self._save_res_right_rect, ">")]:
            col = COL_BTN_HOVER if rect.collidepoint(mx, my) else COL_BTN
            pygame.draw.rect(self.screen, col, rect, border_radius=3)
            pygame.draw.rect(self.screen, (80, 80, 100), rect, 1, border_radius=3)
            txt = self.font.render(label, True, COL_TEXT)
            self.screen.blit(txt, (rect.x + (rect.w - txt.get_width()) // 2,
                                   rect.y + (rect.h - txt.get_height()) // 2))

        self.screen.blit(self.font.render(res_label, True, COL_ACCENT),
                         (right_x + btn_w + 10, btn_y + 4))

        self.screen.blit(self.font.render("[Esc] Cancel", True, (120, 120, 140)),
                         (dx + 12, dy + dh - 22))

    def _draw_load_dialog(self):
        ROW_H = 30
        MAX_VISIBLE = 10
        visible = min(len(self.load_list), MAX_VISIBLE)
        dw = 400
        dh = 52 + visible * ROW_H + 24
        dx = (self.width - dw) // 2
        dy = (self.height - dh) // 2

        box = pygame.Surface((dw, dh), pygame.SRCALPHA)
        box.fill((30, 30, 45, 240))
        self.screen.blit(box, (dx, dy))
        pygame.draw.rect(self.screen, COL_ACCENT, (dx, dy, dw, dh), 1, border_radius=6)

        self.screen.blit(self.font_head.render("Load Location", True, COL_ACCENT),
                         (dx + 12, dy + 12))

        if not self.load_list:
            self.screen.blit(self.font.render("No saved locations.", True, COL_TEXT),
                             (dx + 12, dy + 40))
        else:
            mx, my = pygame.mouse.get_pos()
            self.load_hover = -1
            list_y = dy + 40
            self.load_scroll = max(0, min(self.load_scroll, len(self.load_list) - MAX_VISIBLE))

            for i in range(visible):
                idx = i + self.load_scroll
                if idx >= len(self.load_list):
                    break
                name, filename = self.load_list[idx]
                row_rect = pygame.Rect(dx + 8, list_y + i * ROW_H, dw - 56, ROW_H - 2)
                del_rect = pygame.Rect(dx + dw - 44, list_y + i * ROW_H, 36, ROW_H - 2)

                # Hover highlight for row
                if row_rect.collidepoint(mx, my):
                    self.load_hover = idx
                    pygame.draw.rect(self.screen, (60, 60, 90), row_rect, border_radius=3)
                else:
                    pygame.draw.rect(self.screen, (40, 40, 58), row_rect, border_radius=3)

                display = name if len(name) <= 34 else name[:32] + ".."
                self.screen.blit(self.font.render(display, True, COL_TEXT),
                                 (row_rect.x + 8, row_rect.y + 7))

                # Delete button
                del_hover = del_rect.collidepoint(mx, my)
                del_col = (140, 60, 60) if del_hover else (80, 50, 50)
                pygame.draw.rect(self.screen, del_col, del_rect, border_radius=3)
                xtxt = self.font.render("X", True, (200, 100, 100))
                self.screen.blit(xtxt, (del_rect.x + (del_rect.w - xtxt.get_width()) // 2,
                                        del_rect.y + (del_rect.h - xtxt.get_height()) // 2))

            # Scroll hint
            if len(self.load_list) > MAX_VISIBLE:
                hint = f"Scroll: {self.load_scroll + 1}-{self.load_scroll + visible} of {len(self.load_list)}"
                self.screen.blit(self.font.render(hint, True, (120, 120, 140)),
                                 (dx + 12, dy + dh - 22))

        esc_x = dx + dw - self.font.size("[Esc] Cancel")[0] - 12
        self.screen.blit(self.font.render("[Esc] Cancel", True, (120, 120, 140)),
                         (esc_x, dy + dh - 22))

    # ── Event handling ────────────────────────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            # Route to modal handlers when a dialog is open
            if self.modal == "save":
                self._handle_save_modal_event(event)
                continue
            if self.modal == "save_image":
                self._handle_save_image_modal_event(event)
                continue
            if self.modal == "load":
                self._handle_load_modal_event(event)
                continue

            if event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self._handle_panel_click(event.pos):
                        continue
                    now = time.perf_counter()
                    dx = abs(event.pos[0] - self.last_click_pos[0])
                    dy = abs(event.pos[1] - self.last_click_pos[1])
                    if now - self.last_click_time < 0.35 and dx < 10 and dy < 10:
                        self._start_smooth_zoom(event.pos)
                        self.last_click_time = 0.0
                        continue
                    self.last_click_time = now
                    self.last_click_pos = event.pos
                    self.animating = False
                    self.dragging = True
                    self.drag_start = event.pos
                    self.drag_center_start = (self.center_x, self.center_y)
                elif event.button == 3:
                    self.rotating = True
                    self.rotate_start = event.pos
                    self.rotate_angle_start = self.rotation

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.dragging = False
                elif event.button == 3:
                    self.rotating = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    self._handle_drag(event.pos)
                if self.rotating:
                    self._handle_rotate(event.pos)

            elif event.type == pygame.MOUSEWHEEL:
                if not self._point_in_panel(*pygame.mouse.get_pos()):
                    self.animating = False
                    self._handle_zoom(event.y)

            elif event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)

    def _handle_save_modal_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.modal = None
            elif event.key == pygame.K_RETURN:
                name = self.modal_text.strip()
                if name:
                    self._save_location(name)
                self.modal = None
            elif event.key == pygame.K_BACKSPACE:
                self.modal_text = self.modal_text[:-1]
            else:
                ch = event.unicode
                if ch and ch.isprintable() and len(self.modal_text) < 40:
                    self.modal_text += ch

    def _handle_save_image_modal_event(self, event):
        aspect_name = ASPECT_RATIOS[self.aspect_id][0]
        num_presets = len(RESOLUTION_PRESETS[aspect_name])
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.modal = None
            elif event.key == pygame.K_RETURN:
                name = self.modal_text.strip()
                if name:
                    self._save_image(name)
                self.modal = None
            elif event.key == pygame.K_LEFT:
                self.save_res_idx = (self.save_res_idx - 1) % num_presets
            elif event.key == pygame.K_RIGHT:
                self.save_res_idx = (self.save_res_idx + 1) % num_presets
            elif event.key == pygame.K_BACKSPACE:
                self.modal_text = self.modal_text[:-1]
            else:
                ch = event.unicode
                if ch and ch.isprintable() and len(self.modal_text) < 40:
                    self.modal_text += ch
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if hasattr(self, '_save_res_left_rect') and self._save_res_left_rect.collidepoint(event.pos):
                self.save_res_idx = (self.save_res_idx - 1) % num_presets
            elif hasattr(self, '_save_res_right_rect') and self._save_res_right_rect.collidepoint(event.pos):
                self.save_res_idx = (self.save_res_idx + 1) % num_presets

    def _handle_load_modal_event(self, event):
        ROW_H = 30
        MAX_VISIBLE = 10
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.modal = None
        elif event.type == pygame.MOUSEWHEEL:
            self.load_scroll -= event.y
            self.load_scroll = max(0, min(self.load_scroll,
                                          len(self.load_list) - MAX_VISIBLE))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.load_list:
                return
            visible = min(len(self.load_list), MAX_VISIBLE)
            dw = 400
            dh = 52 + visible * ROW_H + 24
            dx = (self.width - dw) // 2
            dy = (self.height - dh) // 2
            list_y = dy + 40
            mx, my = event.pos

            for i in range(visible):
                idx = i + self.load_scroll
                if idx >= len(self.load_list):
                    break
                row_rect = pygame.Rect(dx + 8, list_y + i * ROW_H, dw - 56, ROW_H - 2)
                del_rect = pygame.Rect(dx + dw - 44, list_y + i * ROW_H, 36, ROW_H - 2)
                if del_rect.collidepoint(mx, my):
                    _name, filename = self.load_list[idx]
                    self._delete_saved_location(filename)
                    return
                if row_rect.collidepoint(mx, my):
                    _name, filename = self.load_list[idx]
                    self._load_saved_location(filename)
                    self.modal = None
                    return

    def _handle_panel_click(self, pos):
        if not self.panel_visible:
            return False
        for rect, _label, action in self.buttons:
            if rect.collidepoint(pos):
                self._do_action(action)
                return True
        return self._point_in_panel(*pos)

    def _handle_keydown(self, event):
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif event.key == pygame.K_TAB:
            self.panel_visible = not self.panel_visible
        elif event.key == pygame.K_c:
            self.color_cycling = not self.color_cycling
        elif event.key == pygame.K_UP:
            self.iter_offset += 100
            self._auto_iter()
            self.needs_compute = True
        elif event.key == pygame.K_DOWN:
            self.iter_offset -= 100
            self._auto_iter()
            self.needs_compute = True
        elif event.key == pygame.K_LEFT:
            self.palette_id = (self.palette_id - 1) % len(PALETTE_NAMES)
            self.needs_colorize = True
        elif event.key == pygame.K_RIGHT:
            self.palette_id = (self.palette_id + 1) % len(PALETTE_NAMES)
            self.needs_colorize = True
        elif event.key == pygame.K_COMMA or event.key == pygame.K_LESS:
            self._do_action("preset_prev")
        elif event.key == pygame.K_PERIOD or event.key == pygame.K_GREATER:
            self._do_action("preset_next")
        elif event.key == pygame.K_LEFTBRACKET:
            self._do_action("fractal_prev")
        elif event.key == pygame.K_RIGHTBRACKET:
            self._do_action("fractal_next")
        elif event.key == pygame.K_s:
            self.modal = "save"
            self.modal_text = ""
        elif event.key == pygame.K_l:
            self._refresh_saved_list()
            if self.load_list:
                self.modal = "load"
                self.load_scroll = 0
                self.load_hover = -1
        elif event.key == pygame.K_a:
            self._do_action("aspect_next")
        elif event.key == pygame.K_r:
            self._do_action("reset")

    def _handle_drag(self, pos):
        dx = pos[0] - self.drag_start[0]
        dy = pos[1] - self.drag_start[1]
        aspect = Decimal(self.render_w) / Decimal(self.render_h)
        zoom_d = Decimal(self.zoom)
        raw_x = Decimal(dx) / Decimal(self.render_w) * aspect / zoom_d
        raw_y = Decimal(dy) / Decimal(self.render_h) / zoom_d
        rc = Decimal(math.cos(self.rotation))
        rs = Decimal(math.sin(self.rotation))
        self.center_x = self.drag_center_start[0] - (raw_x * rc - raw_y * rs)
        self.center_y = self.drag_center_start[1] - (raw_x * rs + raw_y * rc)
        self.needs_compute = True

    def _start_smooth_zoom(self, pos):
        target_fx, target_fy = self._pixel_to_fractal(pos[0], pos[1])
        self.animating = True
        self.anim_start_time = time.perf_counter()
        self.anim_start_cx = self.center_x
        self.anim_start_cy = self.center_y
        self.anim_start_zoom = self.zoom
        self.anim_target_cx = target_fx
        self.anim_target_cy = target_fy
        self.anim_target_zoom = self.zoom * 8.0

    def _update_animation(self):
        if not self.animating:
            return
        t = (time.perf_counter() - self.anim_start_time) / self.anim_duration
        if t >= 1.0:
            t = 1.0
            self.animating = False
        # Smooth ease-in-out
        t = t * t * (3.0 - 2.0 * t)
        # Interpolate zoom exponentially for smooth feel
        log_start = math.log(self.anim_start_zoom)
        log_target = math.log(self.anim_target_zoom)
        self.zoom = math.exp(log_start + (log_target - log_start) * t)
        # Couple center to zoom so the target point stays fixed on screen
        zoom_ratio = Decimal(self.anim_start_zoom / self.zoom)
        self.center_x = self.anim_target_cx - (self.anim_target_cx - self.anim_start_cx) * zoom_ratio
        self.center_y = self.anim_target_cy - (self.anim_target_cy - self.anim_start_cy) * zoom_ratio
        self._auto_iter()
        self.needs_compute = True

    def _handle_rotate(self, pos):
        dx = pos[0] - self.rotate_start[0]
        self.rotation = self.rotate_angle_start + dx * 0.005
        self.needs_compute = True

    def _handle_zoom(self, scroll_y):
        mx, my = pygame.mouse.get_pos()
        fx_before, fy_before = self._pixel_to_fractal(mx, my)

        if scroll_y > 0:
            self.zoom *= 1.3
        elif scroll_y < 0:
            self.zoom /= 1.3

        fx_after, fy_after = self._pixel_to_fractal(mx, my)
        self.center_x += fx_before - fx_after
        self.center_y += fy_before - fy_after

        self._auto_iter()
        self.needs_compute = True

    def _handle_resize(self, new_w, new_h):
        self.width = max(new_w, 64)
        self.height = max(new_h, 64)
        self.screen = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self._update_render_area()
        self.surface = pygame.Surface((self.render_w, self.render_h))
        self._alloc_gpu_buffers()
        self._build_buttons()
        self.needs_compute = True

    def _update_fps(self):
        now = time.perf_counter()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        self.frame_times.append(dt)
        if self.frame_times:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg_dt if avg_dt > 0 else 0.0

    # ── Main loop ─────────────────────────────────────────────

    def run(self):
        try:
            while self.running:
                self.handle_events()
                self._update_animation()

                if self.color_cycling:
                    self.color_offset += 0.005 * self.cycle_speed
                    self.needs_colorize = True

                if self.needs_compute or self.needs_colorize:
                    self.render()

                self.screen.fill((0, 0, 0))
                self.screen.blit(self.surface, (self.render_x, self.render_y))
                self.draw_overlay()
                self.draw_panel()
                self.draw_modal()
                pygame.display.flip()

                self._update_fps()
                self.clock.tick(60)
        finally:
            pygame.quit()


def main():
    viewer = MandelbrotViewer()
    viewer.run()


if __name__ == "__main__":
    main()
