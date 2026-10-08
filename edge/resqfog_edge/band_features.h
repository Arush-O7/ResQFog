// On-device version of ml/features.py, used when readings go over LoRa
// (a raw 256-sample window does not fit in one LoRa packet).
//
// Same steps as the Python code: remove the mean, Hann window, power
// spectrum, sum over 10 bands (0-250 Hz), log10. Checked against the
// Python implementation with real CWRU windows (see ml/check_band_features.py).

#pragma once
#include <math.h>

#define BAND_COUNT 10

static const float BAND_EDGES[BAND_COUNT + 1] = {0, 20, 40, 60, 80, 100, 125, 150, 175, 200, 250};

// In-place radix-2 FFT, n must be a power of two
static void fftInPlace(float* re, float* im, int n) {
  for (int i = 1, j = 0; i < n; i++) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    float angle = -2.0f * (float)M_PI / len;
    float wr = cosf(angle), wi = sinf(angle);
    for (int i = 0; i < n; i += len) {
      float cr = 1.0f, ci = 0.0f;
      for (int k = 0; k < len / 2; k++) {
        int a = i + k, b = i + k + len / 2;
        float tr = re[b] * cr - im[b] * ci;
        float ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr;        im[a] += ti;
        float nr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = nr;
      }
    }
  }
}

// x: n samples of acceleration magnitude in g, fs: sampling rate in Hz
static void computeBandEnergies(const float* x, int n, float fs, float* out) {
  static float re[256], im[256];
  float mean = 0.0f;
  for (int i = 0; i < n; i++) mean += x[i];
  mean /= n;
  for (int i = 0; i < n; i++) {
    float w = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (n - 1));
    re[i] = (x[i] - mean) * w;
    im[i] = 0.0f;
  }
  fftInPlace(re, im, n);

  double sums[BAND_COUNT] = {0};
  for (int k = 0; k <= n / 2; k++) {
    float f = k * fs / n;
    double p = (double)re[k] * re[k] + (double)im[k] * im[k];
    for (int b = 0; b < BAND_COUNT; b++) {
      if (f >= BAND_EDGES[b] && f < BAND_EDGES[b + 1]) { sums[b] += p; break; }
    }
  }
  for (int b = 0; b < BAND_COUNT; b++) out[b] = (float)log10(sums[b] + 1e-12);
}
