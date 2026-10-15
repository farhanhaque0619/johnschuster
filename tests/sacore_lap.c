/*
 * A capture reader that has fallen a whole ring behind, racing the IO thread that overwrites what it
 * is reading (test_sacore_ring.RealThreadTest): sa_in_read must either hand back the right frames or
 * say -2 (lapped), never frames from a ring later. The "device" calls back without pausing; the reader
 * keeps reading at the very edge of being lapped. Not for ThreadSanitizer: those reads race on purpose.
 *   cc -O2 -std=c11 tests/sacore_lap.c -lpthread -lm && ./a.out 2
 */
#include "../native/sacore.c"
#include <stdio.h>

#define F 512u

static _Atomic int stop;

static void *device(void *arg) {                /* input frame k is k (mod 2^24, exact in a float) */
    sa_ring *h = arg;
    static float buf[F];
    for (uint64_t k = 0; !atomic_load(&stop); k++) {
        for (uint32_t i = 0; i < F; i++) buf[i] = (float)((k * F + i) % (1u << 24));
        sa_timeinfo ti = {1.0 + (double)(k * F) / 48000.0, 0, 0};
        sa_in_callback(buf, NULL, F, &ti, 0, h);
    }
    return NULL;
}

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 2;
    sa_ring *h = sa_ring_new(1, 1, 48000.0, 1u << 14);
    pthread_t th;
    pthread_create(&th, NULL, device, h);
    static float dst[480];
    long ok = 0, wrong = 0, lapped = 0;
    uint32_t seed = 1;
    for (double end = sa_now() + seconds; sa_now() < end;) {
        uint64_t w = atomic_load(&h->w);
        if (w < h->cap + 2 * F) continue;
        seed = seed * 1664525u + 1013904223u;
        uint64_t pos = w - h->cap + seed % (2 * F);     /* within two callbacks of being lapped */
        int64_t got = sa_in_read(h, pos, dst, 480);
        if (got == -2) { lapped++; continue; }
        if (got != 480) continue;
        ok++;
        for (uint32_t i = 0; i < 480; i++)
            if (dst[i] != (float)((pos + i) % (1u << 24))) {
                if (wrong++ < 3) fprintf(stderr, "pos %llu frame %u: got %.0f\n", (unsigned long long)pos, i, dst[i]);
                break;
            }
    }
    atomic_store(&stop, 1);
    pthread_join(th, NULL);
    sa_ring_free(h);
    printf("%ld reads accepted, %ld of them with overwritten frames; %ld said lapped\n", ok, wrong, lapped);
    return wrong || !ok || !lapped;
}
