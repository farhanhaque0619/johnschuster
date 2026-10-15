/*
 * sacore's hand-overs between threads, all at once, for a ThreadSanitizer build
 * (test_sacore_ring.TSanTest): an output ring with a "device" thread calling its callback, a
 * producer appending, rewriting, cutting short and starting new runs, and two log readers (one always
 * reading the oldest records, the slots being overwritten); an input ring with a device thread and a
 * reader.
 * TSan reports any access between threads that the rings' atomics don't order; this program also
 * checks what each thread gets.   cc -fsanitize=thread -g tests/sacore_race.c && ./a.out 4
 */
#include "../native/sacore.c"
#include <stdio.h>

#define RATE 48000.0
#define F 256u                          /* frames per device callback */
#define GUARD (4 * F)                   /* the producer stays this far ahead of the device */

static _Atomic int stop;
static _Atomic long problems, played, records, raced, captured;

static void bad(const char *what, double a, double b) {
    if (atomic_fetch_add(&problems, 1) < 10) fprintf(stderr, "%s: %g vs %g\n", what, a, b);
}

static void nap(long us) { struct timespec ts = {us / 1000000, us % 1000000 * 1000}; nanosleep(&ts, NULL); }

static uint32_t lcg(uint32_t *s) { return *s = *s * 1664525u + 1013904223u; }

/* frame p is (p + 1, -(p + 1)): a fade scales both, nothing else can make them disagree */
static void *device_out(void *arg) {
    sa_ring *h = arg;
    static float buf[F * 2];
    for (uint64_t k = 0; !atomic_load(&stop); k++) {
        sa_timeinfo ti = {0, 0, 1.0 + (double)(k * F) / RATE};
        sa_out_callback(NULL, buf, F, &ti, 0, h);
        for (uint32_t i = 0; i < F; i++) {
            float a = buf[2 * i], b = buf[2 * i + 1], p = (float)(k * F + i + 1);
            if (a != -b || a < 0 || a > p) bad("output frame", a, p);
        }
        atomic_fetch_add(&played, F);
        nap(200);
    }
    return NULL;
}

static void *producer(void *arg) {
    sa_ring *h = arg;
    static float x[960 * 2];
    uint64_t st[13];
    uint32_t seed = 1;
    while (!atomic_load(&stop)) {
        sa_ring_state(h, st);
        uint64_t r = st[0], w = st[1], pos, n = 480;
        uint32_t op = lcg(&seed) % 10;
        if (w < r + GUARD || op == 0)                                   /* a new run */
            pos = (w > r + GUARD ? w : r + GUARD) + 1 + lcg(&seed) % 1000;
        else if (op < 3)                                                /* rewrite what isn't played yet */
            pos = w - (w - (r + GUARD) < 300 ? w - (r + GUARD) : 300);
        else if (op == 3 && w > st[2] + 300 && w > r + GUARD + 300) {  /* forget the end of what's written */
            sa_out_cut(h, w - 300, GUARD);
            continue;
        } else
            pos = w;                                                    /* append */
        if (pos + n > r + h->cap) { nap(300); continue; }
        for (uint64_t i = 0; i < n; i++) x[2 * i] = (float)(pos + i + 1), x[2 * i + 1] = -(float)(pos + i + 1);
        sa_out_write(h, pos, x, (uint32_t)n, GUARD);
        nap(50);
    }
    return NULL;
}

/* every record must be whole: frame index and DAC time agree */
static void *log_reader(void *arg) {
    sa_ring *h = arg;
    static sa_log recs[LOG_N];
    uint64_t since = 0, st[13];
    while (!atomic_load(&stop)) {
        uint32_t n = sa_ring_log(h, since, recs, LOG_N);
        for (uint32_t i = 0; i < n; i++) {
            if (recs[i].seq <= since) bad("record order", (double)recs[i].seq, (double)since);
            if (recs[i].frames != F || recs[i].dev_time != 1.0 + (double)recs[i].pos / RATE ||
                recs[i].pos != (recs[i].seq - 1) * F || recs[i].exit_time < recs[i].entry_time)
                bad("torn record", (double)recs[i].pos, recs[i].dev_time);
            since = recs[i].seq;
        }
        atomic_fetch_add(&records, n);
        sa_ring_state(h, st);
        nap(100);
    }
    return NULL;
}

static void *oldest_reader(void *arg) {
    sa_ring *h = arg;
    static sa_log recs[LOG_N];
    while (!atomic_load(&stop)) {
        uint64_t last = atomic_load(&h->log_seq);
        uint32_t n = sa_ring_log(h, last > LOG_N ? last - LOG_N : 0, recs, LOG_N);
        for (uint32_t i = 0; i < n; i++)
            if (recs[i].frames != F || recs[i].dev_time != 1.0 + (double)recs[i].pos / RATE)
                bad("torn oldest record", (double)recs[i].pos, recs[i].dev_time);
        atomic_fetch_add(&raced, (last < LOG_N ? last : LOG_N) - n);
    }
    return NULL;
}

/* input frame p is p (mod 2^24, exact in a float) */
static void *device_in(void *arg) {
    sa_ring *h = arg;
    static float buf[F * 2];
    for (uint64_t k = 0; !atomic_load(&stop); k++) {
        for (uint32_t i = 0; i < F; i++) buf[2 * i] = buf[2 * i + 1] = (float)((k * F + i) % (1u << 24));
        sa_timeinfo ti = {1.0 + (double)(k * F) / RATE, 0, 0};
        sa_in_callback(buf, NULL, F, &ti, 0, h);
        nap(200);
    }
    return NULL;
}

static void *capture(void *arg) {
    sa_ring *h = arg;
    static float x[480 * 2];
    uint64_t pos = 0;
    while (!atomic_load(&stop)) {
        int64_t got = sa_in_read(h, pos, x, 480);
        if (got == -2) { bad("lapped", (double)pos, 0); pos = atomic_load(&h->w); continue; }
        if (got < 0) { nap(100); continue; }
        for (uint32_t i = 0; i < 480; i++)
            if (x[2 * i] != (float)((pos + i) % (1u << 24)) || x[2 * i + 1] != x[2 * i])
                bad("captured frame", x[2 * i], (double)(pos + i));
        pos += 480;
        atomic_fetch_add(&captured, 480);
    }
    return NULL;
}

int main(int argc, char **argv) {
    double seconds = argc > 1 ? atof(argv[1]) : 4;
    sa_ring *out = sa_ring_new(0, 2, RATE, 8192), *in = sa_ring_new(1, 2, RATE, 8192);
    void *(*fn[])(void *) = {device_out, producer, log_reader, oldest_reader, device_in, capture};
    void *arg[] = {out, out, out, out, in, in};
    pthread_t th[6];
    for (int i = 0; i < 6; i++) pthread_create(&th[i], NULL, fn[i], arg[i]);
    nap((long)(seconds * 1e6));
    atomic_store(&stop, 1);
    for (int i = 0; i < 6; i++) pthread_join(th[i], NULL);
    uint64_t st[13];
    sa_ring_state(out, st);
    printf("%ld frames played (%llu underrun events), %ld records read, %ld raced, %ld frames captured, "
           "%ld problems\n", atomic_load(&played), (unsigned long long)st[5], atomic_load(&records),
           atomic_load(&raced), atomic_load(&captured), atomic_load(&problems));
    sa_ring_free(out);
    sa_ring_free(in);
    return atomic_load(&problems) || !atomic_load(&records) || !atomic_load(&captured);
}
