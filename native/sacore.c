/*
 * sacore - the only code that runs on CoreAudio's real-time IO threads in SyncAudio 2.0.
 *
 * macOS calls an audio app's "IO callback" on a real-time thread with a hard deadline: miss it and
 * that slice of sound is skipped (static). Python can't be trusted with that deadline (any of the
 * app's threads may be holding the GIL), so these callbacks are plain C and Python never runs on
 * those threads at all.
 *
 * Two lock-free single-producer/single-consumer rings whose positions are DEVICE FRAME INDICES
 * (time), not "how much was written":
 *
 *   output: the IO thread (consumer) owns r = index of the next frame the device asks for. r
 *           advances by `frames` every callback whether or not audio was there, and by the skipped
 *           amount when macOS skipped IO cycles (its DAC time jumped). So frame index p always
 *           plays at one straight line in host time; Python writes the audio for index p
 *           150 ms - 1 s before it plays.
 *   input:  the IO thread (producer) owns w = device frame index of the next captured frame.
 *
 * The callbacks never take the GIL, never lock, never allocate, never call the OS (the clock is a
 * commpage / vDSO read). They copy, fade at gaps, count, and log one fixed-size record per
 * callback for Python to read.
 *
 * The callbacks have PortAudio's PaStreamCallback signature, so sounddevice can open the stream
 * (blocksize, device, AUHAL setup) and call them directly:
 *   sd._StreamBase(kind='output', callback=<int address of sa_out_callback>, userdata=<ring>)
 *
 * Also here, for the output engine: a ring of received audio by stream sample index and a
 * band-limited resampler (one GIL-free call per block), and a fake IO thread for the tests.
 * Built by native/build.sh; on macOS sacore_mac.c (device facts, listeners) is included at the end.
 */
#ifndef __APPLE__
#define _POSIX_C_SOURCE 200809L     /* clock_gettime, nanosleep and pthreads under -std=c11 */
#endif
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

#define SA_API __attribute__((visibility("default")))
#define SA_VERSION 0x020000
#define LOG_N 4096u                 /* per-callback records kept (power of two): ~3 min at 2048 frames */
#define FADE 64u                    /* frames of linear fade at gap edges (1.3 ms at 48 kHz) */
#define MAX_CH 8u

typedef struct { double inputBufferAdcTime, currentTime, outputBufferDacTime; } sa_timeinfo;  /* PortAudio's */

/* One record per callback. Python reads these to learn the frame-index -> host-time line. */
typedef struct {
    uint64_t seq;           /* 1, 2, 3, ... */
    uint64_t pos;           /* device frame index of the first frame of this buffer */
    double   dev_time;      /* when that frame reaches the DAC (output) / was sampled (input); <= 0 unknown */
    double   current_time;  /* PortAudio's "now" at IOProc entry */
    double   entry_time;    /* our clock when this callback started... */
    double   exit_time;     /* ...and when it returned (its cost) */
    uint32_t frames;
    uint32_t valid;         /* frames that had audio (output) */
    uint32_t skipped;       /* frames macOS skipped just before this buffer (0 normally) */
    uint32_t flags;         /* PortAudio status flags */
} sa_log;
_Static_assert(sizeof(sa_log) == 64, "sacore.LOG_DTYPE mirrors this");

/* A log record as stored: eight 8-byte words, each written and read atomically, so a reader that
 * races the writer gets whole words and can tell from the sequence number whether the record
 * changed under it (a seqlock: see log_put and sa_ring_log). */
typedef struct { _Atomic uint64_t word[8]; } sa_slot;

typedef struct {
    uint32_t kind, channels, cap, mask;     /* kind 0 = output, 1 = input; cap = frames, power of two */
    double rate;
    float *buf;
    _Atomic uint64_t r, w, s;               /* see top; s = where the current run of output audio starts */
    _Atomic uint64_t ps, pe;                /* output: the run before it, which still plays to its end */
    _Atomic uint64_t wz;                    /* input: the IO thread is writing slots up to here (sa_in_read) */
    _Atomic uint64_t callbacks, underrun_frames, underrun_events, skips, skipped_frames, overflows;
    _Atomic uint64_t min_lead_frames, flag_or, max_cb_ns;    /* reset each time Python reads them */
    _Atomic uint32_t armed;                 /* output: count underruns only once Python has written */
    _Atomic uint32_t stall_us;              /* CI: make the next callback this late, once */
    /* the IO thread's own state */
    double next_time, next_cur;             /* when the next buffer is due: device time, PortAudio's "now" */
    int have_next, was_missing;
    _Atomic uint64_t log_seq;
    sa_slot log[LOG_N];
} sa_ring;

#ifdef __APPLE__
static double tick_s;                       /* seconds per mach_absolute_time() tick */
__attribute__((constructor)) static void init_clock(void) {     /* a kernel call: never on an IO thread */
    mach_timebase_info_data_t tb;
    mach_timebase_info(&tb);
    tick_s = (double)tb.numer / tb.denom * 1e-9;
}
#endif

/* The clock PortAudio's timestamps and Python's time.monotonic() use. */
SA_API double sa_now(void) {
#ifdef __APPLE__
    return (double)mach_absolute_time() * tick_s;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

SA_API uint32_t sa_version(void) { return SA_VERSION; }

/* Copy n <= cap frames into / out of a ring at index pos: at most two memcpys (the second when the
 * run wraps past the end of the ring). x == NULL writes silence. */
static void to_ring(float *ring, uint32_t cap, uint32_t ch, uint64_t pos, const float *x, uint64_t n) {
    uint64_t i = pos & (cap - 1), first = n < cap - i ? n : cap - i;
    size_t f = ch * sizeof(float);
    if (x) {
        memcpy(ring + i * ch, x, first * f);
        memcpy(ring, x + first * ch, (n - first) * f);
    } else {
        memset(ring + i * ch, 0, first * f);
        memset(ring, 0, (n - first) * f);
    }
}

static void from_ring(const float *ring, uint32_t cap, uint32_t ch, uint64_t pos, float *x, uint64_t n) {
    uint64_t i = pos & (cap - 1), first = n < cap - i ? n : cap - i;
    size_t f = ch * sizeof(float);
    memcpy(x, ring + i * ch, first * f);
    memcpy(x + first * ch, ring, (n - first) * f);
}

SA_API sa_ring *sa_ring_new(uint32_t kind, uint32_t channels, double rate, uint32_t cap_pow2) {
    if (kind > 1 || channels < 1 || channels > MAX_CH || cap_pow2 < 2 || (cap_pow2 & (cap_pow2 - 1)) || !(rate > 0))
        return NULL;
    sa_ring *h = calloc(1, sizeof(sa_ring));
    if (!h) return NULL;
    h->kind = kind; h->channels = channels; h->rate = rate; h->cap = cap_pow2; h->mask = cap_pow2 - 1;
    h->buf = calloc((size_t)cap_pow2 * channels, sizeof(float));
    if (!h->buf) { free(h); return NULL; }
    atomic_store(&h->min_lead_frames, UINT64_MAX);
    return h;
}

/* Only after the stream is closed: PortAudio must never call back with a freed ring. */
SA_API void sa_ring_free(sa_ring *h) { if (h) { free(h->buf); free(h); } }

/* CI only (sa_ring_debug_stall): spin so this callback is late, the way an overloaded Mac is. */
static void stall(sa_ring *h, double entry) {
    if (!atomic_load_explicit(&h->stall_us, memory_order_relaxed)) return;
    uint32_t us = atomic_exchange_explicit(&h->stall_us, 0, memory_order_relaxed);
    while (sa_now() < entry + us * 1e-6) {}
}

/* If the device's clock moved more than half a buffer past where this buffer was due, macOS
 * skipped IO cycles (overload): those frames were never asked for. Count them and move the index
 * past them, so index -> time stays one straight line and everything after still plays at the right
 * moment. A jump of 2 s or more is not a skip (the device restarted or the clock is off); a backward
 * step is left to Python's clock filter. A buffer without a time (<= 0) is assumed on time, so the
 * next real time is compared with where it should be after it. `cur` is PortAudio's "now" when the
 * IO cycle started. */
static uint32_t detect_skip(sa_ring *h, double t, double cur, uint32_t frames) {
    uint32_t skipped = 0;
    double period = frames / h->rate;
    if (h->have_next && t > 0 && frames) {
        double jump = t - h->next_time;
        if (jump > 0.5 * period && jump < 2.0) {
            /* When this IO cycle itself started late ("now" jumped too), macOS skipped whole cycles, but
             * Bluetooth timestamps wobble by a few ms: a jump within 10 % of a whole number of buffers
             * is exactly that many, or every skip would shift the index->time line by the wobble.
             * Otherwise only the timestamps moved - the device's latency grew - by exactly the jump. */
            double cycles = jump / period, whole = round(cycles);
            int late = cur > 0 && h->next_cur > 0 && cur - h->next_cur > 0.5 * period;
            skipped = late && whole >= 1 && fabs(cycles - whole) < 0.1 ? (uint32_t)whole * frames
                                                                      : (uint32_t)llround(jump * h->rate);
            atomic_fetch_add_explicit(&h->skips, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&h->skipped_frames, skipped, memory_order_relaxed);
        }
    }
    if (t > 0) { h->next_time = t + period; h->have_next = 1; }
    else if (h->have_next) h->next_time += period;
    h->next_cur = cur > 0 ? cur + period : 0;
    return skipped;
}

static void ramp(float *x, uint32_t n, uint32_t ch, int up) {
    for (uint32_t i = 0; i < n; i++) {
        float g = (float)(i + 1) / (float)(n + 1);
        if (!up) g = 1.0f - g;
        for (uint32_t c = 0; c < ch; c++) x[i * ch + c] *= g;
    }
}

/* Copy the audio for device frames [from, to) into a buffer of `frames` starting at frame r (only
 * what falls inside it), softening its edges: a fade-in where it starts inside the buffer or right
 * after audio that was missing or dropped (fade_in), a fade-out where it ends inside the buffer -
 * or, for a run that is `over` (nothing can follow it), also where it ends exactly at the buffer's
 * end. Returns how many frames were copied. */
static uint32_t play(const sa_ring *h, float *o, uint64_t r, uint32_t frames, uint64_t from, uint64_t to,
                     int fade_in, int over) {
    int fade_out = over ? to <= r + frames : to < r + frames;
    if (from < r) from = r;
    if (to > r + frames) to = r + frames;
    if (to <= from) return 0;
    uint32_t a = (uint32_t)(from - r), b = (uint32_t)(to - r), n = b - a < FADE ? b - a : FADE;
    uint32_t ch = h->channels;
    from_ring(h->buf, h->cap, ch, from, o + (size_t)a * ch, b - a);
    if (a > 0 || fade_in) ramp(o + (size_t)a * ch, n, ch, 1);
    if (fade_out) ramp(o + (size_t)(b - n) * ch, n, ch, 0);
    return b - a;
}

/* Publish one log record (the writer half of the seqlock): mark the slot busy, then a release
 * fence, so a reader that sees any new word also sees "busy"; then the words; then the sequence
 * number with release, so a reader that sees it sees all the words. */
static void log_put(sa_ring *h, sa_log *e) {
    uint64_t seq = atomic_load_explicit(&h->log_seq, memory_order_relaxed) + 1, word[8];
    sa_slot *slot = &h->log[seq & (LOG_N - 1)];
    e->seq = seq;
    memcpy(word, e, sizeof word);
    atomic_store_explicit(&slot->word[0], 0, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    for (int i = 1; i < 8; i++) atomic_store_explicit(&slot->word[i], word[i], memory_order_relaxed);
    atomic_store_explicit(&slot->word[0], seq, memory_order_release);
    atomic_store_explicit(&h->log_seq, seq, memory_order_release);
}

/* End of every callback: counters, the worst callback cost, the log record. */
static void finish(sa_ring *h, uint64_t pos, double dev, double cur, double entry, uint32_t frames,
                   uint32_t valid, uint32_t skipped, unsigned long flags) {
    atomic_fetch_add_explicit(&h->callbacks, 1, memory_order_relaxed);
    if (flags) atomic_fetch_or_explicit(&h->flag_or, flags, memory_order_relaxed);
    double exit = sa_now();
    uint64_t ns = exit > entry ? (uint64_t)((exit - entry) * 1e9) : 0;
    if (ns > atomic_load_explicit(&h->max_cb_ns, memory_order_relaxed))        /* only we raise it */
        atomic_store_explicit(&h->max_cb_ns, ns, memory_order_relaxed);
    sa_log e = {0, pos, dev, cur, entry, exit, frames, valid, skipped, (uint32_t)flags};
    log_put(h, &e);
}

SA_API int sa_out_callback(const void *in, void *out, unsigned long frames_ul, const sa_timeinfo *ti,
                           unsigned long flags, void *user) {
    (void)in;
    sa_ring *h = user;
    double entry = sa_now();
    uint32_t frames = (uint32_t)frames_ul, ch = h->channels;
    float *o = out;
    if (frames_ul > h->cap) { memset(o, 0, frames_ul * ch * sizeof(float)); return 0; }  /* never, at our sizes */
    stall(h, entry);
    double dac = ti ? ti->outputBufferDacTime : 0.0, cur = ti ? ti->currentTime : 0.0;
    uint32_t skipped = detect_skip(h, dac, cur, frames);
    uint64_t r = atomic_load_explicit(&h->r, memory_order_relaxed) + skipped;
    uint64_t w = atomic_load_explicit(&h->w, memory_order_acquire);     /* w, s, then ps / pe: see sa_out_write */
    uint64_t s = atomic_load_explicit(&h->s, memory_order_acquire);
    uint64_t ps = atomic_load_explicit(&h->ps, memory_order_relaxed);
    uint64_t pe = atomic_load_explicit(&h->pe, memory_order_relaxed);
    int armed = (int)atomic_load_explicit(&h->armed, memory_order_relaxed);
    if (armed) {
        uint64_t lead = w > r ? w - r : 0;
        if (lead < atomic_load_explicit(&h->min_lead_frames, memory_order_relaxed))
            atomic_store_explicit(&h->min_lead_frames, lead, memory_order_relaxed);
    }
    /* Audio is in the ring for the current run [s, w) and the run before it [ps, pe) (pe <= s: a
     * pe past s is the newer pair, read with the older s, whose run [s, w) is that one); everything
     * else is silence. Every edge between audio and silence is softened. */
    memset(o, 0, (size_t)frames * ch * sizeof(float));
    int fade_in = h->was_missing || skipped;
    uint32_t old = pe <= s ? play(h, o, r, frames, ps, pe, fade_in, 1) : 0;
    uint32_t valid = play(h, o, r, frames, s, w, fade_in, 0);
    /* An underrun is audio missing inside the run Python is writing (positions >= s), not the silence
     * before the first run or between runs. One event per stretch of missing audio. */
    uint64_t from = r > s ? r : s;
    uint32_t expect = r + frames > from ? (uint32_t)(r + frames - from) : 0;
    int missing = valid < expect;
    if (missing && armed) {
        atomic_fetch_add_explicit(&h->underrun_frames, expect - valid, memory_order_relaxed);
        if (!h->was_missing) atomic_fetch_add_explicit(&h->underrun_events, 1, memory_order_relaxed);
    }
    h->was_missing = missing;
    atomic_store_explicit(&h->r, r + frames, memory_order_release);
    finish(h, r, dac, cur, entry, frames, old + valid, skipped, flags);
    return 0;   /* paContinue */
}

SA_API int sa_in_callback(const void *in, void *out, unsigned long frames_ul, const sa_timeinfo *ti,
                          unsigned long flags, void *user) {
    (void)out;
    sa_ring *h = user;
    double entry = sa_now();
    uint32_t frames = (uint32_t)frames_ul, ch = h->channels;
    if (frames_ul > h->cap) return 0;       /* never, at our sizes */
    stall(h, entry);
    double adc = ti ? ti->inputBufferAdcTime : 0.0, cur = ti ? ti->currentTime : 0.0;
    uint32_t skipped = detect_skip(h, adc, cur, frames);
    uint64_t w = atomic_load_explicit(&h->w, memory_order_relaxed);
    /* acquire, before reusing slots: Python's reads of them happened before it stored r */
    uint64_t r = atomic_load_explicit(&h->r, memory_order_acquire);
    /* Say how far the writes below go before making them, so a reader that copied a slot they
     * overwrite finds out (sa_in_read). */
    atomic_store_explicit(&h->wz, w + skipped + frames, memory_order_relaxed);
    atomic_thread_fence(memory_order_release);
    /* Frames macOS skipped never arrived: they are silence at their own indices, so the index stays
     * time (only the last ring-full of them can matter). */
    uint32_t z = skipped < h->cap ? skipped : h->cap;
    to_ring(h->buf, h->cap, ch, w + skipped - z, NULL, z);
    w += skipped;
    to_ring(h->buf, h->cap, ch, w, in, frames);
    if (w + frames - r > h->cap)                        /* Python fell a ring behind: it lost audio */
        atomic_fetch_add_explicit(&h->overflows, 1, memory_order_relaxed);
    atomic_store_explicit(&h->w, w + frames, memory_order_release);
    finish(h, w, adc, cur, entry, frames, frames, skipped, flags);
    return 0;
}

/* ---------------------------------------------------------------- Python side (any thread) */
/* out: r, w, s, callbacks, underrun_frames, underrun_events, skips, skipped_frames, overflows,
 * min_lead_frames, log_seq, flag_or, max_cb_ns. min_lead_frames (UINT64_MAX: no callback yet),
 * flag_or and max_cb_ns cover the time since the previous call. */
SA_API void sa_ring_state(sa_ring *h, uint64_t out[13]) {
    out[0] = atomic_load(&h->r); out[1] = atomic_load(&h->w); out[2] = atomic_load(&h->s);
    out[3] = atomic_load(&h->callbacks); out[4] = atomic_load(&h->underrun_frames);
    out[5] = atomic_load(&h->underrun_events); out[6] = atomic_load(&h->skips);
    out[7] = atomic_load(&h->skipped_frames); out[8] = atomic_load(&h->overflows);
    out[9] = atomic_exchange(&h->min_lead_frames, UINT64_MAX); out[10] = atomic_load(&h->log_seq);
    out[11] = atomic_exchange(&h->flag_or, 0); out[12] = atomic_exchange(&h->max_cb_ns, 0);
}

/* Copy log records with seq > since into dst (at most max, oldest first). Returns how many. A record
 * the IO thread is overwriting right now (Python fell LOG_N records behind) is left out (the reader
 * half of the seqlock: read the sequence number, the words, then the sequence number again). */
SA_API uint32_t sa_ring_log(sa_ring *h, uint64_t since, sa_log *dst, uint32_t max) {
    uint64_t last = atomic_load_explicit(&h->log_seq, memory_order_acquire), q = since + 1;
    if (last >= LOG_N && q < last - LOG_N + 1) q = last - LOG_N + 1;
    uint32_t n = 0;
    for (; q <= last && n < max; q++) {
        sa_slot *slot = &h->log[q & (LOG_N - 1)];
        uint64_t word[8];
        word[0] = atomic_load_explicit(&slot->word[0], memory_order_acquire);
        for (int i = 1; i < 8; i++) word[i] = atomic_load_explicit(&slot->word[i], memory_order_relaxed);
        atomic_thread_fence(memory_order_acquire);
        if (word[0] == q && atomic_load_explicit(&slot->word[0], memory_order_relaxed) == q)
            memcpy(&dst[n++], word, sizeof word);
    }
    return n;
}

/* Output: put n frames at device index pos. pos == w appends; pos > w starts a new run there: the
 * run before it still plays to its end, with a fade-out, then silence until pos; r + guard <= pos < w
 * rewrites audio not yet played. Returns n, or -1 if pos is too close to the device's read position
 * (it may be reading it right now) or more than a ring ahead of it. */
SA_API int64_t sa_out_write(sa_ring *h, uint64_t pos, const float *x, uint32_t n, uint32_t guard) {
    uint64_t r = atomic_load_explicit(&h->r, memory_order_acquire);
    uint64_t w = atomic_load_explicit(&h->w, memory_order_relaxed);    /* we are its only writer */
    if (!n) return 0;
    if (pos < r + guard || pos + n > r + h->cap) return -1;
    to_ring(h->buf, h->cap, h->channels, pos, x, n);
    if (pos > w) {
        /* The IO thread reads w, then s, then ps / pe. Storing the old run's ends, then s, then w means
         * it never sees the new w with the old s, nor the new s with an older run's ends. */
        atomic_store_explicit(&h->ps, atomic_load_explicit(&h->s, memory_order_relaxed), memory_order_relaxed);
        atomic_store_explicit(&h->pe, w, memory_order_relaxed);
        atomic_store_explicit(&h->s, pos, memory_order_release);
    }
    /* stored even when unchanged, so a rewrite is published too */
    atomic_store_explicit(&h->w, pos + n > w ? pos + n : w, memory_order_release);
    atomic_store_explicit(&h->armed, 1, memory_order_relaxed);
    return n;
}

/* Output: forget the audio written from pos on (r + guard <= pos < w, in the current run), so the
 * ring ends there. The producer went back to rewrite from an earlier frame (a delay decrease) and the
 * new audio has only arrived this far: what follows in the ring is the old version. Returns 0, or -1. */
SA_API int sa_out_cut(sa_ring *h, uint64_t pos, uint32_t guard) {
    uint64_t r = atomic_load_explicit(&h->r, memory_order_acquire);
    uint64_t w = atomic_load_explicit(&h->w, memory_order_relaxed);
    if (pos < r + guard || pos >= w || pos < atomic_load_explicit(&h->s, memory_order_relaxed)) return -1;
    atomic_store_explicit(&h->w, pos, memory_order_release);
    return 0;
}

/* Input: copy n frames starting at device index pos and mark them read. Returns n, -1 if they
 * haven't all been captured yet, or -2 if the IO thread overwrote them, before or while they were
 * copied (Python fell a whole ring behind; nothing is copied that can be trusted). */
SA_API int64_t sa_in_read(sa_ring *h, uint64_t pos, float *dst, uint32_t n) {
    uint64_t w = atomic_load_explicit(&h->w, memory_order_acquire);
    if (pos + n > w) return -1;
    if (w - pos > h->cap) return -2;
    from_ring(h->buf, h->cap, h->channels, pos, dst, n);
    /* The reader half of a seqlock: the IO thread says how far it writes before writing (wz), so if
     * any slot copied above was being overwritten, wz says so now. */
    atomic_thread_fence(memory_order_acquire);
    if (atomic_load_explicit(&h->wz, memory_order_relaxed) - pos > h->cap) return -2;
    atomic_store_explicit(&h->r, pos + n, memory_order_release);
    return n;
}

/* CI only: the next callback spins `us` microseconds before doing its work (a forced overload, to
 * check on a real Mac that macOS's clock jumps over the cycles it skips). */
SA_API void sa_ring_debug_stall(sa_ring *h, uint32_t us) {
    atomic_store_explicit(&h->stall_us, us, memory_order_relaxed);
}

/* ------------------------------------------------- content ring + band-limited resampler
 * Received audio by stream sample index n (the network or capture thread writes, the output engine
 * reads; no IO thread). Kept in C so the output engine's heavy lifting is one GIL-free call per
 * block. */
typedef struct {
    uint32_t channels, cap, mask;
    float *buf;
    _Atomic int64_t lo, end;            /* valid samples: [lo, end) */
    _Atomic int empty;
} sa_content;

SA_API sa_content *sa_content_new(uint32_t channels, uint32_t cap_pow2) {
    if (channels < 1 || channels > MAX_CH || cap_pow2 < 8192 || (cap_pow2 & (cap_pow2 - 1))) return NULL;
    sa_content *c = calloc(1, sizeof(sa_content));
    if (!c) return NULL;
    c->channels = channels; c->cap = cap_pow2; c->mask = cap_pow2 - 1;
    c->buf = calloc((size_t)cap_pow2 * channels, sizeof(float));
    if (!c->buf) { free(c); return NULL; }
    atomic_store(&c->empty, 1);
    return c;
}

SA_API void sa_content_free(sa_content *c) { if (c) { free(c->buf); free(c); } }

/* Store k samples at index n. A block from far away (a new stream) starts over; a hole before n (a
 * block that never came) is zero-filled. Returns the number of zero-filled samples (-1: k too big).
 * The oldest samples are declared gone (lo) before their slots are reused, with 4096 samples to
 * spare for a reader that is between reading lo and reading the samples. (A reader racing a
 * restart can still get one block of the old stream mixed with the new; a restart is a jump the
 * output engine splices over anyway.) */
SA_API int64_t sa_content_put(sa_content *c, int64_t n, const float *x, uint32_t k) {
    int64_t cap = c->cap, end = atomic_load_explicit(&c->end, memory_order_relaxed), filled = 0;
    if (k > cap - 4096) return -1;
    if (atomic_load(&c->empty) || n < end - cap || n > end + cap) {
        if (n > end) { atomic_store(&c->lo, n); atomic_store(&c->end, n); }    /* through an empty */
        else { atomic_store(&c->end, n); atomic_store(&c->lo, n); }            /* range, not a wrong one */
        atomic_store(&c->empty, 0);
        end = n;
    }
    int64_t new_end = n + k > end ? n + k : end;
    if (new_end - cap + 4096 > atomic_load_explicit(&c->lo, memory_order_relaxed))
        atomic_store_explicit(&c->lo, new_end - cap + 4096, memory_order_release);
    if (n > end) {
        to_ring(c->buf, c->cap, c->channels, (uint64_t)end, NULL, (uint64_t)(n - end));
        filled = n - end;
    }
    to_ring(c->buf, c->cap, c->channels, (uint64_t)n, x, k);
    atomic_store_explicit(&c->end, new_end, memory_order_release);     /* after the samples */
    return filled;
}

SA_API void sa_content_range(sa_content *c, int64_t out[3] /* lo, end, empty */) {
    out[0] = atomic_load(&c->lo); out[1] = atomic_load(&c->end); out[2] = atomic_load(&c->empty);
}

/* The interpolation filter: `phases` rows of `taps` coefficients, plus one more row (the next whole
 * sample) to interpolate between. Set once at import (sacore.make_filter); never freed, so a
 * resampler running on another thread can't lose it. */
typedef struct { uint32_t taps, phases; float h[]; } sa_filter;
static _Atomic(sa_filter *) g_filter;

SA_API int sa_set_filter(const float *table, uint32_t phases, uint32_t taps) {
    if (taps < 2 || taps > 256 || taps % 2 || phases < 1 || phases > 4096) return -1;
    size_t n = (size_t)(phases + 1) * taps;
    sa_filter *f = malloc(sizeof(sa_filter) + n * sizeof(float));
    if (!f) return -1;
    f->taps = taps; f->phases = phases;
    memcpy(f->h, table, n * sizeof(float));
    atomic_store_explicit(&g_filter, f, memory_order_release);
    return 0;
}

/* dst[j] (+)= gain(j) * content at fractional index pos + j*ratio, band-limited (windowed sinc).
 * gain goes linearly from g0 (j = 0) to g1 (j = n - 1). Content outside [lo, end) counts as silence.
 * Returns how many output frames had no content at their own position. */
SA_API uint32_t sa_resample(sa_content *c, double pos, double ratio, uint32_t n, float *dst,
                            float g0, float g1, int add) {
    const sa_filter *f = atomic_load_explicit(&g_filter, memory_order_acquire);
    uint32_t ch = c->channels, missing = 0;
    int64_t lo = atomic_load_explicit(&c->lo, memory_order_acquire);
    int64_t end = atomic_load_explicit(&c->end, memory_order_acquire);
    if (!f || atomic_load(&c->empty) || !isfinite(pos) || !isfinite(ratio) || fabs(pos) > 1e15 ||
        fabs(pos + ratio * n) > 1e15) {      /* nothing (or nowhere) to play: silence, all of it missing */
        if (!add) memset(dst, 0, (size_t)n * ch * sizeof(float));
        return n;
    }
    uint32_t T = f->taps, P = f->phases;
    float acc[MAX_CH];
    for (uint32_t j = 0; j < n; j++) {
        double p = pos + ratio * j, fl = floor(p);
        int64_t i = (int64_t)fl;
        if (i < lo || i >= end) missing++;
        for (uint32_t q = 0; q < ch; q++) acc[q] = 0;
        double fp = (p - fl) * P;
        uint32_t m = (uint32_t)fp;
        if (m >= P) m = P - 1;
        float a = (float)(fp - m);
        const float *h0 = f->h + (size_t)m * T, *h1 = h0 + T;
        int64_t base = i - (int64_t)(T / 2 - 1);                /* taps cover [i - T/2 + 1, i + T/2] */
        int64_t t0 = lo > base ? lo - base : 0, t1 = end - base < T ? end - base : T;
        for (int64_t t = t0; t < t1; t++) {
            float hh = h0[t] + a * (h1[t] - h0[t]);
            const float *x = c->buf + (size_t)((uint64_t)(base + t) & c->mask) * ch;
            for (uint32_t q = 0; q < ch; q++) acc[q] += hh * x[q];
        }
        float g = n > 1 ? g0 + (g1 - g0) * (float)j / (float)(n - 1) : g1;
        float *d = dst + (size_t)j * ch;
        for (uint32_t q = 0; q < ch; q++) d[q] = add ? d[q] + g * acc[q] : g * acc[q];
    }
    return missing;
}

/* ----------------------------------------------------------- test-only fake CoreAudio IO thread
 * Calls a PortAudio-style callback once per IO cycle on its own thread with a HARD deadline, the
 * way the HAL does: if the callback returns after the cycle's deadline, that buffer is dropped and
 * the cycles whose start already passed are skipped (macOS: "skipping cycle due to overload"), so
 * the next reported DAC time jumps. A late cycle is the platform's fault if the thread woke after
 * the deadline (late_wake), the callback's if it woke in time and still returned late (late_cb).
 * It records what really reached the "speaker", by device frame index (frame k reaches it at
 * t0 + lead + k / true rate), so tests can check absolute timing and continuity at the device.
 * Optional: report jitter ("wobble"), a device clock off by `ppm`, and HAL-side skips every
 * `skip_every` seconds whatever the callback does. Input fakes deliver frame index k as
 * ((k mod 32768) - 16384) / 32768 on every channel: exact in float and in 16-bit PCM, so the index
 * can be read back after the network too.
 * sa_fake_loop instead makes a fake BlackHole that separate processes share (tests/coreaudio on
 * Linux): a tape both map, indexed by device sample time s, which plays at host time s / rate (every
 * process reads the same monotonic clock). As on the HAL, an output cycle runs `frames + safety`
 * ahead of its DAC time and writes its buffer at its sample times - unless it returned late, which
 * leaves silence there, as a dropped cycle does; an input cycle runs that far (plus LOOP_DELAY, so
 * no write can still be under way) behind its ADC time, and reads and clears its sample times. */
#ifndef SA_NO_FAKE
#include <pthread.h>
#include <sched.h>
typedef int (*sa_cb)(const void *, void *, unsigned long, const sa_timeinfo *, unsigned long, void *);
typedef struct {
    pthread_t th; _Atomic int stop; int kind;
    sa_cb cb; void *user;
    uint32_t frames, channels; double rate, lead, wobble, ppm, skip_every;
    float *rec; uint64_t rec_cap;       /* output that reached the device, by frame index */
    float *tape; uint64_t tape_mask, s0;    /* sa_fake_loop: shared tape, sample time of frame 0 */
    double t0;                          /* cycle 0 starts; output frame 0 reaches the DAC at t0 + lead */
    _Atomic uint64_t cycles, late, skipped_cycles, forced, late_wake, late_cb;
    double max_cb_s;
    uint64_t rng;
    int policy;
} sa_fake;

static double urand(uint64_t *s) { *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
                                   return (double)(*s >> 11) / 9007199254740992.0; }

static void sleep_until(double t) {
    double d = t - sa_now();
    if (d <= 0) return;
    struct timespec ts = {(time_t)d, (long)((d - (time_t)d) * 1e9)};
    nanosleep(&ts, NULL);
}

#define LOOP_DELAY 480u              /* frames an input reads behind the last possible write */

static void loop_io(sa_fake *f, uint64_t k, float *buf, int write) {
    uint64_t s = f->s0 + k * f->frames;
    for (uint32_t i = 0; i < f->frames; i++) {
        float *slot = f->tape + ((s + i) & f->tape_mask) * f->channels, *x = buf + (size_t)i * f->channels;
        for (uint32_t c = 0; c < f->channels; c++) {
            if (write) slot[c] = x[c];
            else { x[c] = slot[c]; slot[c] = 0.0f; }
        }
    }
}

static void *fake_run(void *arg) {
    sa_fake *f = arg;
    double true_rate = f->rate * (1.0 + f->ppm * 1e-6), period = f->frames / true_rate;
    { struct sched_param sp; pthread_getschedparam(pthread_self(), &f->policy, &sp); }
    float *buf = calloc((size_t)f->frames * f->channels, sizeof(float));
    float *inb = calloc((size_t)f->frames * f->channels, sizeof(float));
    uint64_t k = 0;
    double next_forced = f->skip_every > 0 ? f->t0 + f->skip_every : 1e300;
    while (buf && inb && !atomic_load(&f->stop)) {
        double start = f->t0 + k * period;
        sleep_until(start);
        if (start >= next_forced) {         /* the HAL itself drops a cycle */
            next_forced += f->skip_every; k++; atomic_fetch_add(&f->forced, 1); continue;
        }
        sa_timeinfo ti = {0, 0, 0};
        ti.currentTime = sa_now();
        double noise = f->wobble > 0 ? (urand(&f->rng) * 2 - 1) * f->wobble : 0;
        if (f->kind == 0) ti.outputBufferDacTime = f->t0 + f->lead + (double)k * f->frames / true_rate + noise;
        else if (f->tape) {
            ti.inputBufferAdcTime = f->t0 - f->lead + (double)k * f->frames / true_rate;
            loop_io(f, k, inb, 0);
        } else {
            ti.inputBufferAdcTime = f->t0 + (double)k * f->frames / true_rate - period + noise;
            for (uint32_t i = 0; i < f->frames; i++)
                for (uint32_t c = 0; c < f->channels; c++)
                    inb[i * f->channels + c] = (float)((int)((k * f->frames + i) % 32768) - 16384) / 32768.0f;
        }
        double t_call = sa_now();
        f->cb(f->kind ? inb : NULL, f->kind ? NULL : buf, f->frames, &ti, 0, f->user);
        double done = sa_now();
        if (done - t_call > f->max_cb_s) f->max_cb_s = done - t_call;
        atomic_fetch_add(&f->cycles, 1);
        double deadline = start + period;
        if (done > deadline) {              /* overload: this buffer is lost, missed cycles skipped */
            atomic_fetch_add(&f->late, 1);
            atomic_fetch_add(t_call > deadline ? &f->late_wake : &f->late_cb, 1);
            uint64_t next = (uint64_t)ceil((done - f->t0) / period);
            if (next <= k + 1) next = k + 1;
            atomic_fetch_add(&f->skipped_cycles, next - (k + 1));
            k = next;
            continue;
        }
        uint64_t at = k * f->frames;
        if (f->kind == 0 && f->rec && at + f->frames <= f->rec_cap)
            memcpy(f->rec + at * f->channels, buf, (size_t)f->frames * f->channels * sizeof(float));
        if (f->kind == 0 && f->tape) loop_io(f, k, buf, 1);
        k++;
    }
    free(buf); free(inb);
    return NULL;
}

static sa_fake *fake_new(int kind, void *cb, void *user, uint32_t frames, uint32_t channels, double rate) {
    sa_fake *f = calloc(1, sizeof(sa_fake));
    if (!f) return NULL;
    f->kind = kind; f->cb = (sa_cb)cb; f->user = user; f->frames = frames; f->channels = channels;
    f->rate = rate;
    f->rng = 0x9E3779B97F4A7C15ULL ^ (uint64_t)(sa_now() * 1e9);
    return f;
}

static sa_fake *fake_go(sa_fake *f, int realtime) {
    pthread_attr_t at;
    pthread_attr_init(&at);
    if (realtime) {                       /* like CoreAudio's IO thread, where we're allowed to */
        struct sched_param sp = {.sched_priority = 80};
        pthread_attr_setinheritsched(&at, PTHREAD_EXPLICIT_SCHED);
        pthread_attr_setschedpolicy(&at, SCHED_FIFO);
        pthread_attr_setschedparam(&at, &sp);
    }
    int err = pthread_create(&f->th, &at, fake_run, f);
    pthread_attr_destroy(&at);
    if (err && realtime) err = pthread_create(&f->th, NULL, fake_run, f);   /* not allowed: ordinary */
    if (err) { free(f->rec); free(f); return NULL; }
    return f;
}

SA_API sa_fake *sa_fake_start(int kind, void *cb, void *user, uint32_t frames, uint32_t channels, double rate,
                              double lead, double wobble, double ppm, double skip_every, uint64_t rec_frames,
                              int realtime) {
    sa_fake *f = fake_new(kind, cb, user, frames, channels, rate);
    if (!f) return NULL;
    f->lead = lead; f->wobble = wobble; f->ppm = ppm; f->skip_every = skip_every;
    f->rec_cap = rec_frames;
    f->rec = rec_frames ? calloc((size_t)rec_frames * channels, sizeof(float)) : NULL;
    if (rec_frames && !f->rec) { free(f); return NULL; }
    f->t0 = sa_now() + 0.05;
    return fake_go(f, realtime);
}

SA_API sa_fake *sa_fake_loop(int kind, void *cb, void *user, uint32_t frames, uint32_t channels, double rate,
                             uint32_t safety, float *tape, uint64_t tape_frames, int realtime) {
    if (!tape || tape_frames < 2u * LOOP_DELAY || (tape_frames & (tape_frames - 1))) return NULL;
    sa_fake *f = fake_new(kind, cb, user, frames, channels, rate);
    if (!f) return NULL;
    uint64_t first = (uint64_t)llround((sa_now() + 0.05) * rate);     /* the first cycle's sample time */
    uint32_t ahead = frames + safety + (kind ? LOOP_DELAY : 0);
    f->tape = tape; f->tape_mask = tape_frames - 1;
    f->s0 = kind == 0 ? first + ahead : first - ahead;
    f->lead = ahead / rate;
    f->t0 = (double)first / rate;
    return fake_go(f, realtime);
}

SA_API void sa_fake_stop(sa_fake *f, uint64_t stats[6] /* cycles, late, skipped, forced, late_wake, late_cb */,
                         double *t0, double *max_cb) {
    atomic_store(&f->stop, 1);
    pthread_join(f->th, NULL);
    stats[0] = f->cycles; stats[1] = f->late; stats[2] = f->skipped_cycles; stats[3] = f->forced;
    stats[4] = f->late_wake; stats[5] = f->late_cb;
    *t0 = f->t0; *max_cb = f->max_cb_s;
}
SA_API int sa_fake_policy(sa_fake *f) { return f->policy; }
SA_API float *sa_fake_rec(sa_fake *f) { return f->rec; }
SA_API void sa_fake_free(sa_fake *f) { if (f) { free(f->rec); free(f); } }
#endif

#ifdef __APPLE__
#include "sacore_mac.c"      /* device facts, overload/default-output listeners (macOS only) */
#endif
