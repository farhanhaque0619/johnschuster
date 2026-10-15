/*
 * sacore_mac.c - the macOS-only half of libsacore, #included at the end of sacore.c on Apple.
 * Plain CoreAudio HAL property calls, no audio IO. Nothing here runs on an IO thread except the
 * listener callbacks, which only bump an atomic counter. All of it is optional: Python reaches it
 * through sacore.mac_call(), so if any of it is missing or fails, audio still plays and only these
 * facts and counts are lost.
 *
 *   sa_device_info()      what PortAudio never tells us: transport (Bluetooth?), latency, safety
 *                         offset, stream latency (PortAudio leaves it out of outputBufferDacTime),
 *                         the IO buffer actually in force and its allowed range, nominal/actual rate
 *   sa_listen()           count a property's notifications: kAudioDeviceProcessorOverload on the
 *                         global and wildcard scopes (PortAudio listens on the output scope, where
 *                         it never fired in any user's diagnostics), latency / rate / alive changes,
 *                         the system default output (replaces polling SwitchAudioSource, which
 *                         forked the whole app every 2 s)
 *   sa_default_output / sa_set_default_output     (replace SwitchAudioSource -c / -s)
 *   sa_set_nominal_rate   CI: change a device's rate under a running stream
 *   sa_set_thread_qos_interactive                 the output engine's thread asks for a fast core
 */
#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <pthread.h>

typedef struct {
    uint32_t id, transport, alive, out_channels, in_channels;
    uint32_t latency, safety, stream_latency, buffer_frames, buffer_min, buffer_max;
    double nominal_rate, actual_rate;
    char uid[256], name[256];
} sa_devinfo;

static OSStatus get(AudioObjectID id, AudioObjectPropertySelector sel, AudioObjectPropertyScope scope,
                    UInt32 size, void *out) {
    AudioObjectPropertyAddress a = {sel, scope, kAudioObjectPropertyElementMain};
    return AudioObjectGetPropertyData(id, &a, 0, NULL, &size, out);
}

static uint32_t get_u32(AudioObjectID id, AudioObjectPropertySelector sel, AudioObjectPropertyScope scope) {
    UInt32 v = 0;
    return get(id, sel, scope, sizeof v, &v) == noErr ? v : 0;
}

static double get_f64(AudioObjectID id, AudioObjectPropertySelector sel) {
    Float64 v = 0;
    return get(id, sel, kAudioObjectPropertyScopeGlobal, sizeof v, &v) == noErr ? v : 0;
}

/* A CFString property as UTF-8 (the same conversion PortAudio uses for device names). */
static int get_str(AudioObjectID id, AudioObjectPropertySelector sel, char *dst, size_t n) {
    CFStringRef s = NULL;
    dst[0] = 0;
    if (get(id, sel, kAudioObjectPropertyScopeGlobal, sizeof s, &s) != noErr || !s) return -1;
    if (!CFStringGetCString(s, dst, (CFIndex)n, kCFStringEncodingUTF8)) dst[0] = 0;
    CFRelease(s);       /* CoreAudio hands out CF objects the caller owns */
    return 0;
}

static uint32_t channels(AudioObjectID id, AudioObjectPropertyScope scope) {
    AudioObjectPropertyAddress a = {kAudioDevicePropertyStreamConfiguration, scope, kAudioObjectPropertyElementMain};
    UInt32 size = 0, ch = 0;
    if (AudioObjectGetPropertyDataSize(id, &a, 0, NULL, &size) != noErr || size < sizeof(AudioBufferList)) return 0;
    AudioBufferList *bl = malloc(size);
    if (bl && AudioObjectGetPropertyData(id, &a, 0, NULL, &size, bl) == noErr)
        for (UInt32 i = 0; i < bl->mNumberBuffers; i++) ch += bl->mBuffers[i].mNumberChannels;
    free(bl);
    return ch;
}

static uint32_t first_stream_latency(AudioObjectID id, AudioObjectPropertyScope scope) {
    AudioObjectPropertyAddress a = {kAudioDevicePropertyStreams, scope, kAudioObjectPropertyElementMain};
    AudioStreamID s[16];
    UInt32 size = sizeof s;
    if (AudioObjectGetPropertyData(id, &a, 0, NULL, &size, s) != noErr || size < sizeof(AudioStreamID)) return 0;
    return get_u32(s[0], kAudioStreamPropertyLatency, kAudioObjectPropertyScopeGlobal);
}

/* Everything we want to know about device `id` in one direction. -1 if it isn't a device now. */
static int fill(AudioObjectID id, int output, sa_devinfo *d) {
    AudioObjectPropertyScope sc = output ? kAudioObjectPropertyScopeOutput : kAudioObjectPropertyScopeInput;
    AudioValueRange range = {0, 0};
    memset(d, 0, sizeof *d);
    if (get_str(id, kAudioObjectPropertyName, d->name, sizeof d->name) != 0) return -1;
    d->id = id;
    d->transport = get_u32(id, kAudioDevicePropertyTransportType, kAudioObjectPropertyScopeGlobal);  /* 'blue' */
    d->alive = get_u32(id, kAudioDevicePropertyDeviceIsAlive, kAudioObjectPropertyScopeGlobal);
    d->out_channels = channels(id, kAudioObjectPropertyScopeOutput);
    d->in_channels = channels(id, kAudioObjectPropertyScopeInput);
    /* the same scopes PortAudio reads them with (pa_mac_core_utilities.c) */
    d->latency = get_u32(id, kAudioDevicePropertyLatency, sc);
    d->safety = get_u32(id, kAudioDevicePropertySafetyOffset, sc);
    d->stream_latency = first_stream_latency(id, sc);
    d->buffer_frames = get_u32(id, kAudioDevicePropertyBufferFrameSize, sc);
    if (get(id, kAudioDevicePropertyBufferFrameSizeRange, sc, sizeof range, &range) == noErr) {
        d->buffer_min = (uint32_t)range.mMinimum;
        d->buffer_max = (uint32_t)range.mMaximum;
    }
    d->nominal_rate = get_f64(id, kAudioDevicePropertyNominalSampleRate);
    d->actual_rate = get_f64(id, kAudioDevicePropertyActualSampleRate);
    get_str(id, kAudioDevicePropertyDeviceUID, d->uid, sizeof d->uid);
    return 0;
}

/* The device PortAudio calls `name` (its names are kAudioObjectPropertyName) that has channels in
 * the wanted direction: AirPods can be two AudioObjects with the same name (A2DP out, HFP in). */
SA_API int sa_device_info(const char *name, int output, sa_devinfo *out) {
    AudioObjectPropertyAddress a = {kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    AudioObjectID ids[256];
    UInt32 size = sizeof ids;
    char found[256];
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, &size, ids) != noErr) return -1;
    for (UInt32 i = 0; i < size / sizeof(AudioObjectID); i++) {
        if (get_str(ids[i], kAudioObjectPropertyName, found, sizeof found) != 0 || strcmp(found, name) != 0) continue;
        if (!channels(ids[i], output ? kAudioObjectPropertyScopeOutput : kAudioObjectPropertyScopeInput)) continue;
        return fill(ids[i], output, out);
    }
    return -1;
}

SA_API int sa_device_info_id(uint32_t id, int output, sa_devinfo *out) { return fill(id, output, out); }

/* -------------------------------------------------------------------- notifications
 * Counters live in a fixed table, never in freed memory: macOS may still be delivering one last
 * notification on its own thread just as Python removes the listener, and bumping a recycled
 * counter once is harmless where writing into freed memory is not. */
#define SA_COUNTERS 1024
static _Atomic uint64_t g_count[SA_COUNTERS];
static _Atomic int g_used[SA_COUNTERS];

SA_API _Atomic uint64_t *sa_counter_new(void) {
    for (int i = 0; i < SA_COUNTERS; i++)
        if (!atomic_exchange(&g_used[i], 1)) { atomic_store(&g_count[i], 0); return &g_count[i]; }
    return NULL;
}

SA_API uint64_t sa_counter_get(_Atomic uint64_t *c) { return atomic_load(c); }

SA_API void sa_counter_free(_Atomic uint64_t *c) {
    if (c >= g_count && c < g_count + SA_COUNTERS) atomic_store(&g_used[c - g_count], 0);
}

static OSStatus bump(AudioObjectID id, UInt32 n, const AudioObjectPropertyAddress *a, void *user) {
    (void)id; (void)n; (void)a;
    atomic_fetch_add_explicit((_Atomic uint64_t *)user, 1, memory_order_relaxed);
    return noErr;
}

/* id 1 (kAudioObjectSystemObject) is the system itself, e.g. for the default output. */
SA_API int sa_listen(uint32_t id, uint32_t selector, uint32_t scope, _Atomic uint64_t *counter) {
    static atomic_flag own_thread = ATOMIC_FLAG_INIT;
    if (!atomic_flag_test_and_set(&own_thread)) {
        /* macOS delivers these notifications on the main thread's run loop unless told otherwise,
         * and a command-line run never spins one: NULL lets the HAL use a thread of its own (once).
         * (Apple lists this property with the deprecated ones in some SDKs; it is still honoured.) */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        AudioObjectPropertyAddress rl = {kAudioHardwarePropertyRunLoop, kAudioObjectPropertyScopeGlobal,
                                         kAudioObjectPropertyElementMain};
#pragma clang diagnostic pop
        CFRunLoopRef none = NULL;
        AudioObjectSetPropertyData(kAudioObjectSystemObject, &rl, 0, NULL, sizeof none, &none);
    }
    AudioObjectPropertyAddress a = {selector, scope, kAudioObjectPropertyElementMain};
    return AudioObjectAddPropertyListener(id, &a, bump, counter);
}

SA_API int sa_unlisten(uint32_t id, uint32_t selector, uint32_t scope, _Atomic uint64_t *counter) {
    AudioObjectPropertyAddress a = {selector, scope, kAudioObjectPropertyElementMain};
    return AudioObjectRemovePropertyListener(id, &a, bump, counter);
}

/* ------------------------------------------------------------- system default output, rates */
SA_API uint32_t sa_default_output(void) {
    return get_u32(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice,
                   kAudioObjectPropertyScopeGlobal);
}

SA_API int sa_set_default_output(uint32_t id) {
    AudioObjectPropertyAddress a = {kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    AudioObjectID v = id;
    return AudioObjectSetPropertyData(kAudioObjectSystemObject, &a, 0, NULL, sizeof v, &v);
}

SA_API int sa_set_nominal_rate(uint32_t id, double rate) {
    AudioObjectPropertyAddress a = {kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal,
                                    kAudioObjectPropertyElementMain};
    Float64 v = rate;
    return AudioObjectSetPropertyData(id, &a, 0, NULL, sizeof v, &v);
}

/* The calling thread (the output engine) asks to be scheduled like the UI: on a performance core,
 * ahead of background work. 0 = done. */
SA_API int sa_set_thread_qos_interactive(void) {
    return pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

/* Selectors and constants Python needs (so it doesn't hard-code four-char codes). */
SA_API void sa_selectors(uint32_t out[11] /* overload, latency, nominal_rate, alive, default_output,
                                             global, output, wildcard, bluetooth, bluetooth_le, virtual */) {
    out[0] = kAudioDeviceProcessorOverload; out[1] = kAudioDevicePropertyLatency;
    out[2] = kAudioDevicePropertyNominalSampleRate; out[3] = kAudioDevicePropertyDeviceIsAlive;
    out[4] = kAudioHardwarePropertyDefaultOutputDevice; out[5] = kAudioObjectPropertyScopeGlobal;
    out[6] = kAudioObjectPropertyScopeOutput; out[7] = kAudioObjectPropertyScopeWildcard;
    out[8] = kAudioDeviceTransportTypeBluetooth; out[9] = kAudioDeviceTransportTypeBluetoothLE;
    out[10] = kAudioDeviceTransportTypeVirtual;
}
