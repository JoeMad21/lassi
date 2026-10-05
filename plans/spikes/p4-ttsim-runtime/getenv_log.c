/*
 * getenv_log.c: an LD_PRELOAD shim for the P4.9 spike (plans/spikes/p4-ttsim-runtime.md).
 *
 * It records which environment variables a process asks for, so the spike can say what tt-metal
 * needs from the environment at run time. Every getenv and secure_getenv call appends one line,
 * "<executable>\t<function>\t<name>\t<set|unset>", to the file named by LASSI_GETENV_LOG, opened
 * once with O_APPEND. Values are never written. Without LASSI_GETENV_LOG it logs nothing. The
 * real functions come from the next object in the lookup order (RTLD_NEXT), so the process sees
 * exactly what it would see without the shim. Used only in the spike's measuring run, outside the
 * sandbox and never on generated code; run.sh compiles it into the run's raw directory.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>

typedef char *(*lookup_fn)(const char *);

static lookup_fn real_getenv;
static lookup_fn real_secure_getenv;
static int log_fd = -2;
static char exe_path[512];

/* Open the log on first use and read this process's executable path; -1 means "do not log". */
static void open_log(void) {
    const char *path;
    ssize_t n;
    if (log_fd != -2) {
        return;
    }
    if (real_getenv == NULL) {
        real_getenv = (lookup_fn)dlsym(RTLD_NEXT, "getenv");
    }
    path = real_getenv ? real_getenv("LASSI_GETENV_LOG") : NULL;
    log_fd = path ? open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644) : -1;
    n = readlink("/proc/self/exe", exe_path, sizeof exe_path - 1);
    exe_path[n > 0 ? n : 0] = '\0';
}

/* Append text to buf at *used, never past cap - 1 bytes. */
static void put(char *buf, size_t cap, size_t *used, const char *text) {
    size_t len = strlen(text);
    if (*used + len > cap - 1) {
        len = cap - 1 - *used;
    }
    memcpy(buf + *used, text, len);
    *used += len;
}

/* Write one log line for a lookup of name by function. */
static void log_lookup(const char *function, const char *name, const char *value) {
    char line[1024];
    size_t used = 0;
    open_log();
    if (log_fd < 0 || name == NULL) {
        return;
    }
    put(line, sizeof line, &used, exe_path);
    put(line, sizeof line, &used, "\t");
    put(line, sizeof line, &used, function);
    put(line, sizeof line, &used, "\t");
    put(line, sizeof line, &used, name);
    put(line, sizeof line, &used, value ? "\tset\n" : "\tunset\n");
    if (write(log_fd, line, used) < 0) {
        log_fd = -1;
    }
}

char *getenv(const char *name) {
    char *value;
    if (real_getenv == NULL) {
        real_getenv = (lookup_fn)dlsym(RTLD_NEXT, "getenv");
    }
    value = real_getenv ? real_getenv(name) : NULL;
    log_lookup("getenv", name, value);
    return value;
}

char *secure_getenv(const char *name) {
    char *value;
    if (real_secure_getenv == NULL) {
        real_secure_getenv = (lookup_fn)dlsym(RTLD_NEXT, "secure_getenv");
    }
    value = real_secure_getenv ? real_secure_getenv(name) : NULL;
    log_lookup("secure_getenv", name, value);
    return value;
}
