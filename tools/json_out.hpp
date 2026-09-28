#pragma once

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

// Minimal JSON builder for the evaluation reports (numbers, strings, arrays,
// objects). Non-finite numbers are written as null.
class JsonOut {
public:
    JsonOut& begin_obj() { sep(); out_ << '{'; first_ = true; return *this; }
    JsonOut& end_obj() { out_ << '}'; first_ = false; return *this; }
    JsonOut& begin_arr() { sep(); out_ << '['; first_ = true; return *this; }
    JsonOut& end_arr() { out_ << ']'; first_ = false; return *this; }
    JsonOut& key(const char* k) { sep(); out_ << '"' << k << "\":"; first_ = true; return *this; }
    JsonOut& num(double v) {
        sep();
        if (std::isfinite(v)) {
            char b[32];
            std::snprintf(b, sizeof(b), "%.6g", v);
            out_ << b;
        } else {
            out_ << "null";
        }
        return *this;
    }
    JsonOut& integer(long long v) { sep(); out_ << v; return *this; }
    JsonOut& boolean(bool v) { sep(); out_ << (v ? "true" : "false"); return *this; }
    JsonOut& null() { sep(); out_ << "null"; return *this; }
    JsonOut& str(const std::string& s) {
        sep();
        out_ << '"';
        for (char c : s) {
            if (c == '"' || c == '\\') out_ << '\\' << c;
            else if (c == '\n') out_ << "\\n";
            else out_ << c;
        }
        out_ << '"';
        return *this;
    }
    JsonOut& kv(const char* k, double v) { return key(k).num(v); }
    JsonOut& kv(const char* k, const std::string& v) { return key(k).str(v); }
    JsonOut& kvb(const char* k, bool v) { return key(k).boolean(v); }
    JsonOut& kvi(const char* k, long long v) { return key(k).integer(v); }
    std::string text() const { return out_.str(); }

private:
    void sep() {
        if (!first_) out_ << ',';
        first_ = false;
    }
    std::ostringstream out_;
    bool first_ = true;
};
