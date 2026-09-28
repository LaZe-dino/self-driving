#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

std::vector<TestCase>& test_registry();
extern int g_test_failures;

struct TestRegistrar {
    TestRegistrar(const std::string& name, std::function<void()> fn) {
        test_registry().push_back({name, fn});
    }
};

#define TEST(name)                                              \
    static void name();                                         \
    static TestRegistrar name##_registrar(#name, name);         \
    static void name()

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::cerr << "    CHECK failed: " #cond " (" << __FILE__ << ":" << __LINE__ \
                      << ")\n";                                                      \
            ++g_test_failures;                                                       \
        }                                                                            \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                          \
    do {                                                                               \
        double va_ = (a), vb_ = (b);                                                   \
        if (!(std::fabs(va_ - vb_) <= (tol))) {                                        \
            std::cerr << "    CHECK_NEAR failed: " #a " = " << va_ << ", " #b " = " << vb_ \
                      << ", tol " << (tol) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            ++g_test_failures;                                                         \
        }                                                                              \
    } while (0)
