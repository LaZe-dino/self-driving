#include "test_main.hpp"

int g_test_failures = 0;

std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> tests;
    return tests;
}

int main() {
    int failed_tests = 0;
    for (const TestCase& t : test_registry()) {
        int before = g_test_failures;
        t.fn();
        bool ok = g_test_failures == before;
        if (!ok) {
            ++failed_tests;
        }
        std::cout << (ok ? "[pass] " : "[FAIL] ") << t.name << "\n";
    }
    std::cout << test_registry().size() - failed_tests << "/" << test_registry().size()
              << " tests passed\n";
    return failed_tests == 0 ? 0 : 1;
}
