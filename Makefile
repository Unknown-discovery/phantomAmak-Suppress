CXX ?= g++
BUILD_DIR ?= build
AMAK_ENABLE_AVX2 ?= 1
CPPFLAGS ?=
PROJECT_CPPFLAGS := -Iinclude -Isrc -DAMAK_ENABLE_AVX2=$(AMAK_ENABLE_AVX2)
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
FP_FLAGS := -ffp-contract=off
THREAD_FLAGS := -pthread
LDFLAGS ?=
LDLIBS ?=
LIB_SOURCES := src/dct.cpp src/dct_kernels.cpp src/rans.cpp src/container.cpp src/runtime.cpp
LIB_OBJECTS := $(LIB_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
APP_SOURCES := src/main.cpp src/benchmark.cpp
APP_OBJECTS := $(APP_SOURCES:%.cpp=$(BUILD_DIR)/%.o)
TEST_OBJECT := $(BUILD_DIR)/tests/tests.o

.PHONY: all test test-scalar sanitize clean
all: $(BUILD_DIR)/amak

$(BUILD_DIR)/amak: $(LIB_OBJECTS) $(APP_OBJECTS)
	$(CXX) $(CXXFLAGS) $(FP_FLAGS) $(THREAD_FLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/amak_tests: $(LIB_OBJECTS) $(TEST_OBJECT)
	$(CXX) $(CXXFLAGS) $(FP_FLAGS) $(THREAD_FLAGS) $^ $(LDFLAGS) $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(PROJECT_CPPFLAGS) $(CXXFLAGS) $(FP_FLAGS) $(THREAD_FLAGS) -MMD -MP -c $< -o $@

test: $(BUILD_DIR)/amak $(BUILD_DIR)/amak_tests
	$(BUILD_DIR)/amak_tests
	python3 tests/cli_smoke.py $(BUILD_DIR)/amak

test-scalar:
	$(MAKE) BUILD_DIR=$(BUILD_DIR)/scalar AMAK_ENABLE_AVX2=0 test

sanitize:
	$(MAKE) BUILD_DIR=$(BUILD_DIR)/sanitize CXXFLAGS='-O1 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address,undefined -fno-omit-frame-pointer' LDFLAGS='-fsanitize=address,undefined' test

clean:
	rm -rf build

-include $(LIB_OBJECTS:.o=.d) $(APP_OBJECTS:.o=.d) $(TEST_OBJECT:.o=.d)
