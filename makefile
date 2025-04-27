# Compiler and flags
CXX = g++

CXXFLAGS = -Wall -std=c++17 -DBOOST_ASIO_NO_DEPRECATED

# Python 3.12 include & link paths
PYTHON_VERSION = 3.12
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
PYTHON_LIB = /usr/lib/python$(PYTHON_VERSION)/config-$(PYTHON_VERSION)-x86_64-linux-gnu
PYTHON_LDFLAGS = -lpython$(PYTHON_VERSION)

# Boost include & link
BOOST_INCLUDE = /usr/include/boost
BOOST_LDFLAGS = -lboost_system

# Project-specific include directory
DIRECTORY_INCLUDE = $(shell pwd)/include

# Directories
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# Main sources
MAIN_SRCS = $(shell find $(SRC_DIR) -name '*.cpp')
MAIN_OUT = $(BUILD_DIR)/main

# Test sources
TEST_SRCS = $(shell find $(TEST_DIR) -name '*.cpp')
TEST_BINS = $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/test/%,$(TEST_SRCS))

# Default target
all: main

# Build main from src recursively
main: $(MAIN_OUT)

$(MAIN_OUT): $(MAIN_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) \
		-I$(PYTHON_INCLUDE) \
		-I$(BOOST_INCLUDE) \
		-I$(DIRECTORY_INCLUDE) \
		$(MAIN_SRCS) -o $@ \
		-L$(PYTHON_LIB) $(PYTHON_LDFLAGS) $(BOOST_LDFLAGS)

# Build all tests
test: $(TEST_BINS)

$(BUILD_DIR)/test/%: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) \
		-I$(PYTHON_INCLUDE) \
		-I$(BOOST_INCLUDE) \
		-I$(DIRECTORY_INCLUDE) \
		$< -o $@ \
		-L$(PYTHON_LIB) $(PYTHON_LDFLAGS) $(BOOST_LDFLAGS)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)

.PHONY: all main test clean
