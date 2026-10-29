#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace executorch::examples::musa {

struct Qwen35Fixture final {
  static constexpr size_t context = 4096;
  static constexpr size_t dimension = 2560;
  static constexpr int64_t vocabulary = 248320;
  static constexpr int64_t image_token = 248056;

  std::vector<int64_t> tokens;
  std::vector<int64_t> positions;
  std::vector<int64_t> image_rows;
  std::vector<float> embeddings;
  std::vector<int64_t> eos;
  std::vector<int64_t> expected;

  bool is_eos(int64_t token) const {
    return std::find(eos.begin(), eos.end(), token) != eos.end();
  }

  int64_t next_position() const {
    return *std::max_element(positions.begin(), positions.end()) + 1;
  }

  bool validate(std::string& error) const {
    if (tokens.empty() || tokens.size() > context ||
        positions.size() != tokens.size() * 3 ||
        image_rows.size() != tokens.size() ||
        embeddings.size() % dimension != 0 ||
        embeddings.size() / dimension > context || eos.empty() ||
        eos.size() > 16 || expected.size() > context) {
      error = "Invalid Qwen3.5 fixture lengths";
      return false;
    }
    for (const auto* values : {&tokens, &eos, &expected}) {
      for (int64_t token : *values) {
        if (token < 0 || token >= vocabulary) {
          error = "Qwen3.5 fixture token is outside the vocabulary";
          return false;
        }
      }
    }
    for (int64_t position : positions) {
      if (position < 0 || position >= static_cast<int64_t>(context)) {
        error = "Qwen3.5 MRoPE position is outside the context";
        return false;
      }
    }
    int64_t next_row = 0;
    for (size_t index = 0; index < tokens.size(); ++index) {
      const int64_t row = image_rows[index];
      if (row < -1 || (row >= 0) != (tokens[index] == image_token) ||
          (row >= 0 && row != next_row++)) {
        error = "Qwen3.5 image rows do not match ordered placeholders";
        return false;
      }
    }
    if (static_cast<size_t>(next_row) != embeddings.size() / dimension) {
      error = "Qwen3.5 image feature count does not match placeholders";
      return false;
    }
    for (float value : embeddings) {
      if (!std::isfinite(value)) {
        error = "Qwen3.5 image features must be finite";
        return false;
      }
    }
    for (size_t index = 0; index + 1 < expected.size(); ++index) {
      if (is_eos(expected[index])) {
        error = "Expected Qwen3.5 generation continues after EOS";
        return false;
      }
    }
    error.clear();
    return true;
  }

  bool validate_generation(size_t maximum_generated, std::string& error) const {
    if (!validate(error)) {
      return false;
    }
    if (maximum_generated == 0 || maximum_generated > context) {
      error = "Maximum generated tokens must be between 1 and 4096";
      return false;
    }
    const size_t available =
        std::min(maximum_generated, context - tokens.size() + 1);
    if (expected.size() > available ||
        (!expected.empty() && !is_eos(expected.back()) &&
         expected.size() != available)) {
      error = "Expected generation has an invalid length or stop condition";
      return false;
    }
    if (static_cast<size_t>(next_position()) + available - 1 > context) {
      error = "Qwen3.5 decode MRoPE would exceed the context";
      return false;
    }
    error.clear();
    return true;
  }

  bool load(const std::string& prefix, std::string& error) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    const uint16_t byte_order = 1;
    if (*reinterpret_cast<const uint8_t*>(&byte_order) != 1) {
      error = "Qwen3.5 fixture format requires a little-endian host";
      return false;
    }
    Qwen35Fixture loaded;
    if (!read_values(prefix + ".tokens.bin", context, loaded.tokens, error) ||
        !read_values(
            prefix + ".positions.bin", context * 3, loaded.positions, error) ||
        !read_values(
            prefix + ".image_rows.bin", context, loaded.image_rows, error) ||
        !read_values(
            prefix + ".embeddings.bin",
            context * dimension,
            loaded.embeddings,
            error) ||
        !read_values(prefix + ".eos.bin", 16, loaded.eos, error) ||
        !read_values(prefix + ".expected.bin", context, loaded.expected, error) ||
        !loaded.validate(error)) {
      return false;
    }
    *this = std::move(loaded);
    return true;
  }

 private:
  template <typename Value>
  static bool read_values(
      const std::string& path,
      size_t maximum,
      std::vector<Value>& values,
      std::string& error) {
    std::ifstream source(path, std::ios::binary | std::ios::ate);
    if (!source) {
      error = "Cannot open Qwen3.5 fixture: " + path;
      return false;
    }
    const auto length = source.tellg();
    if (length < 0 ||
        length > static_cast<std::streamoff>(maximum * sizeof(Value)) ||
        length % sizeof(Value) != 0) {
      error = "Invalid Qwen3.5 fixture byte length: " + path;
      return false;
    }
    values.resize(static_cast<size_t>(length) / sizeof(Value));
    source.seekg(0);
    if (length > 0 &&
        !source.read(reinterpret_cast<char*>(values.data()), length)) {
      error = "Cannot read Qwen3.5 fixture: " + path;
      return false;
    }
    return true;
  }
};

}
