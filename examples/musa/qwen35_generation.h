#pragma once

#include <array>
#include <chrono>
#include <cstring>
#include <iomanip>

#include <executorch/examples/musa/benchmark.h>
#include <executorch/examples/musa/qwen35_fixture.h>
#include <executorch/extension/runner_util/inputs.h>

namespace executorch::examples::musa {

inline void run_qwen35_generation(
    runtime::Method& method,
    const runtime::MethodMeta& metadata,
    const Qwen35Fixture& fixture,
    const std::string& output_path,
    const std::string& dump_prefix,
    uint32_t maximum_generated,
    uint32_t warmup,
    uint32_t repeats) {
  using runtime::Error;
  using runtime::EValue;
  std::string validation_error;
  ET_CHECK_MSG(
      fixture.validate_generation(maximum_generated, validation_error),
      "%s",
      validation_error.c_str());
  ET_CHECK_MSG(
      repeats > 0 && repeats <= 100 && warmup <= 100, "Invalid repeat count");
  ET_CHECK_MSG(
      method.inputs_size() == 5 && method.outputs_size() == 2,
      "Expected Qwen3.5 multimodal step");
  const std::array<std::vector<int32_t>, 7> shapes{
      std::vector<int32_t>{1, 1},
      {1},
      {3, 1},
      {1, 1, 2560},
      {1},
      {1, 248320},
      {1}};
  const std::array<aten::ScalarType, 7> types{
      aten::ScalarType::Long,
      aten::ScalarType::Long,
      aten::ScalarType::Long,
      aten::ScalarType::Float,
      aten::ScalarType::Bool,
      aten::ScalarType::Float,
      aten::ScalarType::Long};
  for (size_t index = 0; index < shapes.size(); ++index) {
    auto information = index < 5 ? metadata.input_tensor_meta(index)
                                : metadata.output_tensor_meta(index - 5);
    ET_CHECK(information.ok());
    ET_CHECK(information->scalar_type() == types[index]);
    auto sizes = information->sizes();
    ET_CHECK(sizes.size() == shapes[index].size());
    for (size_t axis = 0; axis < sizes.size(); ++axis) {
      ET_CHECK(sizes[axis] == shapes[index][axis]);
    }
  }
  Benchmark synchronization(metadata, "", 0, 1);
  std::ofstream report(output_path);
  ET_CHECK_MSG(report.good(), "Cannot open Qwen3.5 generation report");
  report << std::setprecision(17)
         << "{\"version\":1,\"model\":\"Qwen3.5-4B-GGUF\","
         << "\"context_capacity\":4096,\"prefill_mode\":\"serial_token\","
         << "\"controller\":\"native_cpp\",\"vision_encoding_included\":false,"
         << "\"scope\":\"synchronized_prepare_execute_output_validation\","
         << "\"logits_dump_enabled\":" << (!dump_prefix.empty() ? "true" : "false")
         << ",\"expected_tokens_provided\":"
         << (!fixture.expected.empty() ? "true" : "false")
         << ",\"warmup_runs\":" << warmup << ",\"measured_runs\":" << repeats
         << ",\"maximum_generated_tokens\":" << maximum_generated
         << ",\"runs\":[";
  std::vector<int64_t> first_generation;
  for (uint32_t run = 0; run < warmup + repeats; ++run) {
    std::vector<int64_t> generated;
    double prefill_ms = 0;
    double decode_ms = 0;
    size_t executed = 0;
    int64_t predicted = -1;
    const char* stop_reason = "context_limit";
    const auto loop_started = std::chrono::steady_clock::now();
    for (size_t position = 0; position < Qwen35Fixture::context; ++position) {
      synchronization.synchronize();
      const auto started = std::chrono::steady_clock::now();
      const bool prompt = position < fixture.tokens.size();
      int64_t token = prompt ? fixture.tokens[position] : predicted;
      int64_t cache_position = static_cast<int64_t>(position);
      std::array<int64_t, 3> coordinates;
      for (size_t axis = 0; axis < coordinates.size(); ++axis) {
        coordinates[axis] = prompt
            ? fixture.positions[axis * fixture.tokens.size() + position]
            : fixture.next_position() + position - fixture.tokens.size();
        ET_CHECK(coordinates[axis] >= 0 && coordinates[axis] < 4096);
      }
      const int64_t row = prompt ? fixture.image_rows[position] : -1;
      bool use_image = row >= 0;
      std::array<float, Qwen35Fixture::dimension> embedding{};
      if (use_image) {
        std::memcpy(
            embedding.data(),
            fixture.embeddings.data() + row * Qwen35Fixture::dimension,
            sizeof(embedding));
      }
      const std::vector<std::pair<char*, size_t>> input_buffers{
          {reinterpret_cast<char*>(&token), sizeof(token)},
          {reinterpret_cast<char*>(&cache_position), sizeof(cache_position)},
          {reinterpret_cast<char*>(coordinates.data()), sizeof(coordinates)},
          {reinterpret_cast<char*>(embedding.data()), sizeof(embedding)},
          {reinterpret_cast<char*>(&use_image), sizeof(use_image)}};
      auto cleanup = extension::prepare_input_tensors(method, {}, input_buffers);
      ET_CHECK(cleanup.ok());
      ET_CHECK(method.execute() == Error::Ok);
      std::array<EValue, 2> outputs;
      ET_CHECK(method.get_outputs(outputs.data(), outputs.size()) == Error::Ok);
      synchronization.synchronize();
      ET_CHECK(outputs[0].isTensor() && outputs[1].isTensor());
      const auto logits = outputs[0].toTensor();
      const auto selected = outputs[1].toTensor();
      ET_CHECK(
          logits.scalar_type() == aten::ScalarType::Float &&
          logits.numel() == 248320);
      ET_CHECK(
          selected.scalar_type() == aten::ScalarType::Long &&
          selected.numel() == 1);
      predicted = selected.const_data_ptr<int64_t>()[0];
      ET_CHECK(predicted >= 0 && predicted < Qwen35Fixture::vocabulary);
      const auto* values = logits.const_data_ptr<float>();
      int64_t maximum_index = 0;
      for (int64_t index = 0; index < Qwen35Fixture::vocabulary; ++index) {
        ET_CHECK_MSG(std::isfinite(values[index]), "Non-finite Qwen3.5 logit");
        if (values[index] > values[maximum_index]) {
          maximum_index = index;
        }
      }
      ET_CHECK_MSG(
          predicted == maximum_index, "GPU argmax differs from saved logits");
      if (position + 1 >= fixture.tokens.size()) {
        if (!fixture.expected.empty()) {
          ET_CHECK_MSG(
              generated.size() < fixture.expected.size() &&
                  predicted == fixture.expected[generated.size()],
              "Generation differs from expected token IDs at output %zu",
              generated.size());
        }
        generated.push_back(predicted);
      }
      const double elapsed = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      (prompt ? prefill_ms : decode_ms) += elapsed;
      ++executed;
      if (!dump_prefix.empty()) {
        const std::string stem = dump_prefix + "-run-" + std::to_string(run) +
            "-step-" + std::to_string(position);
        std::ofstream dump(stem + ".bin", std::ios::binary);
        dump.write(reinterpret_cast<const char*>(values), logits.nbytes());
        dump.close();
        ET_CHECK_MSG(!dump.fail(), "Cannot write Qwen3.5 logits");
      }
      if (!generated.empty() && fixture.is_eos(predicted)) {
        stop_reason = "eos";
        break;
      }
      if (generated.size() == maximum_generated) {
        stop_reason = "max_new_tokens";
        break;
      }
    }
    ET_CHECK(!generated.empty());
    ET_CHECK(fixture.expected.empty() || generated == fixture.expected);
    if (run == 0) {
      first_generation = generated;
    } else {
      ET_CHECK_MSG(
          generated == first_generation, "Generation changed after state reset");
    }
    const double loop_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - loop_started)
                               .count();
    if (run != 0) {
      report << ',';
    }
    report << "{\"run\":" << run << ",\"warmup\":"
           << (run < warmup ? "true" : "false") << ",\"steps\":" << executed
           << ",\"prompt_tokens\":" << fixture.tokens.size()
           << ",\"decode_steps\":" << generated.size() - 1
           << ",\"prefill_ms\":" << prefill_ms << ",\"decode_ms\":" << decode_ms
           << ",\"loop_ms\":" << loop_ms << ",\"stop_reason\":\"" << stop_reason
           << "\",\"generated_token_ids\":[";
    for (size_t index = 0; index < generated.size(); ++index) {
      if (index != 0) {
        report << ',';
      }
      report << generated[index];
    }
    report << "]}";
    report.flush();
    ET_CHECK(report.good());
    ET_LOG(
        Info,
        "Qwen3.5 run %u: %zu steps, %zu generated, %s",
        run,
        executed,
        generated.size(),
        stop_reason);
  }
  report << "],\"status\":\"completed\"}\n";
  report.close();
  ET_CHECK_MSG(!report.fail(), "Cannot finish Qwen3.5 generation report");
}

}
