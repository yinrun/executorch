#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <unistd.h>

#include <executorch/examples/musa/qwen35_fixture.h>

namespace {

using executorch::examples::musa::Qwen35Fixture;

Qwen35Fixture text_fixture(size_t count = 3) {
  Qwen35Fixture fixture;
  fixture.tokens.assign(count, 42);
  fixture.image_rows.assign(count, -1);
  for (size_t axis = 0; axis < 3; ++axis) {
    for (size_t position = 0; position < count; ++position) {
      fixture.positions.push_back(position);
    }
  }
  fixture.eos = {248044, 248046};
  fixture.expected = {123, 248046};
  return fixture;
}

TEST(MusaQwen35Fixture, TextAndContextBoundary) {
  std::string error;
  auto fixture = text_fixture();
  EXPECT_TRUE(fixture.validate_generation(160, error)) << error;
  EXPECT_EQ(fixture.next_position(), 3);
  fixture = text_fixture(4096);
  fixture.expected = {123};
  EXPECT_TRUE(fixture.validate_generation(160, error)) << error;
  EXPECT_EQ(fixture.next_position(), 4096);
  fixture = text_fixture(4097);
  EXPECT_FALSE(fixture.validate(error));
}

TEST(MusaQwen35Fixture, ImageRowsAndCompressedPositions) {
  auto fixture = text_fixture(5);
  fixture.tokens[1] = fixture.tokens[2] = Qwen35Fixture::image_token;
  fixture.image_rows = {-1, 0, 1, -1, -1};
  fixture.embeddings.assign(2 * Qwen35Fixture::dimension, 0.5f);
  fixture.positions = {0, 1, 1, 3, 4, 0, 1, 1, 3, 4, 0, 1, 2, 3, 4};
  std::string error;
  EXPECT_TRUE(fixture.validate(error)) << error;
  fixture.image_rows[2] = 0;
  EXPECT_FALSE(fixture.validate(error));
  fixture.image_rows[2] = 1;
  fixture.tokens[2] = 42;
  EXPECT_FALSE(fixture.validate(error));
  fixture.tokens[2] = Qwen35Fixture::image_token;
  fixture.embeddings.resize(Qwen35Fixture::dimension);
  EXPECT_FALSE(fixture.validate(error));
}

TEST(MusaQwen35Fixture, NonFiniteFeatures) {
  auto fixture = text_fixture(1);
  fixture.tokens[0] = Qwen35Fixture::image_token;
  fixture.image_rows[0] = 0;
  fixture.embeddings.assign(Qwen35Fixture::dimension, 0.0f);
  std::string error;
  EXPECT_TRUE(fixture.validate(error));
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    fixture.embeddings[0] = invalid;
    EXPECT_FALSE(fixture.validate(error));
  }
}

TEST(MusaQwen35Fixture, InvalidPositionsAndTokens) {
  std::string error;
  for (int64_t invalid : {-1, 4096}) {
    auto fixture = text_fixture();
    fixture.positions[0] = invalid;
    EXPECT_FALSE(fixture.validate(error));
  }
  for (int64_t invalid : {-1, 248320}) {
    auto fixture = text_fixture();
    fixture.tokens[0] = invalid;
    EXPECT_FALSE(fixture.validate(error));
    fixture = text_fixture();
    fixture.expected[0] = invalid;
    EXPECT_FALSE(fixture.validate(error));
    fixture = text_fixture();
    fixture.eos[0] = invalid;
    EXPECT_FALSE(fixture.validate(error));
  }
}

TEST(MusaQwen35Fixture, InvalidLengthsAndRows) {
  std::string error;
  auto fixture = text_fixture(0);
  EXPECT_FALSE(fixture.validate(error));
  fixture = text_fixture();
  fixture.positions.pop_back();
  EXPECT_FALSE(fixture.validate(error));
  fixture = text_fixture();
  fixture.image_rows.pop_back();
  EXPECT_FALSE(fixture.validate(error));
  fixture = text_fixture();
  fixture.image_rows[0] = -2;
  EXPECT_FALSE(fixture.validate(error));
  fixture = text_fixture();
  fixture.eos.clear();
  EXPECT_FALSE(fixture.validate(error));
}

TEST(MusaQwen35Fixture, ExpectedStopsAndBudgets) {
  std::string error;
  auto fixture = text_fixture();
  EXPECT_FALSE(fixture.validate_generation(0, error));
  EXPECT_FALSE(fixture.validate_generation(4097, error));
  EXPECT_FALSE(fixture.validate_generation(1, error));
  fixture.expected = {248046, 42};
  EXPECT_FALSE(fixture.validate(error));
  fixture.expected = {42};
  EXPECT_FALSE(fixture.validate_generation(160, error));
  EXPECT_TRUE(fixture.validate_generation(1, error));
  fixture.expected.clear();
  EXPECT_TRUE(fixture.validate_generation(160, error));
  fixture.positions.back() = 4095;
  EXPECT_FALSE(fixture.validate_generation(2, error));
  EXPECT_TRUE(fixture.validate_generation(1, error));
}

class MusaQwen35FixtureFiles : public ::testing::Test {
 protected:
  void SetUp() override {
    char path[] = "/tmp/executorch-qwen35-fixture-XXXXXX";
    ASSERT_NE(mkdtemp(path), nullptr);
    directory_ = path;
    prefix_ = directory_ + "/fixture";
  }

  void TearDown() override {
    std::filesystem::remove_all(directory_);
  }

  template <typename Value>
  void write(const std::string& suffix, const std::vector<Value>& values) {
    std::ofstream output(prefix_ + suffix, std::ios::binary);
    output.write(
        reinterpret_cast<const char*>(values.data()),
        values.size() * sizeof(Value));
    ASSERT_TRUE(output.good());
  }

  void write_fixture(const Qwen35Fixture& fixture) {
    write(".tokens.bin", fixture.tokens);
    write(".positions.bin", fixture.positions);
    write(".image_rows.bin", fixture.image_rows);
    write(".embeddings.bin", fixture.embeddings);
    write(".eos.bin", fixture.eos);
    write(".expected.bin", fixture.expected);
  }

  std::string directory_;
  std::string prefix_;
};

TEST_F(MusaQwen35FixtureFiles, LoadValidText) {
  const auto source = text_fixture();
  write_fixture(source);
  Qwen35Fixture fixture;
  std::string error;
  ASSERT_TRUE(fixture.load(prefix_, error)) << error;
  EXPECT_EQ(fixture.tokens, source.tokens);
  EXPECT_EQ(fixture.positions, source.positions);
  EXPECT_TRUE(fixture.embeddings.empty());
  EXPECT_TRUE(fixture.validate_generation(160, error));
}

TEST_F(MusaQwen35FixtureFiles, RejectMissingFileWithoutMutatingFixture) {
  auto fixture = text_fixture();
  std::string error;
  EXPECT_FALSE(fixture.load(prefix_, error));
  EXPECT_EQ(fixture.tokens, text_fixture().tokens);
  EXPECT_NE(error.find("Cannot open"), std::string::npos);
}

TEST_F(MusaQwen35FixtureFiles, RejectTruncatedAndOversizedFiles) {
  write_fixture(text_fixture());
  Qwen35Fixture fixture;
  std::string error;
  write(".tokens.bin", std::vector<uint8_t>{1});
  EXPECT_FALSE(fixture.load(prefix_, error));
  EXPECT_NE(error.find("byte length"), std::string::npos);
  write(".tokens.bin", std::vector<int64_t>(4097, 42));
  EXPECT_FALSE(fixture.load(prefix_, error));
  EXPECT_NE(error.find("byte length"), std::string::npos);
}

}
