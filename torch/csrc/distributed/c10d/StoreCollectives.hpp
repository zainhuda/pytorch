#pragma once

#include <c10/macros/Macros.h>
#include <torch/csrc/distributed/c10d/Collectives.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace c10d {

class TORCH_API StoreCollectives : public Collectives {
 public:
  explicit StoreCollectives(
      c10::intrusive_ptr<Store> store,
      int rank,
      int world_size);

  void barrier(
      const std::string& prefix,
      std::chrono::milliseconds timeout = 5min,
      bool block = true) override;

  void broadcast_send(
      const std::string& prefix,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<uint8_t> broadcast_recv(
      const std::string& prefix,
      std::chrono::milliseconds timeout = 5min) override;

  void gather_send(
      const std::string& prefix,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<std::vector<uint8_t>> gather_recv(
      const std::string& prefix,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;

  std::vector<uint8_t> scatter_send(
      const std::string& prefix,
      const std::vector<std::vector<uint8_t>>& data,
      std::chrono::milliseconds timeout = 5min) override;
  std::vector<uint8_t> scatter_recv(
      const std::string& prefix,
      std::chrono::milliseconds timeout = 5min) override;

  std::vector<std::vector<uint8_t>> all_gather(
      const std::string& prefix,
      const std::vector<uint8_t>& data,
      std::chrono::milliseconds timeout = 5min) override;

  int64_t all_sum(
      const std::string& prefix,
      int64_t data,
      std::chrono::milliseconds timeout = 5min) override;

 private:
  c10::intrusive_ptr<Store> store_;
  int rank_;
  int world_size_;
};

} // namespace c10d
