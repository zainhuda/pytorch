#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/StoreCollectives.hpp>
#include <chrono>
#include <exception>
#include <vector>

namespace c10d {

StoreCollectives::StoreCollectives(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int world_size)
    : store_(store), rank_(rank), world_size_(world_size) {}

void StoreCollectives::barrier(
    const std::string& prefix,
    std::chrono::milliseconds timeout,
    bool blocking) {
  StoreTimeoutGuard g{*store_, timeout};

  auto num_members_key = prefix + "/num_members";
  auto last_members_key = prefix + "/last_members";

  auto idx = store_->add(num_members_key, 1);
  store_->set(prefix + "/" + std::to_string(rank_), "joined");

  if (idx == world_size_) {
    store_->set(last_members_key, "<val_ignored>");
  } else if (blocking) {
    try {
      store_->wait({last_members_key});
    } catch (const std::exception& e) {
      std::string msg = "barrier failed -- missing ranks: ";
      for (int i = 0; i < world_size_; i++) {
        if (i == rank_) {
          continue;
        }
        auto key = prefix + "/" + std::to_string(i);
        if (!store_->check({key})) {
          msg += std::to_string(i) + ", ";
        }
      }
      throw std::runtime_error(msg + e.what());
    }
  }
}

void StoreCollectives::broadcast_send(
    const std::string& prefix,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  store_->set(prefix, data);
}

std::vector<uint8_t> StoreCollectives::broadcast_recv(
    const std::string& prefix,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  return store_->get(prefix);
}

void StoreCollectives::gather_send(
    const std::string& prefix,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  auto key = prefix + "/" + std::to_string(rank_);
  store_->set(key, data);
}

std::vector<std::vector<uint8_t>> StoreCollectives::gather_recv(
    const std::string& prefix,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  std::vector<std::string> keys;
  for (int i = 0; i < world_size_; i++) {
    if (i == rank_) {
      continue;
    }
    auto key = prefix + "/" + std::to_string(i);
    keys.emplace_back(key);
  }

  std::vector<std::vector<uint8_t>> results;
  try {
    results = store_->multiGet(keys);
  } catch (const std::exception& e) {
    std::string msg = "gather failed -- missing ranks: ";
    for (int i = 0; i < world_size_; i++) {
      if (i == rank_) {
        continue;
      }
      auto key = prefix + "/" + std::to_string(i);
      if (!store_->check({key})) {
        msg += std::to_string(i) + ", ";
      }
    }
    throw std::runtime_error(msg + e.what());
  }

  // insert local data
  results.insert(results.begin() + rank_, data);
  return results;
}

std::vector<uint8_t> StoreCollectives::scatter_send(
    const std::string& prefix,
    const std::vector<std::vector<uint8_t>>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  std::vector<std::string> keys;
  for (int i = 0; i < world_size_; i++) {
    if (i == rank_) {
      continue;
    }
    auto key = prefix + "/" + std::to_string(i);
    keys.emplace_back(key);
  }
  auto local = data.at(rank_);

  std::vector<std::vector<uint8_t>> to_send{data};

  to_send.erase(to_send.begin() + rank_);

  store_->multiSet(keys, to_send);

  return local;
}

std::vector<uint8_t> StoreCollectives::scatter_recv(
    const std::string& prefix,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  auto key = prefix + "/" + std::to_string(rank_);
  return store_->get(key);
}

std::vector<std::vector<uint8_t>> StoreCollectives::all_gather(
    const std::string& prefix,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  auto local_key = prefix + "/" + std::to_string(rank_);
  store_->set(local_key, data);

  std::vector<std::string> keys;
  for (int i = 0; i < world_size_; i++) {
    auto key = prefix + "/" + std::to_string(i);
    keys.emplace_back(key);
  }

  try {
    return store_->multiGet(keys);
  } catch (const std::exception& e) {
    std::string msg = "all_gather failed -- missing ranks: ";
    for (int i = 0; i < world_size_; i++) {
      if (i == rank_) {
        continue;
      }
      auto key = prefix + "/" + std::to_string(i);
      if (!store_->check({key})) {
        msg += std::to_string(i) + ", ";
      }
    }
    throw std::runtime_error(msg + e.what());
  }
}

int64_t StoreCollectives::all_sum(
    const std::string& prefix,
    int64_t value,
    std::chrono::milliseconds timeout) {
  StoreTimeoutGuard g{*store_, timeout};

  store_->add(prefix, value);

  barrier(prefix, timeout);

  return store_->add(prefix, 0);
}

} // namespace c10d
