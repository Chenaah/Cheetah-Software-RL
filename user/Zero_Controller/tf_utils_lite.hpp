#pragma once

inline void DeleteTensor(TF_Tensor* tensor) {
  if (tensor != nullptr) {
    TF_DeleteTensor(tensor);
  }
}

inline void DeleteTensors(const std::vector<TF_Tensor*>& tensors) {
  for (auto& t : tensors) {
    DeleteTensor(t);
  }
}

inline TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::int64_t* dims, std::size_t num_dims, std::size_t len) {
  if (dims == nullptr) {
    return nullptr;
  }

  return TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
}

inline TF_Tensor* CreateEmptyTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, std::size_t len) {
  return CreateEmptyTensor(data_type, dims.data(), dims.size(), len);
}

inline TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len) {
  auto tensor = CreateEmptyTensor(data_type, dims, num_dims, len);
  if (tensor == nullptr) {
    return nullptr;
  }

  auto tensor_data = TF_TensorData(tensor);
  if (tensor_data == nullptr) {
    DeleteTensor(tensor);
    return nullptr;
  }

  len = std::min(len, TF_TensorByteSize(tensor));
  if (data != nullptr && len != 0) {
    std::memcpy(tensor_data, data, len);
  }

  return tensor;
}

template <typename T>
inline TF_Tensor* CreateTensor(TF_DataType data_type, const std::vector<std::int64_t>& dims, const std::vector<T>& data) {
  return CreateTensor(data_type,
                      dims.data(), dims.size(),
                      data.data(), data.size() * sizeof(T));
}