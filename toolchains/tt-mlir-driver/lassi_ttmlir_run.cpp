// lassi-ttmlir-run: run one tt-mlir flatbuffer through the pinned tt-mlir runtime (task P17.48; the owner's answers
// to OQ-111, option (b), and OQ-112, route A first).
//
// Built by toolchains/tt-mlir-driver.sh against the runtime install named in toolchains/tt-mlir-runtime.pin (tt-mlir
// 5f396fd6ef85780bf1b6b85bef9c7d2f84a85f3c); every file:line below is at that commit. Two modes:
//
//   lassi-ttmlir-run describe <out.ttsys>
//       Write the system descriptor of the device the environment selects (on ttsim: TT_METAL_SIMULATOR) with
//       getCurrentSystemDesc (runtime/include/tt/runtime/runtime.h:44-47), which opens one mesh device and closes it
//       (runtime/lib/common/system_desc.cpp:265-298), then Flatbuffer::store (types.h:279), and read it back.
//
//   lassi-ttmlir-run run <flatbuffer> <descriptor.ttsys> <input files...> <output files...>
//       Load a TT-Metal ("TTM0", include/ttmlir/Target/TTMetal/binary.fbs:61) or TT-NN ("TTNN",
//       include/ttmlir/Target/TTNN/binary.fbs:18) flatbuffer; refuse it unless its schema hash is the runtime's
//       (runtime/lib/binary.cpp:525-545) and the system descriptor it embeds is the recorded one; read each input, a
//       lassi_io file (assets/harness/c/lassi_io.h), in the program's input order; open one 1x1 mesh device; run the
//       program; copy each output to the host; close the device once; write each output as a lassi_io file named by
//       its file's stem, in the program's output order.
//
// The run follows ttrt's own sequence for both runtimes (tools/ttrt/common/run.py:602, :763-765, :815-849;
// tools/ttrt/common/util.py:262-271, :304-310): borrowed host tensors (runtime.h:55-59), each converted with
// toLayout to the layout getLayout gives for that input (runtime.h:217-225; TT-NN: runtime/lib/ttnn/runtime.cpp:
// 776-803 and :823-853; TT-Metal: an empty layout and a no-op, runtime/lib/ttmetal/runtime.cpp:38-61), submit
// (runtime.h:268-270), wait, toHost with untilize (runtime.h:211-212), memcpy into a borrowed host tensor of the
// program's output description (runtime.h:230; TT-Metal copies by stride, runtime/lib/ttmetal/runtime.cpp:606-690;
// TT-NN checks the sizes, runtime/lib/ttnn/runtime.cpp:910-933), and deallocateTensor with force (runtime.h:234).
// Program descriptions come from getProgramInputs and getProgramOutputs (types.h:335-336), whose TensorDesc is the
// logical shape with a contiguous stride (runtime/lib/binary.cpp:164-188 for TT-NN, :323-363 for TT-Metal), so an
// input is the lassi_io array's bytes as they are.
//
// Exit status: 0 when the outputs are written; 2 when the driver refuses before any device is opened (a bad
// argument, flatbuffer, descriptor, or input), with one line "lassi-ttmlir-run: refused: <reason>" on stderr; 1 when
// the runtime throws (its LOG_FATAL and LOG_ASSERT throw, runtime/include/tt/runtime/detail/common/logger.h:388-402),
// with one line "lassi-ttmlir-run: error: <what>". The simulator may end the process itself: after an
// UndefinedBehavior finding ttsim v1.3.4 exited with status 1 and no driver line (plans/spikes/ttmlir-runtime.md,
// 2026-10-30), and an abort ends it by a signal; the executor reads the finding lines and the signal. Nothing here
// chooses a device: the environment does (Agent Rule 9: one open and one close per process, never in a loop).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "lassi_io.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

// TT-NN's configuration, a static object of _ttnncpp.so, which libTTMLIRRuntime always links
// (runtime/lib/CMakeLists.txt:30-53), builds its cache paths from std::getenv("HOME") with no check (tt-metal
// 5280a9cf, ttnn/api/ttnn/config.hpp:23-24; the object is ttnn/core/config.cpp:9), so with HOME unset the program
// faults while the libraries load, before main (plans/spikes/ttmlir-runtime.md, 2026-10-30). The driver sets no
// environment of its own: the ttmlir-ttsim executor passes HOME, a reserved directory inside the sandbox that is
// not collected as a run output, in the environment it records (the owner's answer to OQ-115).

namespace {

namespace rt = ::tt::runtime;
using ::tt::target::DataType;

constexpr const char *kProgram = "lassi-ttmlir-run";
constexpr int kRefused = 2;
constexpr int kRuntimeError = 1;
// The file identifiers the runtime reads (runtime/lib/binary.cpp:468-485).
const std::string kTTMetalId = "TTM0";
const std::string kTTNNId = "TTNN";
const std::string kSystemDescId = "TTSY";
// The suffix of a lassi_io file, which an output's array name leaves out.
const std::string kLassiIoSuffix = ".lassiio";

// A refusal before any device is opened: exit status kRefused.
class Refusal : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

// Return the runtime data type of a lassi_io dtype code. Only the types the runtime supports are mapped
// (runtime/lib/common/utils.cpp:172-196); F16, I8, I64, and F64 have no runtime tensor and are refused.
std::optional<DataType> runtimeType(uint32_t code) {
  switch (code) {
  case LASSI_IO_F32:
    return DataType::Float32;
  case LASSI_IO_BF16:
    return DataType::BFloat16;
  case LASSI_IO_I32:
    return DataType::Int32;
  case LASSI_IO_U32:
    return DataType::UInt32;
  case LASSI_IO_U8:
    return DataType::UInt8;
  default:
    return std::nullopt;
  }
}

// Return the lassi_io dtype code of a runtime data type, or nothing for a type lassi_io cannot hold.
std::optional<uint32_t> lassiType(DataType type) {
  switch (type) {
  case DataType::Float32:
    return LASSI_IO_F32;
  case DataType::BFloat16:
    return LASSI_IO_BF16;
  case DataType::Int32:
    return LASSI_IO_I32;
  case DataType::UInt32:
    return LASSI_IO_U32;
  case DataType::UInt8:
    return LASSI_IO_U8;
  default:
    return std::nullopt;
  }
}

std::string typeName(DataType type) { return ::tt::target::EnumNameDataType(type); }

std::string shapeText(const std::vector<uint32_t> &shape) {
  std::string text = "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    text += (i ? "x" : "") + std::to_string(shape[i]);
  }
  return text + "]";
}

// Return `json` without the whitespace between its tokens; strings are kept as they are.
std::string minified(const std::string &json) {
  std::string out;
  out.reserve(json.size());
  bool inString = false;
  bool escaped = false;
  for (char c : json) {
    if (inString) {
      out += c;
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
    } else if (c == '"') {
      inString = true;
      out += c;
    } else if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
      out += c;
    }
  }
  return out;
}

// Return where the JSON value that starts at `start` ends in minified `json`: the index of the ',' or closing
// bracket that follows it at its own level.
std::size_t valueEnd(const std::string &json, std::size_t start) {
  int depth = 0;
  bool inString = false;
  bool escaped = false;
  for (std::size_t i = start; i < json.size(); ++i) {
    const char c = json[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
    } else if (c == '"') {
      inString = true;
    } else if (c == '{' || c == '[') {
      ++depth;
    } else if (c == '}' || c == ']') {
      if (depth == 0) {
        return i;
      }
      --depth;
    } else if (c == ',' && depth == 0) {
      return i;
    }
  }
  return json.size();
}

// Return the value of the member `key` of the top-level object in minified `json`, or nothing.
std::optional<std::string> topLevelMember(const std::string &json, const std::string &key) {
  const std::string pattern = "\"" + key + "\":";
  int depth = 0;
  bool inString = false;
  bool escaped = false;
  for (std::size_t i = 0; i < json.size(); ++i) {
    const char c = json[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        inString = false;
      }
      continue;
    }
    if (c == '"') {
      const bool atKey = depth == 1 && i > 0 && (json[i - 1] == '{' || json[i - 1] == ',');
      if (atKey && json.compare(i, pattern.size(), pattern) == 0) {
        const std::size_t start = i + pattern.size();
        return json.substr(start, valueEnd(json, start) - start);
      }
      inString = true;
    } else if (c == '{' || c == '[') {
      ++depth;
    } else if (c == '}' || c == ']') {
      --depth;
    }
  }
  return std::nullopt;
}

// Refuse a file that cannot be opened or is shorter than a size-prefixed flatbuffer's head (its size, its root
// offset, and its file identifier: 12 bytes), which the runtime would read past.
void checkReadable(const std::string &path, const std::string &what) {
  std::FILE *probe = std::fopen(path.c_str(), "rb");
  if (probe == nullptr) {
    throw Refusal("cannot open the " + what + " " + path);
  }
  unsigned char head[12];
  const std::size_t got = std::fread(head, 1, sizeof head, probe);
  std::fclose(probe);
  if (got != sizeof head) {
    throw Refusal(path + " is too short to be a " + what);
  }
}

// Load the flatbuffer and refuse it unless it is a TT-Metal or TT-NN binary of this runtime's schema.
rt::Binary loadBinary(const std::string &path) {
  checkReadable(path, "flatbuffer");
  rt::Binary binary = rt::Binary::loadFromPath(path.c_str());
  std::string id;
  try {
    id = binary.getFileIdentifier();
  } catch (const std::exception &) {
    throw Refusal(path + " is not a tt-mlir flatbuffer (no TTM0 or TTNN file identifier)");
  }
  if (id != kTTMetalId && id != kTTNNId) {
    throw Refusal(path + " has the file identifier " + id + ", not TTM0 (TT-Metal) or TTNN (TT-NN)");
  }
  if (!binary.checkSchemaHash()) {
    throw Refusal(path + " was written with another flatbuffer schema (hash " + binary.getSchemaHash() +
                  "); compile it with the pinned tt-mlir");
  }
  return binary;
}

// Refuse the binary unless the system descriptor it embeds is the one recorded at `descPath`, compared as ttrt
// compares them (tools/ttrt/common/util.py:868-907): the binary's system_desc table (runtime/lib/binary.cpp:190-195,
// :289-294) against the recorded root's system_desc member, both as the runtime's JSON. A flatbuffer compiled for
// another descriptor, the mock one included, places L1 buffers the runtime never checks
// (plans/spikes/ttir-d2m.md, section 1).
void checkDescriptor(const rt::Binary &binary, const std::string &descPath) {
  checkReadable(descPath, "recorded system descriptor");
  rt::SystemDesc desc = rt::SystemDesc::loadFromPath(descPath.c_str());
  bool ours = false;
  try {
    ours = desc.getFileIdentifier() == kSystemDescId && desc.checkSchemaHash();
  } catch (const std::exception &) {
    ours = false;
  }
  if (!ours) {
    throw Refusal(descPath + " is not a system descriptor of this runtime's schema");
  }
  const std::optional<std::string> recorded = topLevelMember(minified(desc.asJson()), "system_desc");
  if (!recorded.has_value()) {
    throw Refusal(descPath + " holds no system_desc member");
  }
  if (minified(binary.getSystemDescAsJson()) != *recorded) {
    throw Refusal("the flatbuffer was compiled for another system descriptor than " + descPath +
                  "; compile it with system-desc-path set to that file (a flatbuffer for the mock descriptor never "
                  "runs)");
  }
}

// Return the one program that is not private (ttrt skips private ones, tools/ttrt/common/run.py:660-663).
uint32_t publicProgram(const rt::Binary &binary) {
  std::vector<uint32_t> found;
  for (uint32_t i = 0; i < binary.getNumPrograms(); ++i) {
    if (!binary.isProgramPrivate(i)) {
      found.push_back(i);
    }
  }
  if (found.size() != 1) {
    throw Refusal("the flatbuffer holds " + std::to_string(found.size()) +
                  " public programs; the driver runs a flatbuffer with exactly one");
  }
  const auto mesh = binary.getProgramMeshShape(found[0]);
  if (mesh.first != 1 || mesh.second != 1) {
    throw Refusal("program " + binary.getProgramName(found[0]) + " asks for a " + std::to_string(mesh.first) + "x" +
                  std::to_string(mesh.second) + " mesh; the driver opens one device, a 1x1 mesh");
  }
  return found[0];
}

// Frees a lassi_io array when it goes out of scope.
struct ArrayGuard {
  lassi_io_array array{};
  ArrayGuard() = default;
  ArrayGuard(const ArrayGuard &) = delete;
  ArrayGuard &operator=(const ArrayGuard &) = delete;
  ~ArrayGuard() { lassi_io_free(&array); }
};

// Read input `index` from `path` into a zeroed buffer of the program input's size, after checking its dtype and
// element count against `desc`.
std::vector<std::byte> readInput(const std::string &path, const rt::TensorDesc &desc, std::size_t index) {
  const std::string which = "input " + std::to_string(index) + " (" + path + ")";
  if (desc.isPadded()) {
    throw Refusal(which + ": the program describes a padded input, which the driver does not lay out");
  }
  ArrayGuard guard;
  if (lassi_io_read(path.c_str(), &guard.array) != 0) {
    throw Refusal(which + " is not a readable lassi_io file");
  }
  const std::optional<DataType> type = runtimeType(guard.array.dtype);
  if (!type.has_value()) {
    throw Refusal(which + " has lassi_io dtype code " + std::to_string(guard.array.dtype) +
                  ", which the runtime cannot take; the program wants " + typeName(desc.dataType));
  }
  if (*type != desc.dataType) {
    throw Refusal(which + " holds " + typeName(*type) + " elements; the program's input " +
                  std::to_string(index) + " is " + typeName(desc.dataType) + " " + shapeText(desc.shape));
  }
  if (guard.array.count != desc.volume()) {
    throw Refusal(which + " holds " + std::to_string(guard.array.count) + " elements; the program's input " +
                  std::to_string(index) + " " + shapeText(desc.shape) + " holds " + std::to_string(desc.volume()));
  }
  std::vector<std::byte> buffer(desc.sizeBytes());
  if (guard.array.nbytes > buffer.size()) {
    throw Refusal(which + " holds more bytes than the program's input");
  }
  std::memcpy(buffer.data(), guard.array.data, guard.array.nbytes);
  return buffer;
}

// Return the array name of an output file: its base name without the lassi_io suffix.
std::string outputName(const std::string &path) {
  const std::size_t slash = path.find_last_of('/');
  std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
  if (name.size() > kLassiIoSuffix.size() &&
      name.compare(name.size() - kLassiIoSuffix.size(), kLassiIoSuffix.size(), kLassiIoSuffix) == 0) {
    name.resize(name.size() - kLassiIoSuffix.size());
  }
  return name;
}

// Refuse an output a lassi_io file cannot hold, before any device is opened.
void checkOutput(const rt::TensorDesc &desc, const std::string &path, std::size_t index) {
  if (!lassiType(desc.dataType).has_value()) {
    throw Refusal("output " + std::to_string(index) + " is " + typeName(desc.dataType) +
                  ", which a lassi_io file cannot hold");
  }
  const std::string name = outputName(path);
  if (name.empty() || name.size() > LASSI_IO_MAX_NAME) {
    throw Refusal("output file " + path + " gives no lassi_io array name of 1 to 255 bytes");
  }
}

// Holds the one open mesh device and closes it exactly once: on close(), or when it goes out of scope after an
// error. It is made before any tensor, so every tensor is gone before the device closes.
class OpenDevice {
public:
  explicit OpenDevice(rt::Device device) : device_(std::move(device)) {}
  OpenDevice(const OpenDevice &) = delete;
  OpenDevice &operator=(const OpenDevice &) = delete;
  rt::Device get() const { return *device_; }
  void close() {
    if (device_.has_value()) {
      rt::Device device = std::move(*device_);
      device_.reset();
      rt::closeMeshDevice(device);
    }
  }
  ~OpenDevice() {
    try {
      close();
    } catch (const std::exception &error) {
      std::fprintf(stderr, "%s: error: closing the device after an error failed: %s\n", kProgram, error.what());
    }
  }

private:
  std::optional<rt::Device> device_;
};

// Return the device tensors of the inputs: borrowed host tensors converted to the program's input layouts.
std::vector<rt::Tensor> deviceInputs(const rt::Binary &binary, uint32_t program, rt::Device device,
                                     const std::vector<rt::TensorDesc> &descs,
                                     std::vector<std::vector<std::byte>> &inputs) {
  std::vector<rt::Tensor> tensors;
  tensors.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    rt::Tensor host = rt::createBorrowedHostTensor(inputs[i].data(), descs[i]);
    rt::Layout layout = rt::getLayout(binary, program, static_cast<uint32_t>(i));
    tensors.push_back(rt::toLayout(host, device, layout, /*retain=*/true));
  }
  return tensors;
}

// Copy each output to the host, untilized, into a buffer of its program description, and free its device copy.
std::vector<std::vector<std::byte>> hostOutputs(std::vector<rt::Tensor> &outputs,
                                                const std::vector<rt::TensorDesc> &descs) {
  std::vector<std::vector<std::byte>> buffers;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    std::vector<rt::Tensor> host = rt::toHost(outputs[i], /*untilize=*/true);
    if (host.size() != 1) {
      throw std::runtime_error("output " + std::to_string(i) + " came back as " + std::to_string(host.size()) +
                               " host tensors; one device gives one");
    }
    if (rt::getTensorDataType(host[0]) != descs[i].dataType) {
      throw std::runtime_error("output " + std::to_string(i) + " came back as " +
                               typeName(rt::getTensorDataType(host[0])) + ", not " + typeName(descs[i].dataType));
    }
    std::vector<std::byte> buffer(descs[i].sizeBytes());
    rt::Tensor target = rt::createBorrowedHostTensor(buffer.data(), descs[i]);
    rt::memcpy(target, host[0]);
    rt::deallocateTensor(outputs[i], /*force=*/true);
    buffers.push_back(std::move(buffer));
  }
  return buffers;
}

// Open one 1x1 mesh device, run the program on the inputs, bring the outputs to the host, and close the device.
std::vector<std::vector<std::byte>> execute(const rt::Binary &binary, uint32_t program,
                                            const std::vector<rt::TensorDesc> &inDescs,
                                            std::vector<std::vector<std::byte>> &inputs,
                                            const std::vector<rt::TensorDesc> &outDescs) {
  rt::MeshDeviceOptions options;
  options.meshShape = std::vector<uint32_t>{1, 1};
  OpenDevice device(rt::openMeshDevice(options));
  std::vector<std::vector<std::byte>> results;
  {
    std::vector<rt::Tensor> tensors = deviceInputs(binary, program, device.get(), inDescs, inputs);
    std::vector<rt::Tensor> outputs = rt::submit(device.get(), binary, program, tensors);
    rt::wait(outputs);
    results = hostOutputs(outputs, outDescs);
  }
  device.close();
  return results;
}

// Write each output buffer as a lassi_io file with the program's logical shape and data type.
void writeOutputs(const std::vector<rt::TensorDesc> &descs, const std::vector<std::vector<std::byte>> &buffers,
                  const std::vector<std::string> &paths) {
  for (std::size_t i = 0; i < descs.size(); ++i) {
    const std::vector<uint64_t> dims(descs[i].shape.begin(), descs[i].shape.end());
    const std::string name = outputName(paths[i]);
    const uint32_t code = *lassiType(descs[i].dataType);
    const int status = lassi_io_write(paths[i].c_str(), name.c_str(), code, static_cast<uint32_t>(dims.size()),
                                      dims.empty() ? nullptr : dims.data(), buffers[i].data());
    if (status != 0) {
      throw std::runtime_error("cannot write output " + std::to_string(i) + " to " + paths[i]);
    }
  }
}

// The run mode: check everything that can be checked without a device, then run once and write the outputs.
int runMode(const std::string &flatbuffer, const std::string &descPath, const std::vector<std::string> &files) {
  rt::Binary binary = loadBinary(flatbuffer);
  checkDescriptor(binary, descPath);
  rt::setCompatibleDeviceRuntime(binary);
  const uint32_t program = publicProgram(binary);
  const std::vector<rt::TensorDesc> inDescs = binary.getProgramInputs(program);
  const std::vector<rt::TensorDesc> outDescs = binary.getProgramOutputs(program);
  if (files.size() != inDescs.size() + outDescs.size()) {
    throw Refusal("program " + binary.getProgramName(program) + " takes " + std::to_string(inDescs.size()) +
                  " inputs and gives " + std::to_string(outDescs.size()) + " outputs, but " +
                  std::to_string(files.size()) + " files were named");
  }
  std::vector<std::vector<std::byte>> inputs;
  for (std::size_t i = 0; i < inDescs.size(); ++i) {
    inputs.push_back(readInput(files[i], inDescs[i], i));
  }
  const std::vector<std::string> outPaths(files.begin() + static_cast<std::ptrdiff_t>(inDescs.size()), files.end());
  for (std::size_t i = 0; i < outDescs.size(); ++i) {
    checkOutput(outDescs[i], outPaths[i], i);
  }
  std::vector<std::vector<std::byte>> results = execute(binary, program, inDescs, inputs, outDescs);
  writeOutputs(outDescs, results, outPaths);
  std::printf("%s: ran program %s (%s) with %zu inputs and wrote %zu outputs\n", kProgram,
              binary.getProgramName(program).c_str(), binary.getFileIdentifier().c_str(), inDescs.size(),
              outDescs.size());
  return 0;
}

// The describe mode: write the current device's system descriptor and read it back.
int describeMode(const std::string &out) {
  // getCurrentSystemDesc dispatches on the current device runtime (runtime/lib/runtime.cpp:229-248), and both
  // runtimes call the same function; TT-Metal is chosen so that one is set.
  rt::setCurrentDeviceRuntime(rt::DeviceRuntime::TTMetal);
  rt::SystemDesc desc = rt::getCurrentSystemDesc();
  desc.store(out.c_str());
  rt::SystemDesc back = rt::SystemDesc::loadFromPath(out.c_str());
  if (back.getFileIdentifier() != kSystemDescId || !back.checkSchemaHash() ||
      !topLevelMember(minified(back.asJson()), "system_desc").has_value()) {
    throw std::runtime_error("the descriptor written to " + out + " does not read back as a system descriptor");
  }
  std::printf("%s: wrote the system descriptor %s\n", kProgram, out.c_str());
  return 0;
}

int usage() {
  std::fprintf(stderr,
               "%s: refused: usage: %s describe <out.ttsys> | %s run <flatbuffer> <descriptor.ttsys> "
               "<input files...> <output files...>\n",
               kProgram, kProgram, kProgram);
  return kRefused;
}

} // namespace

int main(int argc, char **argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  try {
    if (args.size() == 2 && args[0] == "describe") {
      return describeMode(args[1]);
    }
    if (args.size() >= 3 && args[0] == "run") {
      return runMode(args[1], args[2], std::vector<std::string>(args.begin() + 3, args.end()));
    }
    return usage();
  } catch (const Refusal &refusal) {
    std::fprintf(stderr, "%s: refused: %s\n", kProgram, refusal.what());
    return kRefused;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s: error: %s\n", kProgram, error.what());
    return kRuntimeError;
  } catch (...) {
    std::fprintf(stderr, "%s: error: an exception that is not a std::exception\n", kProgram);
    return kRuntimeError;
  }
}
