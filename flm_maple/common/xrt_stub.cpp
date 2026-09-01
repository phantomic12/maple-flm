#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_uuid.h"

namespace xrt {

device::~device() = default;
device::device(unsigned int) {}
device::device(const std::string&) {}

bo::~bo() = default;

hw_context::~hw_context() = default;

kernel::~kernel() = default;

}
