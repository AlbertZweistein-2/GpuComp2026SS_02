#pragma once

// Contour tracing is currently host-side. Re-export the serial contour/signal
// declarations so CUDA pipeline code can include parallel stage headers only.
#include "serial/contour_to_signal.hpp"
