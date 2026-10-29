// Prints the Panda's state to stdout as one JSON line (libfranka's RobotState format) every
// <period-ms> milliseconds, until it is interrupted, stdout is closed, or <count> states are
// printed. It only reads the state: it never sends a command, so it can run while the arm is
// hand-guided.
//
// Usage: stream_state <robot-hostname> [period-ms] [count]
//   stream_state 172.16.0.2 1 1 > state.txt     the current state, once

#include <csignal>
#include <iostream>
#include <string>

#include <franka/exception.h>
#include <franka/robot.h>

namespace {

volatile std::sig_atomic_t g_running = 1;

void stop(int) {
  g_running = 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname> [period-ms] [count]" << std::endl;
    return 2;
  }
  const int period = argc >= 3 ? std::stoi(argv[2]) : 20;
  const long count = argc >= 4 ? std::stol(argv[3]) : -1;
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    long received = 0;
    long printed = 0;
    robot.read([&](const franka::RobotState& state) {
      if (received++ % period == 0) {
        std::cout << state << std::endl;
        printed++;
      }
      return g_running && std::cout.good() && (count < 0 || printed < count);
    });
  } catch (const franka::Exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
