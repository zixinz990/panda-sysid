// Clears a reflex of the Panda (franka::Robot::automaticErrorRecovery). It sends no motion; the
// players run the same recovery themselves before they start.
//
// Usage: recover <robot-hostname>

#include <iostream>

#include <franka/exception.h>
#include <franka/robot.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return 2;
  }
  try {
    franka::Robot robot(argv[1], franka::RealtimeConfig::kIgnore);
    std::cout << "Mode before: " << static_cast<int>(robot.readOnce().robot_mode) << std::endl;
    robot.automaticErrorRecovery();
    std::cout << "Mode after: " << static_cast<int>(robot.readOnce().robot_mode) << std::endl;
  } catch (const franka::Exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}
