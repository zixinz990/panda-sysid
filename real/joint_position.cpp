// Moves the Panda along a 1 kHz joint position trajectory with Franka's internal joint
// position controller (for moving between poses; it tracks to hundredths of a degree).
//
// The trajectory (t, q1..q7, as written by scripts/move.py or scripts/plan_campaign.py) is
// absolute; the command at robot time t is q_start + traj(t) - traj(0), where q_start is the
// robot's q_d when the motion starts, so the robot follows the planned path exactly. It refuses
// to start when the robot is not idle or its q differs from traj(0) by more than 0.5 deg on any
// joint, and refuses a trajectory whose rows jump faster than 1 rad/s.
//
// After the motion, or when it is aborted, it writes every received robot state to
// <log.csv>: robot time (ms), q, dq, q_cmd (the command this program computed), q_d, dq_d,
// ddq_d (the robot's desired motion from the commands it received, after libfranka's rate
// limiter and low-pass filter), tau_J, tau_J_d (the internal controller's torque), dtau_J,
// tau_ext_hat_filtered, theta, dtheta, gravity, control_command_success_rate. gravity is the
// gravity torque of Franka's own model at q for the end effector set in Desk (m_total,
// F_x_Ctotal); it is computed after the motion, so the control loop does no extra work.
//
// Usage: joint_position <robot-hostname> <traj.csv> <log.csv> [--rt] [--yes]
// --rt asks for real-time scheduling (needs a PREEMPT_RT kernel); --yes skips the prompt.

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <franka/exception.h>
#include <franka/model.h>
#include <franka/robot.h>

namespace {

constexpr double kDt = 0.001;
constexpr double kStartTolerance = 0.5 * M_PI / 180.0;
constexpr double kMaxRowStep = 1.0 * kDt;  // rad per ms

using Row = std::array<double, 7>;

struct Sample {
  double t_ms;
  Row q, dq, q_cmd, q_d, dq_d, ddq_d, tau_J, tau_J_d, dtau_J, tau_ext, theta, dtheta;
  double success_rate;
  Row gravity;
};

void add_gravity(std::vector<Sample>& log, const franka::Model& model,
                 const franka::RobotState& state) {
  for (Sample& s : log) {
    s.gravity = model.gravity(s.q, state.m_total, state.F_x_Ctotal);
  }
}

void write_log(const std::string& path, const std::vector<Sample>& log) {
  std::ofstream out(path);
  out << "t_ms";
  for (const char* name : {"q", "dq", "q_cmd", "qd", "dqd", "ddqd", "tau_J", "tau_J_d", "dtau_J",
                           "tau_ext", "theta", "dtheta", "gravity"}) {
    for (int j = 1; j <= 7; j++) {
      out << ',' << name << j;
    }
  }
  out << ",success_rate\n";
  out.precision(10);
  for (const Sample& s : log) {
    out << s.t_ms;
    for (const Row* row : {&s.q, &s.dq, &s.q_cmd, &s.q_d, &s.dq_d, &s.ddq_d, &s.tau_J, &s.tau_J_d,
                           &s.dtau_J, &s.tau_ext, &s.theta, &s.dtheta, &s.gravity}) {
      for (double v : *row) {
        out << ',' << v;
      }
    }
    out << ',' << s.success_rate << '\n';
  }
  std::cout << "Wrote " << log.size() << " states to " << path << std::endl;
}

std::vector<Row> load(const std::string& path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("cannot open " + path);
  }
  std::vector<Row> rows;
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#' || line[0] == 't') {
      continue;
    }
    std::stringstream ss(line);
    std::string cell;
    std::vector<double> values;
    while (std::getline(ss, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    if (values.size() != 8) {
      throw std::runtime_error("expected 8 columns (t, q1..q7)");
    }
    if (std::abs(values[0] - kDt * rows.size()) > 1e-6) {
      throw std::runtime_error("rows must be 1 ms apart, starting at t = 0");
    }
    Row q;
    std::copy(values.begin() + 1, values.end(), q.begin());
    for (size_t j = 0; j < 7 && !rows.empty(); j++) {
      if (std::abs(q[j] - rows.back()[j]) > kMaxRowStep) {
        throw std::runtime_error("step too large at t = " + std::to_string(values[0]));
      }
    }
    rows.push_back(q);
  }
  if (rows.size() < 2) {
    throw std::runtime_error("trajectory is empty");
  }
  return rows;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname> <traj.csv> <log.csv> [--rt] [--yes]"
              << std::endl;
    return -1;
  }
  bool realtime = false;
  bool yes = false;
  for (int i = 4; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "--rt") {
      realtime = true;
    } else if (arg == "--yes") {
      yes = true;
    } else {
      std::cerr << "unknown option " << arg << std::endl;
      return -1;
    }
  }
  std::vector<Sample> log;
  std::unique_ptr<franka::Model> model;
  franka::RobotState state;
  try {
    const std::vector<Row> traj = load(argv[2]);
    const size_t n = traj.size();

    franka::Robot robot(argv[1], realtime ? franka::RealtimeConfig::kEnforce
                                          : franka::RealtimeConfig::kIgnore);
    model = std::make_unique<franka::Model>(robot.loadModel());
    state = robot.readOnce();
    if (state.robot_mode == franka::RobotMode::kReflex) {
      std::cout << "Robot is in reflex mode; running automatic error recovery." << std::endl;
      robot.automaticErrorRecovery();
      state = robot.readOnce();
    }
    if (state.robot_mode != franka::RobotMode::kIdle) {
      std::cerr << "Robot is not idle (mode " << static_cast<int>(state.robot_mode)
                << "): release the user stop and unlock the joints in Desk." << std::endl;
      return -1;
    }
    double worst = 0.0;
    for (size_t j = 0; j < 7; j++) {
      worst = std::max(worst, std::abs(state.q[j] - traj[0][j]));
    }
    if (worst > kStartTolerance) {
      std::cerr << "Robot is " << worst * 180.0 / M_PI
                << " deg away from the planned start; plan again from the current state."
                << std::endl;
      return -1;
    }

    // Default behavior of the libfranka 0.8.0 examples (examples_common.cpp).
    robot.setCollisionBehavior(
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0}},
        {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}}, {{20.0, 20.0, 20.0, 20.0, 20.0, 20.0}},
        {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}}, {{10.0, 10.0, 10.0, 10.0, 10.0, 10.0}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});

    std::cout << "Trajectory: " << (n - 1) * kDt << " s, start error " << worst * 180.0 / M_PI
              << " deg." << std::endl
              << "End effector set in Desk: " << state.m_total << " kg, center of mass "
              << state.F_x_Ctotal[0] << ", " << state.F_x_Ctotal[1] << ", " << state.F_x_Ctotal[2]
              << " m from the flange." << std::endl;
    if (!yes) {
      std::cout << "WARNING: the robot will move! Keep the user stop button at hand." << std::endl
                << "Press Enter to continue..." << std::endl;
      std::cin.ignore();
    }

    log.reserve(n + 1000);
    Row q_start{};
    double time = 0.0;
    double t0_ms = 0.0;
    robot.control([&](const franka::RobotState& s, franka::Duration period) -> franka::JointPositions {
      time += period.toSec();
      if (time == 0.0) {
        q_start = s.q_d;
        t0_ms = static_cast<double>(s.time.toMSec());
      }
      const size_t i = std::min(static_cast<size_t>(std::lround(time / kDt)), n - 1);
      Row q;
      for (size_t j = 0; j < 7; j++) {
        q[j] = q_start[j] + traj[i][j] - traj[0][j];
      }
      log.push_back({static_cast<double>(s.time.toMSec()) - t0_ms, s.q, s.dq, q, s.q_d, s.dq_d,
                     s.ddq_d, s.tau_J, s.tau_J_d, s.dtau_J, s.tau_ext_hat_filtered, s.theta,
                     s.dtheta, s.control_command_success_rate});
      franka::JointPositions output(q);
      if (i == n - 1) {
        return franka::MotionFinished(output);
      }
      return output;
    }, franka::ControllerMode::kJointImpedance, true, franka::kDefaultCutoffFrequency);

    state = robot.readOnce();
    double end_error = 0.0;
    for (size_t j = 0; j < 7; j++) {
      end_error = std::max(end_error, std::abs(state.q[j] - (q_start[j] + traj[n - 1][j] - traj[0][j])));
    }
    std::cout << "Finished. Largest joint error at the end: " << end_error * 180.0 / M_PI << " deg."
              << std::endl;
    add_gravity(log, *model, state);
    write_log(argv[3], log);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    if (!log.empty()) {
      add_gravity(log, *model, state);
      write_log(argv[3], log);
    }
    return -1;
  }
  return 0;
}
