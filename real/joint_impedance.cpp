// Joint impedance on the Panda: tau = kp (q_d - q) - kd dq through
// libfranka's torque interface (the robot adds gravity), at 1 kHz.
//
// q_d comes from a 1 kHz trajectory (traj.csv: t, q1..q7, as written by
// scripts/plan_campaign.py). The command at robot time t is q_start + traj(t) - traj(0),
// where q_start is the measured q when the motion starts, so the first torque is the
// damping term only. It refuses to start when the robot is not idle or its q differs
// from traj(0) by more than 0.5 deg on any joint.
//
// The motion is aborted (libfranka cancels it and the robot stops) when any joint
//   - lags its target by more than --max-error rad,
//   - moves faster than --max-speed rad/s,
//   - is commanded more than --max-torque-fraction of its torque limit,
//   - comes within 0.05 rad of its position limit.
//
// After the motion, or when it is aborted, it writes every received robot state to
// <log.csv>: robot time (ms), q, dq, q_d, the computed torque, tau_J, tau_J_d (the
// commanded torque after libfranka's rate limiter and low-pass filter), dtau_J,
// tau_ext_hat_filtered, theta, dtheta, gravity, control_command_success_rate. gravity is
// the gravity torque of Franka's own model at q for the end effector set in Desk
// (m_total, F_x_Ctotal), presumably what the robot adds to the command; it is computed
// after the motion, so the control loop does no extra work.
//
// Usage: joint_impedance <robot-hostname> <traj.csv> <log.csv> [--kp 40] [--kd 20]
//          [--max-error 0.15] [--max-speed 1.0] [--max-torque-fraction 0.5]
//          [--cutoff 100] [--rt] [--yes]
// --kp and --kd take one value or seven comma-separated values. --cutoff 1000 turns
// libfranka's low-pass filter off. --rt asks for real-time scheduling (needs a PREEMPT_RT
// kernel); --yes skips the prompt.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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
// Largest jump of q_d between two rows: the policy's targets jump every 20 ms.
constexpr double kMaxRowStep = 0.1;
constexpr double kLimitMargin = 0.05;

using Row = std::array<double, 7>;

const Row kQMin{{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
const Row kQMax{{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};
const Row kTorqueLimit{{87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0}};

struct Sample {
  double t_ms;
  Row q, dq, q_d, tau_cmd, tau_J, tau_J_d, dtau_J, tau_ext, theta, dtheta;
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
  for (const char* name : {"q", "dq", "qd", "tau_cmd", "tau_J", "tau_J_d", "dtau_J", "tau_ext",
                           "theta", "dtheta", "gravity"}) {
    for (int j = 1; j <= 7; j++) {
      out << ',' << name << j;
    }
  }
  out << ",success_rate\n";
  out.precision(10);
  for (const Sample& s : log) {
    out << s.t_ms;
    for (const Row* row : {&s.q, &s.dq, &s.q_d, &s.tau_cmd, &s.tau_J, &s.tau_J_d, &s.dtau_J,
                           &s.tau_ext, &s.theta, &s.dtheta, &s.gravity}) {
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
    for (size_t j = 0; j < 7; j++) {
      if (!rows.empty() && std::abs(q[j] - rows.back()[j]) > kMaxRowStep) {
        throw std::runtime_error("q_d jumps too far at t = " + std::to_string(values[0]));
      }
      if (q[j] < kQMin[j] + kLimitMargin || q[j] > kQMax[j] - kLimitMargin) {
        throw std::runtime_error("q_d too close to a joint limit at t = " +
                                 std::to_string(values[0]));
      }
    }
    rows.push_back(q);
  }
  if (rows.size() < 2) {
    throw std::runtime_error("trajectory is empty");
  }
  return rows;
}

Row parse_gains(const std::string& text) {
  std::vector<double> values;
  std::stringstream ss(text);
  std::string cell;
  while (std::getline(ss, cell, ',')) {
    values.push_back(std::stod(cell));
  }
  Row gains;
  if (values.size() == 1) {
    gains.fill(values[0]);
  } else if (values.size() == 7) {
    std::copy(values.begin(), values.end(), gains.begin());
  } else {
    throw std::runtime_error("gains take one or seven values: " + text);
  }
  return gains;
}

struct Options {
  Row kp, kd;
  double max_error = 0.15;
  double max_speed = 1.0;
  double max_torque_fraction = 0.5;
  double cutoff = franka::kDefaultCutoffFrequency;
  bool realtime = false;
  bool yes = false;
};

Options parse_options(int argc, char** argv) {
  Options opt;
  opt.kp.fill(40.0);
  opt.kd.fill(20.0);
  for (int i = 4; i < argc; i++) {
    const std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(arg + " needs a value");
      }
      return argv[++i];
    };
    if (arg == "--kp") {
      opt.kp = parse_gains(value());
    } else if (arg == "--kd") {
      opt.kd = parse_gains(value());
    } else if (arg == "--max-error") {
      opt.max_error = std::stod(value());
    } else if (arg == "--max-speed") {
      opt.max_speed = std::stod(value());
    } else if (arg == "--max-torque-fraction") {
      opt.max_torque_fraction = std::stod(value());
    } else if (arg == "--cutoff") {
      opt.cutoff = std::stod(value());
    } else if (arg == "--rt") {
      opt.realtime = true;
    } else if (arg == "--yes") {
      opt.yes = true;
    } else {
      throw std::runtime_error("unknown option " + arg);
    }
  }
  return opt;
}

std::string format_row(const Row& row) {
  std::ostringstream ss;
  ss.precision(4);
  for (size_t j = 0; j < 7; j++) {
    ss << (j ? ", " : "") << row[j];
  }
  return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <robot-hostname> <traj.csv> <log.csv> [--kp 40] [--kd 20] [--max-error 0.15]"
                 " [--max-speed 1.0] [--max-torque-fraction 0.5] [--cutoff 100] [--rt] [--yes]"
              << std::endl;
    return -1;
  }
  std::vector<Sample> log;
  std::unique_ptr<franka::Model> model;
  franka::RobotState state;
  try {
    const Options opt = parse_options(argc, argv);
    const std::vector<Row> traj = load(argv[2]);
    const size_t n = traj.size();

    franka::Robot robot(argv[1], opt.realtime ? franka::RealtimeConfig::kEnforce
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

    std::cout << "Trajectory: " << (n - 1) * kDt << " s, start error " << worst * 180.0 / M_PI
              << " deg." << std::endl
              << "kp: " << format_row(opt.kp) << std::endl
              << "kd: " << format_row(opt.kd) << std::endl
              << "Abort above: error " << opt.max_error << " rad, speed " << opt.max_speed
              << " rad/s, torque " << opt.max_torque_fraction << " of the limits. Low-pass "
              << opt.cutoff << " Hz." << std::endl
              << "End effector set in Desk: " << state.m_total << " kg, center of mass "
              << state.F_x_Ctotal[0] << ", " << state.F_x_Ctotal[1] << ", " << state.F_x_Ctotal[2]
              << " m from the flange." << std::endl;
    if (!opt.yes) {
      std::cout << "WARNING: the robot will move! Keep the user stop button at hand." << std::endl
                << "Press Enter to continue..." << std::endl;
      std::cin.ignore();
    }

    log.reserve(n + 1000);
    Row q_start{};
    double time = 0.0;
    double t0_ms = 0.0;
    auto loop = [&](const franka::RobotState& s, franka::Duration period) -> franka::Torques {
      time += period.toSec();
      if (time == 0.0) {
        q_start = s.q;
        t0_ms = static_cast<double>(s.time.toMSec());
      }
      const size_t i = std::min(static_cast<size_t>(std::lround(time / kDt)), n - 1);
      Row q_d, tau;
      for (size_t j = 0; j < 7; j++) {
        q_d[j] = q_start[j] + traj[i][j] - traj[0][j];
        tau[j] = opt.kp[j] * (q_d[j] - s.q[j]) - opt.kd[j] * s.dq[j];
      }
      log.push_back({static_cast<double>(s.time.toMSec()) - t0_ms, s.q, s.dq, q_d, tau, s.tau_J,
                     s.tau_J_d, s.dtau_J, s.tau_ext_hat_filtered, s.theta, s.dtheta,
                     s.control_command_success_rate});
      for (size_t j = 0; j < 7; j++) {
        std::string reason;
        if (std::abs(q_d[j] - s.q[j]) > opt.max_error) {
          reason = "tracking error";
        } else if (std::abs(s.dq[j]) > opt.max_speed) {
          reason = "speed";
        } else if (std::abs(tau[j]) > opt.max_torque_fraction * kTorqueLimit[j]) {
          reason = "torque";
        } else if (s.q[j] < kQMin[j] + kLimitMargin || s.q[j] > kQMax[j] - kLimitMargin) {
          reason = "joint limit";
        }
        if (!reason.empty()) {
          throw std::runtime_error("Abort at t = " + std::to_string(time) + " s: joint " +
                                   std::to_string(j + 1) + " " + reason);
        }
      }
      franka::Torques output(tau);
      if (i == n - 1) {
        return franka::MotionFinished(output);
      }
      return output;
    };
    robot.control(loop, true, opt.cutoff);

    size_t missed = 0;
    for (size_t k = 1; k < log.size(); k++) {
      missed += static_cast<size_t>(std::max(0.0, log[k].t_ms - log[k - 1].t_ms - 1.0));
    }
    double max_error = 0.0;
    for (const Sample& s : log) {
      for (size_t j = 0; j < 7; j++) {
        max_error = std::max(max_error, std::abs(s.q_d[j] - s.q[j]));
      }
    }
    std::cout << "Finished: " << log.size() << " states, " << missed
              << " missed ticks, largest tracking error " << max_error * 180.0 / M_PI << " deg."
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
