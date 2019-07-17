#include <memory>

#include <gflags/gflags.h>

#include "drake/common/drake_assert.h"
#include "drake/common/find_resource.h"
#include "drake/common/text_logging_gflags.h"
#include "drake/geometry/geometry_visualization.h"
#include "drake/geometry/scene_graph.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/prismatic_joint.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/framework/diagram_builder.h"
#include "drake/systems/controllers/inverse_dynamics_controller.h"
#include "drake/systems/primitives/constant_vector_source.h"
#include "drake/systems/primitives/constant_value_source.h"
#include "Eigen/src/Core/Matrix.h"
#include "drake/systems/primitives/sine.h"
#include "drake/systems/primitives/multiplexer.h"
#include "drake/systems/primitives/adder.h"
#include "drake/systems/trajectory_optimization/direct_transcription.h"
#include "drake/systems/trajectory_optimization/direct_collocation.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/common/trajectories/piecewise_polynomial.h"

namespace drake {
namespace examples {
namespace planar_gripper {
namespace {

using geometry::SceneGraph;

using drake::multibody::MultibodyPlant;
using drake::multibody::Parser;
using drake::multibody::RevoluteJoint;
using drake::math::RigidTransform;
using drake::math::RollPitchYaw;
using drake::multibody::JointActuatorIndex;
using drake::multibody::ModelInstanceIndex;

DEFINE_double(target_realtime_rate, 1.0,
              "Desired rate relative to real time.  See documentation for "
              "Simulator::set_target_realtime_rate() for details.");
DEFINE_double(simulation_time, 8.0,
              "Desired duration of the simulation in seconds.");
DEFINE_double(time_step, 5e-4,
            "If greater than zero, the plant is modeled as a system with "
            "discrete updates and period equal to this time_step. "
            "If 0, the plant is modeled as a continuous system.");
DEFINE_double(brick_z, 0, "Location of the brock on z-axis");
DEFINE_double(fix_input, false, "Fix the actuation inputs to zero?");

template<typename T>
void WeldFingerFrame(multibody::MultibodyPlant<T> *plant) {
  // This function is copied and adapted from planar_gripper_simulation.py
  const double outer_radius = 0.19;
  const double f1_angle = 0;
  const math::RigidTransformd XT(math::RollPitchYaw<double>(0, 0, 0),
                                 Eigen::Vector3d(0, 0, outer_radius));

  // Weld the first finger.
  math::RigidTransformd X_PC1(math::RollPitchYaw<double>(f1_angle, 0, 0),
                              Eigen::Vector3d::Zero());
  X_PC1 = X_PC1 * XT;
  const multibody::Frame<T> &finger1_base_frame =
      plant->GetFrameByName("finger_base");
  plant->WeldFrames(plant->world_frame(), finger1_base_frame, X_PC1);
}

template<typename T>
std::pair<trajectories::PiecewisePolynomial<T>,
    trajectories::PiecewisePolynomial<T>> TrajOptFinger(
    const systems::System<T>* system,
    const systems::Context<T>& context,
    const systems::InputPortIndex actuation_port_index) {
  const int kNumTimeSamples = 21;

  systems::trajectory_optimization::DirectTranscription prog(
      system, context, kNumTimeSamples, actuation_port_index);

  const solvers::VectorXDecisionVariable& u = prog.input();

  // Set the initial and final state constraints.
  systems::BasicVector<double> initial_state(Eigen::VectorXd::Zero(6));
  initial_state[1] = .5;
  initial_state[2] = .7;
  prog.AddLinearConstraint(prog.initial_state() ==
                              initial_state.get_value());

  systems::BasicVector<double> final_state(Eigen::VectorXd::Zero(6));
  final_state[1] = .3;
  final_state[2] = .5;
  prog.AddLinearConstraint(prog.final_state() == final_state.get_value());

  Eigen::Matrix2d R;
  R << 10., 0.,
      0., 10.;
  prog.AddRunningCost(u.transpose() * R * u);

  solvers::SnoptSolver snopt_solver;
  DRAKE_THROW_UNLESS(snopt_solver.is_available());
  const auto result = snopt_solver.Solve(prog, {}, {});

  return std::make_pair(prog.ReconstructStateTrajectory(result),
      prog.ReconstructInputTrajectory(result));
}

int do_main() {
  // paths to models
  auto finger_urdf_path = FindResourceOrThrow(
      "drake/examples/planar_gripper/planar_finger.sdf");
  auto object_urdf_path = FindResourceOrThrow(
      "drake/examples/planar_gripper/1dof_brick.sdf");

  // Make the planar_finger model.
  auto plant =
      std::make_unique<multibody::MultibodyPlant<double>>(FLAGS_time_step);

  // Adds the finger
  Parser(plant.get()).AddModelFromFile(finger_urdf_path, "finger");

  // Adds the object to be manipulated.
  Parser(plant.get()).AddModelFromFile(object_urdf_path, "object");

  WeldFingerFrame<double>(plant.get());

  // Add gravity
  Vector3<double> gravity(0, 0, -9.81);
  plant->mutable_gravity_field().set_gravity_vector(gravity);

  // Now the model is complete.
  plant->Finalize();

  // Create a context for this system:
  auto plant_context = plant->CreateDefaultContext();

  auto actuation_port_index =
      plant->get_actuation_input_port().get_index();

  trajectories::PiecewisePolynomial<double> xtraj;
  trajectories::PiecewisePolynomial<double> utraj;

  // Run trajectory optimization
  std::tie(xtraj, utraj) = TrajOptFinger(plant.get(), *plant_context,
      actuation_port_index);

  // construct diagram for visualization
  systems::DiagramBuilder<double> builder;

  auto scene_graph = builder.AddSystem<geometry::SceneGraph>();
  auto sim_plant = builder.AddSystem<MultibodyPlant>(FLAGS_time_step);

  Parser(sim_plant, scene_graph).AddModelFromFile(finger_urdf_path, "finger");
  Parser(sim_plant).AddModelFromFile(object_urdf_path, "object");

  scene_graph->set_name("scene_graph");
  sim_plant->set_name("sim_plant");

  WeldFingerFrame<double>(sim_plant);

  sim_plant->Finalize();

  geometry::ConnectDrakeVisualizer(&builder, *scene_graph);

  auto sim_plant_source_id = sim_plant->get_source_id().value();
  builder.Connect(sim_plant->get_geometry_poses_output_port(),
      scene_graph->get_source_pose_port(sim_plant_source_id));

  auto diagram = builder.Build();

  systems::Simulator<double> simulator(*diagram);
  simulator.Initialize();

  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  auto* sim_plant_mutable_context =
      &(diagram->GetMutableSubsystemContext(*sim_plant, diagram_context.get()));

  // Visualize the trajectory
  double t = 0.;
  while (t < xtraj.end_time()) {
    auto q = xtraj.value(t);

    std::cout << q.block(0,0,sim_plant->num_positions(),1) << std::endl;

    sim_plant->SetPositions(sim_plant_mutable_context,
        q.block(0,0,sim_plant->num_positions(),1));
    diagram->Publish(*diagram_context);

    t += FLAGS_time_step;
    // std::this_thread::sleep_for(std::chrono::milliseconds(
    //     static_cast<int>(FLAGS_time_step) * 1000));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  return 0;
}

}  // namespace
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "A simple planar gripper example.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  drake::logging::HandleSpdlogGflags();
  return drake::examples::planar_gripper::do_main();
}
