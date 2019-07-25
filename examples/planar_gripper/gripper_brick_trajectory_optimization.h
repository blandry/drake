#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/solvers/mathematical_program.h"

namespace drake {
namespace examples {
namespace planar_gripper {
struct FingerTransition {
  FingerTransition(int m_start_knot_index, int m_end_knot_index,
                   Finger m_finger, BrickFace m_to_face)
      : start_knot_index(m_start_knot_index),
        end_knot_index(m_end_knot_index),
        finger(m_finger),
        to_face(m_to_face) {}
  int start_knot_index;
  int end_knot_index;
  Finger finger;
  BrickFace to_face;
};

/**
 * Given a contact mode sequence, find the finger joint trajectory / brick
 * trajectory and contact forces, such that the object is reoriented.
 */
class GripperBrickTrajectoryOptimization {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(GripperBrickTrajectoryOptimization);

  enum class IntegrationMethod {
    kBackwardEuler,
    kMidpoint,
  };

  struct Options {
    Options(double m_face_shrink_factor, double m_minimum_clearing_distance,
            IntegrationMethod m_integration_method)
        : face_shrink_factor(m_face_shrink_factor),
          minimum_clearing_distance(m_minimum_clearing_distance),
          integration_method{m_integration_method} {}
    double face_shrink_factor = 0.8;
    double minimum_clearing_distance = 0.01;
    IntegrationMethod integration_method = IntegrationMethod::kBackwardEuler;
  };

  /**
   * @param gripper_brick The system for which to plan the trajectory.
   * @param nT The number of knot points.
   * @param initial_contact The assignment of finger to brick face at the
   * initial state.
   * @param finger_transitions All the finger transitions in the trajectory.
   * @note we allow at most one finger in transition at each knot, and each
   * transition must takes at least two time intervals (namely end_knot_index -
   * start_knot_index >= 2).
   */
  GripperBrickTrajectoryOptimization(
      const GripperBrickHelper<double>* gripper_brick, int nT,
      const std::map<Finger, BrickFace>& initial_contact,
      const std::vector<FingerTransition>& finger_transitions,
      const Options& options);

  const std::vector<std::map<Finger, BrickFace>>& finger_face_contacts() const {
    return finger_face_contacts_;
  }

 private:
  void AssignVariableForContactForces(
      const std::map<Finger, BrickFace>& initial_contact,
      const Options& options);

  void AddDynamicConstraint();

  const GripperBrickHelper<double>* const gripper_brick_;
  // number of knots.
  int nT_;
  std::unique_ptr<solvers::MathematicalProgram> prog_;
  // q_vars_.col(i) represents all the q variables at the i'th knot.
  MatrixX<symbolic::Variable> q_vars_;
  // brick_v_y_vars_(i) represents the brick y translational velocity variable
  // at the i'th knot.
  VectorX<symbolic::Variable> brick_v_y_vars_;
  // brick_v_z_vars_(i) represents the brick z translational velocity variable
  // at the i'th knot.
  VectorX<symbolic::Variable> brick_v_z_vars_;
  // brick_omega_x_vars_(i) represents the brick z translational velocity
  // variable at the i'th knot.
  VectorX<symbolic::Variable> brick_omega_x_vars_;
  // f_FB_B_[knot][finger] represents the contact force from the finger (F) to
  // the brick (B) expressed in the brick (B) frame at a given knot for a given
  // finger.
  std::vector<std::unordered_map<Finger, Vector2<symbolic::Variable>>> f_FB_B_;
  // diagram_contexts_[i] is the diagram context for the i'th knot.
  std::vector<std::unique_ptr<systems::Context<double>>> diagram_contexts_;
  std::vector<systems::Context<double>*> plant_mutable_contexts_;
  VectorX<symbolic::Variable> dt_;
  std::vector<FingerTransition> finger_transitions_;
  std::vector<std::map<Finger, BrickFace>> finger_face_contacts_;
};
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
