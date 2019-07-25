#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"

#include <set>
#include <utility>

#include "drake/examples/planar_gripper/brick_static_equilibrium_constraint.h"
#include "drake/examples/planar_gripper/gripper_brick_planning_utils.h"
#include "drake/multibody/inverse_kinematics/kinematic_constraint_utilities.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GripperBrickTrajectoryOptimization::GripperBrickTrajectoryOptimization(
    const GripperBrickHelper<double>* const gripper_brick, int nT,
    const std::map<Finger, BrickFace>& initial_contact,
    const std::vector<FingerTransition>& finger_transitions,
    const Options& options)
    : gripper_brick_{gripper_brick},
      nT_{nT},
      prog_{std::make_unique<solvers::MathematicalProgram>()},
      q_vars_{prog_->NewContinuousVariables(
          gripper_brick_->plant().num_positions(), nT_)},
      brick_v_y_vars_{prog_->NewContinuousVariables(nT_)},
      brick_v_z_vars_{prog_->NewContinuousVariables(nT_)},
      brick_omega_x_vars_{prog_->NewContinuousVariables(nT_)},
      f_FB_B_(nT_),
      plant_mutable_contexts_(nT_),
      dt_(prog_->NewContinuousVariables(nT_ - 1)),
      finger_transitions_(finger_transitions),
      finger_face_contacts_(nT_) {
  // Create the contexts used at each knot.
  diagram_contexts_.reserve(nT_);
  for (int i = 0; i < nT_; ++i) {
    diagram_contexts_.push_back(
        gripper_brick->diagram().CreateDefaultContext());
    plant_mutable_contexts_[i] =
        &(gripper_brick_->diagram().GetMutableSubsystemContext(
            gripper_brick_->plant(), diagram_contexts_[i].get()));
  }
  AssignVariableForContactForces(initial_contact, options);
}

void GripperBrickTrajectoryOptimization::AssignVariableForContactForces(
    const std::map<Finger, BrickFace>& initial_contact,
    const Options& options) {
  // I assume initially the system is in static equilibrium.
  std::vector<std::pair<Finger, BrickFace>> initial_finger_face_contacts;
  for (const auto& finger_face : initial_contact) {
    initial_finger_face_contacts.emplace_back(finger_face.first,
                                              finger_face.second);
    AddFingerTipInContactWithBrickFace(
        *gripper_brick_, finger_face.first, finger_face.second, prog_.get(),
        q_vars_.col(0), plant_mutable_contexts_[0], options.face_shrink_factor);
  }
  auto f_FB_B0 = AddBrickStaticEquilibriumConstraint(
      *gripper_brick_, initial_finger_face_contacts, q_vars_.col(0),
      plant_mutable_contexts_[0], prog_.get());
  int initial_contact_count = 0;
  for (const auto& finger_face : initial_contact) {
    f_FB_B_[0].emplace(finger_face.first, f_FB_B0.col(initial_contact_count++));
  }

  // sort the transitions based on their starting time.
  std::vector<const FingerTransition*> sorted_finger_transitions;
  for (const auto& finger_transition : finger_transitions_) {
    if (finger_transition.end_knot_index - finger_transition.start_knot_index <
        2) {
      throw std::invalid_argument(
          "GripperBrickTrajectoryOptimization: finger transition must take at "
          "least two time intervals.");
    }
    sorted_finger_transitions.push_back(&finger_transition);
  }
  std::sort(sorted_finger_transitions.begin(), sorted_finger_transitions.end(),
            [](const FingerTransition* transition1,
               const FingerTransition* transition2) {
              return transition1->start_knot_index <
                     transition2->start_knot_index;
            });
  if (sorted_finger_transitions[0]->start_knot_index <= 0) {
    throw std::invalid_argument(
        "GripperBrickTrajectoryOptimization: the initial transition cannot "
        "start with knot 0.");
  }
  int last_transition_end_knot = 0;
  finger_face_contacts_[0] = initial_contact;
  for (const auto& finger_transition : sorted_finger_transitions) {
    // From the end of last transtion to the start of this transition.
    for (int i = last_transition_end_knot + 1;
         i <= finger_transition->start_knot_index; ++i) {
      finger_face_contacts_[i] =
          finger_face_contacts_[last_transition_end_knot];
    }
    // During this transition.
    finger_face_contacts_[finger_transition->start_knot_index + 1] =
        finger_face_contacts_[finger_transition->start_knot_index];
    auto it =
        finger_face_contacts_[finger_transition->start_knot_index + 1].find(
            finger_transition->finger);
    if (it ==
        finger_face_contacts_[finger_transition->start_knot_index + 1].end()) {
      throw std::invalid_argument(
          "GripperBrickTrajectoryOptimization: two transitions both move " +
          to_string(finger_transition->finger) + " at knot " +
          std::to_string(finger_transition->start_knot_index + 1));
    }
    finger_face_contacts_[finger_transition->start_knot_index + 1].erase(it);
    for (int i = finger_transition->start_knot_index + 2;
         i <= finger_transition->end_knot_index; ++i) {
      finger_face_contacts_[i] =
          finger_face_contacts_[finger_transition->start_knot_index + 1];
    }
    // End of the transition.
    finger_face_contacts_[finger_transition->end_knot_index].emplace(
        finger_transition->finger, finger_transition->to_face);
    last_transition_end_knot = finger_transition->end_knot_index;
  }
  // Set finger_face_contact_ after the final transition.
  for (int i = last_transition_end_knot; i < nT_; ++i) {
    finger_face_contacts_[i] = finger_face_contacts_[last_transition_end_knot];
  }

  // Now assign the decision variables for the contact forces.
  for (int i = 1;  // The contact force variable for the initial state has been
                   // assigned.
       i < nT_; ++i) {
    for (const auto& finger_face : finger_face_contacts_[i]) {
      auto f = prog_->NewContinuousVariables<2>();
      f_FB_B_[i].emplace(finger_face.first, f);
    }
  }
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
