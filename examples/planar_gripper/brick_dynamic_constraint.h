#pragma once

#include <map>

#include "drake/examples/planar_gripper/gripper_brick.h"
#include "drake/solvers/constraint.h"

namespace drake {
namespace examples {
namespace planar_gripper {
/**
 * Compute the total wrench on the brick.
 *
 *     mg + f_friction + ∑ R_WB * f_FiB_B
 *     τ_friction + ∑ p_BCbi * f_FiB_B
 * where f_friction is the friction force between the brick and the lid, and
 * τ_friction is the friction torque between the brick and the lid.
 * This evaluator is going to be used in enforcing the dynamic constraint on the
 * brick.
 */
class BrickTotalWrenchEvaluator : public solvers::EvaluatorBase {
 public:
  BrickTotalWrenchEvaluator(const GripperBrickHelper<double>* gripper_brick,
                            systems::Context<double>* plant_context,
                            std::map<Finger, BrickFace> finger_faces,
                            double brick_lid_friction_force_magnitude,
                            double brick_lid_friction_torque_magnitude);

  const std::map<Finger, BrickFace>& finger_faces() const {
    return finger_faces_;
  }

  template <typename T>
  void ComposeX(const Eigen::Ref<const VectorX<T>>& q,
                const T& v_brick_translation_y, const T& v_brick_translation_z,
                const T& v_brick_rotation_x, const Matrix2X<T>& f_FB_B,
                VectorX<T>* x) const {
    x->resize(num_vars());
    x->head(gripper_brick_->plant().num_positions()) = q;
    (*x)(q.rows()) = v_brick_translation_y;
    (*x)(q.rows() + 1) = v_brick_translation_z;
    (*x)(q.rows() + 2) = v_brick_rotation_x;
    for (int i = 0; i < f_FB_B.cols(); ++i) {
      x->template segment<2>(q.rows() + 3 + 2 * i) = f_FB_B.col(i);
    }
  }

 private:
  template <typename T>
  void DecomposeX(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* q,
                  T* v_brick_translation_y, T* v_brick_translation_z,
                  T* v_brick_rotation_x, Matrix2X<T>* f_FB_B) const {
    *q = x.head(gripper_brick_->plant().num_positions());
    *v_brick_translation_y = x(gripper_brick_->plant().num_positions());
    *v_brick_translation_z = x(gripper_brick_->plant().num_positions() + 1);
    *v_brick_rotation_x = x(gripper_brick_->plant().num_positions() + 2);
    f_FB_B->resize(2, static_cast<int>(finger_faces_.size()));
    for (int i = 0; i < f_FB_B->cols(); ++i) {
      f_FB_B->col(i) = x.template segment<2>(
          gripper_brick_->plant().num_positions() + 3 + 2 * i);
    }
  }

  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<T>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;
  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::runtime_error(
        "BrickTotalWrenchEvaluator::DoEval doesn't support symbolic "
        "variables.");
  }

  const GripperBrickHelper<double>* const gripper_brick_;
  systems::Context<double>* plant_context_;
  std::map<Finger, BrickFace> finger_faces_;
  double brick_lid_friction_force_magnitude_;
  double brick_lid_friction_torque_magnitude_;
};

/**
 * Enforce the backward Euler integration constraint
 * m(v̇[n+1] - v̇[n]) = mg + f_friction[n+1] + ∑ R_WB[n+1] * f_FiB_B[n+1]
 * I(θ_dot[n+1] - θ_dot[n]) = τ_friction[n+1] + ∑ p_BCbi[n+1] * f_FiB_B[n+1]
 * where f_friction is the friction force between the brick and the lid, and
 * τ_friction is the friction torque between the brick and the lid.
 * The decision variables are q, v_brick, f_FB_B and dt.
 */
class BrickDynamicBackwardEulerConstraint : public solvers::Constraint {
 public:
  BrickDynamicBackwardEulerConstraint(
      const GripperBrickHelper<double>* gripper_brick,
      systems::Context<double>* plant_context,
      std::map<Finger, BrickFace> finger_faces,
      double brick_lid_friction_force_magnitude,
      double brick_lid_friction_torque_magnitude);

  ~BrickDynamicBackwardEulerConstraint() override {}

  template <typename T>
  void ComposeX(const Eigen::Ref<const VectorX<T>>& q_r,
                const T& v_brick_r_translation_y,
                const T& v_brick_r_translation_z, const T& v_brick_r_rotation_x,
                const T& v_brick_l_translation_y,
                const T& v_brick_l_translation_z, const T& v_brick_l_rotation_x,
                const Matrix2X<T>& f_FB_B, const T& dt, VectorX<T>* x) const {
    x->resize(num_vars());
    x->template head(gripper_brick_->plant().num_positions()) = q_r;
    (*x)(q_r.rows()) = v_brick_r_translation_y;
    (*x)(q_r.rows() + 1) = v_brick_r_translation_z;
    (*x)(q_r.rows() + 2) = v_brick_r_rotation_x;
    (*x)(q_r.rows() + 3) = v_brick_l_translation_y;
    (*x)(q_r.rows() + 4) = v_brick_l_translation_z;
    (*x)(q_r.rows() + 5) = v_brick_l_rotation_x;
    for (int i = 0; i < f_FB_B.cols(); ++i) {
      x->template segment<2>(q_r.rows() + 6 + 2 * i) = f_FB_B.col(i);
    }
    (*x)(num_vars() - 1) = dt;
  }

 private:
  template <typename T>
  void DecomposeX(const Eigen::Ref<const VectorX<T>>& x, VectorX<T>* q_r,
                  T* v_brick_r_translation_y, T* v_brick_r_translation_z,
                  T* v_brick_r_rotation_x, T* v_brick_l_translation_y,
                  T* v_brick_l_translation_z, T* v_brick_l_rotation_x,
                  Matrix2X<T>* f_FB_B, T* dt) const {
    *q_r = x.head(gripper_brick_->plant().num_positions());
    *v_brick_r_translation_y = x(q_r->rows());
    *v_brick_r_translation_z = x(q_r->rows() + 1);
    *v_brick_r_rotation_x = x(q_r->rows() + 2);
    *v_brick_l_translation_y = x(q_r->rows() + 3);
    *v_brick_l_translation_z = x(q_r->rows() + 4);
    *v_brick_l_rotation_x = x(q_r->rows() + 5);
    f_FB_B->resize(2, wrench_evaluator_.finger_faces().size());
    for (int i = 0; i < f_FB_B->cols(); ++i) {
      f_FB_B->col(i) = x.template segment<2>(q_r->rows() + 6 + 2 * i);
    }
    *dt = x(x.rows() - 1);
  }

  template <typename T>
  void DoEvalGeneric(const Eigen::Ref<const VectorX<T>>& x,
                     VectorX<T>* y) const;

  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override;
  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override;
  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::runtime_error(
        "BrickDynamicConstraint::DoEval doesn't support symbolic variables.");
  }

  const GripperBrickHelper<double>* const gripper_brick_;
  systems::Context<double>* plant_context_;
  BrickTotalWrenchEvaluator wrench_evaluator_;
};

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
