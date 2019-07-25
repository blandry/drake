#include "drake/examples/planar_gripper/gripper_brick_trajectory_optimization.h"

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace examples {
namespace planar_gripper {
GTEST_TEST(GripperBrickTrajectoryOptimizationTest, TestConstructor) {
  GripperBrickHelper<double> gripper_brick;
  int nT = 9;
  std::map<Finger, BrickFace> initial_contact(
      {{Finger::kFinger1, BrickFace::kNegY},
       {Finger::kFinger2, BrickFace::kNegZ},
       {Finger::kFinger3, BrickFace::kPosY}});
  std::vector<FingerTransition> finger_transitions;
  finger_transitions.emplace_back(5, 7, Finger::kFinger2, BrickFace::kPosY);
  finger_transitions.emplace_back(1, 3, Finger::kFinger1, BrickFace::kNegZ);

  GripperBrickTrajectoryOptimization dut(
      &gripper_brick, nT, initial_contact, finger_transitions,
      GripperBrickTrajectoryOptimization::Options(
          0.8, 0.01,
          GripperBrickTrajectoryOptimization::IntegrationMethod::
              kBackwardEuler));

  EXPECT_EQ(dut.finger_face_contacts()[0], initial_contact);
  EXPECT_EQ(dut.finger_face_contacts()[1], initial_contact);
  std::map<Finger, BrickFace> finger_face_contacts_expected = {
      {Finger::kFinger2, BrickFace::kNegZ},
      {Finger::kFinger3, BrickFace::kPosY}};
  EXPECT_EQ(dut.finger_face_contacts()[2], finger_face_contacts_expected);
  finger_face_contacts_expected.emplace(Finger::kFinger1, BrickFace::kNegZ);
  EXPECT_EQ(dut.finger_face_contacts()[3], finger_face_contacts_expected);
  EXPECT_EQ(dut.finger_face_contacts()[4], finger_face_contacts_expected);
  EXPECT_EQ(dut.finger_face_contacts()[5], finger_face_contacts_expected);
  finger_face_contacts_expected.erase(
      finger_face_contacts_expected.find(Finger::kFinger2));
  EXPECT_EQ(dut.finger_face_contacts()[6], finger_face_contacts_expected);
  finger_face_contacts_expected.emplace(Finger::kFinger2, BrickFace::kPosY);
  EXPECT_EQ(dut.finger_face_contacts()[7], finger_face_contacts_expected);
  EXPECT_EQ(dut.finger_face_contacts()[8], finger_face_contacts_expected);
}
}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake
