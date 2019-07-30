
#include <iostream>
#include <map>
#include <vector>

#include "drake/examples/planar_gripper/dev/contact_mode.h"
#include "drake/examples/planar_gripper/dev/contact_search.h"
#include "drake/examples/planar_gripper/gripper_brick.h"

namespace drake {
namespace examples {
namespace planar_gripper {

std::vector<int> HashContactMode(const ContactMode& cm) {
  auto cps = cm.get_contact_pairs();
  std::vector<int> cm_hash(cps.size());
  for (const auto& cp : cps) {
    cm_hash[cp.first] = cp.second;
  }
  return cm_hash;
}

int DoMain() {
  // 3 fingers and 4 faces, finger can move to adjancent faces
  const int kNumPoints = 3;
  const int kNumFaces = 4;
  std::map<BrickFace, std::set<BrickFace>> gripper_brick_reachability = {
      {BrickFace::kPosZ, {BrickFace::kNegY, BrickFace::kPosY}},
      {BrickFace::kNegZ, {BrickFace::kNegY, BrickFace::kPosY}},
      {BrickFace::kPosY, {BrickFace::kNegZ, BrickFace::kPosZ}},
      {BrickFace::kNegY, {BrickFace::kNegZ, BrickFace::kPosZ}},
  };

  std::vector<int> contact_points(kNumPoints);
  std::vector<int> contact_faces(kNumFaces);
  std::iota(contact_points.begin(), contact_points.end(), 0);
  std::iota(contact_faces.begin(), contact_faces.end(), 0);
  auto cms =
      ContactMode::GenerateAllContactModes(contact_points, contact_faces);

  // group them by finger/face pairs
  std::map<std::vector<int>, const ContactMode*> cms_map;
  for (auto& cm : cms) {
    auto cm_hash = HashContactMode(cm);
    cms_map[cm_hash] = &cm;
  }

  // add each neighbor to the right node
  // rule 1: can only move one finger at a time
  // rule 2: can only move that finger to a face that is reachable from the
  // current one
  for (auto& cm : cms) {
    auto cm_hash = HashContactMode(cm);
    for (int i = 0; i < kNumPoints; i++) {
      auto reachable_faces =
          gripper_brick_reachability[static_cast<BrickFace>(cm_hash[i])];
      for (const auto& face : reachable_faces) {
        auto cm_hash_new = cm_hash;
        cm_hash_new[i] = static_cast<int>(face);
        cm.AddConnectedMode(cms_map[cm_hash_new]);
      }
    }
  }

  // find a contact sequence plan
  std::set<std::pair<int, int>> start_pairs = {
      {static_cast<int>(Finger::kFinger1), static_cast<int>(BrickFace::kNegY)},
      {static_cast<int>(Finger::kFinger2), static_cast<int>(BrickFace::kNegZ)},
      {static_cast<int>(Finger::kFinger3), static_cast<int>(BrickFace::kPosY)},
  };
  ContactMode start(start_pairs);
  auto start_hash = HashContactMode(start);
  start = *cms_map[start_hash];

  std::set<std::pair<int, int>> goal_pairs = {
      {static_cast<int>(Finger::kFinger1), static_cast<int>(BrickFace::kPosZ)},
      {static_cast<int>(Finger::kFinger2), static_cast<int>(BrickFace::kNegY)},
      {static_cast<int>(Finger::kFinger3), static_cast<int>(BrickFace::kNegZ)},
  };
  ContactMode goal(goal_pairs);
  auto goal_hash = HashContactMode(goal);
  goal = *cms_map[goal_hash];

  std::cout << start << std::endl;
  std::cout << goal << std::endl;
  std::cout << "---\n";

  auto path = contact_search::BreadthFirstSearch(start, goal);
  // auto path = contact_search::DepthFirstSearch(start, goal);

  for (const auto& path_cm : path) {
    std::cout << *path_cm << std::endl;
  }

  return 0;
}

}  // namespace planar_gripper
}  // namespace examples
}  // namespace drake

int main() { drake::examples::planar_gripper::DoMain(); }
