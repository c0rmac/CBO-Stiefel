// filename: consensus_points/extrinsic_consensus_point.h

#ifndef EXTRINSIC_CONSENSUS_POINT_H
#define EXTRINSIC_CONSENSUS_POINT_H

#include "base_consensus_point.h"
#include <vector>
#include <Eigen/Core>
#include <string>

class ExtrinsicConsensusPoint : public BaseConsensusPoint {
public:
    ExtrinsicConsensusPoint(int n, int k, int N_particles, double beta_val, BaseObjective* objective_ptr)
        : BaseConsensusPoint(n, k, N_particles, beta_val, objective_ptr) {}

    // Override the calculate method
    MatrixNxK calculate(const std::vector<MatrixNxK>& particles) override;

    std::string get_name() const override { return "Extrinsic"; }
};

#endif // EXTRINSIC_CONSENSUS_POINT_H