//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <math.h>
#ifdef _WIN32
#include <crtdefs.h>
#endif 
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/CPUMatrix.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

BOOST_AUTO_TEST_SUITE(MatrixLearnerTests)

BOOST_FIXTURE_TEST_CASE(FSAdagradTest, RandomSeedFixture)
{
    // compare dense and sparse result
    const size_t dim1 = 256;
    const size_t dim2 = 128;
    const size_t dim3 = 2048;

    // smoothed gradient
    SingleMatrix matSG = SingleMatrix::RandomGaussian(dim1, dim2, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());
    SingleMatrix matSGsparse(matSG.DeepClone());

    // model
    SingleMatrix matM  = SingleMatrix::RandomGaussian(dim1, dim2, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());
    SingleMatrix matMsparse(matM.DeepClone());

    // generates gradient
    SingleMatrix matG1(c_deviceIdZero);
    matG1.AssignTruncateBottomOf(Matrix<float>::RandomUniform(dim2, dim3, c_deviceIdZero, -300.0f, 0.1f, IncrementCounter()), 0);

    SingleMatrix matG1sparseCSC(matG1.DeepClone());
    matG1sparseCSC.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseCSC, true);

    SingleMatrix matG2 = SingleMatrix::RandomGaussian(dim1, dim3, c_deviceIdZero, -1.0f, 1.0f, IncrementCounter());

    SingleMatrix matG(c_deviceIdZero);
    SingleMatrix::MultiplyAndAdd(matG2, false, matG1, true, matG);

    SingleMatrix matGsparseBSC(c_deviceIdZero);
    matGsparseBSC.SwitchToMatrixType(MatrixType::SPARSE, matrixFormatSparseBlockCol, false);
    SingleMatrix::MultiplyAndAdd(matG2, false, matG1sparseCSC, true, matGsparseBSC);

    // copy matGsparse to matGdense and compare with matG
    Matrix<float> matGdense = Matrix<float>::Zeros(dim1, dim2, c_deviceIdZero);
    Matrix<float>::ScaleAndAdd(1, matGsparseBSC, matGdense);

    BOOST_CHECK(matG.IsEqualTo(matGdense, c_epsilonFloatE5));

    // run learner
    double smoothedCount = 1000;
    matSG.FSAdagradUpdate(dim2, matG, matM, smoothedCount, 0.0001, 1.0, 0.9, 0.9);

    smoothedCount = 1000;
    matSGsparse.FSAdagradUpdate(dim2, matGsparseBSC, matMsparse, smoothedCount, 0.0001, 1.0, 0.9, 0.9);

    BOOST_CHECK(matSG.IsEqualTo(matSGsparse, c_epsilonFloatE5));
    BOOST_CHECK(matM.IsEqualTo(matMsparse, c_epsilonFloatE5));
}

BOOST_AUTO_TEST_SUITE_END()
}}}}
