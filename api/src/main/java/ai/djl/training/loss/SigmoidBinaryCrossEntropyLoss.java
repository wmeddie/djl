/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;

/**
 * {@code SigmoidBinaryCrossEntropyLoss} is a type of {@link Loss}.
 *
 * <p>Sigmoid binary cross-entropy loss is defined by: \(L = -\sum_i {label_i * log(prob_i) *
 * posWeight + (1 - label_i) * log(1 - prob_i)}\) where \(prob = \frac{1}{1 + e^{-pred}}\)
 */
public class SigmoidBinaryCrossEntropyLoss extends Loss {

    private float weight;
    private int batchAxis;
    private boolean fromSigmoid;

    /**
     * Performs Sigmoid cross-entropy loss for binary classification.
     *
     * @param weight the weight to apply on the loss value, default 1
     * @param batchAxis the axis that represents the mini-batch, default 0
     * @param fromSigmoid whether the input is from the output of sigmoid, default false
     */
    public SigmoidBinaryCrossEntropyLoss(float weight, int batchAxis, boolean fromSigmoid) {
        super("SigmoidBinaryCrossEntropyLoss");
        this.weight = weight;
        this.batchAxis = batchAxis;
        this.fromSigmoid = fromSigmoid;
    }

    /** Performs Sigmoid cross-entropy loss for binary classification. */
    public SigmoidBinaryCrossEntropyLoss() {
        this(1, 0, false);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray getLoss(NDList label, NDList prediction) {
        NDArray pred = prediction.singletonOrThrow();
        NDArray lab = label.singletonOrThrow();
        lab = lab.reshape(pred.getShape());
        NDArray loss;
        if (!fromSigmoid) {
            // TODO: Add Position weight option
            loss =
                    Activation.relu(pred)
                            .sub(pred.mul(lab))
                            .add(Activation.softrelu(pred.abs().neg()));
        } else {
            double eps = 1e-12;
            loss =
                    pred.add(eps)
                            .log()
                            .mul(lab)
                            .add(NDArrays.sub(1., pred).add(eps).mul(NDArrays.sub(1., lab)));
        }
        if (weight != 1f) {
            loss = loss.mul(weight);
        }
        return loss.mean(excludeBatchAxis(loss, batchAxis));
    }
}
