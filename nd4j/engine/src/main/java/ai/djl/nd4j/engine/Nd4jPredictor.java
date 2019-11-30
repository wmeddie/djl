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

package ai.djl.nd4j.engine;

import ai.djl.inference.BasePredictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;

/** A predictor for the ND4J backend. */
public class Nd4jPredictor<I, O> extends BasePredictor<I, O> {

    /**
     * Makes an Nd4jPredictor.
     *
     * @param copy If it should copy.
     * @param model The model to wrap.
     * @param translator the unused translator.
     */
    public Nd4jPredictor(Nd4jModel model, Translator<I, O> translator, boolean copy) {
        super(model, translator, copy);
    }

    /** {@inheritDoc} */
    @Override
    protected NDList forward(NDList ndList) {
        SameDiff session = ((Nd4jNDManager) model).getSession();
        Nd4jNDManager nd4jNDManager = (Nd4jNDManager) manager;

        Map<String, INDArray> inputs =
                ndList.stream()
                        .map(a -> (Nd4jNDArray) a)
                        .collect(Collectors.toMap(Nd4jNDArray::getName, Nd4jNDArray::getArray));
        // List<Tensor<?>> result = runner.run();
        Map<String, INDArray> outputs = session.outputAll(inputs);

        NDList resultNDList = new NDList();
        Set<Map.Entry<String, INDArray>> entries = outputs.entrySet();
        for (Map.Entry<String, INDArray> entry : entries) {
            NDArray array = nd4jNDManager.create(entry.getValue());
            array.setName(entry.getKey());
            resultNDList.add(array);
        }

        return resultNDList;
    }
}
