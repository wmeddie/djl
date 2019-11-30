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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import java.lang.management.MemoryUsage;

/**
 * The {@code Nd4jEngine} is an implementation of the {@link Engine} based on the <a
 * href="https://www.nd4j.org/">Nd4J Deep Learning Framework</a>.
 *
 * <p>To get an instance of the {@code Nd4jEngine} when it is not the default Engine, call {@link
 * Engine#getEngine(String)} with the Engine name "Nd4J".
 */
public class Nd4jEngine extends Engine {

    Nd4jEngine() {}

    /** {@inheritDoc} */
    @Override
    public Model newModel(Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return "ND4J";
    }

    /** {@inheritDoc} */
    @Override
    public int getGpuCount() {
        return 0;
    }

    /** {@inheritDoc} */
    @Override
    public MemoryUsage getGpuMemory(Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Device defaultDevice() {
        return Device.cpu();
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return "1.0.0-beta5";
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return Nd4jNDManager.newBaseManager();
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return Nd4jNDManager.newBaseManager(device);
    }
}
