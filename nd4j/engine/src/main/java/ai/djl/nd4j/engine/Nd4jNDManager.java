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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.PairList;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jNDManager implements NDManager, AutoCloseable {

    static final Nd4jNDManager SYSTEM_MANAGER = new SystemManager();
    private static int nameAssignment = 1;

    private NDManager parent;
    private String uid;
    private Device device;
    SameDiff graph;
    SameDiff session;
    private Map<String, AutoCloseable> resources;

    private Nd4jNDManager(NDManager parent, Device device, SameDiff graph) {
        this.parent = parent;
        this.device = device;
        this.graph = graph;
        resources = new ConcurrentHashMap<>();
        uid = UUID.randomUUID().toString();
    }

    public static Nd4jNDManager newBaseManager() {
        return SYSTEM_MANAGER.newSubManager();
    }

    public static Nd4jNDManager newBaseManager(Device device) {
        return SYSTEM_MANAGER.newSubManager(device);
    }

    SameDiff getGraph() {
        return graph;
    }

    SameDiff getSession() {
        Nd4jNDManager f = this;
        while (f.session == null) {
            f = (Nd4jNDManager) f.getParentManager();
        }
        return f.session;
    }

    static int nextNameAssignment() {
        return nameAssignment++;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(float[] data, Shape shape) {
        return new Nd4jNDArray(this, session.var(Nd4j.create(data, shape.getShape())));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(int data) {
        return new Nd4jNDArray(this, session.constant(data));
    }

    public Nd4jNDArray create(ByteBuffer data, Shape shape) {
        return new Nd4jNDArray(this, shape, data);
    }

    public NDArray create(INDArray value) {
        return new Nd4jNDArray(this, session.var(value));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createCSR(
            Buffer data, long[] indptr, long[] indices, Shape shape, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createRowSparse(
            Buffer data, Shape dataShape, long[] indices, Shape shape, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void invoke(
            String operation, NDArray[] src, NDArray[] dest, PairList<String, ?> params) {}

    /** {@inheritDoc} */
    @Override
    public NDList invoke(String operation, NDList src, PairList<String, ?> params) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zeros(Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ones(Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray arange(
            Number start, Number stop, Number step, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eye(int rows, int cols, int k, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray linspace(Number start, Number stop, int num, boolean endpoint, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomUniform(
            Number low, Number high, Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomNormal(
            Number loc, Number scale, Shape shape, DataType dataType, Device device) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues, Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray randomMultinomial(int n, NDArray pValues) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getParentManager() {
        return parent;
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return device;
    }

    /** {@inheritDoc} */
    @Override
    public Nd4jNDManager newSubManager() {
        return newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public Nd4jNDManager newSubManager(Device device) {
        Nd4jNDManager manager = new Nd4jNDManager(this, device, graph);
        resources.put(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public void attach(String resourceId, AutoCloseable resource) {
        resources.put(resourceId, resource);
    }

    /** {@inheritDoc} */
    @Override
    public void detach(String resourceId) {
        resources.remove(resourceId);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (AutoCloseable resource : resources.values()) {
            try {
                resource.close();
            } catch (Exception ignore) {
                // ignore
            }
        }
        resources = null;
        parent.detach(uid);
    }

    private static final class SystemManager extends Nd4jNDManager {

        SystemManager() {
            super(null, Device.defaultDevice(), SameDiff.create());
            session = getGraph();
        }

        /** {@inheritDoc} */
        @Override
        public void attach(String resrouceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detach(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
