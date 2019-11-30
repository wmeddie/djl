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
import ai.djl.ndarray.Matrix;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.SparseFormat;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.UUID;
import java.util.function.Predicate;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jNDArray implements NDArray {

    private String uid = UUID.randomUUID().toString();
    private SDVariable tensor;
    private Shape shape;
    private Nd4jNDManager manager;

    Nd4jNDArray(NDManager manager, SDVariable tensor) {
        this.manager = (Nd4jNDManager) manager;
        this.manager.attach(getUid(), this);
        this.tensor = tensor;
    }

    public Nd4jNDArray(NDManager manager, Shape shape, FloatBuffer data) {
        this.manager = (Nd4jNDManager) manager;
        this.manager.attach(getUid(), this);
        tensor = ((Nd4jNDManager) manager).graph.var(Nd4j.create(data.array(), shape.getShape()));
        this.shape = shape;
    }

    Nd4jNDArray(NDManager manager, Shape shape, ByteBuffer data) {
        this.manager = (Nd4jNDManager) manager;
        tensor =
                ((Nd4jNDManager) manager)
                        .graph.var(
                                Nd4j.create(
                                        data.array(),
                                        shape.getShape(),
                                        org.nd4j.linalg.api.buffer.DataType.UBYTE));
        this.shape = shape;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getManager() {
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setName(String name) {}

    /** {@inheritDoc} */
    @Override
    public final String getUid() {
        return uid;
    }

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return Nd4jDataType.fromNd4j(getNd4jDataType());
    }

    /** {@inheritDoc} */
    @Override
    public Device getDevice() {
        return manager.getDevice();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getShape() {
        if (shape == null) {
            shape = new Shape(tensor.getShape());
        }
        return shape;
    }

    public org.nd4j.linalg.api.buffer.DataType getNd4jDataType() {
        return tensor.dataType();
    }

    /** {@inheritDoc} */
    @Override
    public SparseFormat getSparseFormat() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isSparse() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asInDevice(Device device, boolean copy) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asType(DataType dtype, boolean copy) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Matrix asMatrix() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void attachGradient() {}

    /** {@inheritDoc} */
    @Override
    public NDArray getGradient() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer toByteBuffer() {
        return ByteBuffer.wrap(tensor.eval().data().asBytes());
    }

    /** {@inheritDoc} */
    @Override
    public void set(Buffer data) {
        throw new UnsupportedOperationException("Tensor cannot be modified after creation");
    }

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, NDArray value) {}

    /** {@inheritDoc} */
    @Override
    public void set(NDIndex index, Number value) {}

    /** {@inheritDoc} */
    @Override
    public void setScalar(NDIndex index, Number value) {}

    /** {@inheritDoc} */
    @Override
    public NDArray get(NDIndex index) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void copyTo(NDArray array) {}

    /** {@inheritDoc} */
    @Override
    public NDArray booleanMask(NDArray index, int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray zerosLike() {
        return new Nd4jNDArray(manager, manager.graph.zerosLike(tensor));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray onesLike() {
        return new Nd4jNDArray(manager, manager.graph.onesLike(tensor));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray like() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(Number number) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public boolean contentEquals(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray eq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neq(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray gte(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lt(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(Number other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray lte(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray add(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sub(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mul(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray div(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mod(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray pow(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray maximum(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray minimum(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray addi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray subi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray muli(NDArray others) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray divi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toSparse(SparseFormat fmt) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray modi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(Number n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray powi(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray neg() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray negi() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray abs() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray square() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cbrt() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray floor() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray ceil() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray round() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trunc() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray exp() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log10() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray log2() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acos() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atan() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray asinh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray acosh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray atanh() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDegrees() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toRadians() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray max(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray min(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sum(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray prod(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray mean(int[] axes, boolean keepDims) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray trace(int offset, int axis1, int axis2) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int[] indices, int axis) {
        /*Nd4jNDArray axisOp = (Nd4jNDArray) manager.create(axis);
        Operation op =
                manager.getGraph()
                        .opBuilder("Split", "Split_" + Nd4jNDManager.nextNameAssignment())
                        .setAttr("T", getNd4jDataType())
                        .setAttr("num_split", size(axis))
                        .addInput(axisOp.getOutput())
                        .addInput(getOutput())
                        .build();

        NDArray[] result =
                IntStream.range(0, op.numOutputs())
                        .mapToObj((int i) -> new Nd4jNDArray(manager, op.output(i)))
                        .toArray(NDArray[]::new);
        return new NDList(result);*/
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList split(int sections, int axis) {
        /*if (axis < 0 || axis > getShape().dimension()) {
            throw new IllegalArgumentException("Invalid axis value");
        }
        if (sections < 0 || sections > size(axis)) {
            throw new IllegalArgumentException("Invalid numOutputs");
        }
        Nd4jNDArray axisOp = (Nd4jNDArray) manager.create(axis);
        Operation op =
                manager.getGraph()
                        .opBuilder("Split", "Split_" + Nd4jNDManager.nextNameAssignment())
                        .setAttr("T", getNd4jDataType())
                        .setAttr("num_split", sections)
                        .addInput(axisOp.getOutput())
                        .addInput(getOutput())
                        .build();

        NDArray[] result =
                IntStream.range(0, op.numOutputs())
                        .mapToObj((int i) -> new Nd4jNDArray(manager, op.output(i)))
                        .toArray(NDArray[]::new);
        return new NDList(result);*/
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray flatten() {
        return reshape(Arrays.stream(shape.getShape()).sum());
    }

    /** {@inheritDoc} */
    @Override
    public NDArray reshape(Shape shape) {
        return new Nd4jNDArray(manager, tensor.reshape(shape.getShape()));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray expandDims(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray squeeze(int[] axes) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalAnd(NDArray n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalOr(NDArray n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalXor(NDArray n) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray logicalNot() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argSort(int axis, boolean ascending) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray sort() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray softmax(int[] axes, double temperature) {
        SDVariable softmax = manager.graph.nn().softmax(tensor);

        return new Nd4jNDArray(manager, softmax);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum(int axis) {
        return new Nd4jNDArray(manager, manager.graph.cumsum(tensor, false, false, axis));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray cumSum() {
        return new Nd4jNDArray(manager, manager.graph.cumsum(tensor, false, false, 0));
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isInfinite() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray isNaN() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(NDIndex index) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray createMask(Predicate<Number> predicate) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(int axis, long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(long[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray tile(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(int axis, long repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(long[] repeats) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray repeat(Shape desiredShape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray dot(NDArray other) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray clip(Number min, Number max) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray swapAxes(int axis1, int axis2) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray transpose(int... dimensions) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray broadcast(Shape shape) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean shapeEquals(NDArray other) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMax(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray argMin(int axis) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray percentile(Number percentile, int[] dimension) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray median(int[] axes) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray toDense() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDArray nonzero() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEmpty() {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public NDArrayEx getNDArrayInternal() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        tensor = null;
    }

    SDVariable getTensor() {
        return tensor;
    }

    INDArray getArray() {
        return tensor.getArr();
    }
}
