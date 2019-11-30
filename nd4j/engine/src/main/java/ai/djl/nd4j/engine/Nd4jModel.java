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

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.translate.Translator;
import ai.djl.util.PairList;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import org.nd4j.autodiff.samediff.SameDiff;

public class Nd4jModel implements Model {

    private Path modelDir;
    private SameDiff bundle;
    private AtomicBoolean first = new AtomicBoolean(true);

    /** {@inheritDoc} */
    @Override
    public void load(Path modelDir) {
        this.modelDir = modelDir;
        bundle = SameDiff.load(modelDir.toFile(), true);
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String modelName, Map<String, String> options)
            throws IOException, MalformedModelException {
        load(modelPath);
    }

    /** {@inheritDoc} */
    @Override
    public void save(Path modelPath, String modelName) {}

    public SameDiff getSameDiffGraph() {
        return bundle;
    }

    /** {@inheritDoc} */
    @Override
    public Block getBlock() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setBlock(Block block) {}

    /** {@inheritDoc} */
    @Override
    public String getName() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String getProperty(String key) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setProperty(String key, String value) {}

    /** {@inheritDoc} */
    @Override
    public Trainer newTrainer(TrainingConfig trainingConfig) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator) {
        return new Nd4jPredictor<>(this, translator, first.getAndSet(false));
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeInput() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public PairList<String, Shape> describeOutput() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public String[] getArtifactNames() {
        return new String[0];
    }

    /** {@inheritDoc} */
    @Override
    public <T> T getArtifact(String name, Function<InputStream, T> function) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public URL getArtifact(String artifactName) throws IOException {
        if (artifactName == null) {
            throw new IllegalArgumentException("artifactName cannot be null");
        }
        Path file = modelDir.resolve(artifactName);
        if (Files.exists(file) && Files.isReadable(file)) {
            return file.toUri().toURL();
        }
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public InputStream getArtifactAsStream(String name) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDManager getNDManager() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public void setDataType(DataType dataType) {}

    /** {@inheritDoc} */
    @Override
    public DataType getDataType() {
        return DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void cast(DataType dataType) {
        throw new UnsupportedOperationException("Not implemented yet.");
    }

    /** {@inheritDoc} */
    @Override
    public void close() {}
}
