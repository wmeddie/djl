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

import ai.djl.ndarray.types.DataType;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public final class Nd4jDataType {

    private static Map<DataType, org.nd4j.linalg.api.buffer.DataType> toNd4j = createMapToNd4j();
    private static Map<org.nd4j.linalg.api.buffer.DataType, DataType> fromNd4j =
            createMapFromNd4j();

    private Nd4jDataType() {}

    private static Map<DataType, org.nd4j.linalg.api.buffer.DataType> createMapToNd4j() {
        Map<DataType, org.nd4j.linalg.api.buffer.DataType> map = new ConcurrentHashMap<>();
        map.put(DataType.FLOAT32, org.nd4j.linalg.api.buffer.DataType.FLOAT);
        map.put(DataType.FLOAT64, org.nd4j.linalg.api.buffer.DataType.DOUBLE);
        map.put(DataType.INT32, org.nd4j.linalg.api.buffer.DataType.INT);
        map.put(DataType.INT64, org.nd4j.linalg.api.buffer.DataType.LONG);
        map.put(DataType.UINT8, org.nd4j.linalg.api.buffer.DataType.UBYTE);
        return map;
    }

    private static Map<org.nd4j.linalg.api.buffer.DataType, DataType> createMapFromNd4j() {
        Map<org.nd4j.linalg.api.buffer.DataType, DataType> map = new ConcurrentHashMap<>();
        map.put(org.nd4j.linalg.api.buffer.DataType.FLOAT, DataType.FLOAT32);
        map.put(org.nd4j.linalg.api.buffer.DataType.DOUBLE, DataType.FLOAT64);
        map.put(org.nd4j.linalg.api.buffer.DataType.INT, DataType.INT32);
        map.put(org.nd4j.linalg.api.buffer.DataType.LONG, DataType.INT64);
        map.put(org.nd4j.linalg.api.buffer.DataType.UBYTE, DataType.UINT8);
        return map;
    }

    public static DataType fromNd4j(org.nd4j.linalg.api.buffer.DataType nd4jType) {
        return fromNd4j.get(nd4jType);
    }

    public static org.nd4j.linalg.api.buffer.DataType toNd4j(DataType jType) {
        return toNd4j.get(jType);
    }
}
