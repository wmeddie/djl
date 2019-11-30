# DJL - ND4J engine implementation

## Overview

This module contains the ND4J implementation of the Deep Java Library (DJL) EngineProvider.

We don't recommend that developers use classes in this module directly. Use of these classes will couple your code with Tensorflow and make switching between frameworks difficult. Even so, developers are not restricted from using engine-specific features. For more information, see [NDManager#invoke()](https://javadoc.djl.ai/api/0.2.0/ai/djl/ndarray/NDManager.html#invoke-java.lang.String-ai.djl.ndarray.NDList-ai.djl.ndarray.NDList-ai.djl.util.PairList-).

**Right now, the ND4J API is here as a proof of concept. While it can help provide a starting point towards a full Tensorflow implementation, it should not be used in it's current state.**


```

egonzalez@DESKTOP-H086A3U:/mnt/e/src/github/djl-linux/nd4j/engine$ ./gradlew format build run -x test

> Configure project :mxnet:mxnet-engine
[WARN ] Header file has been changed in open source project: mxnet/c_api.h.

> Task :nd4j:engine:run
[main] INFO org.nd4j.linalg.factory.Nd4jBackend - Loaded [CpuBackend] backend
[main] INFO org.nd4j.nativeblas.NativeOpsHolder - Number of threads used for OpenMP: 4
[main] INFO org.nd4j.nativeblas.Nd4jBlas - Number of threads used for OpenMP BLAS: 4
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Backend used: [CPU]; OS: [Linux]
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Cores: [8]; Memory: [4.0GB];
[main] INFO org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner - Blas vendor: [MKL]
Input Shape:
(1, 3)
Softmax:
[0.09003057, 0.24472848, 0.66524094]
ZerosLike:
[0.0, 0.0, 0.0]
OnesLike:
[1.0, 1.0, 1.0]

Deprecated Gradle features were used in this build, making it incompatible with Gradle 7.0.
Use '--warning-mode all' to show the individual deprecation warnings.
See https://docs.gradle.org/6.0/userguide/command_line_interface.html#sec:command_line_warnings

BUILD SUCCESSFUL in 29s
14 actionable tasks: 10 executed, 4 up-to-date
```