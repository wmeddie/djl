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
package ai.djl.test;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Collectors;

public final class CoverageUtils {

    private CoverageUtils() {}

    public static void testGetterSetters(Class<?> baseClass)
            throws IOException, ClassNotFoundException {
        List<Class<?>> list = getClasses(baseClass);
        for (Class<?> clazz : list) {
            Object obj = null;
            if (clazz.isEnum()) {
                obj = clazz.getEnumConstants()[0];
            } else {
                Constructor<?>[] constructors = clazz.getConstructors();
                for (Constructor<?> con : constructors) {
                    try {
                        Class<?>[] types = con.getParameterTypes();
                        Object[] args = new Object[types.length];
                        for (int i = 0; i < args.length; ++i) {
                            args[i] = getMockInstance(types[i], true);
                        }
                        con.setAccessible(true);
                        obj = con.newInstance(args);
                    } catch (ReflectiveOperationException ignore) {
                        // ignore
                    }
                }
            }
            if (obj == null) {
                continue;
            }

            Method[] methods = clazz.getDeclaredMethods();
            for (Method method : methods) {
                String methodName = method.getName();
                int parameterCount = method.getParameterCount();
                try {
                    if (parameterCount == 0
                            && (methodName.startsWith("get")
                                    || methodName.startsWith("is")
                                    || "toString".equals(methodName)
                                    || "hashCode".equals(methodName))) {
                        method.invoke(obj);
                    } else if (parameterCount == 1
                            && (methodName.startsWith("set") || "fromValue".equals(methodName))) {
                        Class<?> type = method.getParameterTypes()[0];
                        method.invoke(obj, getMockInstance(type, true));
                    } else if ("equals".equals(methodName)) {
                        method.invoke(obj, obj);
                        method.invoke(obj, (Object) null);
                        Class<?> type = method.getParameterTypes()[0];
                        method.invoke(obj, getMockInstance(type, true));
                    }
                } catch (ReflectiveOperationException ignore) {
                    // ignore
                }
            }
        }
    }

    private static List<Class<?>> getClasses(Class<?> clazz)
            throws IOException, ClassNotFoundException {
        URL url = clazz.getProtectionDomain().getCodeSource().getLocation();
        String path = url.getPath();

        if (!"file".equalsIgnoreCase(url.getProtocol())) {
            return Collections.emptyList();
        }

        List<Class<?>> classList = new ArrayList<>();

        Path classPath = Paths.get(path);
        if (Files.isDirectory(classPath)) {
            Collection<Path> files =
                    Files.walk(Paths.get(path))
                            .filter(p -> Files.isRegularFile(p) && p.toString().endsWith(".class"))
                            .collect(Collectors.toList());
            for (Path file : files) {
                Path p = classPath.relativize(file);
                String className = p.toString();
                className = className.substring(0, className.lastIndexOf("."));
                className = className.replace(File.separatorChar, '.');

                try {
                    classList.add(Class.forName(className));
                } catch (ExceptionInInitializerError | NoClassDefFoundError ignore) {
                    // ignore
                }
            }
        } else if (path.toLowerCase().endsWith(".jar")) {
            try (JarFile jarFile = new JarFile(path)) {
                Enumeration<JarEntry> en = jarFile.entries();
                while (en.hasMoreElements()) {
                    JarEntry entry = en.nextElement();
                    String fileName = entry.getName();
                    if (fileName.endsWith(".class")) {
                        fileName = fileName.substring(0, fileName.lastIndexOf("."));
                        fileName = fileName.replace('/', '.');
                        try {
                            classList.add(Class.forName(fileName));
                        } catch (ExceptionInInitializerError ignore) {
                            // ignore
                        }
                    }
                }
            }
        }

        return classList;
    }

    private static Object getMockInstance(Class<?> clazz, boolean useConstructor) {
        if (clazz.isPrimitive()) {
            if (clazz == Boolean.TYPE) {
                return Boolean.TRUE;
            }
            if (clazz == Character.TYPE) {
                return '0';
            }
            if (clazz == Byte.TYPE) {
                return (byte) 0;
            }
            if (clazz == Short.TYPE) {
                return (short) 0;
            }
            if (clazz == Integer.TYPE) {
                return 0;
            }
            if (clazz == Long.TYPE) {
                return 0L;
            }
            if (clazz == Float.TYPE) {
                return 0f;
            }
            if (clazz == Double.TYPE) {
                return 0d;
            }
        }

        if (clazz.isAssignableFrom(String.class)) {
            return "";
        }

        if (clazz.isAssignableFrom(List.class)) {
            return new ArrayList<>();
        }

        if (clazz.isAssignableFrom(Set.class)) {
            return new HashSet<>();
        }

        if (clazz.isAssignableFrom(Map.class)) {
            return new HashMap<>();
        }

        if (clazz.isEnum()) {
            return clazz.getEnumConstants()[0];
        }

        if (clazz.isInterface()) {
            return newProxyInstance(clazz);
        }

        if (useConstructor) {
            Constructor<?>[] constructors = clazz.getConstructors();
            for (Constructor<?> con : constructors) {
                try {
                    Class<?>[] types = con.getParameterTypes();
                    Object[] args = new Object[types.length];
                    for (int i = 0; i < args.length; ++i) {
                        args[i] = getMockInstance(types[i], false);
                    }
                    con.setAccessible(true);
                    return con.newInstance(args);
                } catch (ReflectiveOperationException ignore) {
                    // ignore
                }
            }
        }

        return null;
    }

    @SuppressWarnings("rawtypes")
    private static Object newProxyInstance(Class<?> clazz) {
        return Proxy.newProxyInstance(
                clazz.getClassLoader(), new Class[] {clazz}, (proxy, method, args) -> null);
    }
}
