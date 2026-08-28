#!/usr/bin/env bash
# Конфигурация сборки для CI. Общая для всех шардов и для финальной задачи:
# ccache ключует по всей командной строке компилятора, поэтому расхождение хотя бы
# в одном флаге обнулило бы попадания и шардирование потеряло бы смысл.
set -euo pipefail

: "${CUDA_ARCHS:=75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real}"
: "${GCC_CPU:=16}"
: "${GCC_CUDA:=15}"

# Внутри контейнера каталог принадлежит другому пользователю, и git отказывается
# с ним работать ("dubious ownership"). Без этого cmake не определяет коммит, и
# образ получает version "build 0, commit unknown-dirty" вместо настоящего.
git config --global --add safe.directory "$PWD" || true

# Политика маршрутов — чистая функция без зависимостей от CUDA. Проверяем до
# многочасовой сборки ядер, а не после.
g++ -std=c++17 -O0 -o /tmp/rp-test tests/test-cuda-fattn-route-policy.cpp
/tmp/rp-test "$PWD"
echo "тест маршрутов пройден"

cmake -B build -G Ninja \
    -DCMAKE_C_COMPILER="gcc-${GCC_CPU}" \
    -DCMAKE_CXX_COMPILER="g++-${GCC_CPU}" \
    -DCMAKE_CUDA_HOST_COMPILER="g++-${GCC_CUDA}" \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined -L/usr/local/cuda/lib64/stubs" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_BACKEND_DL=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON
