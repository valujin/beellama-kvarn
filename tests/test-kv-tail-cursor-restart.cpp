// Волна 43. Охрана починки вращающегося курсора записи f16-хвоста.
//
// ДЕФЕКТ. llama_kv_tail_store::acquire выбирает слот арены по вращающемуся
// курсору, своему на каждую последовательность:
//     slot = seq_id*arena_stride + (write_cursors[seq_id] + offset) % arena_stride
// Курсор обнулялся только в clear() и при восстановлении состояния. При начале
// НОВОГО запроса в том же слоте он оставался там, куда его довёл предыдущий,
// и новый запрос раскладывал свой хвост со сдвигом, равным длине предыдущего.
// Содержимое строк то же, места другие. Вращение по арене (KVARN_WHT) и
// внимание, складывающее строки в порядке арены, превращали сдвиг в другое
// округление и в конце концов в другой ответ.
//
// ПРОВЕРКА. Одна и та же последовательность коммитится дважды с перезапуском
// между прогонами; множество И ПОРЯДОК слотов обязаны совпасть. Развёртка по
// длине N — ровно та, что поймала дефект на стенде (кратные и некратные длине
// арены, граничные значения).
//
// ОТРИЦАТЕЛЬНЫЙ КОНТРОЛЬ. Без reset_write_cursor второй прогон даёт сдвиг на N:
// это проверяется явно, чтобы тест не выродился в тавтологию.

#include "llama-kv-cache-tail.h"

#include <cstdio>
#include <vector>

static std::vector<int32_t> commit_run(llama_kv_tail_store & store, uint32_t n) {
    std::vector<int32_t> slots;
    slots.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        const llama_kv_tail_identity id { /*.stream =*/ 0, /*.cell =*/ i, /*.generation =*/ 1 };
        slots.push_back(store.commit(0, id, (llama_pos) i, (uint64_t) i, UINT32_MAX));
    }
    return slots;
}

int main() {
    const uint32_t cases[] = { 1, 63, 64, 127, 128, 129, 130 };
    int failures = 0;

    for (uint32_t n : cases) {
        llama_kv_tail_store store(128, 1, 1, 129, 0);
        const std::vector<int32_t> first = commit_run(store, n);
        store.seq_rm(0, -1, -1);
        store.reset_write_cursor(0);
        const std::vector<int32_t> again = commit_run(store, n);
        if (again != first) {
            fprintf(stderr,
                    "N=%u: перезапуск дал ДРУГУЮ раскладку хвоста "
                    "(первый слот %d против %d) — курсор не сброшен\n",
                    n, first.empty() ? -1 : first[0], again.empty() ? -1 : again[0]);
            ++failures;
        }
    }

    // Отрицательный контроль: без сброса раскладка обязана сдвинуться.
    // Берём N, не кратное длине арены, иначе сдвиг был бы нулевым сам по себе.
    {
        const uint32_t n = 64;
        llama_kv_tail_store store(128, 1, 1, 129, 0);
        const std::vector<int32_t> first = commit_run(store, n);
        store.seq_rm(0, -1, -1);
        const std::vector<int32_t> again = commit_run(store, n);
        if (again == first) {
            fprintf(stderr,
                    "отрицательный контроль не сработал: без сброса курсора "
                    "раскладка совпала, значит тест ничего не сторожит\n");
            ++failures;
        }
    }

    if (failures != 0) {
        fprintf(stderr, "test-kv-tail-cursor-restart: отказов %d\n", failures);
        return 1;
    }
    printf("test-kv-tail-cursor-restart: перезапуск последовательности не зависит "
           "от длины предыдущего запроса\n");
    return 0;
}
