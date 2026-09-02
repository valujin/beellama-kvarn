// Охрана патча волны 41 (kvarn-w41-sealed-restore): признак запечатанности
// группы KVarN переживает восстановление состояния.
//
// ЧТО ЛОМАЛОСЬ. `allocation_group_sealed` — липкий признак «группа была
// заполнена целиком, её префикс лежит в записях, слот F16 ей больше не нужен».
// Он живёт ТОЛЬКО в памяти объекта llama_kv_cache и меняется ТОЛЬКО в
// find_slot. Любое восстановление состояния KVarN подменяет ЖИВОЙ кэш
// метаданных свежим клоном (llama_kv_cache_kvarn::state_read:
// make_metadata_cache() -> clone_logical_state_from() -> metadata.swap()), а
// clone_logical_state_from признак не переносил. У клона вектор пуст, засев в
// find_slot видит несовпадение длины и обнуляет ВСЕ признаки. После этого
// каждая неполная группа-дыра снова считается живой незавершённой и занимает
// слот F16, а слотов при одной последовательности всего два (кольцо равно
// 2*n_seq_max). Две дыры одной чётности -> бросок
// "structured KV live groups alias one F16 stage slot" посреди декодирования.
// Это и есть отказ из issues #1 и #2 этого репозитория.
//
// ЧТО ОХРАНЯЕТСЯ ЗДЕСЬ. Причина, а не одно из её следствий. Следствий два, и
// какое из них наступит, решает случай — на какие слоты кольца попали дыры:
//   * дыры на ОДНОМ слоте -> бросок "structured KV live groups alias one F16
//     stage slot" (issues #1 и #2, `--parallel 1`, кольцо всего из двух слотов);
//   * дыры на РАЗНЫХ слотах -> броска нет, но снимается и запрет на дозапись
//     (llama-kv-cache.cpp:2534-2537), и аллокатор садится ПРЯМО В ДЫРУ, чей
//     префикс лежит в записях, а не в стейдже. Это уже тихая порча.
// Поэтому тест проверяет саму причину: пережил ли признак восстановление.
//
// Порядок: одна последовательность, кэш KVarN, две ЗАПЕЧАТАННЫЕ неполные группы
// (их делает откат, переходящий границу группы назад — то же самое штатно
// делает спекуляция на каждом шаге генерации), затем сохранение и
// восстановление состояния последовательности, затем ещё декоды. После
// восстановления обе дыры обязаны остаться запечатанными.
//
// Отрицательный контроль проведён: с откаченным патчем волны 41 после
// восстановления структурная ветка печатает НОЛЬ запечатанных групп вместо
// двух, а следующие два декода садятся в ячейки 255 и 639 — то есть ровно в
// дыры. С правкой признак сохраняется и дыры остаются обойдёнными.

#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// Синтетические модели из test-generate-models объявляют context_length = 128.
// При окне 128 хвост KVarN совпадает с окном целиком, представление становится
// native_exact, и llama_model строит ОБЫЧНЫЙ llama_kv_cache без групповой
// раскладки — то есть структурная ветка find_slot, которую этот тест и
// охраняет, не исполняется. Поэтому здесь делается копия модели с увеличенным
// context_length; веса и токенизатор те же.
static std::string widen_context_length(const std::string & src, uint32_t n_ctx_train) {
    ggml_context * meta = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &meta };
    gguf_context * gctx = gguf_init_from_file(src.c_str(), gp);
    if (gctx == nullptr) {
        return std::string();
    }
    const int64_t arch_key = gguf_find_key(gctx, "general.architecture");
    if (arch_key < 0) {
        gguf_free(gctx);
        ggml_free(meta);
        return std::string();
    }
    const std::string key = std::string(gguf_get_val_str(gctx, arch_key)) + ".context_length";

    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, gctx);
    gguf_set_val_u32(out, key.c_str(), n_ctx_train);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        ggml_tensor * t = ggml_get_tensor(meta, gguf_get_tensor_name(gctx, i));
        if (t == nullptr) {
            gguf_free(out);
            gguf_free(gctx);
            ggml_free(meta);
            return std::string();
        }
        gguf_add_tensor(out, t);
    }

    const std::string dst =
        (std::filesystem::temp_directory_path() / "beellama-sealed-restore-model.gguf").string();
    const bool ok = gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
    gguf_free(gctx);
    ggml_free(meta);
    return ok ? dst : std::string();
}

// Проверка, что структурная ветка аллокатора действительно исполняется, и учёт
// живых незавершённых групп. Без неё тест вырождается молча: любая смена
// умолчаний (политика хвоста, объединённый кэш, тип кэша) может увести модель
// на обычный llama_kv_cache. Наблюдаемый признак — строка KVARN-SEED, которую
// печатает САМА структурная ветка под LLAMA_KVARN_DEBUG_STAGE.
static bool              g_structural_branch_seen = false;
static int               g_sealed_partial_seen    = 0;
static ggml_log_callback g_prev_log_callback      = nullptr;
static void *            g_prev_log_user_data     = nullptr;

static void stage_log_probe(ggml_log_level level, const char * text, void * user_data) {
    if (text != nullptr && std::strstr(text, "KVARN-SEED") != nullptr) {
        g_structural_branch_seen = true;
        int sealed = 0;
        for (const char * p = text; (p = std::strstr(p, ",зап")) != nullptr; p += 4) {
            ++sealed;
        }
        if (sealed > g_sealed_partial_seen) {
            g_sealed_partial_seen = sealed;
        }
    }
    if (g_prev_log_callback != nullptr) {
        g_prev_log_callback(level, text, g_prev_log_user_data);
    }
    (void) user_data;
}

static bool decode_one(llama_context * ctx, llama_batch & batch, llama_pos pos) {
    common_batch_clear(batch);
    common_batch_add(batch, 1, pos, { 0 }, false);
    return llama_decode(ctx, batch) == 0;
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified    = true;
    // Кольцо слотов F16 равно 2*n_seq_max. Одна последовательность — два слота,
    // ровно как в конфигурациях из обеих issue (`--parallel 1`).
    params.n_parallel    = 1;
    params.n_ctx         = 2048;
    params.n_batch       = 512;
    params.n_ubatch      = 512;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED) {
        fprintf(stderr, "%s : KVarN cache is not selected, the structural allocator would not run\n", __func__);
        return 1;
    }
    if (!params.kv_unified) {
        fprintf(stderr, "%s : unified cache is required, the structural allocator needs n_stream == 1\n", __func__);
        return 1;
    }
    if (params.n_parallel != 1) {
        fprintf(stderr, "%s : this test reproduces the two-slot ring, it needs exactly one sequence\n", __func__);
        return 1;
    }

    ggml_backend_load_all();

    const std::string wide_model = widen_context_length(params.model.path, 8192);
    if (wide_model.empty()) {
        fprintf(stderr, "%s : failed to rewrite the model context length\n", __func__);
        return 1;
    }
    params.model.path = wide_model;

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_batch batch = llama_batch_init(1, 0, 1);

    llama_log_get(&g_prev_log_callback, &g_prev_log_user_data);
    llama_log_set(stage_log_probe, nullptr);
    setenv("LLAMA_KVARN_DEBUG_STAGE", "1", 1);

    // Размер группы KVarN. Держится в одном месте, чтобы тест не расходился с
    // KVAR_N_GROUP молча: расхождение видно по числу запечатанных дыр ниже.
    const llama_pos group = 128;

    // Две дыры делаются на группах 1 и 3 — они одной чётности, слот у обеих
    // 1 + ((g - 1) % 2) == 1. Голова после них встанет на группу 5, слот тоже 1.
    // Это ровно та тройка, которая в замере дала group=73 против group=67.
    const std::vector<llama_pos> hole_groups = { 1, 3 };

    // Группы раскладываются по ЯЧЕЙКАМ, а не по позициям, и каждая оставленная
    // дыра сдвигает одно относительно другого ровно на одну ячейку. Поэтому
    // счёт ведётся по ячейкам: cell == pos + (число уже оставленных дыр).
    llama_pos pos = 0;
    llama_pos cell = 0;
    size_t next_hole = 0;
    const llama_pos cell_end = 6*group;   // дойти головой до группы 5
    while (cell < cell_end) {
        if (!decode_one(ctx, batch, pos)) {
            fprintf(stderr, "%s : llama_decode failed while filling position %d\n", __func__, int(pos));
            llama_batch_free(batch);
            return 1;
        }
        ++pos;
        ++cell;

        // Как только группа заполнилась целиком, откатить один токен назад.
        // Группа остаётся ЗАПЕЧАТАННОЙ и неполной — дописывать в неё нельзя,
        // она становится постоянной дырой. Это ровно то, что делает
        // спекулятивный откат сервера, перешагнувший границу группы назад.
        if (next_hole < hole_groups.size() && cell == (hole_groups[next_hole] + 1)*group) {
            const llama_pos p0 = pos - 1;
            if (!llama_memory_seq_rm(mem, 0, p0, -1)) {
                fprintf(stderr, "%s : KVarN refused the one-token suffix rollback at %d\n", __func__, int(p0));
                llama_batch_free(batch);
                return 1;
            }
            pos = p0;
            ++next_hole;
            // Освободившаяся ячейка остаётся дырой: следующий декод обязан её
            // обойти и уйти в следующую группу. Ячейка при этом не переиспользуется,
            // поэтому счётчик ячеек не откатывается.
            if (!decode_one(ctx, batch, pos)) {
                fprintf(stderr, "%s : llama_decode failed right after the rollback at %d\n", __func__, int(pos));
                llama_batch_free(batch);
                return 1;
            }
            ++pos;
            ++cell;
        }
    }

    if (!g_structural_branch_seen) {
        fprintf(stderr, "%s : the structural KVarN allocator did not run; this test guards nothing\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    if (g_sealed_partial_seen < int(hole_groups.size())) {
        fprintf(stderr,
                "%s : expected at least %zu sealed incomplete groups, the allocator reported %d; "
                "the rollback no longer leaves holes and this test guards nothing\n",
                __func__, hole_groups.size(), g_sealed_partial_seen);
        llama_batch_free(batch);
        return 1;
    }

    // Сохранение и восстановление состояния последовательности. Это тот же
    // путь, которым сервер восстанавливает контрольную точку промпта, и именно
    // он подменяет кэш метаданных клоном.
    constexpr llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_SELF_CONTAINED;
    const size_t state_size = llama_state_seq_get_size_ext(ctx, 0, flags);
    if (state_size == 0) {
        fprintf(stderr, "%s : the context reported an empty sequence state\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    std::vector<uint8_t> state(state_size);
    if (llama_state_seq_get_data_ext(ctx, state.data(), state.size(), 0, flags) != state_size) {
        fprintf(stderr, "%s : failed to save the sequence state\n", __func__);
        llama_batch_free(batch);
        return 1;
    }
    if (llama_state_seq_set_data_ext(ctx, state.data(), state.size(), 0, flags) == 0) {
        fprintf(stderr, "%s : failed to restore the sequence state\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    // ОХРАНЯЕМОЕ МЕСТО. Счётчик сбрасывается, чтобы смотреть ТОЛЬКО на то, что
    // структурная ветка видит ПОСЛЕ восстановления.
    g_sealed_partial_seen = 0;

    // Декоды не должны ни падать (узкое кольцо -> бросок), ни садиться в дыру
    // (широкое кольцо -> тихая дозапись). Оба исхода отсекаются одной
    // проверкой ниже: дыры обязаны остаться запечатанными.
    for (int i = 0; i < 3*int(group) + 1; ++i) {
        if (!decode_one(ctx, batch, pos)) {
            fprintf(stderr,
                    "%s : llama_decode failed %d decodes after the state restore at position %d\n",
                    __func__, i + 1, int(pos));
            fprintf(stderr,
                    "%s : this is the sealed-group regression: the restore dropped "
                    "allocation_group_sealed and the sealed holes claimed F16 stage slots again\n",
                    __func__);
            llama_batch_free(batch);
            return 1;
        }
        ++pos;
    }

    if (g_sealed_partial_seen < int(hole_groups.size())) {
        fprintf(stderr,
                "%s : after the sequence state restore the allocator sees %d sealed incomplete groups "
                "instead of %zu\n", __func__, g_sealed_partial_seen, hole_groups.size());
        fprintf(stderr,
                "%s : this is the sealed-group regression: allocation_group_sealed did not survive the "
                "metadata cache swap, so the holes claim F16 stage slots and accept new cells whose "
                "prefix is no longer in the stage\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    unsetenv("LLAMA_KVARN_DEBUG_STAGE");
    llama_log_set(g_prev_log_callback, g_prev_log_user_data);

    fprintf(stderr,
            "%s : %zu sealed incomplete groups survived a sequence state restore, %d decodes after it: OK\n",
            __func__, hole_groups.size(), 3*int(group) + 1);

    llama_batch_free(batch);
    return 0;
}
