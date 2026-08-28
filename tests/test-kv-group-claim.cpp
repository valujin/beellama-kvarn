// Охрана десятого патча стопки (kvarn-group-claim, шестнадцатая волна).
//
// ЧТО ЛОМАЛОСЬ. Структурный аллокатор KVarN раскладывает ячейки группами по
// KVAR_N_GROUP. Совместимость кандидата с группой проверялась только по УЖЕ
// занятым ячейкам, а зарезервированные этим же вызовом ячейки для cells ещё
// пусты. Поэтому один шаг декодирования с несколькими последовательностями в
// одном ubatch складывал их токены в одну группу. На следующем шаге такая
// группа несовместима уже ни с одной из них: она остаётся навсегда
// незавершённой и держит слот F16 из кольца в tail_groups слотов. Через
// tail_groups шагов кольцо кончается, и find_slot возвращает пусто при почти
// пустом кэше — llama_decode отвечает "failed to find a memory slot".
//
// ЧТО ОХРАНЯЕТСЯ ЗДЕСЬ. Ровно наблюдаемое следствие: много шагов подряд, на
// каждом одна и та же четвёрка последовательностей в ОДНОМ ubatch, общий кэш
// (n_stream == 1, иначе структурная ветка не включается), кэш KVarN. Ёмкость
// не должна утекать от шага к шагу: суммарно кладётся заведомо меньше токенов,
// чем вмещает кэш, и каждый декод обязан пройти.
//
// Отрицательный контроль проведён: с откаченным хунком group_claim тест падает
// на шаге, кратном глубине кольца, а с правкой проходит все шаги.

#include "arg.h"
#include "common.h"
#include "gguf.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// Синтетические модели из test-generate-models объявляют context_length = 128.
// Политика хвоста берёт окно как min(n_ctx_train, n_ctx), а при окне 128 хвост
// KVarN совпадает с окном целиком, представление становится native_exact, и
// llama_model строит ОБЫЧНЫЙ llama_kv_cache без групповой раскладки — то есть
// структурная ветка find_slot, которую этот тест и охраняет, не исполняется.
//
// Поэтому здесь делается копия модели с увеличенным context_length. Больше
// ничего не меняется: веса и токенизатор те же.
// Проверка, что структурная ветка аллокатора действительно исполняется.
//
// Без неё тест вырождается молча: любая смена умолчаний (политика хвоста,
// объединённый кэш, тип кэша) может увести модель на обычный llama_kv_cache, и
// шестьдесят четыре шага пройдут, ничего не охраняя. Наблюдаемый признак —
// строка KVARN-SEED, которую печатает САМА структурная ветка под
// LLAMA_KVARN_DEBUG_STAGE; своего кода в продукт для этого добавлять не нужно.
static bool              g_structural_branch_seen = false;
static ggml_log_callback g_prev_log_callback      = nullptr;
static void *            g_prev_log_user_data     = nullptr;

static int g_structural_live_groups = 0;

static void structural_branch_log_probe(ggml_log_level level, const char * text, void * user_data) {
    if (text != nullptr && std::strstr(text, "KVARN-SEED") != nullptr) {
        g_structural_branch_seen = true;
        // Каждая живая незавершённая группа печатается как "<группа>(<ячеек>)->s<слот>".
        int live = 0;
        for (const char * p = text; (p = std::strstr(p, "->s")) != nullptr; p += 3) {
            ++live;
        }
        g_structural_live_groups = std::max(g_structural_live_groups, live);
    }
    if (g_prev_log_callback != nullptr) {
        g_prev_log_callback(level, text, g_prev_log_user_data);
    }
    (void) user_data;
}

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
        (std::filesystem::temp_directory_path() / "beellama-group-claim-model.gguf").string();
    const bool ok = gguf_write_to_file(out, dst.c_str(), false);
    gguf_free(out);
    gguf_free(gctx);
    ggml_free(meta);
    return ok ? dst : std::string();
}

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified    = true;
    params.n_parallel    = 4;
    params.n_ctx         = 2048;
    params.n_batch       = 512;
    params.n_ubatch      = 512;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    // Тест бессмыслен без структурной ветки аллокатора: она включается только
    // на кэше KVarN и только при общем кэше. Молчаливое вырождение недопустимо.
    if (params.kvarn.type == LLAMA_KVARN_TYPE_DISABLED) {
        fprintf(stderr, "%s : KVarN cache is not selected, the structural allocator would not run\n", __func__);
        return 1;
    }
    if (!params.kv_unified) {
        fprintf(stderr, "%s : unified cache is required, the structural allocator needs n_stream == 1\n", __func__);
        return 1;
    }
    if (params.n_parallel < 2) {
        fprintf(stderr, "%s : at least two sequences are required to mix them inside one ubatch\n", __func__);
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

    const int n_seq = params.n_parallel;

    // Глубина кольца слотов F16 равна tail_groups и в рабочих конфигурациях не
    // превышает двух десятков; шагов берётся кратно больше, чтобы падение
    // «через tail_groups шагов» гарантированно попало внутрь прогона.
    const int n_steps = 64;

    // Токенов кладётся 4 * (1 + 64) = 260 при кэше на 2048 ячеек: если ёмкость
    // не утекает, места хватает с восьмикратным запасом.
    llama_batch batch = llama_batch_init(n_seq, 0, 1);

    // Затравка — ОДИН ubatch со всеми последовательностями сразу, и головы у
    // всех ещё стоят на нуле. Это и есть та сходимость голов, при которой в
    // рабочей нагрузке (шестнадцатая волна: общий кэш, четыре одновременных
    // запроса) все токены шага садились в одну группу. Затравка по одной
    // последовательности за вызов разводит головы по своим группам и опыт
    // разрушает: проверено, при ней отрицательный контроль проходит.
    common_batch_clear(batch);
    for (int s = 0; s < n_seq; ++s) {
        common_batch_add(batch, 1, 0, { s }, false);
    }
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode the seeding batch\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    // Первые три шага прогоняются с включённой отладкой стейджа. Структурная
    // ветка печатает KVARN-SEED, как только в кэше есть хоть одна живая
    // незавершённая группа помимо якорной, — а к третьему шагу она есть в
    // ЛЮБОМ случае, и с правкой, и без неё. Признак поэтому проверяет ровно
    // одно: что ветка вообще исполняется, а не что тест вырождается молча.
    llama_log_get(&g_prev_log_callback, &g_prev_log_user_data);
    llama_log_set(structural_branch_log_probe, nullptr);
    setenv("LLAMA_KVARN_DEBUG_STAGE", "1", 1);

    for (int step = 1; step <= n_steps; ++step) {
        if (step == 4) {
            unsetenv("LLAMA_KVARN_DEBUG_STAGE");
            llama_log_set(g_prev_log_callback, g_prev_log_user_data);
            if (!g_structural_branch_seen) {
                fprintf(stderr,
                        "%s : the structural KVarN allocator did not run; this test guards nothing\n",
                        __func__);
                llama_batch_free(batch);
                return 1;
            }
            if (g_structural_live_groups < 1) {
                fprintf(stderr,
                        "%s : the structural allocator reported no live unfinished group\n", __func__);
                llama_batch_free(batch);
                return 1;
            }
        }
        common_batch_clear(batch);
        for (int s = 0; s < n_seq; ++s) {
            common_batch_add(batch, 1, step, { s }, false);
        }
        const int rc = llama_decode(ctx, batch);
        if (rc != 0) {
            fprintf(stderr,
                    "%s : llama_decode failed at step %d of %d (rc=%d) with %d sequences in one ubatch\n",
                    __func__, step, n_steps, rc, n_seq);
            fprintf(stderr,
                    "%s : this is the group-claim regression: mixed-sequence groups stay live forever\n",
                    __func__);
            llama_batch_free(batch);
            return 1;
        }
    }

    // Ёмкость не утекла: у каждой последовательности ровно n_steps + 1 позиций.
    llama_memory_t mem = llama_get_memory(ctx);
    for (int s = 0; s < n_seq; ++s) {
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, s);
        if (pos_max != n_steps) {
            fprintf(stderr, "%s : sequence %d ends at position %d, expected %d\n",
                    __func__, s, int(pos_max), n_steps);
            llama_batch_free(batch);
            return 1;
        }
    }

    fprintf(stderr, "%s : %d steps x %d sequences in one ubatch on a unified KVarN cache: OK\n",
            __func__, n_steps, n_seq);

    llama_batch_free(batch);
    return 0;
}
