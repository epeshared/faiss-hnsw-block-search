#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/random.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

namespace {

struct Args {
    int d = 1024;
    int64_t nb = 1000000;
    int64_t nq = 1000;
    int k = 10;
    int m = 32;
    int efc = 128;
    int efs = 128;
    int threads = 1;
    int reps = 5;
    int warmup = 1;
    int mode = 0;
    int frontier_window = 1;
    int batch_threshold = 32;
    int refill_topk = 0;
    int64_t limit_nb = 0;
    int64_t limit_nq = 0;
    int64_t recall_nq = 0;
    std::string xb_path;
    std::string xq_path;
    std::string index_path;
};

struct NpyArrayF32 {
    std::vector<float> data;
    int64_t rows = 0;
    int64_t cols = 0;
};

int parse_int(const char* s) {
    return std::atoi(s);
}

int64_t parse_int64(const char* s) {
    return std::atoll(s);
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--d") && i + 1 < argc) {
            args.d = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--nb") && i + 1 < argc) {
            args.nb = parse_int64(argv[++i]);
        } else if (!std::strcmp(argv[i], "--nq") && i + 1 < argc) {
            args.nq = parse_int64(argv[++i]);
        } else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) {
            args.k = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--M") && i + 1 < argc) {
            args.m = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--efC") && i + 1 < argc) {
            args.efc = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--efS") && i + 1 < argc) {
            args.efs = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
            args.threads = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) {
            args.reps = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
            args.warmup = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) {
            args.mode = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--frontier-window") && i + 1 < argc) {
            args.frontier_window = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--batch-threshold") && i + 1 < argc) {
            args.batch_threshold = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--refill-topk") && i + 1 < argc) {
            args.refill_topk = parse_int(argv[++i]);
        } else if (!std::strcmp(argv[i], "--limit-nb") && i + 1 < argc) {
            args.limit_nb = parse_int64(argv[++i]);
        } else if (!std::strcmp(argv[i], "--limit-nq") && i + 1 < argc) {
            args.limit_nq = parse_int64(argv[++i]);
        } else if (!std::strcmp(argv[i], "--recall-nq") && i + 1 < argc) {
            args.recall_nq = parse_int64(argv[++i]);
        } else if (!std::strcmp(argv[i], "--xb") && i + 1 < argc) {
            args.xb_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--xq") && i + 1 < argc) {
            args.xq_path = argv[++i];
        } else if (!std::strcmp(argv[i], "--index-path") && i + 1 < argc) {
            args.index_path = argv[++i];
        }
    }
    return args;
}

bool file_exists(const std::string& path) {
    if (path.empty()) {
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    return input.good();
}

double compute_recall_at_k(
        const std::vector<faiss::idx_t>& ref,
        const std::vector<faiss::idx_t>& pred,
        int64_t nq,
        int k) {
    if (nq <= 0 || k <= 0) {
        return 0.0;
    }
    int64_t hits = 0;
    for (int64_t qi = 0; qi < nq; ++qi) {
        const faiss::idx_t* ref_row = ref.data() + qi * k;
        const faiss::idx_t* pred_row = pred.data() + qi * k;
        for (int j = 0; j < k; ++j) {
            const faiss::idx_t target = pred_row[j];
            for (int t = 0; t < k; ++t) {
                if (ref_row[t] == target) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return double(hits) / double(nq * k);
}

NpyArrayF32 load_npy_f32(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        std::fprintf(stderr, "failed to open npy file: %s\n", path.c_str());
        std::exit(2);
    }

    char magic[6];
    input.read(magic, sizeof(magic));
    if (input.gcount() != (std::streamsize)sizeof(magic) ||
        std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fprintf(stderr, "invalid npy magic: %s\n", path.c_str());
        std::exit(2);
    }

    char version[2];
    input.read(version, sizeof(version));
    if (input.gcount() != (std::streamsize)sizeof(version)) {
        std::fprintf(stderr, "failed to read npy version: %s\n", path.c_str());
        std::exit(2);
    }

    uint32_t header_len = 0;
    if (version[0] == 1) {
        uint16_t v = 0;
        input.read(reinterpret_cast<char*>(&v), sizeof(v));
        header_len = v;
    } else {
        input.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
    }
    if (!input) {
        std::fprintf(stderr, "failed to read npy header length: %s\n", path.c_str());
        std::exit(2);
    }

    std::string header(header_len, '\0');
    input.read(header.data(), header_len);
    if (!input) {
        std::fprintf(stderr, "failed to read npy header: %s\n", path.c_str());
        std::exit(2);
    }

    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos) {
        std::fprintf(stderr, "unsupported npy dtype in %s, expected float32\n", path.c_str());
        std::exit(2);
    }
    if (header.find("True") != std::string::npos) {
        std::fprintf(stderr, "fortran_order=True is not supported: %s\n", path.c_str());
        std::exit(2);
    }

    const size_t l = header.find('(');
    const size_t comma = header.find(',', l);
    const size_t r = header.find(')', comma);
    if (l == std::string::npos || comma == std::string::npos || r == std::string::npos) {
        std::fprintf(stderr, "failed to parse npy shape: %s\n", path.c_str());
        std::exit(2);
    }

    const int64_t rows = std::stoll(header.substr(l + 1, comma - l - 1));
    int64_t cols = 1;
    const std::string col_str = header.substr(comma + 1, r - comma - 1);
    if (col_str.find_first_not_of(' ') != std::string::npos) {
        cols = std::stoll(col_str);
    }

    NpyArrayF32 out;
    out.rows = rows;
    out.cols = cols;
    out.data.resize(static_cast<size_t>(rows * cols));
    input.read(reinterpret_cast<char*>(out.data.data()), out.data.size() * sizeof(float));
    if (!input) {
        std::fprintf(stderr, "failed to read npy payload: %s\n", path.c_str());
        std::exit(2);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    omp_set_num_threads(args.threads);

    std::vector<float> xb;
    std::vector<float> xq;
    if (!args.xb_path.empty() || !args.xq_path.empty()) {
        if (args.xb_path.empty() || args.xq_path.empty()) {
            std::fprintf(stderr, "both --xb and --xq must be provided together\n");
            return 2;
        }
        NpyArrayF32 xb_arr = load_npy_f32(args.xb_path);
        NpyArrayF32 xq_arr = load_npy_f32(args.xq_path);
        if (args.limit_nb > 0 && args.limit_nb < xb_arr.rows) {
            xb_arr.data.resize(static_cast<size_t>(args.limit_nb * xb_arr.cols));
            xb_arr.rows = args.limit_nb;
        }
        if (args.limit_nq > 0 && args.limit_nq < xq_arr.rows) {
            xq_arr.data.resize(static_cast<size_t>(args.limit_nq * xq_arr.cols));
            xq_arr.rows = args.limit_nq;
        }
        args.nb = xb_arr.rows;
        args.nq = xq_arr.rows;
        args.d = static_cast<int>(xb_arr.cols);
        if (xq_arr.cols != xb_arr.cols) {
            std::fprintf(stderr, "query dim mismatch: xb d=%lld xq d=%lld\n",
                    static_cast<long long>(xb_arr.cols),
                    static_cast<long long>(xq_arr.cols));
            return 2;
        }
        xb = std::move(xb_arr.data);
        xq = std::move(xq_arr.data);
    } else {
        xb.resize(static_cast<size_t>(args.nb) * args.d);
        xq.resize(static_cast<size_t>(args.nq) * args.d);
        faiss::float_rand(xb.data(), xb.size(), 1235);
        faiss::float_rand(xq.data(), xq.size(), 1234);
    }

    std::unique_ptr<faiss::Index> index_storage;
    faiss::IndexHNSWSQ* index = nullptr;
    if (!args.index_path.empty() && file_exists(args.index_path)) {
        index_storage = faiss::read_index_up(args.index_path.c_str());
        index = dynamic_cast<faiss::IndexHNSWSQ*>(index_storage.get());
        if (!index) {
            std::fprintf(stderr, "cached index is not IndexHNSWSQ: %s\n", args.index_path.c_str());
            return 2;
        }
    } else {
        auto owned = std::make_unique<faiss::IndexHNSWSQ>(
                args.d,
                faiss::ScalarQuantizer::QT_bf16,
                args.m,
                faiss::METRIC_INNER_PRODUCT);
        index = owned.get();
        index->hnsw.efConstruction = args.efc;
        index->hnsw.efSearch = args.efs;
#ifdef FAISS_ENABLE_LEVEL0_BATCHED_EXPANSION
        index->hnsw.level0_search_mode = args.mode;
        index->hnsw.level0_frontier_window = args.frontier_window;
        index->hnsw.level0_batch_threshold = args.batch_threshold;
        index->hnsw.level0_refill_topk = args.refill_topk;
#endif
        index->add(args.nb, xb.data());
        if (!args.index_path.empty()) {
            faiss::write_index(index, args.index_path.c_str());
        }
        index_storage = std::move(owned);
    }

    index->hnsw.efSearch = args.efs;
#ifdef FAISS_ENABLE_LEVEL0_BATCHED_EXPANSION
    index->hnsw.level0_search_mode = args.mode;
    index->hnsw.level0_frontier_window = args.frontier_window;
    index->hnsw.level0_batch_threshold = args.batch_threshold;
    index->hnsw.level0_refill_topk = args.refill_topk;
#endif

    std::vector<float> D(static_cast<size_t>(args.nq) * args.k);
    std::vector<faiss::idx_t> I(static_cast<size_t>(args.nq) * args.k);

    faiss::SearchParametersHNSW params;
    params.efSearch = args.efs;
#ifdef FAISS_ENABLE_LEVEL0_BATCHED_EXPANSION
    params.level0_search_mode = args.mode;
    params.level0_frontier_window = args.frontier_window;
    params.level0_batch_threshold = args.batch_threshold;
    params.level0_refill_topk = args.refill_topk;
#endif

    for (int i = 0; i < args.warmup; ++i) {
        index->search(args.nq, xq.data(), args.k, D.data(), I.data(), &params);
    }

    double total_s = 0.0;
    for (int i = 0; i < args.reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        index->search(args.nq, xq.data(), args.k, D.data(), I.data(), &params);
        auto t1 = std::chrono::steady_clock::now();
        total_s += std::chrono::duration<double>(t1 - t0).count();
    }

    double recall = -1.0;
    if (args.recall_nq > 0) {
        const int64_t gt_nq = std::min<int64_t>(args.recall_nq, args.nq);
        std::vector<float> gt_D(static_cast<size_t>(gt_nq) * args.k);
        std::vector<faiss::idx_t> gt_I(static_cast<size_t>(gt_nq) * args.k);
        faiss::IndexFlatIP gt_index(args.d);
        gt_index.add(args.nb, xb.data());
        gt_index.search(gt_nq, xq.data(), args.k, gt_D.data(), gt_I.data());
        recall = compute_recall_at_k(gt_I, I, gt_nq, args.k);
    }

    const double avg_s = total_s / std::max(args.reps, 1);
    const double qps = double(args.nq) / avg_s;
    std::printf(
            "bench_hnsw_sq_amx_level0 d=%d nb=%lld nq=%lld M=%d efC=%d efS=%d mode=%d frontier_window=%d batch_threshold=%d refill_topk=%d avg_search_s=%.6f qps=%.2f recall=%.6f xb=%s xq=%s index=%s\n",
            args.d,
            static_cast<long long>(args.nb),
            static_cast<long long>(args.nq),
            args.m,
            args.efc,
            args.efs,
            args.mode,
            args.frontier_window,
            args.batch_threshold,
            args.refill_topk,
            avg_s,
            qps,
            recall,
            args.xb_path.empty() ? "<synthetic>" : args.xb_path.c_str(),
            args.xq_path.empty() ? "<synthetic>" : args.xq_path.c_str(),
            args.index_path.empty() ? "<none>" : args.index_path.c_str());

    return 0;
}