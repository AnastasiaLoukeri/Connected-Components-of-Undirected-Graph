#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <pthread.h>
#include <atomic>

using namespace std;
using u32 = uint32_t;
using u64 = uint64_t;
using clk = chrono::steady_clock;


struct MMEntry { u32 r, c; };

struct CSR {
    u32 n = 0;
    vector<u64> row_ptr;
    vector<u32> col_idx;
};


bool read_matrix_market_coo(const string &path, vector<MMEntry> &coords, bool &symmetric, u32 &max_index) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "Cannot open '%s'\n", path.c_str()); return false; }

    char buf[65536]; 
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return false; }
    string header(buf);
    symmetric = (header.find("symmetric") != string::npos);

    do { if (!fgets(buf, sizeof(buf), f)) { fclose(f); return false; } } while (buf[0] == '%');

    unsigned long long rows=0, cols=0, nnz=0;
    sscanf(buf, "%llu %llu %llu", &rows, &cols, &nnz);

    coords.reserve(symmetric ? nnz * 2 : nnz);
    max_index = 0;

    while (fgets(buf, sizeof(buf), f)) {
        if (buf[0] == '%') continue;
        char* p = buf;
        u32 r = 0, c = 0;
        
        while(*p >= '0' && *p <= '9') { r = r * 10 + (*p - '0'); ++p; }
        while(*p == ' ' || *p == '\t') ++p;
        while(*p >= '0' && *p <= '9') { c = c * 10 + (*p - '0'); ++p; }

        if(r > 0 && c > 0) {
            coords.push_back({r - 1, c - 1});
            if (r - 1 > max_index) max_index = r - 1;
            if (c - 1 > max_index) max_index = c - 1;
            if (symmetric && r != c) coords.push_back({c - 1, r - 1});
        }
    }
    fclose(f);
    return true;
}

//  PTHREADS CSR BUILD 
struct CSRThreadData {
    int tid, num_threads;
    const vector<MMEntry>* coords;
    vector<u64>* deg;
    vector<u64>* cur;
    CSR* G;
    u32 n;
};

void* degree_count_thread(void* arg){
    auto* td = (CSRThreadData*)arg;
    size_t start = (size_t)td->tid * td->coords->size() / td->num_threads;
    size_t end   = (td->tid == td->num_threads - 1) ? td->coords->size() : (size_t)(td->tid + 1) * td->coords->size() / td->num_threads;

    for(size_t i = start; i < end; i++) {
        __sync_fetch_and_add(&(*td->deg)[(*td->coords)[i].r], 1);
    }
    return nullptr;
}

void* edge_insert_thread(void* arg){
    auto* td = (CSRThreadData*)arg;
    size_t start = (size_t)td->tid * td->coords->size() / td->num_threads;
    size_t end   = (td->tid == td->num_threads - 1) ? td->coords->size() : (size_t)(td->tid + 1) * td->coords->size() / td->num_threads;

    for(size_t i = start; i < end; i++){
        u32 u = (*td->coords)[i].r;
        u32 v = (*td->coords)[i].c;
        u64 pos = __sync_fetch_and_add(&(*td->cur)[u], 1);
        td->G->col_idx[pos] = v;
    }
    return nullptr;
}

void* sort_thread(void* arg){
    auto* td = (CSRThreadData*)arg;
    u32 start = (u32)((u64)td->tid * td->n / td->num_threads);
    u32 end   = (td->tid == td->num_threads - 1) ? td->n : (u32)((u64)(td->tid + 1) * td->n / td->num_threads);

    for(u32 i = start; i < end; i++){
        u64 s = td->G->row_ptr[i], t = td->G->row_ptr[i+1];
        if(t > s) sort(td->G->col_idx.begin() + s, td->G->col_idx.begin() + t);
    }
    return nullptr;
}

CSR build_csr_pthreads(u32 n, const vector<MMEntry> &coords, int num_threads){
    CSR G; G.n = n;
    G.row_ptr.assign(n + 1, 0);
    
    vector<pthread_t> threads(num_threads);
    vector<CSRThreadData> td(num_threads);

    // Degree Counting
    for(int i=0; i<num_threads; i++) {
        td[i] = {i, num_threads, &coords, &G.row_ptr, nullptr, &G, n};
        pthread_create(&threads[i], nullptr, degree_count_thread, &td[i]);
    }
    for(int i=0; i<num_threads; i++) pthread_join(threads[i], nullptr);

    // Prefix sum 
    u64 sum = 0;
    for(u32 i=0; i<n; i++) { u64 d = G.row_ptr[i]; G.row_ptr[i] = sum; sum += d; }
    G.row_ptr[n] = sum;

    G.col_idx.resize(sum);
    vector<u64> cur = G.row_ptr;

    //  Edge Insertion
    for(int i=0; i<num_threads; i++) {
        td[i].cur = &cur;
        pthread_create(&threads[i], nullptr, edge_insert_thread, &td[i]);
    }
    for(int i=0; i<num_threads; i++) pthread_join(threads[i], nullptr);

    //  Sorting
    for(int i=0; i<num_threads; i++) {
        pthread_create(&threads[i], nullptr, sort_thread, &td[i]);
    }
    for(int i=0; i<num_threads; i++) pthread_join(threads[i], nullptr);

    return G;
}


struct CCThreadData {
    int tid, num_threads;
    const CSR* G;
    vector<u32>* labels;
    std::atomic<bool>* global_changed;
    pthread_barrier_t* barrier;
    size_t* iterations;
};

void* cc_thread_func(void* arg) {
    auto* td = (CCThreadData*)arg;
    u32 n = td->G->n;
    
    // Static Partitioning
    u32 start = (u32)((u64)td->tid * n / td->num_threads);
    u32 end   = (td->tid == td->num_threads - 1) ? n : (u32)((u64)(td->tid + 1) * n / td->num_threads);

    // Αρχικοποίηση ετικετών
    for(u32 i = start; i < end; i++) (*td->labels)[i] = i;

    bool local_changed;
    
    while (true) {
        // Συγχρονισμός πριν από κάθε νέα επανάληψη
        pthread_barrier_wait(td->barrier);
        
        // Μόνο ένα thread μηδενίζει το global flag
        if (td->tid == 0) {
            td->global_changed->store(false, std::memory_order_relaxed);
            (*td->iterations)++;
        }
        
        // Περιμένουμε να μηδενιστεί το flag
        pthread_barrier_wait(td->barrier);
        local_changed = false;

        //  Label Propagation 
        for (u32 u = start; u < end; u++) {
            u64 s = td->G->row_ptr[u], t = td->G->row_ptr[u+1];
            u32 min_label = (*td->labels)[u];
            for (u64 p = s; p < t; p++) {
                u32 v_label = (*td->labels)[td->G->col_idx[p]];
                if (v_label < min_label) min_label = v_label;
            }
            if (min_label < (*td->labels)[u]) {
                (*td->labels)[u] = min_label;
                local_changed = true;
            }
        }

        if (local_changed) td->global_changed->store(true, std::memory_order_relaxed);

        // Περιμένουμε να τελειώσουν όλα τα threads τη Φάση 1
        pthread_barrier_wait(td->barrier);

        // Pointer Jumping 
        for (u32 u = start; u < end; u++) {
            u32 current = (*td->labels)[u];
            while (current != (*td->labels)[current]) {
                current = (*td->labels)[current];
            }
            (*td->labels)[u] = current;
        }

        // Περιμένουμε να τελειώσουν όλα τα threads τη Φάση 2
        pthread_barrier_wait(td->barrier);

        // Ελέγχουμε αν χρειάζεται άλλη επανάληψη
        if (!td->global_changed->load(std::memory_order_relaxed)) {
            break; 
        }
    }
    return nullptr;
}

vector<u32> compute_connected_components_pthreads(const CSR &G, int num_threads, size_t &iters) {
    u32 n = G.n;
    vector<u32> labels(n);
    std::atomic<bool> global_changed(true);
    iters = 0;

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, nullptr, num_threads);

    vector<pthread_t> threads(num_threads);
    vector<CCThreadData> td(num_threads);

    for (int i = 0; i < num_threads; i++) {
        td[i] = {i, num_threads, &G, &labels, &global_changed, &barrier, &iters};
        pthread_create(&threads[i], nullptr, cc_thread_func, &td[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], nullptr);
    }

    pthread_barrier_destroy(&barrier);

    // Κανονικοποίηση ετικετών (Σειριακά)
    vector<u32> unique_labels(labels.begin(), labels.end());
    sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
    unordered_map<u32, u32> label_map;
    for (size_t i = 0; i < unique_labels.size(); i++) label_map[unique_labels[i]] = i;
    for (u32 i = 0; i < n; i++) labels[i] = label_map[labels[i]];

    return labels;
}

vector<u32> compute_connected_components_serial(const CSR &G, size_t &iters) {
    u32 n = G.n;
    vector<u32> labels(n);
    
    // Αρχικοποίηση
    for(u32 i = 0; i < n; i++) {
        labels[i] = i;
    }

    iters = 0;
    bool changed;
    
    do {
        changed = false;
        iters++;
        
        //  Label Propagation
        for (u32 u = 0; u < n; u++) {
            u64 s = G.row_ptr[u], t = G.row_ptr[u+1];
            u32 min_label = labels[u];
            for (u64 p = s; p < t; p++) {
                u32 v_label = labels[G.col_idx[p]];
                if (v_label < min_label) min_label = v_label;
            }
            if (min_label < labels[u]) {
                labels[u] = min_label;
                changed = true;
            }
        }

        //  Pointer Jumping
        for (u32 u = 0; u < n; u++) {
            u32 current = labels[u];
            while (current != labels[current]) {
                current = labels[current];
            }
            labels[u] = current;
        }
    } while (changed);

    // Κανονικοποίηση ετικετών για άμεση σύγκριση
    vector<u32> unique_labels(labels.begin(), labels.end());
    sort(unique_labels.begin(), unique_labels.end());
    unique_labels.erase(unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());
    
    unordered_map<u32, u32> label_map;
    for (size_t i = 0; i < unique_labels.size(); i++) {
        label_map[unique_labels[i]] = i;
    }
    for (u32 i = 0; i < n; i++) {
        labels[i] = label_map[labels[i]];
    }

    return labels;
}

bool verify_correctness(const vector<u32>& baseline, const vector<u32>& test) {
    if (baseline.size() != test.size()) return false;
    for (size_t i = 0; i < baseline.size(); i++) {
        if (baseline[i] != test[i]) return false;
    }
    return true;
}


int main(int argc, char **argv){
    if(argc < 2){ 
        fprintf(stderr,"Usage: %s graph.mtx\n",argv[0]); 
        return 1; 
    }
    string path = argv[1];

    vector<MMEntry> coords;
    bool symmetric = false;
    u32 max_index = 0;

    auto t0 = clk::now();
    if(!read_matrix_market_coo(path, coords, symmetric, max_index)){ 
        fprintf(stderr,"Failed to read matrix\n"); 
        return 1; 
    }
    u32 n = max_index + 1;

    
    int build_threads = 4; 
    auto t1 = clk::now();
    CSR G = build_csr_pthreads(n, coords, build_threads);
    auto t2 = clk::now();

    coords.clear();
    coords.shrink_to_fit();

    double read_time = chrono::duration<double>(t1-t0).count();
    double build_time = chrono::duration<double>(t2-t1).count();

    printf(" GRAPH INFO \n");
    printf("File: %s\n", path.c_str());
    printf("Nodes: %u, Directed Edges: %zu\n", G.n, G.col_idx.size());
    printf("I/O Time: %.3f s | CSR Build Time (Threads: %d): %.3f s\n\n", read_time, build_threads, build_time);

    //  ΣΕΙΡΙΑΚΗ ΕΚΤΕΛΕΣΗ (Baseline) 
    printf(" SERIAL EXECUTION \n");
    size_t serial_iters = 0;
    auto t_seq_start = clk::now();
    vector<u32> serial_labels = compute_connected_components_serial(G, serial_iters);
    auto t_seq_end = clk::now();
    
    double serial_time = chrono::duration<double>(t_seq_end - t_seq_start).count();
    unordered_set<u32> comps_serial(serial_labels.begin(), serial_labels.end());
    
    printf("CCs found: %zu | Iterations: %zu | Time: %.3f s\n\n", comps_serial.size(), serial_iters, serial_time);

    // PThreads Scaling
    printf(" PTHREADS SCALING & VERIFICATION \n");
    printf("%-10s %-12s %-12s %-10s %-10s\n", "Threads", "Time (s)", "Speedup", "Iters", "Status");
    printf("----------------------------------------------------------\n");

    vector<int> thread_counts = {1, 2, 4, 8, 16}; 
    
    for (int t : thread_counts) {
        size_t par_iters = 0;
        
        auto t_par_start = clk::now();
        vector<u32> par_labels = compute_connected_components_pthreads(G, t, par_iters);
        auto t_par_end = clk::now();
        
        double par_time = chrono::duration<double>(t_par_end - t_par_start).count();
        double speedup = serial_time / par_time;
        
        bool is_correct = verify_correctness(serial_labels, par_labels);
        string status = is_correct ? "PASS" : "FAIL";

        printf("%-10d %-12.3f %-12.2fx %-10zu %-10s\n", t, par_time, speedup, par_iters, status.c_str());
    }

    return 0;
}