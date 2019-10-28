"""
CUDA code as text, auto-generated in Python.
"""

################################################################################
### Global helpers #############################################################
################################################################################

letters = "abcdefghijklmnopqrstuvwxyz"

def get_setters_helper(d, op):
    code = " "*8 + "*(grid"
    for i in range(d-1, -1, -1):
        L = letters[i]
        n = d - i - 1
        code += f" + {op}_{L}{'*N'*n}"
    code += ") = 1;\n"
    return code

def get_setters(d):
    max_s = get_setters_helper(d, "max")
    min_s = get_setters_helper(d, "min")
    return max_s + min_s

################################################################################
### Code for simple GPU algorithm ##############################################
################################################################################

signature_pair = """
__global__ void f(int8_t* grid, int32_t* present, int32_t* idx_to_pair,
                    int32_t ns, int32_t N, int32_t tile_size)
{
    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x + gridDim.x * blockIdx.y;
    int globalThreadNum  = threadNumInBlock + blockNumInGrid * threadsPerBlock;
    int num_pairs = (ns*(ns-1)/2);
    for (int i = 0; i < tile_size; i++) {
        int curr_pair = globalThreadNum*tile_size + i;
        if (curr_pair >= num_pairs) { return; }
        int idx = idx_to_pair[curr_pair];
        int idy = idx_to_pair[curr_pair + num_pairs];
"""

def get_code_pair(d):
    code = signature_pair
    for i in range(d):
        L = letters[i]
        code += " "*8 + f"int32_t {L}1 = present[idx{'+ns'*i}];\n"
        code += " "*8 + f"int32_t {L}2 = present[idy{'+ns'*i}];\n"
        code += " "*8 + f"int32_t max_{L} = max({L}1, {L}2);\n"
        code += " "*8 + f"int32_t min_{L} = min({L}1, {L}2);\n"
    code += get_setters(d)
    code += " "*4 + "}\n}"
    return code

################################################################################
### Code for hyper-splitting GPU algorithm #####################################
################################################################################

signature_hyper = """
__global__ void f(int8_t* grid, int32_t* present, int32_t ns, int32_t N)
{
    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x + gridDim.x * blockIdx.y;
    int globalThreadNum  = threadNumInBlock + blockNumInGrid * threadsPerBlock;
    // The thread ID is the position of the first point.
    if (globalThreadNum >= ns) {{ return; }}
"""

def get_code_hyper(d):
    code = signature_hyper
    code += "  // Get the first point\n"
    for i in range(d):
        L = letters[i]
        code += f"  int32_t {L}1 = present[globalThreadNum{'+ns'*i}];\n"
    code += "  for (int i = 0; i < globalThreadNum; i++) {\n"
    for i in range(d):
        L = letters[i]
        code += " "*8 + f"int32_t {L}2 = present[i{'+ns'*i}];\n"
        code += " "*8 + f"int32_t max_{L} = max({L}1, {L}2);\n"
        code += " "*8 + f"int32_t min_{L} = min({L}1, {L}2);\n"
    code += get_setters(d)
    code += "  }\n}"
    return code

################################################################################
### Code for grid GPU algorithm ################################################
################################################################################

signature_grid = """__global__ void f(int8_t* grid, int32_t N, int32_t N_D)
{
    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x + gridDim.x * blockIdx.y;
    int globalThreadNum  = threadNumInBlock + blockNumInGrid * threadsPerBlock;
    int idx = globalThreadNum;
    if (idx >= N_D) { return; }
    // Every thread has a local point defined by its index.
"""

def get_code_grid(d):
    code = signature_grid
    mid_1 = " "*4 + "int32_t q = *(grid"
    mid_2 = " "*8 + "q = *(grid"
    code_2 = ""
    maxmins = ""

    for i in range(d-1, -1, -1):
        L = letters[i]
        code += f"  int32_t {L}1 = idx%N; idx = idx/N;\n"
        mid_1 += f"+{L}1{'*N'*(d-i-1)}"
        mid_2 += f"+{L}2{'*N'*(d-i-1)}"
        code_2 += " "*8 + f"int32_t {L}2 = idy%N; idy = idy/N;\n"
        maxmins += " "*8 + f"int32_t min_{L} = min({L}1, {L}2);\n"
        maxmins += " "*8 + f"int32_t max_{L} = max({L}1, {L}2);\n"
    
    code += mid_1 + ");\n  if (q == 0) { return; }\n"
    code += "  for (int i = 0; i < globalThreadNum; i++) {\n"
    code += " "*8 + "int idy = i;\n"
    code += code_2 + mid_2 + ");\n"
    code += " "*8 + "if (q == 0) { continue; }\n" + maxmins
    code += get_setters(d) + "  }\n}"
    return code

