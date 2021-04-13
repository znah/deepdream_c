#include <stdio.h>
#include <math.h> /* TODO: remove */

typedef unsigned char uint8;

enum {
    MAX_TENSOR_DIMS = 4,
    MAX_OP_INPUTS = 4
};

typedef struct {int x, y;} ivec2;

typedef struct {
    float *val, *grad;
    int size, ndim;
    int shape[MAX_TENSOR_DIMS];
} tensor_t;
typedef const tensor_t * tensor_ptr_t;

typedef struct { int ofs; } const_op_t;  /* weights offset in .pb file */
typedef struct { ivec2 stride; } conv_op_t;
typedef struct { ivec2 stride, ksize; } maxpool_op_t;
typedef struct { int depth_radius; float alpha, beta, bias; } lrn_op_t;

typedef struct op_ref_t op_ref_t;
typedef enum {RUN_FORWARD, RUN_BACKWARD} run_mode_t;
typedef void op_func(const op_ref_t*, run_mode_t);

struct op_ref_t {
    op_func *run;
    const void *params;
    const char *name;
    const tensor_ptr_t output;
    const tensor_ptr_t input[MAX_OP_INPUTS];
};

#include "inception.inc"
/* Include the generated model code. We care about:

const chat * model_filename     -- where to load weights from
tensor_t g_data;                -- input tensor
op_ref_t g_ops[g_ops_num]       -- neural net operations list
void *_func(...)                -- forward declarations of net ops that we 
                                   implement below
*/

/*************** IEEE 74 32-bit float parsing (little-endian) ****************/
enum {
  FLOAT32_SIZE = 4,
  FLOAT32_MANTISSA_BIT = (1<<23)
};
float float32_exp_table[256];

void init_float32_exp() {
    int i;
    float a=1.0, b=1.0;
    for (i=0; i<128; ++i) {
        float32_exp_table[127-i] = a;
        float32_exp_table[127+i] = b;
        a /= 2.0;
        b *= 2.0;
    }
    float32_exp_table[0] = float32_exp_table[1]; /* subnomal */
    float32_exp_table[2] = 1.0/0.0;            /* infinity */
}

float parse_float32(const uint8 * p) {
    float sign = 1-(p[3]>>7)*2;
    int exponent = ((p[3]&0x7f)<<1) + (p[2]>>7);
    int mantissa = ((p[2]&0x7f)<<16) + (p[1]<<8) + p[0];
    if (exponent == 255 && mantissa != 0) {
        return 0.0/0.0; /* Not a Number */
    }
    if (exponent > 0) { /* normal number */
        mantissa |= FLOAT32_MANTISSA_BIT;
    }
    return sign * ((float)mantissa / FLOAT32_MANTISSA_BIT)
                * float32_exp_table[exponent];
}

void init_consts() {
    FILE *f=fopen(model_filename, "rb");
    const op_ref_t * op = g_ops;
    for (; op<g_ops+g_ops_num; ++op) {
        int i;
        const const_op_t * params;
        float * val;
        if (op->run != &const_func)
          continue;
        params = (const_op_t*) (op->params);
        val = op->output->val;
        fseek(f, params->ofs, SEEK_SET);
        for (i=0; i<op->output->size; ++i) {
          uint8 buf[4];
          if (fread(buf, FLOAT32_SIZE, 1, f) != 1)
            return;
          val[i] = parse_float32(buf);
        }

    }
    fclose(f);
}

float minf(float a, float b) { return a<b ? a : b; }
float maxf(float a, float b) { return a>b ? a : b; }

int min(int a, int b) { return a<b ? a : b; }
int max(int a, int b) { return a>b ? a : b; }

void fill_zeros(const tensor_t * t) {
    int i=0;
    for (; i<t->size; ++i) t->val[i] = 0.0;
}


void const_func(const op_ref_t * op, run_mode_t mode) {}

void conv2d_func(const op_ref_t * op, run_mode_t mode) {
    if (mode==RUN_FORWARD) {
        const tensor_t * input = op->input[0];
        const tensor_t * kernel = op->input[1]; /* kh, kw, ci, co*/
        const tensor_t * output = op->output;
        const int kh = kernel->shape[0], kw = kernel->shape[1];
        const int ci = kernel->shape[2], co = kernel->shape[3];
        const int h = input->shape[0], w = input->shape[1];
        const int wo = output->shape[1];
        ivec2 stride = {1, 1};
        if (op->params != NULL) {
            stride = ((conv_op_t *)op->params)->stride;
        }
        const float * kmat = kernel->val;
        int x, y, sx, sy, i, o;
        fill_zeros(output);

        for (sy=-kh/2; sy<kh-kh/2; ++sy)
        for (sx=-kw/2; sx<kw-kw/2; ++sx, kmat += ci*co)
        for (y=max(0, sy); y<min(h, h+sy); ++y) if ((y-sy+1)%stride.y==0)
        for (x=max(0, sx); x<min(w, w+sx); ++x) if ((x-sx+1)%stride.x==0) {
            const float * irow = input->val + (y*w+x)*ci;
            float * orow = output->val + ((y-sy)/stride.y*wo+(x-sx)/stride.x)*co;
            for (i=0; i<ci; ++i)
            for (o=0; o<co; ++o) {
                orow[o] += irow[i]*kmat[i*co + o];
            }
        }
    }
}

void biasadd_func(const op_ref_t * op, run_mode_t mode) {
    if (mode==RUN_FORWARD) {
        const float * bias = op->input[1]->val;
        int bias_len = op->input[1]->size;
        float * val = op->output->val;
        int i;
        for (i=0; i < op->output->size; ++i) {
            val[i] += bias[i%bias_len];
        }
    } // do nothing on the backward pass
}


void relu_func(const op_ref_t * op, run_mode_t mode) {
    float * val = op->output->val;
    int i, size=op->output->size;
    if (mode==RUN_FORWARD) {
        for (i=0; i<size; ++i) {
            val[i] = maxf(val[i], 0.0);
        }
    } else {
        float * grad = op->output->grad;
        for (i=0; i<size; ++i) {
            if (val[i] == 0.0) grad[i] = 0.0;
        }
    }
}

float sqr(float v) { return v*v; }

void lrn_func(const op_ref_t * op, run_mode_t mode) {
    if (mode==RUN_FORWARD) {
        int size = op->input[0]->size;
        int ndim = op->input[0]->ndim;
        int row_len = op->input[0]->shape[ndim-1];
        const float * input = op->input[0]->val;
        float * output = op->output->val;
        const lrn_op_t p = *(const lrn_op_t*)(op->params);
        int r = p.depth_radius;
        int row, i;
        fill_zeros(op->output);

        for (row=0; row<size; row+=row_len) {
            float ssum = 0.0;
            for (i=0; i<r; ++i) ssum += sqr(input[row+i]);
            for (i=0; i<row_len; ++i) {
                if (i+r<row_len) ssum += sqr(input[row+i+r]);
                output[row+i] = input[row+i] / pow(p.bias + ssum*p.alpha, p.beta);
                if (i-r>=0) ssum -= sqr(input[row+i-r]);
            }
        }
    }
}

void maxpool_func(const op_ref_t * op, run_mode_t mode) {
    if (mode==RUN_FORWARD) {
        const tensor_t * input = op->input[0];
        const tensor_t * output = op->output;
        const maxpool_op_t p = *(const maxpool_op_t*)(op->params);
        const int kh = p.ksize.y, kw = p.ksize.x;
        const int h = input->shape[0], w = input->shape[1], c = input->shape[2];
        const int wo = output->shape[1];
        int sx, sy, x, y, i;
        fill_zeros(output);

        for (sy=-kh/2; sy<kh-kh/2; ++sy)
        for (sx=-kw/2; sx<kw-kw/2; ++sx)
        for (y=max(0, sy); y<min(h, h+sy); ++y) if ((y-sy+1)%p.stride.y==0)
        for (x=max(0, sx); x<min(w, w+sx); ++x) if ((x-sx+1)%p.stride.x==0) {
            const float * irow = input->val + (y*w+x)*c;
            float * orow = output->val + ((y-sy)/p.stride.y*wo+(x-sx)/p.stride.x)*c;
            for (i=0; i<c; ++i) {
              orow[i] = maxf(orow[i], irow[i]);
            }
        }
    }
}

void concatv2_func(const op_ref_t * op, run_mode_t mode) {
    if (mode==RUN_FORWARD) {
        const tensor_t * output = op->output;
        const int co = output->shape[2];
        int input_i = 0, ofs = 0;
        for (; input_i<MAX_OP_INPUTS; ++input_i) {
            int s=0, d=ofs, ci;
            const tensor_t * input = op->input[input_i];
            if (input == NULL) break;
            ci = input->shape[2];
            while (s<input->size) {
                int i;
                for (i=0; i<ci; ++i) {
                    output->val[d+i] = input->val[s+i];
                }
                s += ci; d += co;
            }
            ofs += ci;
        }
    }
}

void avgpool_func(const op_ref_t * op, run_mode_t mode) {
    if (mode==RUN_FORWARD) {
        const float * input = op->input[0]->val;
        float * output = op->output->val;
        const int depth = op->output->size;
        const int n = op->input[0]->size; 
        const float scale = 1.0/(n/depth);
        int i;
        fill_zeros(op->output);
        for (i=0; i < n; ++i) {
            output[i%depth] += input[i];
        }
        for (i=0; i < depth; ++i) {
            output[i] *= scale;
        }
    }
}

void matmul_func(const op_ref_t * op, run_mode_t mode) {
    conv2d_func(op, mode);
}

void softmax_func(const op_ref_t * op, run_mode_t mode) {
    if (mode==RUN_FORWARD) {
        const float * input = op->input[0]->val;
        float * output = op->output->val;
        const int size = op->output->size;
        float max_logit = input[0], exp_sum=0.0;
        int i;
        for (i=1; i<size; ++i) {
            max_logit = maxf(max_logit, input[i]);
        }
        for (i=0; i<size; ++i) {
            float e = exp(input[i]-max_logit);
            output[i] = e;
            exp_sum += e;
        }
        for (i=0; i<size; ++i) {
            output[i] /= exp_sum;
        }
    }
}


void write_tensor(const char *name, const tensor_t *tensor) {
    FILE *f = fopen(name, "wb");
    fwrite(tensor->val, sizeof(float), tensor->size, f);
    fclose(f);
}

void forward(int target) {
    int i;
    char fn[1024];
    for (i=0; i<=target; ++i) {
        /*printf("%s\n", g_ops[i].name);*/
        g_ops[i].run(g_ops+i, RUN_FORWARD);
        sprintf(fn, "out/%s", g_ops[i].name);
        write_tensor(fn, g_ops[i].output);
    }
}

void backward(int target) {
    int i;
    for (i=target; i>=0; --i) {
      g_ops[i].run(g_ops+i, RUN_BACKWARD);
    }
}


int run_tests();

int main(int arvc, const char * argv[]) {
    int i=0;
    FILE *f;

    run_tests();
  
    init_float32_exp();
    init_consts();


    f = fopen("input.dat", "rb");
    fread(g_data_val, FLOAT32_SIZE, g_data.size, f);
    fclose(f);

    for (int i=0; i<1; ++i)
        forward(g_ops_num-1);

    for (i=0; i<arvc; ++i) {
        printf("%s\n", argv[i]);
    }

    return 0;
}

float test_pulse_4x4_val[4*4] = {0, 0, 0, 0,
                                 0, 1, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 1};
tensor_t test_pulse_4x4 = {test_pulse_4x4_val, NULL, 4*4, 3, {4, 4, 1}};

float test_pulse_3x3_val[3*3] = {0, 0, 0,
                                 0, 1, 0,
                                 0, 0, 0};
tensor_t test_pulse_3x3 = {test_pulse_3x3_val, NULL, 3*3, 4, {3, 3, 1, 1}};

float test_gx_3x3_val[3*3] = {1, 2, 3,
                              4, 5, 6,
                              7, 8, 9};
tensor_t test_gx_3x3 = {test_gx_3x3_val, NULL, 3*3, 4, {3, 3, 1, 1}};

float test_out_4x4_val[4*4];
tensor_t test_out_4x4 = {test_out_4x4_val, NULL, 4*4, 3, {4, 4, 1}};

float test_out_2x2_val[4*4];
tensor_t test_out_2x2 = {test_out_2x2_val, NULL, 2*2, 3, {2, 2, 1}};

conv_op_t test_conv_s1 = {1, 1};
conv_op_t test_conv_s2 = {2, 2};
op_ref_t test_ops[] = {
    {conv2d_func, &test_conv_s1, "test_conv_3x3", &test_out_4x4, {&test_pulse_4x4, &test_gx_3x3}},
    {conv2d_func, &test_conv_s2, "test_conv_3x3s2", &test_out_2x2, {&test_pulse_4x4, &test_gx_3x3}},
};

void print_tensor(const tensor_t *t) {
    int i, j, k=0;
    printf("ndim: %d, size: %d  (%d, %d, %d, %d)\n", t->ndim, t->size,
        t->shape[0], t->shape[1], t->shape[2], t->shape[3]);
    for (i=0; i<t->shape[0]; ++i) {
        for (j=0; j<t->shape[1]; ++j) {
            printf("%.2f  ", t->val[k]); ++k;
        }
        printf("\n");
    }
}

void run_op(op_ref_t * op) {
    printf("\n%s\n", op->name);
    op->run(op, RUN_FORWARD);
    /*write_tensor("test_out", op->output);*/
    print_tensor(op->output);
}

int run_tests() {
    run_op(test_ops+0);
    run_op(test_ops+1);

    return 1;
}