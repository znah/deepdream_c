#include <stdio.h>  /* printf fopen fclose fseek fread fwrite */
#include <math.h>   /* TODO: remove */

/****************************** Neural Net data strucutres **********************************/
/*                  Minimal set definitions used by "inception.inc"                         */
/********************************************************************************************/
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



/********************************************************************************************/
/*                                Utilities and heplers                                     */
/********************************************************************************************/

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


/*************** Math functions ****************/

float minf(float a, float b) { return a<b ? a : b; }
float maxf(float a, float b) { return a>b ? a : b; }

int min(int a, int b) { return a<b ? a : b; }
int max(int a, int b) { return a>b ? a : b; }

float sqr(float v) { return v*v; }



/********************************************************************************************/
/*                              Neural Net layer functions                                  */
/********************************************************************************************/

void placeholder_func(const op_ref_t * op, run_mode_t mode) {}
void const_func(const op_ref_t * op, run_mode_t mode) {}

void conv2d_func(const op_ref_t * op, run_mode_t mode) {
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
    const float * kmat = (mode == RUN_FORWARD) ? kernel->val : kernel->grad;
    int x, y, sx, sy, i, o;

    for (sy=-kh/2; sy<kh-kh/2; ++sy)
    for (sx=-kw/2; sx<kw-kw/2; ++sx, kmat += ci*co)
    for (y=max(0, sy); y<min(h, h+sy); ++y) if ((y-sy+1)%stride.y==0)
    for (x=max(0, sx); x<min(w, w+sx); ++x) if ((x-sx+1)%stride.x==0) {
        int o_ofs = ((y-sy)/stride.y*wo+(x-sx)/stride.x)*co;
        if (mode == RUN_FORWARD) {
            const float * irow = input->val + (y*w+x)*ci;
            float * orow = output->val + o_ofs;
            for (i=0; i<ci; ++i)
            for (o=0; o<co; ++o) {
                orow[o] += irow[i]*kmat[i*co + o];
            }
        } else {
            float * irow = input->grad + (y*w+x)*ci;
            const float * orow = output->grad + o_ofs;
            for (o=0; o<co; ++o)
            for (i=0; i<ci; ++i)  {
                irow[i] += orow[o]*kmat[o*ci + i];
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

void lrn_func(const op_ref_t * op, run_mode_t mode) {
    int size = op->input[0]->size;
    int ndim = op->input[0]->ndim;
    int depth = op->input[0]->shape[ndim-1];
    const float * input = op->input[0]->val;
    float * output = op->output->val;
    const lrn_op_t p = *(const lrn_op_t*)(op->params);
    int r = p.depth_radius;
    int row, i, j;
    float norm, norm_pow, ssum;
    const float alpha_beta_2 = -2.0*p.alpha*p.beta;

    for (row=0; row<size; row+=depth) {
        for (i=0; i<depth; ++i) {
            ssum = 0.0;
            for (j=max(i-r, 0); j<min(i+r+1, depth); ++j) {
                ssum += sqr(input[row+j]);
            }
            norm = p.bias + ssum*p.alpha;
            norm_pow = pow(norm, -p.beta);
            if (mode==RUN_FORWARD) {
                output[row+i] = input[row+i] * norm_pow;
            } else {
                float out_grad = op->output->grad[row+i];
                float activations_ab2 = output[row+i]*alpha_beta_2;
                for (j=max(i-r, 0); j<min(i+r+1, depth); ++j) {
                    float grad = input[row+j] * activations_ab2 / norm;
                    if (i==j) {
                        grad += norm_pow;
                    }
                    grad *= out_grad;
                    op->input[0]->grad[row+j] += grad;
                }
            }
        }
    }
}

void maxpool_func(const op_ref_t * op, run_mode_t mode) {
    const tensor_t * input = op->input[0];
    const tensor_t * output = op->output;
    const maxpool_op_t p = *(const maxpool_op_t*)(op->params);
    const int kh = p.ksize.y, kw = p.ksize.x;
    const int hi = input->shape[0], wi = input->shape[1], c = input->shape[2];
    const int ho = output->shape[0], wo = output->shape[1];
    int x, y, i, xi, yi, o_ofs=0;

    for (y=0, yi=p.stride.y-1-kh/2; y<ho; ++y, yi+=p.stride.y)
    for (x=0, xi=p.stride.x-1-kw/2; x<wo; ++x, xi+=p.stride.x)
    for (i=0; i<c; ++i, ++o_ofs) {
        int xk, yk, argmax=0;
        float v_max = -1e10;
        for (yk=max(yi, 0); yk<min(yi+kh, hi); ++yk)
        for (xk=max(xi, 0); xk<min(xi+kw, wi); ++xk) {
            int i_ofs = (yk*wi+xk)*c + i;
            float v = input->val[i_ofs];
            if (v > v_max) { 
                v_max = v;
                argmax = i_ofs;
            }
        }
        if (mode==RUN_FORWARD) {
            output->val[o_ofs] = v_max;
        } else {
            input->grad[argmax] += output->grad[o_ofs];
        }
    }
}

void concatv2_func(const op_ref_t * op, run_mode_t mode) {
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
            if (mode==RUN_FORWARD) {
                for (i=0; i<ci; ++i) {
                    output->val[d+i] = input->val[s+i];
                }
            } else {
                for (i=0; i<ci; ++i) {
                    input->grad[s+i] += output->grad[d+i];
                }
            }
            s += ci; d += co;
        }
        ofs += ci;
    }
}

void avgpool_func(const op_ref_t * op, run_mode_t mode) {
    const int depth = op->output->size;
    const int n = op->input[0]->size; 
    const float scale = 1.0/(n/depth);
    int i;
    if (mode==RUN_FORWARD) {
        const float * input = op->input[0]->val;
        float * output = op->output->val;
        for (i=0; i < n; ++i) {
            output[i%depth] += input[i]*scale;
        }
    } else {
        const float * out_grad = op->output->grad;
        float * in_grad = op->input[0]->grad;
        for (i=0; i < n; ++i) {
            in_grad[i] += out_grad[i%depth]*scale;
        }
    }
}

void matmul_func(const op_ref_t * op, run_mode_t mode) {
    conv2d_func(op, mode);
}

void softmax_func(const op_ref_t * op, run_mode_t mode) {
    const int size = op->output->size;
    int i;
    if (mode==RUN_FORWARD) {
        const float * input = op->input[0]->val;
        float * output = op->output->val;
        float max_logit = input[0], exp_sum=0.0;
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
    } else {
        /* grad_x = (grad_softmax - sum(grad_softmax * softmax)) * softmax */
        const float * output = op->output->val;
        const float * out_grad = op->output->grad;
        float * in_grad = op->input[0]->grad;
        float sum = 0.0;
        for (i=0; i<size; ++i) {
            sum += output[i]*out_grad[i];
        }
        for (i=0; i<size; ++i) {
            in_grad[i] += (out_grad[i] - sum) * output[i];
        }
    }
}


/********************************************************************************************/
/*                              Initialization and support                                  */
/********************************************************************************************/

void init_consts() {
    FILE *f=fopen(model_filename, "rb");
    const op_ref_t * op = g_ops;
    for (; op<g_ops+g_ops_num; ++op) {
        int i, j, k;
        const const_op_t * params;
        float * val;
        if (op->run != &const_func)
          continue;
        params = (const_op_t*) (op->params);
        val = op->output->val;
        fseek(f, params->ofs, SEEK_SET);
        for (i=0; i<op->output->size; ++i) {
          uint8 buf[FLOAT32_SIZE];
          if (fread(buf, FLOAT32_SIZE, 1, f) != 1)
            return;
          val[i] = parse_float32(buf);
        }
        if (op->output->ndim == 4) {
            /* compute tranposed kernel for backward pass and store it in 'grad' */
            float * grad = op->output->grad;
            const int h = op->output->shape[2];
            const int w = op->output->shape[3];
            for (int k=0; k<op->output->size; k += w*h) {
                for (i=0; i < h; ++i)
                for (j=0; j < w; ++j) {
                    grad[k + j*h + i] = val[k + i*w + j];
                }
            }
        }

    }
    fclose(f);
}

void fill_zeros(const tensor_t * t) {
    int i=0;
    for (; i<t->size; ++i) {
        t->val[i] = 0.0;
        t->grad[i] = 0.0;
    }
}


void write_data(const char *name, const int size, const float * data) {
    FILE *f = fopen(name, "wb");
    fwrite(data, sizeof(float), size, f);
    fclose(f);
}

void reset_buffers() {
    const op_ref_t * op = g_ops;
    for (; op != g_ops+g_ops_num; ++op) {
        if (op->run != const_func && op->run != placeholder_func) { 
            fill_zeros(op->output);
        }
    }
}

void forward(int target) {
    int i;
    reset_buffers();
    for (i=0; i<=target; ++i) {
        g_ops[i].run(g_ops+i, RUN_FORWARD);
    }
}

void backward(int target) {
    int i;
    for (i=target; i>=0; --i) {
      g_ops[i].run(g_ops+i, RUN_BACKWARD);
    }
}

int validate_buffer(const char *fn, const int size, const float * buf) {
    int i, err=0;
    float v, acc_v=0.0, max_dv=0.0;
    uint8 tmp[FLOAT32_SIZE];
    FILE *f = fopen(fn, "rb");
    if (f == NULL)
        return err; /* skip if file doesn't exist */
    for (i=0; i<size; ++i) {
          if (fread(tmp, FLOAT32_SIZE, 1, f) != 1) {
              printf("tensor is larger than expected!");
              err = 1;
              break;
          }
          v = parse_float32(tmp);
          acc_v += fabsf(v);
          max_dv = maxf(max_dv, fabsf(v-buf[i]));
    }
    if (i == size) {
        acc_v /= size;
        v = max_dv/acc_v;
        printf("relerr: %.3e, abserr: %.3e", v, max_dv);
        if (fread(tmp, FLOAT32_SIZE, 1, f) != 0) {
            printf(", tensor is smaller than expected! ");
            err = 1;
        }
        if (v>1e-3) {
            printf(", error is large! ");
            err = 1;
        }
    }
    fclose(f);
    printf(" --- %s\n", fn);
    return err;
}

const char * testdata_fn = "test/data";

void print_top_scores() {
    const int top_n = 5;
    float score[top_n+1];
    int index[top_n+1];
    int i, j, k;
    for (i=0; i<top_n; ++i) {
        score[i] = -1e10;
    }
    for (i=0; i<g_prob.size; ++i) {
        score[top_n] = g_prob.val[i];
        index[top_n] = i;
        for (j=top_n-1; j>=0; --j) {
            float v;
            if (score[j] >= score[j+1]) {
                break;
            }
            v = score[j]; score[j] = score[j+1]; score[j+1] = v;
            k = index[j]; index[j] = index[j+1]; index[j+1] = k;
        }
    }
    printf("\nTop %d scores:\n", top_n);
    for (i=0; i<top_n; ++i) {
        j = index[i];
        printf("  %.3f %d %s\n", score[i], j, pred_labels[j]);
    }
    printf("\n");
}

void validate_model() {
    int i;
    char fn[1024];
    const op_ref_t * op;

    printf("validating the model\n");
    FILE * f = fopen(testdata_fn, "rb");
    if (f == NULL) {
        printf("undable to load %s!", testdata_fn);
        return;
    }
    fread(g_data.val, FLOAT32_SIZE, g_data.size, f);
    fclose(f);

    reset_buffers();

    for (i=0; i<g_ops_num; ++i) {
        op = g_ops+i;
        op->run(op, RUN_FORWARD);
        sprintf(fn, "test/%s", op->name);
        if (validate_buffer(fn, op->output->size, op->output->val) != 0) {
            write_data(fn+5, op->output->size, op->output->val);
            return;
        }
    }
    print_top_scores();
    g_prob.grad[162] = 1.0;

    for (i=g_ops_num-1; i>=0; --i) {
        op = g_ops+i;
        sprintf(fn, "test/grad_%s", op->name);
        if (validate_buffer(fn, op->output->size, op->output->grad) != 0) {
            write_data(fn+5, op->output->size, op->output->grad);
            return;
        }
        op->run(op, RUN_BACKWARD);
    }

}


int main(int arvc, const char * argv[]) {
    int i=0;
  
    init_float32_exp();
    init_consts();

    validate_model();

    for (int i=0; i<10; ++i) {
        forward(g_ops_num-1);
        backward(g_ops_num-1);
    }

    for (i=0; i<arvc; ++i) {
        printf("%s\n", argv[i]);
    }

    return 0;
}