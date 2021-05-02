/********************************************************************************************


We didn't have Tensorflow, PyTorch, JAX, Caffe or any other differentiable
programming framework. We didn't have BLAS and any library for image loading,
saving and resizing. We refrained from using dynamic memory allocation and
ditched the assumption that the CPU uses IEEE-754 standard to represent
floating-point numbers. We restricted the code to the subset of C89
language, excluding preprocessor macros and many other things. Our only
includes were <stdio.h> and <math.h>. We made sure that the program
takes less than a second to compile with any compiler we tried.
Not that it all was needed to make the program portable and easy to build,
but when you start removing dependencies, the tendency is to push it
as far as you can.


********************************************************************************************/

#include <stdio.h>  /* (s)printf fopen fclose fseek fread fwrite fflush */


/****************************** Neural Net data strucutres **********************************/
/*               Minimal set of definitions required for "inception.inc"                    */
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


typedef struct {
    const tensor_t * data; 
    int ofs;  /* weights offset in .pb file */
} const_t; 
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

int parse_int16(const uint8 *p) {
    return p[0] + (p[1]<<8);
}

int parse_int32(const uint8 *p) {
    return p[0] + (p[1]<<8) + (p[2]<<16) + (p[3]<<24);
}

void store_int32(uint8 *p, int v) {
    p[0] = v&0xff;
    p[1] = (v>>8)&0xff;
    p[2] = (v>>16)&0xff;
    p[3] = (v>>24)&0xff;
}

void fill_zeros(int n, float * a) {
    int i=0;
    for (; i<n; ++i) {
        a[i] = 0.0;
    }
}

/*************** IEEE-754 32-bit float parsing (little-endian) ****************/
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
    float32_exp_table[255] = 1.0/0.0;            /* infinity */
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
float clipf(float x, float a, float b) { return maxf(a, minf(x, b)); }

int min(int a, int b) { return a<b ? a : b; }
int max(int a, int b) { return a>b ? a : b; }

float sqr(float x) { return x*x; }
float absf(float x) { return x>=0.0 ? x : -x; }

float sign(float v) {
    if (v>0.0) { return 1.0;}
    else if (v<0.0) { return -1.0; }
    return 0.0;
}

const float LOG_2 = 0.6931471805599453f;

float expf(float x) {
    float x2, exp, y, s;
    int i;
    if (x>0.0) {
        return 1.0/expf(-x);
    }
    x2 = x / LOG_2;
    i = (int)x2;
    if (127+i < 1) {
        /* we could have used subnormals for very small numbers, but this
        doesn't seem to make meaningful difference here */
        return 0.0f; 
    }
    /* our trusty float exp2 table happens to be useful */
    exp = float32_exp_table[127+i];
    /* Taylor series */
    x = (x2-i)*LOG_2; /*x in (-LOG_2, 0] */
    y = x;
    s = y;
    for (i=2; i<10; ++i) {
        y *= x/i;
        s += y;
    }
    return exp*(1.0+s);
}

/********************************************************************************************/
/*                              Neural Net layer functions                                  */
/********************************************************************************************/

void placeholder_func(const op_ref_t * op, run_mode_t mode) {}

/* Matrix-vector product function: y += xA, where A[n][m], x[n], y[m].

Let's take a moment and appreciate this little function. Our program is going to
spend 80%-90% of time executing this code. Most of what we call Machine Learning,
Artificial Intelligence, Computational Physics, Numerical Simulation, etc. boils
down to Linear Algebra and Matrix Multiplications. It's not surprising that plenty
engineering efforts went into making this code run as fast as possible. Real world
implementations of matrix-matrix and matrix-vector product functions are much more
sophisticated than the one below. They use cache-optimal memory layouts,
multiprocessing, special CPU instructions and dedicated hardware units. */

void mul_matvec(float *y, const float *x, const float *A, int n, int m) {
    int i, j;
    for (i=0; i<n; ++i)
    for (j=0; j<m; ++j) {
        y[j] += x[i] * A[i*m + j];
    }
}

void conv2d_func(const op_ref_t * op, run_mode_t mode) {
    const tensor_t * input = op->input[0];
    const tensor_t * kernel = op->input[1]; /* kh, kw, ci, co*/
    const tensor_t * output = op->output;
    const int kh = kernel->shape[0], kw = kernel->shape[1];
    const int ci = kernel->shape[2], co = kernel->shape[3];
    const int h = op->input[0]->shape[0], w = op->input[0]->shape[1];
    const int wo = op->output->shape[1];
    ivec2 stride = {1, 1};
    const float * kmat = (mode == RUN_FORWARD) ? kernel->val : kernel->grad;
    int x, y, sx, sy;
    if (op->params != NULL) {
        stride = ((conv_op_t *)op->params)->stride;
    }
    if (mode == RUN_FORWARD) {
        fill_zeros(output->size, output->val);
    }

    for (sy=-kh/2; sy<kh-kh/2; ++sy)
    for (sx=-kw/2; sx<kw-kw/2; ++sx, kmat += ci*co)
    for (y=max(0, sy); y<min(h, h+sy); ++y) if ((y-sy+1)%stride.y==0)
    for (x=max(0, sx); x<min(w, w+sx); ++x) if ((x-sx+1)%stride.x==0) {
        int o = ((y-sy)/stride.y*wo+(x-sx)/stride.x)*co;
        int i = (y*w+x)*ci;
        if (mode == RUN_FORWARD) {
            mul_matvec(output->val+o, input->val+i, kmat, ci, co);
        } else {
            mul_matvec(input->grad+i, output->grad+o, kmat, co, ci);
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
    } /* do nothing on the backward pass */
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

float root4(float y) {
    float x = 1.0;
    int i = 0;
    if (y>1.0)  { return 1.0f/root4(1.0f/y); }
    if (y==0.0) { return 0.0; }
    for (; i<8; ++i) {
        x = 0.75f*x + y/(4.0f*x*x*x);     
    }
    return x;
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
            /* norm_pow = pow(norm, -p.beta), but beta==0.75 for Inception V1,
            so I use a custom root4 function to avoid depending on math.h */
            norm_pow = root4(norm)/norm;
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
        fill_zeros(depth, output);
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
            float e = expf(input[i]-max_logit);
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
    const const_t * c = g_consts;
    for (; c<g_consts+g_consts_num; ++c) {
        int i, j, k;
        float * val = c->data->val;
        fseek(f, c->ofs, SEEK_SET);
        for (i=0; i<c->data->size; ++i) {
          uint8 buf[FLOAT32_SIZE];
          if (fread(buf, FLOAT32_SIZE, 1, f) != 1)
            return;
          val[i] = parse_float32(buf);
        }
        if (c->data->ndim == 4) {
            /* compute tranposed kernel for backward pass and store it in 'grad' */
            float * grad = c->data->grad;
            const int h = c->data->shape[2];
            const int w = c->data->shape[3];
            for (k=0; k<c->data->size; k += w*h) {
                for (i=0; i < h; ++i)
                for (j=0; j < w; ++j) {
                    grad[k + j*h + i] = val[k + i*w + j];
                }
            }
        }
    }
    fclose(f);
}


void write_data(const char *name, const int size, const float * data) {
    FILE *f = fopen(name, "wb");
    fwrite(data, sizeof(float), size, f);
    fclose(f);
}

void reset_gradients() {
    const op_ref_t * op = g_ops;
    for (; op != g_ops+g_ops_num; ++op) {
        fill_zeros(op->output->size, op->output->grad);
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
          acc_v += absf(v);
          max_dv = maxf(max_dv, absf(v-buf[i]));
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

void print_top_scores() {
    enum { top_n = 5 };
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
    reset_gradients();
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


/********************************************************************************************/
/*                                  Image manipulation                                      */
/********************************************************************************************/
enum { 
    MAX_IMAGE_SIZE  = 4096,
    BMP_HEADER_SIZE = 14+40
};

uint8 g_img[MAX_IMAGE_SIZE*MAX_IMAGE_SIZE][3];  /* BGR image data */
int g_img_width, g_img_height;
const float imagenet_mean_bgr[3] = {104.0, 116.7, 122.7};


int load_bmp(const char * fn) {
    FILE *f = fopen(fn, "rb");
    uint8 buf[BMP_HEADER_SIZE];
    int i, r, image_offset, width, height, bpp, row_padding;
    int ofs, y_step, y;
    if (f == NULL) {
        printf("ERROR: unable to open '%s'\n", fn);
        return 1;
    }
    r = fread(buf, BMP_HEADER_SIZE, 1, f);
    if (r != 1 || buf[0] != 'B' || buf[1] != 'M') {
        printf("ERROR: BMP header parsing error\n");
        fclose(f);
        return 1;
    }
    image_offset = parse_int32(buf+10);
    width = parse_int32(buf+18);
    height = parse_int32(buf+22);
    bpp = parse_int16(buf+28);
    if (max(width, height) > MAX_IMAGE_SIZE || bpp != 24) {
        printf("ERROR: only 24-bit RGB images, max(width, height) <= %d supported\n", MAX_IMAGE_SIZE);
        fclose(f);
        return 1;
    }
    row_padding = width % 4;
    if (height < 0) {  /* is top-bottom row order? */
        height = -height;
        ofs = 0;
        y_step = width;
    } else {
        ofs = (height-1)*width;
        y_step = -width;
    }
    fseek(f, image_offset, SEEK_SET);
    for (y=0; y<height; ++y, ofs += y_step) {
        if (fread(g_img + ofs, width*3, 1, f) != 1) {
            printf("ERROR: image data loading error\n");
            fclose(f);
            return 1;
        }
        fread(buf, row_padding, 1, f);
    }
    fclose(f);
    g_img_width = width;
    g_img_height = height;
    printf("loaded '%s' (%dx%d)\n", fn, width, height);
    return 0;
}

int save_bmp(const char * fn) {
    uint8 header[BMP_HEADER_SIZE] = {
         66, 77, 54, 76,  2,  0,  0,  0,  0,  0, 54,  0,  0,  0, 40,  0,  0,  0,
        224,  0,  0,  0,224,  0,  0,  0,  1,  0, 24,  0,  0,  0,  0,  0,  0, 76,
          2,  0,196, 14,  0,  0,196, 14,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0
    };
    const uint8 zero[4] = {0, 0, 0, 0};
    int y, padding;
    FILE *f = fopen(fn, "wb");
    if (f == NULL) {
        printf("ERROR: unable to open '%s' for writing\n", fn);
        return 1;
    }
    padding = g_img_width % 4;
    store_int32(header+18, g_img_width);
    store_int32(header+22, g_img_height);
    store_int32(header+34, (g_img_width*3+padding)*g_img_height);
    fwrite(header, BMP_HEADER_SIZE, 1, f);
    for (y=g_img_height-1; y>=0; y -= 1) {
        fwrite(g_img + y*g_img_width, 1, g_img_width*3, f);
        if (padding) {
            fwrite(zero, 1, padding, f);
        }
    }
    fclose(f);
    return 0;
}

void copy_img2net(int ofs_x, int ofs_y) {
    const int tile_size = g_data.shape[0];
    int x, y, c, dst = 0;
    for (y=0; y<tile_size; ++y)
    for (x=0; x<tile_size; ++x) {
        int src = (ofs_y+y)%g_img_height * g_img_width + (ofs_x+x)%g_img_width;
        for (c=0; c<3; ++c, ++dst ) {
            g_data.val[dst] = g_img[src][c]-imagenet_mean_bgr[c];
        }
    }
}

void copy_net2img(int ofs_x, int ofs_y) {
    const int tile_size = g_data.shape[0];
    int x, y, c, src, dst;
    for (y=0; y<min(tile_size, g_img_height); ++y)
    for (x=0; x<min(tile_size, g_img_width); ++x) {
        int src = (y*tile_size + x)*3;
        int dst = (ofs_y+y)%g_img_height * g_img_width + (ofs_x+x)%g_img_width;
        for (c=0; c<3; ++c ) {
            g_img[dst][c] = clipf(g_data.val[src+c]+imagenet_mean_bgr[c]+0.5, 0.0, 255.0);
        }
    }
}

void print_bar(const char * c, int layer_i, int target) {
    const int skip = 10;
    int k;
    layer_i /= skip;
    printf("\r[");
    for (k=0; k<target/skip; ++k) {
        printf("%s", k==layer_i?c:"-");
    }
    printf("]");
    fflush(stdout);
}

void forward(int target) {
    int i;
    for (i=0; i<=target; ++i) {
        if (i%1==0) {
            print_bar(">", i, target);
        }
        g_ops[i].run(g_ops+i, RUN_FORWARD);
    }
}

void backward(int target) {
    int i;
    for (i=target; i>=0; --i) {
        if (i%1==0) {
            print_bar("<", i, target);
        }
        g_ops[i].run(g_ops+i, RUN_BACKWARD);
    }
}

void run_adversarial() {
    int pass, i;
    copy_img2net(0, 0);
    forward(g_ops_num-1);
    print_top_scores();

    for (pass=0; pass<5; ++pass) {
        g_prob.grad[852] = 1.0;  /* tennis ball */
        backward(g_ops_num-1);
        for (i=0; i<g_data.size; ++i) {
            g_data.val[i] += sign(g_data.grad[i])*0.5;
        }
        forward(g_ops_num-1);
        reset_gradients();
        print_top_scores();
    }
    copy_net2img(0, 0);
    save_bmp("t.bmp");
}

int find_layer(const char * name) {
    int i=0;
    for (; i<g_ops_num; ++i) {
        if (g_ops[i].name == name) {
            return i;
        }
    }
    return -1;
}

void sample_bilinear(int x, int y, uint8 * dst) {
    int u = x&0xff, v = y&0xff;
    int w = g_img_width, o, c;
    x = x>>8; y = y>>8;
    o = y*w+x;
    for (c=0; c<3; ++c) {
        uint8 a = (g_img[o][c]*(255-u) + g_img[o+1][c]*u)>>8;
        uint8 b = (g_img[o+w][c]*(255-u) + g_img[o+w+1][c]*u)>>8;
        dst[c] = (a*(255-v) + b*v)>>8;
    }
}

void upscale_image() {
    int new_width = g_img_width*14/10;
    int new_height = g_img_height*14/10;
    int x, y;
    for (y=new_height-1; y>=0; --y) {
        int sy = (y<<8)*10/14;
        for (x=new_width-1;  x>=0; --x) {
            int sx = (x<<8)*10/14;
            sample_bilinear(sx, sy, g_img[y*new_width + x]);
        }
    }
    g_img_width = new_width;
    g_img_height = new_height;
}

void render_octave(int target_i) {
    const int tile_size = g_data.shape[0];
    const int step_n = 20;
    const tensor_t * target = g_ops[target_i].output;
    float acc;
    int x, y, i, step;
    int tile_h = (g_img_height+tile_size-1)/tile_size;
    int tile_w = (g_img_width+tile_size-1)/tile_size;
    int tile_n=tile_w*tile_w;
    for (step=0; step<step_n; ++step) {
        int sx=step*79, sy=step*127;
        int tile_count=0;
        for (y=0; y<g_img_height; y+=tile_size)
        for (x=0; x<g_img_width; x+=tile_size, ++tile_count) {
            copy_img2net(x+sx, y+sy);
            forward(target_i);
            reset_gradients();
            for (i=0; i<target->size; ++i) {
                target->grad[i] = target->val[i];
            }
            backward(target_i);
            acc = 0.0;
            for (i=0; i<g_data.size; ++i) {
                acc += absf(g_data.grad[i]);
            }
            acc /= g_data.size;
            for (i=0; i<g_data.size; ++i) {
                g_data.val[i] += 1.5*g_data.grad[i]/acc;
            }
            copy_net2img(x+sx, y+sy);
            printf(" %d/%d %d/%d ", tile_count+1, tile_n, step+1, step_n);
        }
    }
}

void run_deepdream(int octave_n) {
    int target_i = find_layer("inception_4c_output"); /*pool4 4c*/ // pool3_3x3_s2 pool3_3x3_s2
    int octave;
    char s[128];
    for (octave=0; octave<octave_n; ++octave) {
        if (octave > 0) {
            upscale_image();
        }
        render_octave(target_i);
        sprintf(s, "out%d.bmp", octave);
        save_bmp(s);
        printf("saved %dx%d\n", g_img_width, g_img_height);
    }
}

int main(int arvc, const char * argv[]) {
    int x, y, c, i, pass;
    const int data_w = 224;
    FILE * f;
  
    init_float32_exp();
    init_consts();

    if (load_bmp("cat_dog224.bmp") != 0) {
        return 1;
    }
    //copy_img2net(0, 0);
    //validate_model();
    
    //run_adversarial();
    run_deepdream(7);


    /*printf("\033[2J\033[H");*/

    return 0;
}
