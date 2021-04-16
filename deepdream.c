#include <stdio.h>  /* (s)printf fopen fclose fseek fread fwrite fflush */
#include <math.h>   /* TODO: remove */

/****************************** Neural Net data strucutres **********************************/
/*                  Minimal set of definitions used in "inception.inc"                      */
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
float clipf(float x, float a, float b) { return maxf(a, minf(x, b)); }

int min(int a, int b) { return a<b ? a : b; }
int max(int a, int b) { return a>b ? a : b; }

float sqr(float v) { return v*v; }

float sign(float v) {
    if (v>0.0) { return 1.0;}
    else if (v<0.0) { return -1.0; }
    return 0.0;
}




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
    const float * kmat = (mode == RUN_FORWARD) ? kernel->val : kernel->grad;
    int x, y, sx, sy, i, o;
    if (op->params != NULL) {
        stride = ((conv_op_t *)op->params)->stride;
    }

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

enum {MAX_IMAGE_SIZE = 4096};

uint8 g_img_data[MAX_IMAGE_SIZE*MAX_IMAGE_SIZE*3];  /* BGR image data */
int g_img_width, g_img_height;

int parse_int16(const uint8 *p) {
    return p[0] + (p[1]<<8);
}

int parse_int32(const uint8 *p) {
    return p[0] + (p[1]<<8) + (p[2]<<16) + (p[2]<<24);
}

void store_int32(uint8 *p, int v) {
    p[0] = v&0xff;
    p[1] = (v>>8)&0xff;
    p[2] = (v>>16)&0xff;
    p[3] = (v>>24)&0xff;
}


enum { BMP_HEADER_SIZE=14+40 };

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
        y_step = width*3;
    } else {
        ofs = (height-1)*width*3;
        y_step = -width*3;
    }
    fseek(f, image_offset, SEEK_SET);
    for (y=0; y<height; ++y, ofs += y_step) {
        if (fread(g_img_data + ofs, width*3, 1, f) != 1) {
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
        fwrite(g_img_data + y*g_img_width*3, 1, g_img_width*3, f);
        fwrite(zero, 1, padding, f);
    }
    fclose(f);
    return 0;
}

void print_bar(const char * c, int i, int target) {
    const int skip = 10;
    int k;
    i /= skip;
    printf("\r[");
    for (k=0; k<target/skip; ++k) {
        printf("%s", k==i?c:"-");
    }
    printf("]");
    fflush(stdout);
}

void forward(int target) {
    int i;
    reset_buffers();
    for (i=0; i<=target; ++i) {
        if (i%10==0) {
            print_bar(">", i, target);
        }
        g_ops[i].run(g_ops+i, RUN_FORWARD);
    }
}

void backward(int target) {
    int i;
    for (i=target; i>=0; --i) {
        if (i%10==0) {
            print_bar("<", i, target);
        }
        g_ops[i].run(g_ops+i, RUN_BACKWARD);
    }
}



const float imagenet_mean_bgr[3] = {104.0, 116.7, 122.7};

int main(int arvc, const char * argv[]) {
    int x, y, c, i;
    const int data_w = 224;
    FILE * f;
  
    init_float32_exp();
    init_consts();

    if (load_bmp("cat_dog224.bmp") != 0) {
        return 1;
    }

    f = fopen("t.dat", "wb");
    fwrite(g_img_data, 1, g_img_height*g_img_width*3, f);
    fclose(f);

    for (y=0; y<min(data_w, g_img_height); ++y)
    for (x=0; x<min(data_w, g_img_width); ++x)
    for (c=0; c<3; ++c ){
        float v = g_img_data[(y*g_img_width+x)*3+c];
        v -= imagenet_mean_bgr[c];
        g_data.val[(y*data_w + x)*3+c] = v;
    }

    //validate_model();
    forward(g_ops_num-1);
    print_top_scores();

    for (c=0; c<5; ++c) {
        g_prob.grad[852] = 1.0; /* tennis ball */
        backward(g_ops_num-1);
        for (i=0; i<g_data.size; ++i) {
            g_data.val[i] += sign(g_data.grad[i])*0.5;
        }
        reset_buffers();
        forward(g_ops_num-1);
        print_top_scores();
    }

    for (y=0; y<min(data_w, g_img_height); ++y)
    for (x=0; x<min(data_w, g_img_width); ++x)
    for (c=0; c<3; ++c ){
        float v = g_data.val[(y*data_w + x)*3+c];
        v += imagenet_mean_bgr[c];
        g_img_data[(y*g_img_width+x)*3+c] = clipf(v, 0.0, 255.0);
    }    

    save_bmp("t.bmp");

    // for (int i=0; i<1; ++i) {
    //     forward(g_ops_num-1);
    //     backward(g_ops_num-1);
    // }

    // for (i=0; i<arvc; ++i) {
    //     printf("%s\n", argv[i]);
    // }

    return 0;
}