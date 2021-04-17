import numpy as np
import tensorflow as tf

# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

model_fn = 'InceptionV1.pb'

graph_def_buf = open(model_fn, 'rb').read()
graph_def = tf.compat.v1.GraphDef.FromString(graph_def_buf)
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')

DT_FLOAT = 1
FLOAT_SIZE = 4



total_mem = 0
last_const_ofs = 0
aliases = {}
tensors = []
op_list = []

def get_cname(s):
    s = s.split(':')[0]
    return 'g_'+s.replace('/', '_')

def resolve(s):
    while s in aliases:
        s = aliases[s]
    return s

def make_tensor(tensor, is_const):
    global total_mem
    if tensor.dtype != tf.float32:
        return False
    shape = tensor.shape
    if not is_const:
        shape = shape[1:]    # strip batch dim
        if len(shape) == 1:  # unify matmul and conv2d
            shape = (1, 1) + shape
    else:
        if len(shape) == 2:  # unify matmul and conv2d
            shape = (1, 1) + shape
    ndim = len(shape)
    name = get_cname(tensor.name)
    #mul_shape_str = '*'.join(map(str, shape))
    shape_str = '{'+','.join(map(str, shape))+'}'
    #tensors.append(f'\nfloat {name}_val[{mul_shape_str}];\n')
    size = np.prod(shape)
    val_ptr = f"g_mem+{total_mem}"
    total_mem += size
    grad_ptr = 'NULL'
    if True:  # not is_const:
        grad_ptr = f"g_mem+{total_mem}"
        total_mem += size
        #grad = name+'_grad'
        #tensors.append(f'float {grad}[{mul_shape_str}];\n')
    tensors.append(
        f'const tensor_t {name} = {{ {val_ptr}, {grad_ptr}, {size}, {ndim}, {shape_str} }};\n')
    return True

export_op_types = 'Placeholder Const Conv2D BiasAdd Relu LRN ConcatV2 MaxPool AvgPool MatMul Softmax'.split()

for op in graph.get_operations():
    name = get_cname(op.name)
    output_name = get_cname(op.outputs[0].name)
    inputs = [resolve(get_cname(t.name)) for t in op.inputs]
    if op.type in ['ConcatV2', 'Reshape']:
        inputs = inputs[:-1]  # drop last input (axis or shape)
    if op.type in ['Identity', 'Reshape', 'Relu', 'BiasAdd']:  # in-place ops
        aliases[output_name] = inputs[0]
        output_name = resolve(output_name)
    else:
        if not make_tensor(op.outputs[0], op.type == 'Const'):
            continue  # unsupported tensor type, skipping

    if op.type == 'Identity':
        continue
    params = 'NULL'
    const_ofs = -1
    if op.type == 'Const':
        data = op.node_def.attr['value'].tensor.tensor_content
        const_ofs = graph_def_buf.find(data, last_const_ofs)
        last_const_ofs = const_ofs + len(data)
        #tensors.append(f'const const_op_t {name}_op = {{ /*ofs*/ {ofs} }};\n')
        #params = '&'+name+'_op'
    elif op.type == 'Conv2D':
        _, sy, sx, _ = op.node_def.attr['strides'].list.i
        if sx != 1 or sy != 1:
            tensors.append(
                f'const conv_op_t {name}_op = {{ /*stride*/ {{{sx}, {sy}}} }};\n')
            params = '&'+name+'_op'
    elif op.type == 'LRN':
        attr = op.node_def.attr
        depth_radius = attr['depth_radius'].i
        alpha, beta, bias = attr['alpha'].f, attr['beta'].f, attr['bias'].f
        assert np.abs(beta-0.75) < 1e-8
        tensors.append(f'const lrn_op_t {name}_op = {{ /*depth_radius*/ {depth_radius}, ' +
                f'/*alpha*/ {alpha}, /*beta*/ {beta}, /*bias*/ {bias} }};\n')
        params = '&'+name+'_op'
    elif op.type == 'MaxPool':
        _, sy, sx, _ = op.node_def.attr['strides'].list.i
        _, ky, kx, _ = op.node_def.attr['ksize'].list.i
        tensors.append(
            f'const maxpool_op_t {name}_op = {{ /*stride*/ {{{sx}, {sy}}}, /*ksize*/ {{{kx}, {ky}}} }};\n')
        params = '&'+name+'_op'

    if op.type in export_op_types:
        inputs = ', '.join(['&'+s for s in inputs]) if inputs else 'NULL'
        op_list.append(
            f' {{ &{op.type.lower()}_func, {params}, "{name[2:]}", &{output_name}, {{ {inputs} }}, {const_ofs} }},\n')


with open('inception.inc', 'w') as f:
    f.write(f'const char * model_filename = "{model_fn}";\n\n')

    f.write(f'float g_mem[{total_mem}];\n\n');

    f.writelines(tensors)

    f.write('\n')
    for op in export_op_types:
        f.write(
            f'void {op.lower()}_func(const op_ref_t * op, run_mode_t mode);\n')
    f.write(f'\nconst int g_ops_num = {len(op_list)};\n')
    f.write('const op_ref_t g_ops[] = {\n')
    f.writelines(op_list)
    f.write('};\n')
    labels = open('ImageNet_standard.txt').read().split('\n')[:-1]
    f.write('\nenum {CLASS_N = %d};\n'%len(labels))
    f.write('const char * pred_labels[] = {\n')
    labels = open('ImageNet_standard.txt').read().split('\n')[:-1]
    for s in labels:
        f.write(f'  "{s}",\n')
    f.write('};\n')

print('total_mem', total_mem*FLOAT_SIZE/2**20)


# del graph_def.node[0]  # placeholder

# @tf.function
# def run_model(x):
#     prob = tf.import_graph_def(graph_def, {'data':x}, ['prob:0'])[0]
#     obj = prob[0, 162]
#     grad = tf.gradients(obj, x)
#     return prob, grad

# x = np.float32(np.random.rand(1, 224, 224, 3))
# obj, g = run_model(x)
# print(g)

# import time
# start = time.time()
# n = 100
# for i in range(n):
#     run_model(x)
# print((time.time()-start)/n*1000)

