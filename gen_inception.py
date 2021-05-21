import os
import sys
import numpy as np
import tensorflow as tf

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
consts = []
op_list = []

def get_cname(s):
    s = s.split(':')[0]
    chunks = s.split('/')
    if len(chunks) == 2 and chunks[0] == chunks[1]:
        chunks = chunks[:1]
    return 'g_'+'_'.join(chunks)

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
        shape = shape[1:]  # remove batch dimension (we assume it's always 1)
        if len(shape) == 1:
            # adding dummy spatial dimensions to matmul
            # inputs and outputs to unify them with conv2d
            shape = (1, 1) + shape
    else:
        if len(shape) == 2:
            # 2d weight matrices are used by 'matmul' layers, we add
            # dummy filter sizes to unify them with 'conv2d' layers
            shape = (1, 1) + shape
    ndim = len(shape)
    name = get_cname(tensor.name)
    shape_str = '{'+','.join(map(str, shape))+'}'
    size = np.prod(shape)
    val_ptr = f"g_mem+{total_mem}"
    total_mem += size
    grad_ptr = f"g_mem+{total_mem}"
    total_mem += size
    tensors.append(
        f'const tensor_t {name} = {{ {val_ptr}, {grad_ptr}, {size}, {ndim}, {shape_str} }};\n')
    return True

export_op_types = 'Placeholder Conv2D BiasAdd Relu LRN ConcatV2 MaxPool AvgPool MatMul Softmax'.split()

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
    if op.type == 'Const':
        data = op.node_def.attr['value'].tensor.tensor_content
        const_ofs = graph_def_buf.find(data, last_const_ofs)
        last_const_ofs = const_ofs + len(data)
        consts.append(f'  {{ &{output_name}, {const_ofs} }},\n')
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
            f' {{ &{op.type.lower()}_func, {params}, "{name[2:]}", &{output_name}, {{ {inputs} }} }},\n')


with open('inception.inc', 'w') as f:
    f.write(f'const char * model_filename = "{model_fn}";\n\n')

    f.write(f'float g_mem[{total_mem}];\n\n');

    f.writelines(tensors)
    f.write('\nenum { g_consts_num = %d };\n'%len(consts))
    f.write('const const_t g_consts[] = {\n')
    f.writelines(consts)
    f.write('};\n')

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

print('generated "inception.inc"')
print('%.1f MB memory used for networ'%(total_mem*FLOAT_SIZE/2**20))

if len(sys.argv) > 1 and sys.argv[1] == 'test':
    print('\ngenerating test data...')
    os.makedirs('test', exist_ok=True)

    bmp_data = open('cat_dog224.bmp', 'rb').read()
    input_image = tf.image.decode_bmp(bmp_data)
    input_image = tf.cast(input_image, tf.float32)

    tensors = [n.name+':0' for n in graph_def.node if len(n.input)]

    imagenet_mean = np.float32([122.7, 116.7, 104.0])

    @tf.function
    def run_model(x, target_label=162):
        data = (x-imagenet_mean)[None,:,:,::-1] # rgb->bgr
        data = tf.identity(data, 'data')
        outputs = tf.import_graph_def(graph_def, {'data':data}, tensors, name='')
        outputs = dict(zip(tensors, outputs))
        objective = outputs['prob:0'][0, target_label]

        grad_tensors = [t.op.inputs[0] for t in outputs.values() if t.op.inputs[0].op.type != 'Reshape']
        grads = tf.gradients(objective, grad_tensors)
        grads = {'grad_'+t.name:g for t, g in zip(grad_tensors, grads)}
        outputs.update(grads)

        outputs['data'] = data
        outputs = {get_cname(k)[2:]:v for k, v in outputs.items()}
        return outputs

    outputs = run_model(input_image)
    for name, v in outputs.items():
        if v is not None:
            v.numpy().tofile('test/'+name)